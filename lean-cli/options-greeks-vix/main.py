# region imports
from AlgorithmImports import *
from datetime import timedelta
# endregion


class OptionsGreeksVix(QCAlgorithm):
    """
    Options lane v2 (paper-first):
    - regime-aware direction (calls in risk-on, puts in risk-off)
    - dynamic delta/DTE by volatility regime
    - liquidity + spread filters (OI/volume/bid-ask quality)
    - open-buffer timing discipline (avoid first N minutes)
    - explicit time stop + PnL stop/take-profit exits
    """

    def initialize(self):
        self.set_start_date(int(self.get_parameter("start_year") or 2021), 1, 1)
        self.set_end_date(int(self.get_parameter("end_year") or 2025), 12, 31)
        self.set_cash(float(self.get_parameter("initial_cash") or 100000))

        # Risk/execution controls
        self.max_contracts = int(self.get_parameter("max_contracts") or 2)
        self.cooldown_days = int(self.get_parameter("cooldown_days") or 3)
        self.take_profit = float(self.get_parameter("take_profit") or 0.30)
        self.stop_loss = float(self.get_parameter("stop_loss") or -0.16)
        self.max_holding_days = int(self.get_parameter("max_holding_days") or 6)
        self.max_alloc_pct = float(self.get_parameter("max_alloc_pct") or 0.02)
        self.min_option_price = float(self.get_parameter("min_option_price") or 0.35)
        self.open_buffer_minutes = int(self.get_parameter("open_buffer_minutes") or 30)
        self.daily_trade_limit = int(self.get_parameter("daily_trade_limit") or 1)

        # Microstructure / liquidity
        self.min_open_interest = int(self.get_parameter("min_open_interest") or 600)
        self.min_volume = int(self.get_parameter("min_volume") or 100)
        self.max_spread_pct = float(self.get_parameter("max_spread_pct") or 0.18)

        # Vol regime / contract targeting
        self.target_delta_low_vol = float(self.get_parameter("target_delta_low_vol") or 0.34)
        self.target_delta_high_vol = float(self.get_parameter("target_delta_high_vol") or 0.24)
        self.min_dte_low_vol = int(self.get_parameter("min_dte_low_vol") or 10)
        self.max_dte_low_vol = int(self.get_parameter("max_dte_low_vol") or 35)
        self.min_dte_high_vol = int(self.get_parameter("min_dte_high_vol") or 6)
        self.max_dte_high_vol = int(self.get_parameter("max_dte_high_vol") or 21)
        self.high_vol_atr_pct = float(self.get_parameter("high_vol_atr_pct") or 0.022)
        self.high_vol_vixy_ratio = float(self.get_parameter("high_vol_vixy_ratio") or 1.03)

        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.vixy = self.add_equity("VIXY", Resolution.DAILY).symbol

        self.spy_sma50 = self.sma(self.spy, 50, Resolution.DAILY)
        self.spy_sma200 = self.sma(self.spy, 200, Resolution.DAILY)
        self.vixy_sma20 = self.sma(self.vixy, 20, Resolution.DAILY)
        self.spy_atr20 = self.atr(self.spy, 20, MovingAverageType.WILDERS, Resolution.DAILY)

        opt = self.add_option("SPY", Resolution.MINUTE)
        opt.set_filter(lambda u: u.strikes(-12, 12).expiration(timedelta(days=4), timedelta(days=45)))
        self.option_symbol = opt.symbol

        self.open_option_symbols = set()
        self.open_meta = {}  # symbol -> {entry_time, entry_price}
        self.next_entry_time = self.start_date

        self.trades_today = 0
        self.current_day = None

        self.set_benchmark(self.spy)
        self.set_warm_up(timedelta(days=230))

    def _is_high_vol_regime(self) -> bool:
        if not (self.spy_atr20.is_ready and self.vixy_sma20.is_ready):
            return False

        spy_px = float(self.securities[self.spy].price)
        vixy_px = float(self.securities[self.vixy].price)
        if spy_px <= 0:
            return False

        atr_pct = float(self.spy_atr20.current.value) / spy_px
        vixy_ratio = vixy_px / float(self.vixy_sma20.current.value) if float(self.vixy_sma20.current.value) > 0 else 1.0
        return atr_pct >= self.high_vol_atr_pct or vixy_ratio >= self.high_vol_vixy_ratio

    def _risk_direction(self) -> OptionRight:
        if not (self.spy_sma50.is_ready and self.spy_sma200.is_ready):
            return OptionRight.CALL

        spy_px = float(self.securities[self.spy].price)
        uptrend = spy_px > float(self.spy_sma50.current.value) and float(self.spy_sma50.current.value) > float(self.spy_sma200.current.value)

        if uptrend and not self._is_high_vol_regime():
            return OptionRight.CALL
        return OptionRight.PUT

    def _time_window_ok(self) -> bool:
        # NY-equity clock assumptions in LEAN US equities context.
        minute_of_day = self.time.hour * 60 + self.time.minute
        open_min = 9 * 60 + 30
        close_guard = 15 * 60 + 55
        return (open_min + self.open_buffer_minutes) <= minute_of_day <= close_guard

    def _select_contract(self, chain: OptionChain, right: OptionRight, target_delta: float, min_dte: int, max_dte: int):
        best = None
        best_score = 1e9

        for c in chain:
            if c.right != right:
                continue

            dte = (c.expiry.date() - self.time.date()).days
            if dte < min_dte or dte > max_dte:
                continue

            bid = float(c.bid_price)
            ask = float(c.ask_price)
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            mid = (bid + ask) / 2.0
            if mid < self.min_option_price:
                continue

            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > self.max_spread_pct:
                continue

            vol = float(getattr(c, "volume", 0) or 0)
            oi = float(getattr(c, "open_interest", 0) or 0)
            if vol < self.min_volume or oi < self.min_open_interest:
                continue

            if c.greeks is None:
                continue
            delta = abs(float(c.greeks.delta))
            gamma = abs(float(c.greeks.gamma))
            if delta <= 0:
                continue

            # Reward tighter spreads/liquidity + target-delta fit + modest gamma.
            score = (
                abs(delta - target_delta)
                + 0.0015 * dte
                + 0.45 * spread_pct
                - 0.00001 * min(200000.0, vol + oi)
                - 0.08 * min(0.10, gamma)
            )

            if score < best_score:
                best_score = score
                best = c

        return best

    def _manage_open_positions(self):
        for sym in list(self.open_option_symbols):
            holding = self.portfolio[sym]
            if not holding.invested:
                self.open_option_symbols.discard(sym)
                self.open_meta.pop(sym, None)
                continue

            sec = self.securities[sym]
            px = float(sec.price)
            avg = float(holding.average_price)
            if avg <= 0:
                continue

            pnl = (px - avg) / avg
            expiry = sym.id.date
            dte = (expiry.date() - self.time.date()).days

            meta = self.open_meta.get(sym, {})
            entry_time = meta.get("entry_time", self.time)
            holding_days = max(0, (self.time.date() - entry_time.date()).days)

            exit_now = (
                dte <= 2
                or holding_days >= self.max_holding_days
                or pnl >= self.take_profit
                or pnl <= self.stop_loss
            )

            if exit_now:
                self.liquidate(sym, f"risk_exit dte={dte} days={holding_days} pnl={pnl:.2%}")
                self.open_option_symbols.discard(sym)
                self.open_meta.pop(sym, None)

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        if self.current_day != self.time.date():
            self.current_day = self.time.date()
            self.trades_today = 0

        self._manage_open_positions()

        if self.time < self.next_entry_time:
            return
        if self.trades_today >= self.daily_trade_limit:
            return
        if not self._time_window_ok():
            return

        # Keep single-leg exposure at a time for strict risk control.
        if any(self.portfolio[s].invested for s in self.open_option_symbols):
            return

        chain = data.option_chains.get(self.option_symbol)
        if chain is None:
            return

        high_vol = self._is_high_vol_regime()
        right = self._risk_direction()

        if high_vol:
            target_delta = self.target_delta_high_vol
            min_dte = self.min_dte_high_vol
            max_dte = self.max_dte_high_vol
        else:
            target_delta = self.target_delta_low_vol
            min_dte = self.min_dte_low_vol
            max_dte = self.max_dte_low_vol

        contract = self._select_contract(chain, right, target_delta, min_dte, max_dte)
        if contract is None:
            return

        ask = float(contract.ask_price)
        mark = ask if ask > 0 else float(contract.last_price)
        if mark <= 0:
            return

        # Volatility-aware allocation scaling.
        spy_px = float(self.securities[self.spy].price)
        atr_pct = float(self.spy_atr20.current.value) / spy_px if (self.spy_atr20.is_ready and spy_px > 0) else self.high_vol_atr_pct
        vol_scale = max(0.45, min(1.15, self.high_vol_atr_pct / max(0.005, atr_pct)))

        alloc_cash = max(0.0, float(self.portfolio.cash) * self.max_alloc_pct * vol_scale)
        qty = int(alloc_cash // (mark * 100.0))
        qty = max(1, min(qty, self.max_contracts))

        if (mark * 100.0 * qty) > float(self.portfolio.cash):
            return

        ticket = self.market_order(contract.symbol, qty, tag=f"entry right={right} high_vol={high_vol}")
        if ticket is not None:
            self.open_option_symbols.add(contract.symbol)
            self.open_meta[contract.symbol] = {
                "entry_time": self.time,
                "entry_price": mark,
            }
            self.trades_today += 1
            self.next_entry_time = self.time + timedelta(days=self.cooldown_days)
