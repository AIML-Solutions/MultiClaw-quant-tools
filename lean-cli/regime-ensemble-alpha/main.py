# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
# endregion


class RegimeEnsembleAlpha(QCAlgorithm):
    """
    Multi-factor allocator v2:
    - trend + multi-horizon momentum ensemble
    - volatility-normalized alpha with correlation penalty
    - regime-aware gross exposure
    - drawdown circuit-breaker + defensive floor
    - turnover-aware rebalancing
    """

    def initialize(self):
        self.set_start_date(int(self.get_parameter("start_year") or 2018), 1, 1)
        self.set_end_date(int(self.get_parameter("end_year") or 2025), 12, 31)
        self.set_cash(float(self.get_parameter("initial_cash") or 100000))

        self.max_weight = float(self.get_parameter("max_weight") or 0.16)
        self.rebalance_days = int(self.get_parameter("rebalance_days") or 1)
        self.max_drawdown = float(self.get_parameter("max_drawdown") or 0.11)
        self.min_rebalance_delta = float(self.get_parameter("min_rebalance_delta") or 0.012)
        self.defensive_floor = float(self.get_parameter("defensive_floor") or 0.12)

        self.mom_fast_w = float(self.get_parameter("mom_fast_weight") or 0.30)
        self.mom_mid_w = float(self.get_parameter("mom_mid_weight") or 0.35)
        self.mom_slow_w = float(self.get_parameter("mom_slow_weight") or 0.35)

        self.target_portfolio_vol = float(self.get_parameter("target_portfolio_vol") or 0.14)
        self.corr_penalty_weight = float(self.get_parameter("corr_penalty_weight") or 0.20)

        self.risk_assets = ["SPY", "QQQ", "IWM", "XLE", "XLF", "XLV", "EEM", "HYG"]
        self.def_assets = ["TLT", "GLD", "UUP"]
        self.regime_symbols = ["SPY", "VIXY", "HYG", "LQD"]

        self.symbols = {}
        all_tickers = list(dict.fromkeys(self.risk_assets + self.def_assets + self.regime_symbols))
        for t in all_tickers:
            sec = self.add_equity(t, Resolution.DAILY)
            self.symbols[t] = sec.symbol

        self.include_crypto = str(self.get_parameter("include_crypto") or "true").lower() in ("1", "true", "yes", "y")
        self.crypto_assets = []
        if self.include_crypto:
            for t in ["BTCUSD", "ETHUSD"]:
                sym = self.add_crypto(t, Resolution.DAILY, Market.BITFINEX).symbol
                self.symbols[t] = sym
                self.crypto_assets.append(t)

        self.mom21 = {}
        self.mom63 = {}
        self.mom126 = {}
        self.sma20 = {}
        self.sma100 = {}
        self.atr20 = {}

        for sym in self.symbols.values():
            self.mom21[sym] = self.roc(sym, 21, Resolution.DAILY)
            self.mom63[sym] = self.roc(sym, 63, Resolution.DAILY)
            self.mom126[sym] = self.roc(sym, 126, Resolution.DAILY)
            self.sma20[sym] = self.sma(sym, 20, Resolution.DAILY)
            self.sma100[sym] = self.sma(sym, 100, Resolution.DAILY)
            self.atr20[sym] = self.atr(sym, 20, MovingAverageType.SIMPLE, Resolution.DAILY)

        self.spy_sma200 = self.sma(self.symbols["SPY"], 200, Resolution.DAILY)
        self.vixy_sma30 = self.sma(self.symbols["VIXY"], 30, Resolution.DAILY)
        self.hyg_sma20 = self.sma(self.symbols["HYG"], 20, Resolution.DAILY)
        self.lqd_sma20 = self.sma(self.symbols["LQD"], 20, Resolution.DAILY)

        self.peak_equity = float(self.portfolio.total_portfolio_value)
        self.last_rebalance = self.start_date - timedelta(days=10)

        self.set_benchmark(self.symbols["SPY"])
        self.set_warm_up(timedelta(days=260))

        self.schedule.on(
            self.date_rules.every_day(self.symbols["SPY"]),
            self.time_rules.after_market_open(self.symbols["SPY"], 45),
            self.rebalance,
        )

    def _is_ready(self) -> bool:
        core = [self.spy_sma200, self.vixy_sma30, self.hyg_sma20, self.lqd_sma20]
        if not all(ind.is_ready for ind in core):
            return False

        for sym in self.symbols.values():
            if not (
                self.mom21[sym].is_ready
                and self.mom63[sym].is_ready
                and self.mom126[sym].is_ready
                and self.sma20[sym].is_ready
                and self.sma100[sym].is_ready
                and self.atr20[sym].is_ready
            ):
                return False
        return True

    def _regime_score(self) -> int:
        spy = float(self.securities[self.symbols["SPY"]].price)
        vixy = float(self.securities[self.symbols["VIXY"]].price)

        score = 0
        if spy > float(self.spy_sma200.current.value):
            score += 1
        if vixy < float(self.vixy_sma30.current.value):
            score += 1
        if float(self.hyg_sma20.current.value) >= float(self.lqd_sma20.current.value):
            score += 1

        breadth = 0
        for t in self.risk_assets:
            s = self.symbols[t]
            px = float(self.securities[s].price)
            if px > float(self.sma20[s].current.value):
                breadth += 1
        if breadth >= int(len(self.risk_assets) * 0.60):
            score += 1

        return score

    def _corr_to_spy(self, sym: Symbol, lookback: int = 90) -> float:
        if sym == self.symbols["SPY"]:
            return 1.0

        h = self.history([sym, self.symbols["SPY"]], lookback, Resolution.DAILY)
        if h.empty:
            return 0.0

        try:
            c = h.close.unstack(level=0)
        except Exception:
            return 0.0

        if sym not in c.columns or self.symbols["SPY"] not in c.columns:
            return 0.0

        c = c[[sym, self.symbols["SPY"]]].dropna()
        if c.shape[0] < 25:
            return 0.0

        r1 = np.diff(np.log(c[sym].values))
        r2 = np.diff(np.log(c[self.symbols["SPY"]].values))
        if len(r1) < 10 or np.std(r1) < 1e-12 or np.std(r2) < 1e-12:
            return 0.0

        return float(np.corrcoef(r1, r2)[0, 1])

    def _asset_alpha(self, sym: Symbol, corr_to_spy: float) -> float:
        px = float(self.securities[sym].price)
        if px <= 0:
            return -999.0

        mom = (
            self.mom_fast_w * float(self.mom21[sym].current.value)
            + self.mom_mid_w * float(self.mom63[sym].current.value)
            + self.mom_slow_w * float(self.mom126[sym].current.value)
        )

        sma20 = float(self.sma20[sym].current.value)
        stretch = abs((px - sma20) / sma20) if sma20 > 0 else 0.0

        atr = float(self.atr20[sym].current.value)
        vol_proxy = max(0.003, atr / px)

        corr_penalty = self.corr_penalty_weight * max(0.0, corr_to_spy - 0.70)

        # momentum preferred, overstretch penalized, risk-normalized, high-beta overlap penalized
        raw = (mom - 0.26 * stretch - corr_penalty) / vol_proxy
        return raw

    def _market_realized_vol(self) -> float:
        spy = self.symbols["SPY"]
        px = float(self.securities[spy].price)
        atr = float(self.atr20[spy].current.value)
        if px <= 0:
            return self.target_portfolio_vol
        # Daily ATR/price -> annualized rough proxy
        return max(0.05, min(0.60, (atr / px) * np.sqrt(252.0)))

    def _target_exposure(self, regime_score: int, drawdown: float, realized_vol: float) -> float:
        if drawdown >= self.max_drawdown:
            base = 0.18
        elif regime_score >= 4:
            base = 0.95
        elif regime_score == 3:
            base = 0.78
        elif regime_score == 2:
            base = 0.58
        elif regime_score == 1:
            base = 0.42
        else:
            base = 0.28

        # Vol targeting overlay
        vol_scale = self.target_portfolio_vol / max(0.06, realized_vol)
        vol_scale = max(0.45, min(1.20, vol_scale))
        return max(0.12, min(0.98, base * vol_scale))

    def _current_weight(self, sym: Symbol) -> float:
        tpv = float(self.portfolio.total_portfolio_value)
        if tpv <= 0:
            return 0.0
        return float(self.portfolio[sym].holdings_value) / tpv

    def rebalance(self):
        if self.is_warming_up or not self._is_ready():
            return

        if (self.time.date() - self.last_rebalance.date()).days < self.rebalance_days:
            return

        tpv = float(self.portfolio.total_portfolio_value)
        self.peak_equity = max(self.peak_equity, tpv)
        drawdown = (self.peak_equity - tpv) / self.peak_equity if self.peak_equity > 0 else 0.0

        regime = self._regime_score()
        realized_vol = self._market_realized_vol()
        target_exposure = self._target_exposure(regime, drawdown, realized_vol)

        active_tickers = self.risk_assets + self.def_assets + self.crypto_assets

        corr_cache = {}
        for t in active_tickers:
            corr_cache[t] = self._corr_to_spy(self.symbols[t])

        candidates = []
        for t in active_tickers:
            sym = self.symbols[t]
            sc = self._asset_alpha(sym, corr_cache[t])

            # Regime modifiers
            if t in self.risk_assets and regime <= 1:
                sc *= 0.40
            if t in self.def_assets and regime <= 1:
                sc *= 1.35
            if t in self.crypto_assets and regime <= 2:
                sc *= 0.30

            candidates.append((t, sym, sc))

        positives = [(t, s, a) for t, s, a in candidates if a > 0]
        if not positives:
            # Defensive fallback
            for kvp in self.portfolio:
                if kvp.value.invested:
                    self.liquidate(kvp.key, "no_positive_alpha")
            self.set_holdings(self.symbols["TLT"], 0.10)
            self.set_holdings(self.symbols["GLD"], 0.08)
            self.last_rebalance = self.time
            return

        positives.sort(key=lambda x: x[2], reverse=True)
        top_n = int(self.get_parameter("top_n") or 6)
        sel = positives[:top_n]

        # Alpha + inverse-vol blend
        raw_targets = {}
        blend_sum = 0.0
        for t, sym, alpha in sel:
            px = float(self.securities[sym].price)
            atr = float(self.atr20[sym].current.value)
            vol = max(0.003, atr / max(1e-9, px))
            inv_vol = 1.0 / vol
            blend = max(0.0, alpha) * np.sqrt(inv_vol)
            raw_targets[sym] = blend
            blend_sum += blend

        targets = {}
        if blend_sum > 0:
            for sym, blend in raw_targets.items():
                w = (blend / blend_sum) * target_exposure
                w = min(self.max_weight, max(0.0, w))
                targets[sym] = w

        # Enforce defensive floor in weaker regimes.
        if regime <= 1:
            def_sym = [self.symbols["TLT"], self.symbols["GLD"], self.symbols["UUP"]]
            floor_each = self.defensive_floor / len(def_sym)
            for s in def_sym:
                targets[s] = max(targets.get(s, 0.0), floor_each)

        gross = sum(targets.values())
        if gross > 0:
            scale = min(1.0, target_exposure / gross)
            for sym in list(targets.keys()):
                targets[sym] *= scale

        # Flatten symbols not targeted (with turnover threshold)
        for kvp in self.portfolio:
            sym = kvp.key
            h = kvp.value
            if h.invested and sym not in targets:
                if abs(self._current_weight(sym)) >= self.min_rebalance_delta:
                    self.liquidate(sym, "out_of_target")

        # Apply target holdings with turnover guard
        for sym, w in targets.items():
            current_w = self._current_weight(sym)
            if abs(w - current_w) < self.min_rebalance_delta:
                continue
            self.set_holdings(sym, w)

        self.last_rebalance = self.time
        self.debug(
            f"rebalance regime={regime} drawdown={drawdown:.2%} mkt_vol={realized_vol:.2%} "
            f"target_exposure={target_exposure:.2f} holdings={len(targets)}"
        )
