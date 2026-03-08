# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
# endregion


class StatarbSpreadEngine(QCAlgorithm):
    """
    Statistical arbitrage v2:
    - rolling hedge ratio with beta clipping
    - correlation + half-life gating
    - z-score entry/exit with stop-z and max-hold safeguards
    - beta-aware dollar-neutral sizing
    - pair-level cooldown and risk controls
    """

    def initialize(self):
        self.set_start_date(int(self.get_parameter("start_year") or 2018), 1, 1)
        self.set_end_date(int(self.get_parameter("end_year") or 2025), 12, 31)
        self.set_cash(float(self.get_parameter("initial_cash") or 100000))

        self.lookback = int(self.get_parameter("lookback") or 110)
        self.entry_z = float(self.get_parameter("entry_z") or 2.1)
        self.exit_z = float(self.get_parameter("exit_z") or 0.35)
        self.stop_z = float(self.get_parameter("stop_z") or 3.7)
        self.max_active_pairs = int(self.get_parameter("max_active_pairs") or 2)
        self.pair_risk_pct = float(self.get_parameter("pair_risk_pct") or 0.05)
        self.cooldown_days = int(self.get_parameter("cooldown_days") or 2)

        self.min_corr = float(self.get_parameter("min_corr") or 0.55)
        self.min_half_life = float(self.get_parameter("min_half_life") or 3.0)
        self.max_half_life = float(self.get_parameter("max_half_life") or 45.0)
        self.max_holding_days = int(self.get_parameter("max_holding_days") or 14)
        self.max_pair_loss_pct = float(self.get_parameter("max_pair_loss_pct") or 0.022)

        self.beta_min = float(self.get_parameter("beta_min") or 0.35)
        self.beta_max = float(self.get_parameter("beta_max") or 2.75)

        self.pair_defs = [
            ("SPY", "QQQ"),
            ("XLE", "XOP"),
            ("XLF", "KRE"),
            ("XLK", "QQQ"),
            ("IWM", "SPY"),
        ]

        self.symbols = {}
        for t in sorted(set([x for p in self.pair_defs for x in p])):
            self.symbols[t] = self.add_equity(t, Resolution.DAILY).symbol

        self.active = {}
        self.cooldown_until = {}

        self.set_benchmark(self.symbols["SPY"])
        self.set_warm_up(timedelta(days=self.lookback + 30))

        self.schedule.on(
            self.date_rules.every_day(self.symbols["SPY"]),
            self.time_rules.after_market_open(self.symbols["SPY"], 70),
            self.evaluate_pairs,
        )

    def _pair_key(self, a: str, b: str) -> str:
        return f"{a}/{b}"

    def _history_pair(self, a: Symbol, b: Symbol):
        h = self.history([a, b], self.lookback, Resolution.DAILY)
        if h.empty:
            return None

        try:
            close = h.close.unstack(level=0)
        except Exception:
            return None

        if a not in close.columns or b not in close.columns:
            return None

        close = close[[a, b]].dropna()
        if close.shape[0] < max(35, int(self.lookback * 0.75)):
            return None
        return close

    def _half_life(self, spread: np.ndarray) -> float:
        if len(spread) < 20:
            return np.inf

        s_lag = spread[:-1]
        ds = spread[1:] - spread[:-1]
        var = float(np.var(s_lag))
        if var <= 1e-12:
            return np.inf

        beta = float(np.cov(ds, s_lag)[0, 1] / var)
        # Mean-reverting process needs beta < 0 in ds = a + beta*s_lag + e
        if beta >= -1e-8:
            return np.inf

        hl = -np.log(2.0) / beta
        if hl <= 0 or hl > 1e6:
            return np.inf
        return float(hl)

    def _calc_stats(self, a: Symbol, b: Symbol):
        close = self._history_pair(a, b)
        if close is None:
            return None

        x = np.log(close[b].values)
        y = np.log(close[a].values)

        dx = np.diff(x)
        dy = np.diff(y)
        if len(dx) < 20 or np.std(dx) < 1e-12 or np.std(dy) < 1e-12:
            return None

        corr = float(np.corrcoef(dy, dx)[0, 1])

        var_x = float(np.var(x))
        if var_x <= 1e-12:
            return None

        beta = float(np.cov(y, x)[0, 1] / var_x)
        beta = max(self.beta_min, min(self.beta_max, beta))

        spread = y - beta * x
        mu = float(np.mean(spread))
        sd = float(np.std(spread))
        if sd <= 1e-9:
            return None

        z = float((spread[-1] - mu) / sd)
        half_life = self._half_life(spread)

        return {
            "beta": beta,
            "z": z,
            "mu": mu,
            "sd": sd,
            "corr": corr,
            "half_life": half_life,
            "spread": float(spread[-1]),
        }

    def _symbol_in_active(self, sym: Symbol) -> bool:
        for st in self.active.values():
            if st["a"] == sym or st["b"] == sym:
                return True
        return False

    def _open_pair(self, a: Symbol, b: Symbol, side: int, key: str, beta: float, z: float):
        # side = +1 -> long A / short B ; side = -1 -> short A / long B
        tpv = float(self.portfolio.total_portfolio_value)
        notional = tpv * self.pair_risk_pct

        pa = float(self.securities[a].price)
        pb = float(self.securities[b].price)
        if pa <= 0 or pb <= 0:
            return False

        b_abs = max(0.2, abs(beta))
        dollar_a = notional / (1.0 + b_abs)
        dollar_b = notional - dollar_a

        qa = int(max(0, np.floor(dollar_a / pa)))
        qb = int(max(0, np.floor(dollar_b / pb)))
        if qa <= 0 or qb <= 0:
            return False

        if side > 0:
            self.market_order(a, qa, False, f"pair_open {key} longA z={z:.2f}")
            self.market_order(b, -qb, False, f"pair_open {key} shortB z={z:.2f}")
        else:
            self.market_order(a, -qa, False, f"pair_open {key} shortA z={z:.2f}")
            self.market_order(b, qb, False, f"pair_open {key} longB z={z:.2f}")

        self.active[key] = {
            "a": a,
            "b": b,
            "side": side,
            "beta": beta,
            "opened": self.time,
            "entry_notional": max(1.0, qa * pa + qb * pb),
        }
        return True

    def _close_pair(self, key: str, reason: str):
        st = self.active.get(key)
        if not st:
            return
        a = st["a"]
        b = st["b"]
        self.liquidate(a, f"pair_close {key} {reason}")
        self.liquidate(b, f"pair_close {key} {reason}")
        self.cooldown_until[key] = self.time + timedelta(days=self.cooldown_days)
        del self.active[key]

    def _pair_pnl_pct(self, st: dict) -> float:
        a = st["a"]
        b = st["b"]
        entry_notional = float(st.get("entry_notional", 1.0))
        pnl_abs = float(self.portfolio[a].unrealized_profit) + float(self.portfolio[b].unrealized_profit)
        return pnl_abs / max(1e-6, entry_notional)

    def evaluate_pairs(self):
        if self.is_warming_up:
            return

        # 1) Manage active pairs
        for a_t, b_t in self.pair_defs:
            key = self._pair_key(a_t, b_t)
            if key not in self.active:
                continue

            st = self.active[key]
            stats = self._calc_stats(st["a"], st["b"])
            if stats is None:
                continue

            z = stats["z"]
            holding_days = max(0, (self.time.date() - st["opened"].date()).days)
            pnl_pct = self._pair_pnl_pct(st)

            exit_condition = abs(z) <= self.exit_z
            stop_condition = abs(z) >= self.stop_z
            time_condition = holding_days >= self.max_holding_days
            pnl_stop = pnl_pct <= -self.max_pair_loss_pct

            if exit_condition or stop_condition or time_condition or pnl_stop:
                self._close_pair(key, f"z={z:.2f}|days={holding_days}|pnl={pnl_pct:.2%}")

        # 2) Open new pairs
        if len(self.active) >= self.max_active_pairs:
            return

        for a_t, b_t in self.pair_defs:
            if len(self.active) >= self.max_active_pairs:
                break

            key = self._pair_key(a_t, b_t)
            if key in self.active:
                continue

            cd = self.cooldown_until.get(key)
            if cd and self.time < cd:
                continue

            a = self.symbols[a_t]
            b = self.symbols[b_t]

            # Prevent symbol overlap across simultaneously active pairs.
            if self._symbol_in_active(a) or self._symbol_in_active(b):
                continue

            stats = self._calc_stats(a, b)
            if stats is None:
                continue

            z = stats["z"]
            corr = stats["corr"]
            half_life = stats["half_life"]
            beta = stats["beta"]

            gate_ok = (
                corr >= self.min_corr
                and self.min_half_life <= half_life <= self.max_half_life
            )
            if not gate_ok:
                continue

            # Dynamic threshold: longer half-life requires stronger dislocation.
            dyn_entry = self.entry_z + (0.30 if half_life > 28 else 0.0)

            if z >= dyn_entry:
                # A rich vs B -> short A / long B
                self._open_pair(a, b, -1, key, beta, z)
                self.debug(f"open {key} SHORT_A z={z:.2f} corr={corr:.2f} hl={half_life:.1f}")
            elif z <= -dyn_entry:
                # A cheap vs B -> long A / short B
                self._open_pair(a, b, +1, key, beta, z)
                self.debug(f"open {key} LONG_A z={z:.2f} corr={corr:.2f} hl={half_life:.1f}")
