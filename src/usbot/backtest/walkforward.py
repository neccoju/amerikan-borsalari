"""Walk-forward comparison of an adaptive vs. a static factor-weighting rule.

Uses only price-derived factors (reconstructable point-in-time from history), so
the comparison is honest and look-ahead-safe. At each rebalance the adaptive rule
updates factor weights from the trailing information coefficient of each factor;
the static rule keeps equal weights. Both are run through the same backtest
engine and their metrics returned side by side.

This validates the adaptive mechanism out-of-sample without relying on stored
fundamental/news history (which isn't point-in-time available for free).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..learning import update_weights
from .engine import BacktestConfig, BacktestResult, run_backtest


def _momentum_factor(history: pd.DataFrame, lookback: int) -> pd.Series:
    if len(history) <= lookback:
        return pd.Series(dtype=float)
    return (history.iloc[-1] / history.iloc[-lookback - 1] - 1.0)


def _trend_factor(history: pd.DataFrame, period: int) -> pd.Series:
    if len(history) <= period:
        return pd.Series(dtype=float)
    ma = history.tail(period).mean()
    return (history.iloc[-1] / ma - 1.0)


def _vol_factor(history: pd.DataFrame, window: int = 21) -> pd.Series:
    rets = history.pct_change().tail(window)
    if len(rets) < 2:
        return pd.Series(dtype=float)
    return -rets.std()   # low-vol preferred -> negate


PRICE_FACTORS = {
    "momentum": lambda h: _momentum_factor(h, 126),
    "trend": lambda h: _trend_factor(h, 100),
    "lowvol": lambda h: _vol_factor(h, 21),
}


@dataclass
class WalkForwardComparison:
    adaptive: BacktestResult
    static: BacktestResult

    def summary(self) -> dict:
        return {"adaptive": self.adaptive.summary(), "static": self.static.summary(),
                "adaptive_minus_static_cagr": self.adaptive.metrics.cagr - self.static.metrics.cagr}


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 2 or s.std(ddof=0) == 0:
        return pd.Series(50.0, index=s.index)
    return 50.0 + 10.0 * (s - s.mean()) / s.std(ddof=0)


def _make_weight_fn(adaptive: bool, lr: float, top_n: int, max_weight: float):
    """Build a weight_fn closure with its own learning state (per backtest run)."""
    state = {"weights": {f: 1.0 / len(PRICE_FACTORS) for f in PRICE_FACTORS},
             "last_scores": None, "last_px": None}

    def weight_fn(asof: pd.Timestamp, history: pd.DataFrame) -> dict:
        # 1. adaptive update from realized return since last rebalance
        if adaptive and state["last_scores"] is not None and state["last_px"] is not None:
            common = history.columns.intersection(state["last_px"].index)
            cur = history.iloc[-1][common]
            ret = (cur / state["last_px"][common] - 1.0)
            ic = {}
            for f, sc in state["last_scores"].items():
                j = pd.concat([sc.rename("s"), ret.rename("r")], axis=1).dropna()
                if len(j) >= 3 and j["s"].nunique() >= 2:
                    ic[f] = float(j["s"].rank().corr(j["r"].rank()) or 0.0)
                else:
                    ic[f] = 0.0
            state["weights"] = update_weights(state["weights"], ic, lr=lr,
                                              min_w=0.05, max_w=max_weight)

        # 2. compute factor scores now (point-in-time) and blend with weights
        scores = {f: _zscore(fn(history)) for f, fn in PRICE_FACTORS.items()}
        syms = sorted({s for v in scores.values() for s in v.index})
        if not syms:
            return {}
        comp = pd.Series(0.0, index=syms)
        for f, w in state["weights"].items():
            comp = comp.add(scores[f].reindex(syms).fillna(50.0) * w, fill_value=0.0)

        state["last_scores"] = {f: scores[f] for f in scores}
        state["last_px"] = history.iloc[-1]

        top = comp.sort_values(ascending=False).head(top_n)
        top = top[top > 0]
        if top.empty:
            return {}
        w = 1.0 / len(top)
        return {s: min(w, max_weight) for s in top.index}

    return weight_fn


def walk_forward_compare(prices: dict[str, pd.DataFrame], config: BacktestConfig,
                         *, lr: float = 0.5, top_n: int = 10,
                         max_weight: float = 0.15) -> WalkForwardComparison:
    """Run adaptive vs static factor weighting through the backtest engine."""
    adaptive = run_backtest(prices, _make_weight_fn(True, lr, top_n, max_weight), config)
    static = run_backtest(prices, _make_weight_fn(False, lr, top_n, max_weight), config)
    return WalkForwardComparison(adaptive=adaptive, static=static)
