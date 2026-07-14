"""Look-ahead-safe backtest of the live technical-momentum specification.

Unlike the single-lookback ``momentum_weight_fn``, this replicates the
price-derived legs of the *production* technical score
(``scoring/technical_score.py``) at every rebalance, reusing the very same
indicator functions the live scorer calls:

- multi-lookback momentum blend (21/63/126/252d, weights 0.15/0.25/0.30/0.30)
  with the Jegadeesh-Titman **12-1** skip-month long leg,
- Barroso-Santa-Clara **risk-adjusted** (volatility-scaled) momentum,
- distance from the 52-week high (drawdown).

These are the honest subset of the composite that can be reconstructed
point-in-time from free price history; fundamental/news/insider legs lack free
point-in-time history and stay out of scope (see ``engine.py``). The result is a
faithful out-of-sample test of the momentum methodology actually shipped — pair
it with ``--trials N`` to read its Deflated Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.technical import momentum, momentum_12_1, risk_adjusted_momentum
from ..scoring.normalize import percentile_rank

# lookbacks/weights mirror scoring.yaml `technical.momentum_*`
_LOOKBACKS = [21, 63, 126, 252]
_MOM_W = [0.15, 0.25, 0.30, 0.30]
# blend across the three price-derived legs (renormalized from technical_score's
# 0.25 momentum / 0.12 risk-adj / 0.09 drawdown contributions)
_LEG_W = {"mom": 0.55, "ram": 0.27, "dd": 0.18}


def _drawdown_52w(series: pd.Series, window: int = 252) -> float:
    tail = series.tail(window)
    if len(tail) < 2:
        return float("nan")
    peak = float(tail.max())
    return float(series.iloc[-1] / peak - 1.0) if peak > 0 else float("nan")


def composite_scores(history: pd.DataFrame) -> pd.Series:
    """Cross-sectional 0..100 technical-momentum score from a close panel."""
    mom_blend: dict[str, float] = {}
    ram: dict[str, float] = {}
    dd: dict[str, float] = {}
    for sym in history.columns:
        s = history[sym].dropna()
        if len(s) < 30:
            continue
        vals: list[float] = []
        weights: list[float] = []
        for lb, w in zip(_LOOKBACKS, _MOM_W):
            v = momentum_12_1(s) if lb >= 200 else momentum(s, lb)
            if v is not None and not pd.isna(v):
                vals.append(v)
                weights.append(w)
        mom_blend[sym] = float(np.average(vals, weights=weights)) if vals else np.nan
        ram[sym] = risk_adjusted_momentum(s)
        dd[sym] = _drawdown_52w(s)

    mom_rank = percentile_rank(pd.Series(mom_blend, dtype="float64"), higher_better=True)
    if mom_rank.empty:
        return pd.Series(dtype=float)
    ram_rank = percentile_rank(pd.Series(ram, dtype="float64"), higher_better=True)
    dd_rank = percentile_rank(pd.Series(dd, dtype="float64"), higher_better=True)
    syms = mom_rank.index
    return (_LEG_W["mom"] * mom_rank
            + _LEG_W["ram"] * ram_rank.reindex(syms).fillna(50.0)
            + _LEG_W["dd"] * dd_rank.reindex(syms).fillna(50.0))


def composite_momentum_weight_fn(top_n: int = 10, max_weight: float = 0.15):
    """Point-in-time weight_fn: top-N by the live technical-momentum composite."""
    def _fn(asof: pd.Timestamp, history: pd.DataFrame) -> dict:
        comp = composite_scores(history)
        if comp.empty:
            return {}
        top = comp.sort_values(ascending=False).head(top_n)
        if top.empty:
            return {}
        w = 1.0 / len(top)
        return {sym: min(w, max_weight) for sym in top.index}
    return _fn
