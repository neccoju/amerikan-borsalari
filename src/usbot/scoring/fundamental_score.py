"""Fundamental sub-score from yfinance metrics, cross-sectionally ranked."""
from __future__ import annotations

import pandas as pd

from .normalize import percentile_rank

# direction: True = higher is better, False = lower is better
_DIRECTION = {
    "revenue_growth": True,
    "earnings_growth": True,
    "profit_margin": True,
    "return_on_equity": True,
    "free_cash_flow_yield": True,
    "debt_to_equity": False,
    "valuation_pe": False,
}


def fundamental_scores(fundamentals: dict[str, dict], cfg: dict) -> pd.Series:
    """0..100 fundamental score per symbol. Missing metrics -> neutral (50)."""
    syms = list(fundamentals.keys())
    if not syms:
        return pd.Series(dtype=float)

    weights = cfg.get("weights", {})
    # Per-metric cross-sectional rank.
    metric_ranks: dict[str, pd.Series] = {}
    for metric, higher_better in _DIRECTION.items():
        raw = pd.Series({s: fundamentals[s].get(metric) for s in syms}, dtype="float64")
        metric_ranks[metric] = percentile_rank(raw, higher_better=higher_better)

    scores = {}
    total_w = sum(weights.get(m, 0) for m in _DIRECTION) or 1.0
    for sym in syms:
        acc = 0.0
        for metric in _DIRECTION:
            w = weights.get(metric, 0)
            acc += w * metric_ranks[metric].get(sym, 50.0)
        scores[sym] = acc / total_w
    return pd.Series(scores)
