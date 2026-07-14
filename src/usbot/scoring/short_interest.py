"""Short-interest factor.

High short interest predicts lower future returns (Asquith–Pathak–Ritter 2005;
Boehmer–Jones–Zhang 2008), and because short sellers are informed a *rising*
short book is bearish while short *covering* is bullish (Diether–Lee–Werner
2009; the aggregate-timing analogue is Rapach–Ringgenberg–Zhou 2016). We map
both the short-interest LEVEL (inverse: low = bullish) and its month-over-month
CHANGE (inverse: rising = bearish) to a 0..100 score, staying neutral (50)
wherever the data is missing so absent short data never penalizes a name.

Inputs come straight from the yfinance ``.info`` fundamentals already fetched
(``short_percent_float``, ``short_ratio``, ``shares_short``,
``shares_short_prior``) — no extra network call.
"""
from __future__ import annotations

import pandas as pd

from .normalize import percentile_rank


def short_interest_scores(fundamentals: dict[str, dict], universe: list[str] | None = None,
                          level_weight: float = 0.6) -> pd.Series:
    """0..100 per-symbol short-interest score. Missing data -> neutral (50)."""
    syms = list(universe) if universe else list(fundamentals.keys())
    if not syms:
        return pd.Series(dtype=float)

    # LEVEL: prefer % of float; fall back to days-to-cover (short ratio). Lower
    # short interest -> higher score, so higher_better=False.
    level_raw: dict[str, float | None] = {}
    for s in syms:
        m = fundamentals.get(s, {})
        v = m.get("short_percent_float")
        if v is None:
            v = m.get("short_ratio")
        level_raw[s] = v
    level = percentile_rank(pd.Series(level_raw, dtype="float64"), higher_better=False)

    # CHANGE: month-over-month growth in shares short. Rising short interest is
    # bearish, so higher_better=False.
    change_raw: dict[str, float | None] = {}
    for s in syms:
        m = fundamentals.get(s, {})
        cur, prior = m.get("shares_short"), m.get("shares_short_prior")
        change_raw[s] = ((cur - prior) / prior) if (cur is not None and prior) else None
    change = percentile_rank(pd.Series(change_raw, dtype="float64"), higher_better=False)

    lw = min(1.0, max(0.0, level_weight))
    score = level * lw + change * (1.0 - lw)
    return score.reindex(syms)


def short_interest_highlights(fundamentals: dict[str, dict], universe: list[str] | None = None,
                              limit: int = 12) -> list[dict]:
    """Most-shorted names (by % of float) for the report, with MoM change."""
    syms = list(universe) if universe else list(fundamentals.keys())
    rows: list[dict] = []
    for s in syms:
        m = fundamentals.get(s, {})
        pct = m.get("short_percent_float")
        if not isinstance(pct, (int, float)):
            continue
        cur, prior = m.get("shares_short"), m.get("shares_short_prior")
        chg = ((cur - prior) / prior) if (cur is not None and prior) else None
        rows.append({"symbol": s, "pct_float": float(pct),
                     "short_ratio": m.get("short_ratio"), "change": chg})
    rows.sort(key=lambda r: r["pct_float"], reverse=True)
    return rows[:limit]
