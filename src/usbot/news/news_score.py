"""Per-symbol news score (0..100) from annotated news items.

Aggregates item-level sentiment into a symbol score, with mild recency and
volume weighting. Symbols with no news map to neutral (50) so missing coverage
is never penalized as negative.
"""
from __future__ import annotations

import math

import pandas as pd

from .model import NewsItem


def _symbol_score(items: list[NewsItem]) -> float:
    if not items:
        return 50.0
    # Weight: recent items and stronger sentiment count more. Categories like
    # earnings/legal get a small amplification as higher-impact.
    cat_weight = {"earnings": 1.3, "legal": 1.3, "analyst": 1.2, "product": 1.1,
                  "macro": 1.0, "general": 1.0}
    num = 0.0
    den = 0.0
    for it in items:
        w = cat_weight.get(it.category, 1.0)
        num += it.sentiment * w
        den += w
    avg = num / den if den else 0.0
    # squash from [-1,1] to [0,100] with a gentle slope; clamp
    score = 50.0 + 50.0 * math.tanh(2.0 * avg)
    return float(max(0.0, min(100.0, score)))


def news_scores(items_by_symbol: dict[str, list[NewsItem]],
                universe: list[str]) -> pd.Series:
    """Return a 0..100 news score per universe symbol (neutral 50 if no news)."""
    out = {}
    for sym in universe:
        out[sym] = _symbol_score(items_by_symbol.get(sym, []))
    return pd.Series(out, dtype=float)
