"""Per-symbol institutional (13F) score from quarter-over-quarter changes.

Aggregates the signed signals across tracked funds: consensus accumulation
(several funds opening/adding) pushes a symbol above neutral; broad distribution
(trimming/exiting) below. No data for a symbol -> neutral (50). Deliberately
gentle because 13F is delayed (slow confirmation, not a trigger).
"""
from __future__ import annotations

import pandas as pd

from .model import HoldingChange


def institutional_scores(changes: list[HoldingChange], universe: list[str],
                         points_per_fund: float = 8.0,
                         max_swing: float = 35.0) -> pd.Series:
    agg: dict[str, float] = {}
    for c in changes:
        agg[c.symbol] = agg.get(c.symbol, 0.0) + c.signed_weight

    out = {}
    for sym in universe:
        net = agg.get(sym)
        if net is None:
            out[sym] = 50.0
            continue
        swing = max(-max_swing, min(max_swing, net * points_per_fund))
        out[sym] = float(max(0.0, min(100.0, 50.0 + swing)))
    return pd.Series(out, dtype=float)
