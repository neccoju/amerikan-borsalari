"""Per-symbol congressional-trading score (0..100).

Disclosure data is delayed and noisy, so this is treated as a weak-to-moderate
signal: net recent buying nudges a symbol above neutral, net selling below.
Symbols with no disclosed trades map to neutral (50).
"""
from __future__ import annotations

import math

import pandas as pd

from .model import CongressTrade


def congress_scores(trades: list[CongressTrade], universe: list[str]) -> pd.Series:
    """Aggregate signed dollar flow per symbol into a 0..100 score.

    Uses a compressive transform so a single large disclosure doesn't dominate;
    multiple buyers across members on the same name push the score higher.
    """
    flow: dict[str, float] = {}
    buyers: dict[str, set] = {}
    sellers: dict[str, set] = {}
    for t in trades:
        flow[t.symbol] = flow.get(t.symbol, 0.0) + t.signed_amount
        if t.txn_type == "buy":
            buyers.setdefault(t.symbol, set()).add(t.politician)
        else:
            sellers.setdefault(t.symbol, set()).add(t.politician)

    out = {}
    for sym in universe:
        f = flow.get(sym)
        if f is None:
            out[sym] = 50.0
            continue
        # log-compress dollar flow (sign-preserving), scale to +/- points
        mag = math.log10(abs(f) + 1.0)          # ~0..7+
        direction = 1.0 if f >= 0 else -1.0
        base = 50.0 + direction * min(30.0, mag * 6.0)
        # consensus bonus: several distinct buyers (or sellers) strengthens signal
        nb = len(buyers.get(sym, ()))
        ns = len(sellers.get(sym, ()))
        consensus = (nb - ns) * 3.0
        score = base + max(-15.0, min(15.0, consensus))
        out[sym] = float(max(0.0, min(100.0, score)))
    return pd.Series(out, dtype=float)
