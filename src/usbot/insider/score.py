"""Per-symbol insider score (0..100) from Form 4 trades.

Emphasis follows the literature: opportunistic open-market BUYS (especially
clustered across several insiders, and by senior officers) push a name well
above neutral; open-market sales nudge modestly below (they are far noisier).
Names with no insider activity map to neutral (50).
"""
from __future__ import annotations

import math

import pandas as pd

from .model import InsiderTrade


def insider_scores(trades: list[InsiderTrade], universe: list[str]) -> pd.Series:
    """Aggregate Form 4 activity into a 0..100 per-symbol score."""
    buy_val: dict[str, float] = {}
    sell_val: dict[str, float] = {}
    buyers: dict[str, set] = {}
    senior_buy: dict[str, bool] = {}
    for t in trades:
        if t.is_open_market_buy:
            buy_val[t.symbol] = buy_val.get(t.symbol, 0.0) + t.value
            buyers.setdefault(t.symbol, set()).add(t.insider)
            senior_buy[t.symbol] = senior_buy.get(t.symbol, False) or t.is_senior
        elif t.is_open_market_sell:
            sell_val[t.symbol] = sell_val.get(t.symbol, 0.0) + t.value

    out: dict[str, float] = {}
    for sym in universe:
        bv = buy_val.get(sym, 0.0)
        sv = sell_val.get(sym, 0.0)
        if bv <= 0 and sv <= 0:
            out[sym] = 50.0
            continue
        score = 50.0
        if bv > 0:
            # log-compressed buy magnitude (a single huge buy can't dominate)
            score += min(28.0, math.log10(bv + 1.0) * 5.0)
            n_buyers = len(buyers.get(sym, ()))
            if n_buyers >= 2:                       # cluster buy — strongest case
                score += min(15.0, (n_buyers - 1) * 6.0)
            if senior_buy.get(sym):                 # C-suite / director conviction
                score += 5.0
        if sv > 0:
            # sales are noisy; penalize only lightly, and only net of buys
            score -= min(15.0, math.log10(sv + 1.0) * 2.5)
        out[sym] = float(max(0.0, min(100.0, score)))
    return pd.Series(out, dtype=float)
