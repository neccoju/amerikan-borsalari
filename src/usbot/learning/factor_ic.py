"""Factor information coefficient (IC): how well last period's factor scores
predicted this period's realized returns.

IC = Spearman rank correlation between a factor's prior cross-sectional scores
and the subsequently realized per-symbol returns. This is computed only with
data available after the fact (prior scores vs. later returns), so it is
look-ahead-safe by construction.
"""
from __future__ import annotations

import pandas as pd


def realized_returns(price_history: dict[str, pd.DataFrame], since_date: str,
                     symbols: list[str]) -> pd.Series:
    """Per-symbol total return from ``since_date`` close to the latest close.

    Uses adj_close when present. Symbols lacking data on/after ``since_date`` are
    omitted (NaN-free output).
    """
    out: dict[str, float] = {}
    cutoff = pd.Timestamp(since_date)
    for sym in symbols:
        df = price_history.get(sym)
        if df is None or df.empty:
            continue
        col = "adj_close" if "adj_close" in df.columns else "close"
        s = df[col].astype(float)
        s.index = pd.to_datetime(s.index)
        past = s[s.index <= cutoff]
        if past.empty or s.empty:
            continue
        p0, p1 = float(past.iloc[-1]), float(s.iloc[-1])
        if p0 > 0:
            out[sym] = p1 / p0 - 1.0
    return pd.Series(out, dtype=float)


def compute_factor_ic(prev_scores: dict[str, dict[str, float]] | dict[str, pd.Series],
                      realized: pd.Series) -> dict[str, float]:
    """Spearman IC per factor between prior scores and realized returns.

    ``prev_scores`` maps factor -> {symbol: score} (or factor -> Series). Returns
    {factor: ic in [-1, 1]}; factors with <3 overlapping observations -> 0.0.
    """
    ic: dict[str, float] = {}
    ret = realized.dropna()
    for factor, scores in prev_scores.items():
        s = scores if isinstance(scores, pd.Series) else pd.Series(scores, dtype=float)
        joined = pd.concat([s.rename("score"), ret.rename("ret")], axis=1).dropna()
        if len(joined) < 3 or joined["score"].nunique() < 2:
            ic[factor] = 0.0
            continue
        # Spearman = Pearson of ranks (avoids a scipy dependency).
        corr = joined["score"].rank().corr(joined["ret"].rank())
        ic[factor] = float(corr) if pd.notna(corr) else 0.0
    return ic
