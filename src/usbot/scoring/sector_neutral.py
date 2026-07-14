"""Sector-neutralization of composite scores.

A raw cross-sectional factor score rewards whatever sector happens to be in
favour, so the top ranks pile into one or two hot sectors (in a recent live run
the growth *and* defensive lists both overloaded on memory semis). Standard quant
practice is to score each name on how much it beats its OWN sector, not the whole
market. We do this softly: subtract a fraction ``strength`` of each sector's mean
edge, so a name in a hot sector is judged against its peers rather than carried by
the sector, while within-sector ordering is preserved.

``strength`` 0 = untouched, 1 = fully sector-neutral. The cross-sectional mean is
preserved (the adjustment nets to ~0 across names), so absolute-score thresholds
elsewhere (e.g. the Active sleeve's entry score) keep their meaning.
"""
from __future__ import annotations

import pandas as pd


def sector_neutralize(scores: pd.Series, sectors: dict[str, str],
                      strength: float = 0.0) -> pd.Series:
    """Pull each score toward its within-sector relative rank. Names with an
    unknown sector are left unchanged."""
    if strength <= 0 or scores is None or scores.empty or not sectors:
        return scores
    strength = min(1.0, float(strength))
    df = pd.DataFrame({"score": scores.astype(float)})
    df["sector"] = [sectors.get(s) for s in df.index]
    global_mean = float(df["score"].mean())
    sector_mean = df.groupby("sector")["score"].transform("mean")  # NaN for unknown sector
    adj = df["score"] - strength * (sector_mean - global_mean)
    adj = adj.where(sector_mean.notna(), df["score"])              # unknown sector -> unchanged
    return adj.clip(0.0, 100.0)
