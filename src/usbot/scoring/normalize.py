"""Cross-sectional normalization helpers (map raw factor values to 0..100)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_rank(values: pd.Series, higher_better: bool = True) -> pd.Series:
    """Rank to 0..100 percentile. NaNs map to 50 (neutral) to avoid penalizing
    missing data as if it were worst-in-class."""
    s = values.astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(50.0, index=s.index)
    ranks = valid.rank(pct=True)
    if not higher_better:
        ranks = 1.0 - ranks
    out = pd.Series(50.0, index=s.index)
    out.loc[valid.index] = ranks * 100.0
    return out


def clamp01_100(x: float) -> float:
    return float(min(100.0, max(0.0, x)))
