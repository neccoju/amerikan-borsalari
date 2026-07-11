"""Technical sub-score: blends trend, momentum, RSI/MACD, breakout, drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import clamp01_100, percentile_rank


def _rsi_band_score(rsi: float, low: float, high: float) -> float:
    """Reward healthy momentum (~50-70), penalize overbought/oversold extremes."""
    if pd.isna(rsi):
        return 50.0
    if rsi >= 80 or rsi <= 20:
        return 25.0
    if low <= rsi <= high:
        return 80.0
    return 55.0


def technical_scores(indicators: dict[str, dict], cfg: dict) -> pd.Series:
    """Compute 0..100 technical score per symbol from precomputed indicators."""
    syms = list(indicators.keys())
    if not syms:
        return pd.Series(dtype=float)

    lookbacks = cfg.get("momentum_lookbacks", [21, 63, 126, 252])
    mom_w = cfg.get("momentum_weights", [0.15, 0.25, 0.30, 0.30])
    low, high = cfg.get("rsi_low", 30), cfg.get("rsi_high", 70)

    # Build a blended momentum value per symbol, then cross-sectionally rank it.
    # The 12M leg uses the classic 12-1 specification (Jegadeesh & Titman 1993):
    # skip the most recent month so the short-term reversal effect doesn't
    # contaminate the trend signal.
    mom_blend = {}
    for sym in syms:
        ind = indicators[sym]
        vals, weights = [], []
        for lb, w in zip(lookbacks, mom_w):
            v = ind.get(f"mom_{lb}")
            if lb >= 200:
                v121 = ind.get("mom_12_1")
                if v121 is not None and not pd.isna(v121):
                    v = v121
            if v is not None and not pd.isna(v):
                vals.append(v)
                weights.append(w)
        mom_blend[sym] = float(np.average(vals, weights=weights)) if vals else np.nan
    mom_rank = percentile_rank(pd.Series(mom_blend), higher_better=True)

    # Risk-adjusted momentum (Barroso & Santa-Clara 2015): same trend per unit
    # of realized volatility — prefers smooth trends over crash-prone ones.
    ram = pd.Series({s: indicators[s].get("risk_adj_mom", np.nan) for s in syms})
    ram_rank = percentile_rank(ram, higher_better=True)

    # Drawdown: closer to 0 (near highs) better
    dd = pd.Series({s: indicators[s].get("drawdown_52w", np.nan) for s in syms})
    dd_rank = percentile_rank(dd, higher_better=True)

    scores = {}
    for sym in syms:
        ind = indicators[sym]
        trend = 0.0
        for key in ("above_sma50", "above_sma200", "golden_cross"):
            v = ind.get(key)
            if not pd.isna(v):
                trend += v
        trend_score = (trend / 3.0) * 100.0  # 0..100

        rsi_score = _rsi_band_score(ind.get("rsi", np.nan), low, high)
        macd_score = 70.0 if ind.get("macd_hist", 0) and ind["macd_hist"] > 0 else 40.0
        vb = ind.get("vol_breakout", np.nan)
        vb_score = clamp01_100(50.0 + (vb - 1.0) * 40.0) if not pd.isna(vb) else 50.0

        composite = (
            0.28 * trend_score
            + 0.25 * mom_rank.get(sym, 50.0)
            + 0.12 * ram_rank.get(sym, 50.0)
            + 0.10 * rsi_score
            + 0.08 * macd_score
            + 0.09 * dd_rank.get(sym, 50.0)
            + 0.08 * vb_score
        )
        scores[sym] = clamp01_100(composite)
    return pd.Series(scores)
