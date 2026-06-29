"""Macro regime score + regime label used as an exposure multiplier."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..indicators.technical import sma
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MacroRegime:
    score: float            # 0..100 (broad-market health)
    label: str              # "risk_on" | "neutral" | "risk_off"
    exposure_multiplier: float  # scales portfolio gross exposure
    detail: dict


def _trend_ok(df: pd.DataFrame, period: int) -> bool | None:
    if df is None or "close" not in df or len(df) < period:
        return None
    close = df["close"].astype(float)
    ma = sma(close, period)
    if ma.isna().all():
        return None
    return bool(close.iloc[-1] > ma.iloc[-1])


def compute_macro_regime(macro: dict[str, pd.DataFrame], cfg: dict) -> MacroRegime:
    """Derive a market-wide regime from index trends + VIX level.

    Keyless: relies on yfinance proxies (SPY/QQQ/IWM above 200DMA, VIX level).
    """
    ma_period = cfg.get("ma_period", 200)
    detail: dict = {}
    trend_flags = []
    for key in ("spy", "qqq", "iwm"):
        ok = _trend_ok(macro.get(key), ma_period)
        detail[f"{key}_above_{ma_period}dma"] = ok
        if ok is not None:
            trend_flags.append(ok)

    trend_component = (sum(trend_flags) / len(trend_flags) * 100.0) if trend_flags else 50.0

    # VIX level
    vix_df = macro.get("vix")
    vix_level = None
    if vix_df is not None and "close" in vix_df and len(vix_df):
        vix_level = float(vix_df["close"].iloc[-1])
    detail["vix"] = vix_level

    risk_off_vix = cfg.get("vix_risk_off", 25.0)
    risk_on_vix = cfg.get("vix_risk_on", 16.0)
    if vix_level is None:
        vix_component = 50.0
    elif vix_level >= risk_off_vix:
        vix_component = 20.0
    elif vix_level <= risk_on_vix:
        vix_component = 85.0
    else:
        # linear interpolation between thresholds
        frac = (risk_off_vix - vix_level) / (risk_off_vix - risk_on_vix)
        vix_component = 20.0 + frac * (85.0 - 20.0)

    score = 0.6 * trend_component + 0.4 * vix_component

    if score >= 65:
        label, mult = "risk_on", 1.0
    elif score <= 40:
        label, mult = "risk_off", 0.5
    else:
        label, mult = "neutral", 0.8

    log.info("Macro regime: %s (score=%.1f, vix=%s)", label, score, vix_level)
    return MacroRegime(score=score, label=label, exposure_multiplier=mult, detail=detail)
