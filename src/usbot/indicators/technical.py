"""Pure-pandas technical indicators (no pandas-ta dependency for robustness)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # Edge cases: all-gains (no losses) -> 100; flat (no moves) -> neutral 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (0..100): volume-weighted RSI-like flow indicator."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"].astype(float)
    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0).rolling(period).sum()
    neg = rmf.where(delta < 0, 0.0).rolling(period).sum()
    ratio = pos / neg.replace(0.0, np.nan)
    out = 100 - (100 / (1 + ratio))
    return out.where(neg != 0, 100.0)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative signed volume."""
    if not {"close", "volume"}.issubset(df.columns):
        return pd.Series(dtype=float)
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"].astype(float)).cumsum()


def momentum(series: pd.Series, lookback: int) -> float:
    """Total return over ``lookback`` trading days. NaN if insufficient history."""
    if len(series) <= lookback:
        return float("nan")
    past = series.iloc[-lookback - 1]
    last = series.iloc[-1]
    if past == 0 or pd.isna(past) or pd.isna(last):
        return float("nan")
    return float(last / past - 1.0)


def realized_vol(series: pd.Series, window: int = 21) -> float:
    rets = series.pct_change().dropna().tail(window)
    if len(rets) < 2:
        return float("nan")
    return float(rets.std() * np.sqrt(252))


def compute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    """Compute a snapshot of indicators for the most recent bar.

    Returns a flat dict of scalar features used by the technical score. Missing
    history yields NaN values (handled gracefully downstream).
    """
    close = df["close"].astype(float)
    out: dict[str, float] = {}

    out["price"] = float(close.iloc[-1]) if len(close) else float("nan")
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    out["sma50"] = float(sma50.iloc[-1]) if not sma50.isna().all() else float("nan")
    out["sma200"] = float(sma200.iloc[-1]) if not sma200.isna().all() else float("nan")
    out["above_sma50"] = float(out["price"] > out["sma50"]) if out["sma50"] == out["sma50"] else float("nan")
    out["above_sma200"] = float(out["price"] > out["sma200"]) if out["sma200"] == out["sma200"] else float("nan")
    out["golden_cross"] = float(out["sma50"] > out["sma200"]) if (
        out["sma50"] == out["sma50"] and out["sma200"] == out["sma200"]
    ) else float("nan")

    rsi_series = rsi(close, cfg.get("rsi_period", 14))
    out["rsi"] = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else float("nan")

    _, _, hist = macd(close, cfg.get("macd_fast", 12), cfg.get("macd_slow", 26),
                      cfg.get("macd_signal", 9))
    out["macd_hist"] = float(hist.iloc[-1]) if len(hist) else float("nan")

    out["atr"] = float(atr(df, cfg.get("atr_period", 14)).iloc[-1]) if len(df) > cfg.get("atr_period", 14) else float("nan")
    out["realized_vol"] = realized_vol(close)

    for lb in cfg.get("momentum_lookbacks", [21, 63, 126, 252]):
        out[f"mom_{lb}"] = momentum(close, lb)

    # Drawdown from 52-week (252d) high
    hi = close.tail(252).max()
    out["drawdown_52w"] = float(close.iloc[-1] / hi - 1.0) if hi and hi == hi else float("nan")

    # Volume breakout: last volume vs 20d average
    if "volume" in df:
        vol = df["volume"].astype(float)
        avg20 = vol.tail(20).mean()
        out["vol_breakout"] = float(vol.iloc[-1] / avg20) if avg20 else float("nan")
    return out
