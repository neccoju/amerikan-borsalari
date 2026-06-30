"""Sector-rotation analytics from sector-ETF price history.

Pure data layer (no plotting): multi-horizon returns, relative strength vs SPY,
RRG quadrant classification, and the Estimated Smart-Money Rotation Proxy. All
functions degrade gracefully on thin/missing data.

The proxy is NOT actual dollar flow — it is a momentum/strength/volume composite,
clearly labelled as such wherever it is shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators.technical import mfi, obv

# trading-day lookbacks
_HORIZONS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}


@dataclass
class SectorRow:
    etf: str
    name: str
    ret: dict[str, float] = field(default_factory=dict)   # horizon -> return
    rs_vs_spy: float = 0.0          # relative strength vs SPY (1M rel return, %)
    momentum: float = 0.0          # change in relative strength (RRG y-axis)
    rrg_x: float = 100.0           # RRG relative-strength ratio (centered ~100)
    rrg_y: float = 100.0           # RRG momentum ratio (centered ~100)
    quadrant: str = "n/a"
    proxy: float = 0.0             # smart-money proxy, -100..+100
    direction: str = "neutral"     # inflow | outflow | neutral
    rsi: float = float("nan")
    macd_hist: float = float("nan")


def _ret(series: pd.Series, n: int) -> float:
    s = series.dropna()
    if len(s) <= n:
        return float("nan")
    p0, p1 = float(s.iloc[-n - 1]), float(s.iloc[-1])
    return (p1 / p0 - 1.0) if p0 else float("nan")


def _zscore_last(series: pd.Series, window: int = 60) -> float:
    s = series.dropna().tail(window)
    if len(s) < 5 or s.std(ddof=0) == 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / s.std(ddof=0))


def _classify_quadrant(rs_ratio: float, mom_ratio: float) -> str:
    strong = rs_ratio >= 100.0
    rising = mom_ratio >= 100.0
    if strong and rising:
        return "Leading"
    if strong and not rising:
        return "Weakening"
    if not strong and not rising:
        return "Lagging"
    return "Improving"


def compute_sector_rows(prices: dict[str, pd.DataFrame], sector_etfs: dict[str, str],
                        bench: str = "SPY") -> list[SectorRow]:
    """Build per-sector analytics. Returns [] if the benchmark is unavailable."""
    bench_df = prices.get(bench)
    if bench_df is None or bench_df.empty or "close" not in bench_df:
        return []
    bench_close = bench_df["close"].astype(float)
    bench_1m = _ret(bench_close, _HORIZONS["1M"])

    # relative-strength series vs SPY for RRG (ratio * 100), and its momentum
    rows: list[SectorRow] = []
    for etf, name in sector_etfs.items():
        df = prices.get(etf)
        if df is None or df.empty or "close" not in df:
            continue
        close = df["close"].astype(float)
        row = SectorRow(etf=etf, name=name)
        row.ret = {h: _ret(close, n) for h, n in _HORIZONS.items()}

        sec_1m = row.ret.get("1M", float("nan"))
        row.rs_vs_spy = ((sec_1m - bench_1m) * 100.0
                         if not (np.isnan(sec_1m) or np.isnan(bench_1m)) else 0.0)

        # RRG: relative price ratio vs SPY, normalized to ~100, plus its 1W momentum
        aligned = pd.concat([close.rename("s"), bench_close.rename("b")], axis=1).dropna()
        if len(aligned) > _HORIZONS["1M"] + 5:
            rel = (aligned["s"] / aligned["b"])
            rel_norm = 100.0 * rel / rel.tail(_HORIZONS["3M"]).mean()
            row.rrg_x = float(rel_norm.iloc[-1])
            mom = rel_norm.iloc[-1] / float(rel_norm.iloc[-_HORIZONS["1W"] - 1]) * 100.0 \
                if len(rel_norm) > _HORIZONS["1W"] + 1 else 100.0
            row.rrg_y = float(mom)
            row.momentum = row.rrg_y - 100.0
        row.quadrant = _classify_quadrant(row.rrg_x, row.rrg_y)

        # technicals for the proxy / table
        try:
            from ..indicators.technical import rsi as _rsi, macd as _macd
            row.rsi = float(_rsi(close).iloc[-1])
            row.macd_hist = float(_macd(close)[2].iloc[-1])
        except Exception:  # noqa: BLE001
            pass

        row.proxy = _smart_money_proxy(df, close, bench_close, row)
        rows.append(row)

    # normalize proxy across sectors to -100..+100 and set direction
    _normalize_proxy(rows)
    rows.sort(key=lambda r: r.rs_vs_spy, reverse=True)
    return rows


def _smart_money_proxy(df: pd.DataFrame, close: pd.Series, bench_close: pd.Series,
                       row: SectorRow) -> float:
    """Raw proxy = weighted blend of RS change, volume z, MFI change, OBV trend,
    price momentum. Normalized cross-sectionally afterwards."""
    # relative-strength change (1W)
    rs_change = row.momentum
    # volume z-score
    vol_z = _zscore_last(df["volume"].astype(float)) if "volume" in df else 0.0
    # MFI change
    mfi_series = mfi(df)
    mfi_change = 0.0
    if len(mfi_series.dropna()) > 6:
        mfi_change = float(mfi_series.iloc[-1] - mfi_series.iloc[-6])
    # OBV trend (sign of recent slope, scaled)
    obv_series = obv(df)
    obv_trend = 0.0
    if len(obv_series.dropna()) > 10:
        recent = obv_series.dropna().tail(10)
        obv_trend = float(np.sign(recent.iloc[-1] - recent.iloc[0]))
    # price momentum (1M return %)
    price_mom = (row.ret.get("1M") or 0.0) * 100.0

    raw = (0.35 * rs_change
           + 0.25 * (vol_z * 10.0)
           + 0.20 * mfi_change
           + 0.10 * (obv_trend * 10.0)
           + 0.10 * price_mom)
    return float(raw)


def _normalize_proxy(rows: list[SectorRow], threshold: float = 8.0) -> None:
    if not rows:
        return
    vals = np.array([r.proxy for r in rows], dtype=float)
    peak = np.max(np.abs(vals)) or 1.0
    for r in rows:
        r.proxy = float(np.clip(r.proxy / peak * 100.0, -100.0, 100.0))
        if r.proxy >= threshold:
            r.direction = "inflow"
        elif r.proxy <= -threshold:
            r.direction = "outflow"
        else:
            r.direction = "neutral"


def rotation_summary(rows: list[SectorRow]) -> str:
    """One-line risk-on/off rotation read for the dashboard header."""
    if not rows:
        return "Sector data unavailable."
    inflow = [r.name for r in rows if r.direction == "inflow"][:3]
    outflow = [r.name for r in rows if r.direction == "outflow"][:3]
    cyclical = {"Technology", "Consumer Discretionary", "Communication Services",
                "Industrials", "Financials"}
    lead_cyc = sum(1 for r in rows[:4] if r.name in cyclical)
    tilt = ("toward growth/risk-on sectors" if lead_cyc >= 2
            else "toward defensive/risk-off sectors")
    parts = []
    if outflow:
        parts.append(f"{', '.join(outflow)} weakening")
    if inflow:
        parts.append(f"{', '.join(inflow)} strengthening")
    lead = "; ".join(parts) if parts else "Mixed sector signals"
    return f"{lead}. Leadership appears to be rotating {tilt}."


def sankey_pairs(rows: list[SectorRow], threshold: float = 8.0,
                 max_flows: int = 8) -> list[tuple[str, str, float]]:
    """Pair strongest outflow sectors with strongest inflow sectors.

    Returns [(source_sector, target_sector, value)] for the Sankey chart.
    """
    outflows = sorted([r for r in rows if r.proxy <= -threshold], key=lambda r: r.proxy)
    inflows = sorted([r for r in rows if r.proxy >= threshold],
                     key=lambda r: r.proxy, reverse=True)
    pairs: list[tuple[str, str, float]] = []
    i = j = 0
    out_rem = [abs(r.proxy) for r in outflows]
    in_rem = [abs(r.proxy) for r in inflows]
    while i < len(outflows) and j < len(inflows) and len(pairs) < max_flows:
        val = min(out_rem[i], in_rem[j])
        if val <= 0:
            break
        pairs.append((f"{outflows[i].name} ▼", f"{inflows[j].name} ▲", round(val, 2)))
        out_rem[i] -= val
        in_rem[j] -= val
        if out_rem[i] <= 1e-6:
            i += 1
        if in_rem[j] <= 1e-6:
            j += 1
    return pairs
