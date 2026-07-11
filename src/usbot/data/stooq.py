"""Stooq daily-price fallback (keyless).

When the yfinance batch returns nothing for a symbol (transient miss, rename,
source hiccup), Stooq's free CSV endpoint usually still has the daily history:

    https://stooq.com/q/d/l/?s=<symbol>.us&i=d
    -> Date,Open,High,Low,Close,Volume

Prices are split-adjusted like Yahoo's unadjusted-close convention, so frames
are interchangeable for indicators/valuation. No dividends/adj_close columns —
corporate-action credits simply skip symbols served from here (best-effort).
"""
from __future__ import annotations

import io

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

_URL = "https://stooq.com/q/d/l/?s={sym}.us&i=d"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) usbot research"}


def fetch_stooq_daily(symbol: str, period_days: int = 420,
                      timeout: float = 15.0) -> pd.DataFrame | None:
    """Daily OHLCV for one symbol from Stooq, in our frame schema. None on miss."""
    import requests

    sym = symbol.lower().replace("-", ".")  # BRK-B -> brk.b (Stooq convention)
    try:
        resp = requests.get(_URL.format(sym=sym), headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:  # noqa: BLE001
        log.debug("stooq fetch failed for %s: %s", symbol, exc)
        return None
    if not text or text.lstrip().startswith("<") or "No data" in text[:100]:
        return None
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:  # noqa: BLE001
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    needed = ("date", "open", "high", "low", "close")
    if any(k not in cols for k in needed):
        return None
    df = df.rename(columns={cols[k]: k for k in cols})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    if df.empty:
        return None
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].astype(float).tail(period_days)
    return out if len(out) >= 2 else None
