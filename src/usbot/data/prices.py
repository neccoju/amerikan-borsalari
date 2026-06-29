"""Price & fundamental fetching via yfinance (keyless).

Design guarantees:
- Per-ticker isolation: one failing symbol never aborts the batch.
- Cache-first: avoids hammering yfinance and tolerates transient outages.
- Returns plain pandas; downstream code does not depend on yfinance objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.logging import get_logger
from ..utils.retry import with_retry
from .cache import Cache

log = get_logger(__name__)


@dataclass
class PriceData:
    """Holds OHLCV history per symbol and tracks fetch errors."""

    history: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __getitem__(self, sym: str) -> pd.DataFrame:
        return self.history[sym]

    def __contains__(self, sym: str) -> bool:
        return sym in self.history

    @property
    def symbols(self) -> list[str]:
        return list(self.history.keys())


@with_retry(attempts=3, base_delay=2.0)
def _download(symbols: list[str], period_days: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        tickers=" ".join(symbols),
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty frame")
    return df


def _extract_single(df: pd.DataFrame, sym: str, multi: bool) -> pd.DataFrame | None:
    try:
        sub = df[sym] if multi else df
        sub = sub.dropna(how="all")
        if sub.empty:
            return None
        sub = sub.rename(
            columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
            }
        )
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in sub]
        return sub[keep]
    except Exception:  # noqa: BLE001
        return None


def fetch_prices(symbols: list[str], period_days: int = 420,
                 cache: Cache | None = None) -> PriceData:
    """Fetch daily OHLCV for ``symbols``. Cache-first, fail-soft per symbol."""
    out = PriceData()
    to_download: list[str] = []

    if cache is not None:
        for sym in symbols:
            if cache.fresh(f"px_{sym}"):
                cached = cache.load(f"px_{sym}")
                if cached is not None and not cached.empty:
                    out.history[sym] = cached
                    continue
            to_download.append(sym)
    else:
        to_download = list(symbols)

    if not to_download:
        log.info("All %d symbols served from cache", len(symbols))
        return out

    log.info("Downloading %d symbols from yfinance", len(to_download))
    try:
        raw = _download(to_download, period_days)
    except Exception as exc:  # noqa: BLE001
        msg = f"yfinance batch download failed: {exc}"
        log.error(msg)
        out.errors.append(msg)
        return out

    multi = isinstance(raw.columns, pd.MultiIndex)
    for sym in to_download:
        sub = _extract_single(raw, sym, multi)
        if sub is None or sub.empty:
            out.errors.append(f"no price data: {sym}")
            continue
        out.history[sym] = sub
        if cache is not None:
            cache.save(f"px_{sym}", sub)
    log.info("Fetched %d/%d symbols (%d errors)",
             len(out.history), len(symbols), len(out.errors))
    return out


_FUNDAMENTAL_FIELDS = {
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "profit_margin": "profitMargins",
    "return_on_equity": "returnOnEquity",
    "debt_to_equity": "debtToEquity",
    "valuation_pe": "trailingPE",
    "free_cash_flow": "freeCashflow",
    "market_cap": "marketCap",
    "beta": "beta",
    "sector": "sector",
}


def _fundamentals_one(sym: str) -> tuple[str, dict[str, float]]:
    """Fetch one symbol's fundamentals. Never raises; returns (sym, metrics)."""
    import yfinance as yf

    try:
        info = yf.Ticker(sym).info or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("fundamentals failed for %s: %s", sym, exc)
        return sym, {}
    metrics: dict[str, float] = {}
    for key, yf_key in _FUNDAMENTAL_FIELDS.items():
        val = info.get(yf_key)
        if val is not None:
            metrics[key] = val
    if metrics.get("free_cash_flow") and metrics.get("market_cap"):
        metrics["free_cash_flow_yield"] = metrics["free_cash_flow"] / metrics["market_cap"]
    return sym, metrics


def fetch_fundamentals(symbols: list[str], max_workers: int = 12) -> dict[str, dict[str, float]]:
    """Best-effort fundamentals via yfinance ``.info``, fetched in parallel.

    Returns {symbol: {metric: value}}. Per-ticker isolation; missing fields are
    simply absent and never abort the batch.
    """
    from concurrent.futures import ThreadPoolExecutor

    result: dict[str, dict[str, float]] = {}
    if not symbols:
        return result
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sym, metrics in pool.map(_fundamentals_one, symbols):
            if metrics:
                result[sym] = metrics
    log.info("Fundamentals fetched for %d/%d symbols", len(result), len(symbols))
    return result
