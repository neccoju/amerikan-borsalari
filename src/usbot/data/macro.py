"""Macro data fetch.

Phase 1 uses keyless yfinance proxies (SPY/QQQ/IWM/^VIX/^TNX). FRED is wired as
an optional enrichment in Phase 2 when FRED_API_KEY is present; absence simply
means we rely on the yfinance proxies and the report notes FRED was skipped.
"""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from .cache import Cache
from .prices import fetch_prices

log = get_logger(__name__)


def fetch_macro_series(macro_tickers: dict[str, str], period_days: int = 420,
                       cache: Cache | None = None) -> dict[str, pd.DataFrame]:
    """Fetch macro proxy series via yfinance. Returns {label: ohlcv df}."""
    symbols = list(macro_tickers.values())
    pdata = fetch_prices(symbols, period_days=period_days, cache=cache)
    out: dict[str, pd.DataFrame] = {}
    for label, sym in macro_tickers.items():
        if sym in pdata:
            out[label] = pdata[sym]
        else:
            log.warning("macro proxy missing: %s (%s)", label, sym)
    return out


def fetch_fred_series(series_ids: list[str], api_key: str) -> dict[str, pd.Series]:
    """Optional FRED enrichment (Phase 2). Returns {} on any failure."""
    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        return {sid: fred.get_series(sid) for sid in series_ids}
    except Exception as exc:  # noqa: BLE001
        log.warning("FRED fetch skipped: %s", exc)
        return {}
