"""S&P 500 constituent list.

Tries a dynamic fetch (Wikipedia via pandas) first; falls back to a bundled
static seed snapshot so the universe is always available with zero network/keys.
The seed is a liquid, sector-diverse subset — Phase 2 swaps in the full dynamic
list with point-in-time handling (see docs/research_notes.md, survivorship bias).
"""
from __future__ import annotations

from ..utils.logging import get_logger

log = get_logger(__name__)

# Static seed snapshot (large, liquid names across all 11 GICS sectors).
SP500_SEED: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO",
    "ACN", "TXN", "QCOM", "IBM", "NOW", "INTU", "AMAT", "MU",
    # Communication services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX",
    # Consumer staples
    "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "MO", "PM",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SPGI", "C",
    # Health care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE", "RTX", "LMT", "DE", "UNP", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Real estate
    "PLD", "AMT", "EQIX", "SPG",
    # Materials
    "LIN", "SHW", "FCX", "APD", "NEM",
]


def _fetch_sp500() -> tuple[list[str], dict[str, str]]:
    """One Wikipedia fetch -> (symbols, {symbol: GICS sector}). Raises on failure."""
    from .wiki import read_wikipedia_tables

    tbl = read_wikipedia_tables(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    syms_raw = tbl["Symbol"].astype(str).str.replace(".", "-", regex=False)
    syms = [s.strip().upper() for s in syms_raw if s and s != "nan"]
    sectors: dict[str, str] = {}
    sec_col = next((c for c in tbl.columns if "GICS Sector" in str(c)), None)
    if sec_col is not None:
        for sym, sec in zip(syms_raw, tbl[sec_col].astype(str)):
            sym = sym.strip().upper()
            if sym and sec and sec != "nan":
                sectors[sym] = sec.strip()
    return syms, sectors


_CACHE: tuple[list[str], dict[str, str]] | None = None


def get_sp500_constituents(dynamic: bool = True) -> tuple[list[str], dict[str, str]]:
    """(symbols, {symbol: GICS sector}); the sector map is the free, full-coverage
    source used for sector caps + the dashboard treemap (yfinance .info is
    rate-limited from CI and covers only a fraction of the universe)."""
    global _CACHE
    if dynamic:
        if _CACHE is not None:
            return _CACHE
        try:
            syms, sectors = _fetch_sp500()
            if len(syms) >= 400:
                log.info("Loaded %d S&P 500 symbols dynamically", len(syms))
                _CACHE = (syms, sectors)
                return _CACHE
            log.warning("Dynamic S&P 500 list too short (%d); using seed", len(syms))
        except Exception as exc:  # noqa: BLE001
            log.warning("Dynamic S&P 500 fetch failed (%s); using static seed", exc)
    log.info("Using static S&P 500 seed (%d symbols)", len(SP500_SEED))
    return list(SP500_SEED), {}


def get_sp500(dynamic: bool = True) -> list[str]:
    """Return S&P 500 symbols. Dynamic fetch with static fallback."""
    return get_sp500_constituents(dynamic)[0]
