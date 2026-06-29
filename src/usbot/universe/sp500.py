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


def get_sp500(dynamic: bool = True) -> list[str]:
    """Return S&P 500 symbols. Dynamic fetch with static fallback."""
    if dynamic:
        try:
            import pandas as pd

            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            syms = [s.strip().upper() for s in syms if s and s != "nan"]
            if len(syms) >= 400:
                log.info("Loaded %d S&P 500 symbols dynamically", len(syms))
                return syms
            log.warning("Dynamic S&P 500 list too short (%d); using seed", len(syms))
        except Exception as exc:  # noqa: BLE001
            log.warning("Dynamic S&P 500 fetch failed (%s); using static seed", exc)
    log.info("Using static S&P 500 seed (%d symbols)", len(SP500_SEED))
    return list(SP500_SEED)
