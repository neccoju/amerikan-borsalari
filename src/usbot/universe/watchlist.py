"""Curated small/mid-cap growth watchlist + benchmark/defensive ETFs.

These names bypass the large-cap market-cap floor (watchlist_relax_filters) so
quality small/mid caps can be tracked without scanning every microcap daily.
"""
from __future__ import annotations

# High-quality small/mid-cap growth candidates (illustrative, editable).
SMALL_MID_WATCHLIST: list[str] = [
    "CRWD", "DDOG", "SNOW", "NET", "ZS", "PANW", "MDB", "TEAM", "HUBS",
    "TTD", "ROKU", "RBLX", "U", "PLTR", "SHOP", "SE", "MELI",
    "ENPH", "FSLR", "ON", "ALB",
    "DKNG", "ABNB", "UBER", "LULU", "DECK",
]

# Benchmark ETFs shown on the dashboard (cards + Portfolio-vs-Benchmarks).
BENCHMARK_ETFS: dict[str, str] = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000",
    "VTI": "Total US Market", "GLD": "Gold", "TLT": "20Y Treasuries", "SGOV": "0-3M T-Bills",
}

# The 11 GICS sector SPDRs for sector-rotation / RRG analysis (+ SPY benchmark).
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}

# ETFs used for benchmark / defensive / cash-like exposure (not for scoring).
ETF_UNIVERSE: list[str] = sorted(set(
    ["SPY", "QQQ", "RSP", "IWM", "TLT", "IEF", "SHV", "BIL", "GLD"]
    + list(BENCHMARK_ETFS) + list(SECTOR_ETFS)
))


def get_watchlist() -> list[str]:
    return list(SMALL_MID_WATCHLIST)


def get_etfs() -> list[str]:
    return list(ETF_UNIVERSE)


def get_benchmark_etfs() -> dict[str, str]:
    return dict(BENCHMARK_ETFS)


def get_sector_etfs() -> dict[str, str]:
    return dict(SECTOR_ETFS)
