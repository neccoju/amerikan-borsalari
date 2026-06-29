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

# ETFs used for benchmark / defensive / cash-like exposure (not for scoring).
ETF_UNIVERSE: list[str] = [
    "SPY", "QQQ", "RSP", "IWM",     # benchmarks / breadth
    "TLT", "IEF", "SHV", "BIL",     # rates / cash-like
    "XLP", "XLU", "XLV",            # defensive sectors
    "GLD",                          # diversifier
]


def get_watchlist() -> list[str]:
    return list(SMALL_MID_WATCHLIST)


def get_etfs() -> list[str]:
    return list(ETF_UNIVERSE)
