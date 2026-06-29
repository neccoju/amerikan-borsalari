"""Normalized congressional trade record (source-agnostic)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# Rough midpoints for the standard disclosure amount ranges (USD).
_AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8000,
    "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000,
    "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000,
    "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000,
    "$5,000,001 - $25,000,000": 15000000,
}


@dataclass
class CongressTrade:
    symbol: str
    chamber: str            # "house" | "senate"
    politician: str
    txn_type: str           # "buy" | "sell"
    traded_date: dt.date | None
    filed_date: dt.date | None
    amount_range: str = ""
    party: str = ""

    @property
    def amount_mid(self) -> float:
        return float(_AMOUNT_MIDPOINTS.get(self.amount_range.strip(), 10000))

    @property
    def signed_amount(self) -> float:
        """+ for buys, - for sells (by approximate dollar size)."""
        return self.amount_mid if self.txn_type == "buy" else -self.amount_mid


def normalize_txn_type(raw: str) -> str | None:
    """Map various disclosure verbs to buy/sell, or None if not a trade."""
    r = (raw or "").lower()
    if "purchase" in r or r in ("buy", "p"):
        return "buy"
    if "sale" in r or "sell" in r or r in ("s", "sale_full", "sale_partial"):
        return "sell"
    return None
