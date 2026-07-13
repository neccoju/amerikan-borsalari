"""Insider (SEC Form 4) trade model + routine/opportunistic classification.

Corporate insiders must file Form 4 within 2 business days of trading their own
company's stock — far fresher than 13F (quarterly, +45 days). The signal that
predicts returns is NOT all insider activity: Cohen, Malloy & Pomorski (2012)
show that "routine" traders (who trade the same calendar month every year) carry
no predictive power, while "opportunistic" open-market purchases do (~+0.82%/mo).

We approximate that decomposition with what a single run can see:
- transaction code "P" (open-market purchase) is the strong buy signal;
  "S" (open-market sale) is noisy (diversification/taxes/10b5-1) and weighted down;
- planned (10b5-1) trades are treated as routine when the flag is present;
- CLUSTER buys — several distinct insiders buying the same name in the window —
  are the highest-conviction case.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# Form 4 transaction codes we care about (open-market only).
OPEN_MARKET_BUY = "P"
OPEN_MARKET_SELL = "S"


@dataclass
class InsiderTrade:
    symbol: str
    insider: str
    title: str                 # CEO/CFO/Director/10% owner/...
    code: str                  # Form 4 transaction code (P, S, A, M, ...)
    shares: float
    price: float
    date: dt.date | None
    is_planned: bool = False   # 10b5-1 pre-arranged -> treated as routine

    @property
    def value(self) -> float:
        return self.shares * self.price

    @property
    def is_open_market_buy(self) -> bool:
        return self.code == OPEN_MARKET_BUY and not self.is_planned

    @property
    def is_open_market_sell(self) -> bool:
        return self.code == OPEN_MARKET_SELL

    @property
    def is_senior(self) -> bool:
        t = (self.title or "").lower()
        return any(k in t for k in ("chief", "ceo", "cfo", "coo", "president",
                                    "chair", "10%", "director"))


def classify(code: str, is_planned: bool) -> str:
    """'opportunistic_buy' | 'sell' | 'other' — the label the score keys on."""
    if code == OPEN_MARKET_BUY and not is_planned:
        return "opportunistic_buy"
    if code == OPEN_MARKET_SELL:
        return "sell"
    return "other"
