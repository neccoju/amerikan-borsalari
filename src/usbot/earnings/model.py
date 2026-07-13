"""Earnings event models for the PEAD signal and the earnings blackout."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class EarningsSurprise:
    symbol: str
    period: dt.date            # fiscal period end / report date
    eps_actual: float
    eps_estimate: float
    surprise_pct: float        # (actual - estimate) / |estimate|, as a fraction


@dataclass
class UpcomingEarnings:
    symbol: str
    date: dt.date
    hour: str = ""             # bmo (before open) / amc (after close) when known
