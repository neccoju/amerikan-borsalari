"""Trading-day and US market calendar helpers.

Uses pandas-market-calendars (NYSE) when available, with a hard-coded NYSE
holiday fallback so date logic never depends on a network call.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from .logging import get_logger

log = get_logger(__name__)

# Minimal fallback holiday set (fixed-date + observed) used only if
# pandas-market-calendars is unavailable. Not exhaustive; the library is preferred.
_FALLBACK_FIXED_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas
}


@lru_cache(maxsize=1)
def _nyse_calendar():
    try:
        import pandas_market_calendars as mcal

        return mcal.get_calendar("NYSE")
    except Exception as exc:  # noqa: BLE001
        log.warning("pandas-market-calendars unavailable (%s); using weekend+fixed fallback", exc)
        return None


def is_trading_day(day: dt.date | None = None) -> bool:
    """Return True if ``day`` (default: today, NY date) is a US equity trading day."""
    day = day or dt.date.today()
    cal = _nyse_calendar()
    if cal is not None:
        sched = cal.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        return not sched.empty
    # Fallback: weekdays minus a few fixed holidays.
    if day.weekday() >= 5:
        return False
    return (day.month, day.day) not in _FALLBACK_FIXED_HOLIDAYS


def is_last_trading_day_of_month(day: dt.date | None = None) -> bool:
    """True if ``day`` is the final trading day of its month (rebalance trigger)."""
    day = day or dt.date.today()
    if not is_trading_day(day):
        return False
    # Walk forward to month end; if no trading day after ``day`` within the month, it's the last.
    probe = day + dt.timedelta(days=1)
    while probe.month == day.month:
        if is_trading_day(probe):
            return False
        probe += dt.timedelta(days=1)
    return True


def market_status(day: dt.date | None = None) -> str:
    """Human-readable status string for reports/logs."""
    day = day or dt.date.today()
    if is_trading_day(day):
        return "open"
    if day.weekday() >= 5:
        return "weekend"
    return "holiday"
