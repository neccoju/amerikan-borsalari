"""Post-earnings-announcement drift (PEAD) score + earnings blackout.

Bernard & Thomas (1989/90): prices keep drifting in the direction of the
earnings surprise for ~60 trading days after the announcement — investors
underreact to the news in current earnings for future earnings. We turn the most
recent surprise into a 0..100 score that DECAYS to neutral over the drift window,
so a name is boosted right after a positive beat and fades back as the drift is
presumed arbitraged away.

The blackout is the risk-control complement: a name reporting within a few days
should NOT get a fresh Active-sleeve entry — that would be an unintended bet on
the binary earnings outcome, not on the signal.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .model import EarningsSurprise, UpcomingEarnings

# drift window (trading days ~ 60; use calendar days for a simple decay)
DRIFT_DAYS = 63


def pead_scores(surprises: list[EarningsSurprise], universe: list[str],
                today: dt.date | None = None, drift_days: int = DRIFT_DAYS) -> pd.Series:
    """0..100 PEAD score from each name's most recent surprise, time-decayed."""
    today = today or dt.date.today()
    latest: dict[str, EarningsSurprise] = {}
    for s in surprises:
        cur = latest.get(s.symbol)
        if cur is None or s.period > cur.period:
            latest[s.symbol] = s

    out: dict[str, float] = {}
    for sym in universe:
        s = latest.get(sym)
        if s is None:
            out[sym] = 50.0
            continue
        age = (today - s.period).days
        if age < 0 or age > drift_days:
            out[sym] = 50.0
            continue
        decay = 1.0 - age / drift_days                     # 1 at report -> 0 at window end
        # compress the surprise: +/-25% surprise ~ full tilt; clamp
        tilt = max(-1.0, min(1.0, s.surprise_pct / 0.25))
        out[sym] = float(max(0.0, min(100.0, 50.0 + tilt * 30.0 * decay)))
    return pd.Series(out, dtype=float)


def earnings_blackout(upcoming: list[UpcomingEarnings], today: dt.date | None = None,
                      days_ahead: int = 5) -> set[str]:
    """Symbols reporting within ``days_ahead`` calendar days (inclusive) — the
    Active sleeve should not open a NEW position in these before the print."""
    today = today or dt.date.today()
    horizon = today + dt.timedelta(days=days_ahead)
    return {u.symbol for u in upcoming if today <= u.date <= horizon}
