"""Price data-quality gates.

Bad input silently corrupts every downstream decision (scores, fills, P/L), so
each run screens the fetched history for the classic failure modes:

- **stale series** — last bar older than the most recent completed session
  (source stopped updating; the book would be valued on old prices),
- **suspect jump** — an extreme 1-day move with no split recorded (usually a
  bad print or an unadjusted corporate action),
- **non-positive close** — corrupt row.

Flags are surfaced in the report/dashboard Data Quality panel; they do NOT drop
symbols automatically (a genuine -40% earnings gap must not be censored) —
they exist so a human can spot data problems the day they start.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from ..utils.dates import is_trading_day
from ..utils.logging import get_logger

log = get_logger(__name__)

# |1d return| above this with no split event -> suspect print
EXTREME_MOVE = 0.35


@dataclass
class QualityReport:
    checked: int = 0
    stale: list[str] = field(default_factory=list)        # symbol (last bar date)
    suspects: list[str] = field(default_factory=list)     # symbol (move, no split)
    corrupt: list[str] = field(default_factory=list)      # non-positive close

    @property
    def flags(self) -> list[str]:
        out = [f"{s} (stale)" for s in self.stale]
        out += [f"{s} (suspect move)" for s in self.suspects]
        out += [f"{s} (corrupt close)" for s in self.corrupt]
        return out


def _last_expected_session(today: dt.date | None = None) -> dt.date:
    """Most recent trading day on/before ``today`` (the newest bar we can expect)."""
    day = today or dt.date.today()
    for _ in range(10):
        if is_trading_day(day):
            return day
        day -= dt.timedelta(days=1)
    return day


def validate_prices(history: dict[str, pd.DataFrame], today: dt.date | None = None,
                    stale_grace_days: int = 3) -> QualityReport:
    """Screen per-symbol frames for staleness / suspect prints / corrupt rows."""
    rep = QualityReport()
    expected = _last_expected_session(today)
    stale_cutoff = expected - dt.timedelta(days=stale_grace_days)

    for sym, df in history.items():
        if df is None or df.empty or "close" not in df:
            continue
        rep.checked += 1
        close = df["close"].astype(float)

        last = close.iloc[-1]
        if not last > 0:
            rep.corrupt.append(sym)
            continue

        try:
            last_bar = pd.Timestamp(df.index[-1]).date()
            if last_bar < stale_cutoff:
                rep.stale.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass

        if len(close) >= 2 and close.iloc[-2] > 0:
            move = abs(last / float(close.iloc[-2]) - 1.0)
            if move > EXTREME_MOVE:
                split_today = False
                if "splits" in df.columns:
                    split_today = float(df["splits"].iloc[-1] or 0.0) not in (0.0, 1.0)
                if not split_today:
                    rep.suspects.append(sym)

    if rep.flags:
        log.warning("Price quality flags: %s", "; ".join(rep.flags[:12]))
    return rep
