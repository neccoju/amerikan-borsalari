"""Earnings ingestion via Finnhub (free tier).

- ``/stock/earnings?symbol=`` -> historical actual vs estimate EPS (PEAD signal),
  bounded to the top-N scored names and rate-limited.
- ``/calendar/earnings?from=&to=`` -> a single call listing every upcoming report
  in the window (the blackout set). One request covers the whole universe.

All wrapped: any failure degrades to a clean skip.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .model import EarningsSurprise, UpcomingEarnings

log = get_logger(__name__)

_SURPRISE_URL = "https://finnhub.io/api/v1/stock/earnings"
_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/earnings"


@dataclass
class EarningsResult:
    surprises: list[EarningsSurprise] = field(default_factory=list)
    upcoming: list[UpcomingEarnings] = field(default_factory=list)
    enabled: bool = False
    skip_reason: str = ""
    errors: list[str] = field(default_factory=list)


def _parse_surprises(symbol: str, rows: list[dict], since: dt.date) -> list[EarningsSurprise]:
    out: list[EarningsSurprise] = []
    for r in rows or []:
        try:
            period = dt.date.fromisoformat(str(r.get("period"))[:10])
        except (TypeError, ValueError):
            continue
        if period < since:
            continue
        actual, est = r.get("actual"), r.get("estimate")
        if actual is None or est is None or float(est) == 0:
            continue
        actual, est = float(actual), float(est)
        sp = r.get("surprisePercent")
        surprise = float(sp) / 100.0 if sp is not None else (actual - est) / abs(est)
        out.append(EarningsSurprise(symbol=symbol, period=period, eps_actual=actual,
                                    eps_estimate=est, surprise_pct=surprise))
    return out


def fetch_earnings(symbols: list[str], api_key: str | None, *,
                   lookback_days: int = 90, days_ahead: int = 10,
                   max_symbols: int = 120, rate_per_min: int = 55,
                   timeout: float = 10.0) -> EarningsResult:
    res = EarningsResult()
    if not api_key:
        res.skip_reason = "missing FINNHUB_API_KEY"
        return res
    import requests

    from ..utils.ratelimit import get_limiter

    limiter = get_limiter("finnhub", rate_per_min)
    session = requests.Session()
    today = dt.date.today()
    since = today - dt.timedelta(days=lookback_days)
    res.enabled = True

    # ---- upcoming calendar: one call for the whole universe window ----
    limiter.acquire()
    try:
        resp = session.get(_CALENDAR_URL, params={
            "from": today.isoformat(),
            "to": (today + dt.timedelta(days=days_ahead)).isoformat(),
            "token": api_key}, timeout=timeout)
        resp.raise_for_status()
        want = set(symbols)
        for r in (resp.json() or {}).get("earningsCalendar") or []:
            sym = str(r.get("symbol", "")).upper()
            if sym not in want:
                continue
            try:
                d = dt.date.fromisoformat(str(r.get("date"))[:10])
            except (TypeError, ValueError):
                continue
            res.upcoming.append(UpcomingEarnings(sym, d, str(r.get("hour", "") or "")))
    except Exception as exc:  # noqa: BLE001
        res.errors.append(f"calendar: {exc}")

    # ---- per-symbol surprises (bounded, globally rate-limited) ----
    for sym in symbols[:max_symbols]:
        limiter.acquire()
        try:
            resp = session.get(_SURPRISE_URL, params={"symbol": sym, "token": api_key},
                               timeout=timeout)
            resp.raise_for_status()
            res.surprises.extend(_parse_surprises(sym, resp.json() or [], since))
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{sym}: {exc}")
    log.info("Earnings: %d surprises, %d upcoming reports",
             len(res.surprises), len(res.upcoming))
    return res
