"""Congressional trade ingestion from free public datasets.

Primary sources are the community-maintained house/senate stock-watcher JSON
dumps (keyless). These can change or rate-limit, so every failure degrades to
"no data" and the report notes it. Per-record parsing is defensive: a malformed
row is skipped, never fatal.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from ..utils.retry import with_retry
from .model import CongressTrade, normalize_txn_type

log = get_logger(__name__)

HOUSE_URL = ("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/"
             "data/all_transactions.json")
SENATE_URL = ("https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
              "aggregate/all_transactions.json")


@dataclass
class CongressResult:
    trades: list[CongressTrade] = field(default_factory=list)
    enabled: bool = False
    skip_reason: str = ""
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _parse_date(s: str) -> dt.date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:len(fmt) + 2] if "T" in fmt else s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


@with_retry(attempts=2, base_delay=1.0)
def _get_json(url: str):
    import requests

    r = requests.get(url, timeout=45, headers={"User-Agent": "usbot/0.1 (research)"})
    r.raise_for_status()
    return r.json()


def _parse_house(rows, universe: set[str], since: dt.date) -> list[CongressTrade]:
    out = []
    for row in rows:
        try:
            sym = (row.get("ticker") or "").strip().upper()
            if not sym or sym in ("--", "N/A") or (universe and sym not in universe):
                continue
            ttype = normalize_txn_type(row.get("type", ""))
            if ttype is None:
                continue
            traded = _parse_date(row.get("transaction_date", ""))
            if traded and traded < since:
                continue
            out.append(CongressTrade(
                symbol=sym, chamber="house",
                politician=row.get("representative", "") or "",
                txn_type=ttype, traded_date=traded,
                filed_date=_parse_date(row.get("disclosure_date", "")),
                amount_range=row.get("amount", "") or "",
            ))
        except Exception:  # noqa: BLE001 - skip malformed rows
            continue
    return out


def _parse_senate(rows, universe: set[str], since: dt.date) -> list[CongressTrade]:
    out = []
    for row in rows:
        try:
            sym = (row.get("ticker") or "").strip().upper()
            if not sym or sym in ("--", "N/A") or (universe and sym not in universe):
                continue
            ttype = normalize_txn_type(row.get("type", "") or row.get("transaction_type", ""))
            if ttype is None:
                continue
            traded = _parse_date(row.get("transaction_date", ""))
            if traded and traded < since:
                continue
            out.append(CongressTrade(
                symbol=sym, chamber="senate",
                politician=row.get("senator", "") or "",
                txn_type=ttype, traded_date=traded,
                filed_date=_parse_date(row.get("disclosure_date", "")),
                amount_range=row.get("amount", "") or "",
            ))
        except Exception:  # noqa: BLE001
            continue
    return out


def fetch_congress_trades(universe: list[str], *, lookback_days: int = 90,
                          house_url: str = HOUSE_URL,
                          senate_url: str = SENATE_URL) -> CongressResult:
    """Fetch recent congressional trades filtered to ``universe``. Graceful skip."""
    uni = {s.upper() for s in universe}
    since = dt.date.today() - dt.timedelta(days=lookback_days)
    res = CongressResult()
    got_any = False

    for label, url, parser in (("house", house_url, _parse_house),
                               ("senate", senate_url, _parse_senate)):
        try:
            data = _get_json(url)
            rows = data if isinstance(data, list) else data.get("transactions", data)
            trades = parser(rows, uni, since)
            res.trades.extend(trades)
            res.sources.append(label)
            got_any = True
            log.info("Congress %s: %d trades in universe (last %dd)",
                     label, len(trades), lookback_days)
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"congress {label}: {exc}")
            log.warning("Congress %s fetch failed: %s", label, exc)

    if not got_any:
        res.enabled = False
        res.skip_reason = "congress sources unreachable"
    else:
        res.enabled = True
    return res
