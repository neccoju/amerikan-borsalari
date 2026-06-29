"""Congressional trade ingestion with multiple fallback sources.

Order of attempts (each graceful):
  1. House/Senate stock-watcher public JSON (keyless) — tries the modern
     virtual-hosted ``s3.<region>`` host first, then the legacy ``s3-<region>``.
  2. Finnhub ``congressional-trading`` endpoint (if FINNHUB_API_KEY present) —
     per-symbol over the universe; works only if the tier exposes it.

Every failure degrades to "no data" and the report notes it; per-record parsing
is defensive so a malformed row is skipped, never fatal.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..config.secrets import Secrets
from ..utils.logging import get_logger
from ..utils.retry import with_retry
from .model import CongressTrade, normalize_txn_type

log = get_logger(__name__)

# Modern virtual-hosted host (dot) first, then legacy (dash) as fallback.
HOUSE_URLS = [
    "https://house-stock-watcher-data.s3.us-west-2.amazonaws.com/data/all_transactions.json",
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
]
SENATE_URLS = [
    "https://senate-stock-watcher-data.s3.us-west-2.amazonaws.com/aggregate/all_transactions.json",
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
]


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
            return dt.datetime.strptime(s.split("T")[0] if fmt == "%Y-%m-%d" else s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


@with_retry(attempts=2, base_delay=1.0)
def _get_json(url: str):
    import requests

    r = requests.get(url, timeout=45, headers={
        "User-Agent": "Mozilla/5.0 (compatible; usbot/0.1; research)",
        "Accept": "application/json,*/*",
    })
    r.raise_for_status()
    return r.json()


def _first_working(urls: list[str], errors: list[str], label: str):
    """Return JSON from the first URL that succeeds, else None."""
    for url in urls:
        try:
            return _get_json(url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"congress {label} ({url.split('//')[1].split('/')[0]}): {exc}")
    return None


def _parse_rows(rows, chamber: str, member_key: str, universe: set[str],
                since: dt.date) -> list[CongressTrade]:
    out = []
    for row in rows or []:
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
                symbol=sym, chamber=chamber,
                politician=row.get(member_key, "") or "",
                txn_type=ttype, traded_date=traded,
                filed_date=_parse_date(row.get("disclosure_date", "")),
                amount_range=row.get("amount", "") or "",
            ))
        except Exception:  # noqa: BLE001 - skip malformed rows
            continue
    return out


def _fetch_stockwatcher(universe: set[str], since: dt.date, res: CongressResult) -> bool:
    got = False
    house = _first_working(HOUSE_URLS, res.errors, "house")
    if house is not None:
        rows = house if isinstance(house, list) else house.get("transactions", house)
        res.trades.extend(_parse_rows(rows, "house", "representative", universe, since))
        res.sources.append("house")
        got = True
    senate = _first_working(SENATE_URLS, res.errors, "senate")
    if senate is not None:
        rows = senate if isinstance(senate, list) else senate.get("transactions", senate)
        res.trades.extend(_parse_rows(rows, "senate", "senator", universe, since))
        res.sources.append("senate")
        got = True
    return got


def _parse_quiver_rows(rows, universe: set[str], since: dt.date) -> list[CongressTrade]:
    """Parse Quiver Quant live/congresstrading rows into CongressTrade records."""
    out = []
    for a in rows or []:
        try:
            sym = (a.get("Ticker") or "").strip().upper()
            if not sym or (universe and sym not in universe):
                continue
            ttype = normalize_txn_type(a.get("Transaction", ""))
            if ttype is None:
                continue
            traded = _parse_date(a.get("TransactionDate", ""))
            if traded and traded < since:
                continue
            house = (a.get("House", "") or "").lower()
            chamber = "senate" if "senate" in house else "house"
            out.append(CongressTrade(
                symbol=sym, chamber=chamber,
                politician=a.get("Representative", "") or a.get("Senator", "") or "",
                txn_type=ttype, traded_date=traded,
                filed_date=_parse_date(a.get("ReportDate", "")),
                amount_range=a.get("Range", "") or "",
                party=a.get("Party", "") or "",
            ))
        except Exception:  # noqa: BLE001 - skip malformed rows
            continue
    return out


def _fetch_quiver(universe: set[str], since: dt.date, api_key: str,
                  res: CongressResult) -> bool:
    """Quiver Quantitative bulk congressional-trading (one call). Graceful."""
    import requests

    try:
        r = requests.get(
            "https://api.quiverquant.com/beta/live/congresstrading",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=45,
        )
        if r.status_code != 200:
            res.errors.append(f"quiver congress: HTTP {r.status_code}")
            return False
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        res.errors.append(f"quiver congress: {exc}")
        return False
    trades = _parse_quiver_rows(rows, universe, since)
    if trades:
        res.trades.extend(trades)
        res.sources.append("quiver")
        return True
    return False


def _fetch_finnhub(universe: list[str], since: dt.date, api_key: str,
                   res: CongressResult, max_symbols: int = 80) -> bool:
    """Per-symbol Finnhub congressional-trading. Works only if the tier allows it."""
    import requests

    to = dt.date.today()
    got = False
    for sym in list(universe)[:max_symbols]:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/stock/congressional-trading",
                params={"symbol": sym, "from": since.isoformat(), "to": to.isoformat(),
                        "token": api_key},
                timeout=20,
            )
            if r.status_code != 200:
                res.errors.append(f"finnhub congress {sym}: HTTP {r.status_code}")
                if r.status_code in (401, 403):
                    break  # tier doesn't expose it — stop trying
                continue
            data = (r.json() or {}).get("data", [])
            for a in data:
                ttype = normalize_txn_type(a.get("transactionType", ""))
                if ttype is None:
                    continue
                res.trades.append(CongressTrade(
                    symbol=sym, chamber="congress", politician=a.get("name", "") or "",
                    txn_type=ttype, traded_date=_parse_date(a.get("transactionDate", "")),
                    filed_date=_parse_date(a.get("filingDate", "")),
                    amount_range=_finnhub_amount(a),
                ))
                got = True
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"finnhub congress {sym}: {exc}")
            continue
    if got:
        res.sources.append("finnhub")
    return got


def _finnhub_amount(a: dict) -> str:
    lo, hi = a.get("amountFrom"), a.get("amountTo")
    if lo and hi:
        return f"${int(lo):,} - ${int(hi):,}"
    return a.get("amount", "") or ""


def fetch_congress_trades(universe: list[str], *, lookback_days: int = 90,
                          secrets: Secrets | None = None) -> CongressResult:
    """Fetch recent congressional trades filtered to ``universe``. Graceful skip."""
    uni = {s.upper() for s in universe}
    since = dt.date.today() - dt.timedelta(days=lookback_days)
    res = CongressResult()

    # 1. Quiver Quantitative (preferred when a key is configured — single bulk call).
    got = False
    if secrets is not None and secrets.has("QUIVER_API_KEY"):
        got = _fetch_quiver(uni, since, secrets.get("QUIVER_API_KEY"), res)

    # 2. Free public stock-watcher datasets.
    if not got:
        got = _fetch_stockwatcher(uni, since, res)

    # 3. Finnhub (premium-gated; stops early on 401/403).
    if not got and secrets is not None and secrets.has("FINNHUB_API_KEY"):
        log.info("Congress: prior sources failed; trying Finnhub fallback")
        got = _fetch_finnhub(universe, since, secrets.get("FINNHUB_API_KEY"), res)

    if not got:
        res.enabled = False
        res.skip_reason = "congress sources unreachable (quiver/public datasets/finnhub)"
    else:
        res.enabled = True
        log.info("Congress: %d trades via %s", len(res.trades), "+".join(res.sources))
    return res
