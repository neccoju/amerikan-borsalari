"""Insider (Form 4) ingestion via Finnhub's free insider-transactions endpoint.

Finnhub aggregates SEC Form 4 filings and serves them per symbol on the free
tier (``/stock/insider-transactions``), which is far more reliable from CI than
parsing EDGAR ownership XML. Bounded to the top-N scored names (like news) and
rate-limited so it shares the Finnhub quota politely. Everything is wrapped so
any failure degrades to a clean skip.

Transaction codes follow SEC Form 4: ``P`` open-market purchase, ``S``
open-market sale, ``A`` grant, ``M`` option exercise, etc. Only P/S carry the
signal (see insider.score).
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .model import InsiderTrade

log = get_logger(__name__)

_URL = "https://finnhub.io/api/v1/stock/insider-transactions"


@dataclass
class InsiderResult:
    trades: list[InsiderTrade] = field(default_factory=list)
    enabled: bool = False
    skip_reason: str = ""
    symbols_seen: int = 0
    errors: list[str] = field(default_factory=list)


def _parse_rows(symbol: str, rows: list[dict], since: dt.date) -> list[InsiderTrade]:
    out: list[InsiderTrade] = []
    for r in rows or []:
        code = str(r.get("transactionCode", "")).upper().strip()
        if code not in ("P", "S"):
            continue
        tdate = None
        for key in ("transactionDate", "filingDate"):
            raw = r.get(key)
            if raw:
                try:
                    tdate = dt.date.fromisoformat(str(raw)[:10])
                    break
                except ValueError:
                    continue
        if tdate is not None and tdate < since:
            continue
        shares = abs(float(r.get("share", 0) or 0))
        price = float(r.get("transactionPrice", 0) or 0)
        if shares <= 0 or price <= 0:
            continue
        out.append(InsiderTrade(
            symbol=symbol, insider=str(r.get("name", "")).strip(),
            title=str(r.get("position", "") or "").strip(),
            code=code, shares=shares, price=price, date=tdate,
            is_planned=bool(r.get("isPlanned", False))))
    return out


def fetch_insider_trades(symbols: list[str], api_key: str | None, *,
                         lookback_days: int = 90, max_symbols: int = 120,
                         rate_per_min: int = 40, timeout: float = 10.0) -> InsiderResult:
    """Form 4 buys/sells for up to ``max_symbols`` names in the last window."""
    res = InsiderResult()
    if not api_key:
        res.skip_reason = "missing FINNHUB_API_KEY"
        return res
    if not symbols:
        res.skip_reason = "no target symbols"
        return res
    import requests

    session = requests.Session()
    since = dt.date.today() - dt.timedelta(days=lookback_days)
    frm, to = since.isoformat(), dt.date.today().isoformat()
    interval = 60.0 / max(1, rate_per_min)
    res.enabled = True
    for i, sym in enumerate(symbols[:max_symbols]):
        try:
            resp = session.get(_URL, params={"symbol": sym, "from": frm, "to": to,
                                             "token": api_key}, timeout=timeout)
            resp.raise_for_status()
            data = (resp.json() or {}).get("data") or []
            res.trades.extend(_parse_rows(sym, data, since))
            res.symbols_seen += 1
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{sym}: {exc}")
        if i + 1 < min(len(symbols), max_symbols):
            time.sleep(interval)
    log.info("Insider: %d Form-4 P/S trades across %d symbols",
             len(res.trades), res.symbols_seen)
    return res
