"""News ingestion from free APIs with graceful skip.

Providers (in priority order):
  1. Finnhub  company-news      (needs FINNHUB_API_KEY)
  2. Alpha Vantage NEWS_SENTIMENT (needs ALPHA_VANTAGE_API_KEY)

If no key is configured, ``fetch_news`` returns an empty result with a skip
reason — the bot keeps running and the news factor stays disabled. Per-symbol
isolation: one failing ticker never aborts the batch.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..config.secrets import Secrets
from ..utils.logging import get_logger
from ..utils.retry import with_retry
from .model import NewsItem, dedup

log = get_logger(__name__)


@dataclass
class NewsResult:
    items: dict[str, list[NewsItem]] = field(default_factory=dict)  # symbol -> items
    provider: str = "none"
    enabled: bool = False
    skip_reason: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.items.values())


def fetch_news(symbols: list[str], secrets: Secrets, *, days: int = 3,
               max_per_symbol: int = 10) -> NewsResult:
    """Fetch recent news for symbols. Chooses an available provider or skips."""
    if secrets.has("FINNHUB_API_KEY"):
        return _fetch_finnhub(symbols, secrets.get("FINNHUB_API_KEY"), days, max_per_symbol)
    if secrets.has("ALPHA_VANTAGE_API_KEY"):
        return _fetch_alpha_vantage(symbols, secrets.get("ALPHA_VANTAGE_API_KEY"),
                                    max_per_symbol)
    log.info("News skipped: no FINNHUB_API_KEY / ALPHA_VANTAGE_API_KEY")
    return NewsResult(provider="none", enabled=False,
                      skip_reason="missing FINNHUB_API_KEY / ALPHA_VANTAGE_API_KEY")


@with_retry(attempts=3, base_delay=1.0)
def _finnhub_call(client, symbol: str, frm: str, to: str):
    return client.company_news(symbol, _from=frm, to=to)


def _fetch_finnhub(symbols, api_key, days, max_per_symbol) -> NewsResult:
    res = NewsResult(provider="finnhub", enabled=True)
    try:
        import finnhub
    except Exception as exc:  # noqa: BLE001
        return NewsResult(provider="finnhub", enabled=False,
                          skip_reason=f"finnhub package missing: {exc}")
    client = finnhub.Client(api_key=api_key)
    to = dt.date.today()
    frm = to - dt.timedelta(days=days)
    for sym in symbols:
        try:
            raw = _finnhub_call(client, sym, frm.isoformat(), to.isoformat()) or []
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"news {sym}: {exc}")
            continue
        items = []
        for a in raw[:max_per_symbol]:
            ts = a.get("datetime")
            items.append(NewsItem(
                symbol=sym,
                headline=a.get("headline", "") or "",
                url=a.get("url", "") or "",
                source=a.get("source", "") or "",
                summary=a.get("summary", "") or "",
                published_at=dt.datetime.utcfromtimestamp(ts) if ts else None,
            ))
        items = dedup(items)
        if items:
            res.items[sym] = items
    log.info("Finnhub news: %d items across %d symbols", res.total, len(res.items))
    return res


def _fetch_alpha_vantage(symbols, api_key, max_per_symbol) -> NewsResult:
    """Alpha Vantage NEWS_SENTIMENT fallback. Free tier is heavily rate-limited
    (~25 req/day), so we cap aggressively and fail soft."""
    res = NewsResult(provider="alpha_vantage", enabled=True)
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        return NewsResult(provider="alpha_vantage", enabled=False,
                          skip_reason=f"requests missing: {exc}")
    # AV supports multiple tickers per call; batch to conserve the daily quota.
    tickers = ",".join(symbols[:50])
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "NEWS_SENTIMENT", "tickers": tickers, "apikey": api_key,
                    "limit": 200},
            timeout=30,
        )
        r.raise_for_status()
        feed = (r.json() or {}).get("feed", [])
    except Exception as exc:  # noqa: BLE001
        return NewsResult(provider="alpha_vantage", enabled=True,
                          skip_reason="", errors=[f"AV news: {exc}"])
    counts: dict[str, int] = {}
    for a in feed:
        for ts_obj in a.get("ticker_sentiment", []):
            sym = ts_obj.get("ticker")
            if sym not in symbols or counts.get(sym, 0) >= max_per_symbol:
                continue
            res.items.setdefault(sym, []).append(NewsItem(
                symbol=sym,
                headline=a.get("title", "") or "",
                url=a.get("url", "") or "",
                source=a.get("source", "") or "",
                summary=a.get("summary", "") or "",
            ))
            counts[sym] = counts.get(sym, 0) + 1
    for sym in list(res.items):
        res.items[sym] = dedup(res.items[sym])
    log.info("Alpha Vantage news: %d items across %d symbols", res.total, len(res.items))
    return res
