"""Normalized news data model, source-agnostic."""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field


@dataclass
class NewsItem:
    symbol: str
    headline: str
    url: str = ""
    source: str = ""
    summary: str = ""
    published_at: dt.datetime | None = None
    # Filled in by the sentiment stage:
    sentiment: float = 0.0          # -1..+1
    label: str = "neutral"          # positive | negative | neutral
    category: str = "general"       # earnings | analyst | legal | product | macro | general

    @property
    def hash(self) -> str:
        """Stable hash for dedup: symbol + normalized headline."""
        norm = " ".join(self.headline.lower().split())
        return hashlib.sha1(f"{self.symbol}|{norm}".encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.summary}".strip()


def dedup(items: list[NewsItem]) -> list[NewsItem]:
    """Drop duplicate headlines per symbol (keeps first occurrence)."""
    seen: set[str] = set()
    out: list[NewsItem] = []
    for it in items:
        h = it.hash
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out
