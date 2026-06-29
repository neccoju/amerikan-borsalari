"""Build the daily active universe and apply liquidity/size filters."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.logging import get_logger
from .sp500 import get_sp500
from .watchlist import get_etfs, get_watchlist

log = get_logger(__name__)


@dataclass
class Universe:
    symbols: list[str] = field(default_factory=list)        # tradable equities
    etfs: list[str] = field(default_factory=list)           # benchmark/defensive
    watchlist: set[str] = field(default_factory=set)        # relaxed-filter names
    dropped: dict[str, str] = field(default_factory=dict)   # symbol -> reason

    @property
    def all_symbols(self) -> list[str]:
        return sorted(set(self.symbols) | set(self.etfs))


def build_universe(settings: dict) -> Universe:
    """Assemble candidate symbols from config (filters applied later, post-price)."""
    ucfg = settings.get("universe", {})
    source = ucfg.get("source", "sp500")

    candidates: list[str] = []
    if source in ("sp500", "both"):
        candidates += get_sp500(dynamic=True)
    watch = get_watchlist() if ucfg.get("include_watchlist", True) else []
    if source in ("watchlist", "both") or ucfg.get("include_watchlist", True):
        candidates += watch

    # De-dup preserving order, cap size.
    seen: set[str] = set()
    ordered: list[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    max_names = int(ucfg.get("max_names", 600))
    if len(ordered) > max_names:
        log.info("Capping universe %d -> %d", len(ordered), max_names)
        ordered = ordered[:max_names]

    return Universe(
        symbols=ordered,
        etfs=get_etfs(),
        watchlist=set(watch),
    )


def apply_liquidity_filters(universe: Universe, prices, fundamentals: dict,
                            settings: dict) -> Universe:
    """Drop illiquid / penny / sub-cap names using fetched prices & fundamentals.

    Watchlist names skip the market-cap floor when watchlist_relax_filters is set.
    """
    ucfg = settings.get("universe", {})
    min_cap = float(ucfg.get("min_market_cap", 0))
    min_dvol = float(ucfg.get("min_avg_dollar_volume", 0))
    min_price = float(ucfg.get("min_price", 0))
    relax_watch = bool(ucfg.get("watchlist_relax_filters", True))

    kept: list[str] = []
    for sym in universe.symbols:
        if sym not in prices:
            universe.dropped[sym] = "no_price_data"
            continue
        df = prices[sym]
        if df.empty or "close" not in df:
            universe.dropped[sym] = "empty_price"
            continue
        last_close = float(df["close"].iloc[-1])
        if last_close < min_price:
            universe.dropped[sym] = f"price<{min_price}"
            continue
        # 20-day average dollar volume
        tail = df.tail(20)
        if "volume" in tail:
            adv = float((tail["close"] * tail["volume"]).mean())
            if adv < min_dvol:
                universe.dropped[sym] = "illiquid"
                continue
        is_watch = sym in universe.watchlist
        cap = fundamentals.get(sym, {}).get("market_cap")
        if cap is not None and cap < min_cap and not (is_watch and relax_watch):
            universe.dropped[sym] = "below_min_cap"
            continue
        kept.append(sym)

    log.info("Liquidity filter: kept %d, dropped %d", len(kept), len(universe.dropped))
    universe.symbols = kept
    return universe
