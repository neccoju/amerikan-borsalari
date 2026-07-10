"""Build the daily active universe and apply liquidity/size filters."""
from __future__ import annotations

from dataclasses import dataclass, field


from ..utils.logging import get_logger
from .sp500 import get_sp500_constituents
from .sp1500 import get_sp400_constituents, get_sp600_constituents
from .watchlist import get_etfs, get_watchlist

log = get_logger(__name__)


@dataclass
class Universe:
    symbols: list[str] = field(default_factory=list)        # tradable equities
    etfs: list[str] = field(default_factory=list)           # benchmark/defensive
    watchlist: set[str] = field(default_factory=set)        # relaxed-filter names
    dropped: dict[str, str] = field(default_factory=dict)   # symbol -> reason
    # GICS sector per symbol from the constituent tables (full coverage, free);
    # used for sector caps + the dashboard treemap instead of yfinance .info.
    sectors: dict[str, str] = field(default_factory=dict)

    @property
    def all_symbols(self) -> list[str]:
        return sorted(set(self.symbols) | set(self.etfs))


def build_universe(settings: dict) -> Universe:
    """Assemble candidate symbols from config (filters applied later, post-price)."""
    ucfg = settings.get("universe", {})
    source = ucfg.get("source", "sp500")

    candidates: list[str] = []
    sectors: dict[str, str] = {}
    # sp500 | sp1500 | broad | watchlist | both
    if source in ("sp500", "sp1500", "broad", "both"):
        syms, secs = get_sp500_constituents(dynamic=True)
        candidates += syms
        sectors.update(secs)
    if source in ("sp1500", "broad"):
        for getter in (get_sp400_constituents, get_sp600_constituents):
            syms, secs = getter(dynamic=True)
            candidates += syms
            sectors.update(secs)
    watch = get_watchlist() if ucfg.get("include_watchlist", True) else []
    if source in ("watchlist", "both") or ucfg.get("include_watchlist", True):
        candidates += watch

    # De-dup preserving order. (Size is bounded later by the price/volume filter +
    # max_names cap, after we have liquidity data — not by arbitrary truncation.)
    seen: set[str] = set()
    ordered: list[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    log.info("Universe '%s': %d candidate symbols (%d with GICS sector)",
             source, len(ordered), len(sectors))

    return Universe(
        symbols=ordered,
        etfs=get_etfs(),
        watchlist=set(watch),
        sectors=sectors,
    )


def apply_price_liquidity_filter(universe: Universe, prices, settings: dict) -> Universe:
    """Price/volume pre-filter (NO fundamentals needed), then cap to max_names.

    Runs before fundamentals are fetched so the (slow, per-ticker) fundamentals
    call is bounded to survivors. Keeps the most liquid names when capping.
    """
    ucfg = settings.get("universe", {})
    min_dvol = float(ucfg.get("min_avg_dollar_volume", 0))
    min_price = float(ucfg.get("min_price", 0))
    max_names = int(ucfg.get("max_names", 600))

    scored: list[tuple[str, float]] = []   # (symbol, avg_dollar_volume)
    for sym in universe.symbols:
        df = prices[sym] if sym in prices else None
        if df is None or df.empty or "close" not in df:
            universe.dropped[sym] = "no_price_data"
            continue
        last_close = float(df["close"].iloc[-1])
        if last_close < min_price:
            universe.dropped[sym] = f"price<{min_price}"
            continue
        tail = df.tail(20)
        adv = float((tail["close"] * tail["volume"]).mean()) if "volume" in tail else 0.0
        if adv < min_dvol:
            universe.dropped[sym] = "illiquid"
            continue
        scored.append((sym, adv))

    # keep most-liquid first, then cap
    scored.sort(key=lambda kv: -kv[1])
    kept = [s for s, _ in scored]
    if len(kept) > max_names:
        for s in kept[max_names:]:
            universe.dropped[s] = "over_max_names"
        kept = kept[:max_names]
    log.info("Price/volume filter: kept %d, dropped %d", len(kept), len(universe.dropped))
    universe.symbols = kept
    return universe


def apply_marketcap_filter(universe: Universe, fundamentals: dict, settings: dict) -> Universe:
    """Drop sub-cap names using fundamentals. Watchlist names skip the floor when
    watchlist_relax_filters is set. Missing cap data -> kept (don't over-prune)."""
    ucfg = settings.get("universe", {})
    min_cap = float(ucfg.get("min_market_cap", 0))
    relax_watch = bool(ucfg.get("watchlist_relax_filters", True))
    if min_cap <= 0:
        return universe

    kept: list[str] = []
    for sym in universe.symbols:
        is_watch = sym in universe.watchlist
        cap = fundamentals.get(sym, {}).get("market_cap")
        if cap is not None and cap < min_cap and not (is_watch and relax_watch):
            universe.dropped[sym] = "below_min_cap"
            continue
        kept.append(sym)
    log.info("Market-cap filter: kept %d", len(kept))
    universe.symbols = kept
    return universe
