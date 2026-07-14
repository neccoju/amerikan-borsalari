"""Finnhub fundamentals fallback (free tier: /stock/metric).

yfinance ``.info`` is heavily rate-limited from datacenter IPs — the last CI
runs covered only ~36/750 names, starving the fundamental factor. Finnhub's
basic-financials endpoint is on the free tier and serves the same core metrics.

Unit normalization matters: the fundamental score ranks each metric
cross-sectionally, so mixing conventions WITHIN a metric corrupts the ranking.
Finnhub reports growth/margin/ROE in percent and market cap in millions; both
are converted here to yfinance conventions (decimals, absolute USD).
"""
from __future__ import annotations


from ..utils.logging import get_logger

log = get_logger(__name__)

_BASE = "https://finnhub.io/api/v1/stock/metric"

# our_field -> (finnhub metric key, scale to yfinance convention)
_FIELD_MAP: dict[str, tuple[str, float]] = {
    "revenue_growth": ("revenueGrowthTTMYoy", 0.01),      # % -> decimal
    "earnings_growth": ("epsGrowthTTMYoy", 0.01),         # % -> decimal
    "profit_margin": ("netProfitMarginTTM", 0.01),        # % -> decimal
    "return_on_equity": ("roeTTM", 0.01),                 # % -> decimal
    "debt_to_equity": ("totalDebt/totalEquityQuarterly", 100.0),  # ratio -> yf percent style
    "valuation_pe": ("peTTM", 1.0),
    "market_cap": ("marketCapitalization", 1e6),          # $M -> $
    "beta": ("beta", 1.0),
}


def _one(session, sym: str, api_key: str, timeout: float) -> dict[str, float]:
    resp = session.get(_BASE, params={"symbol": sym, "metric": "all", "token": api_key},
                       timeout=timeout)
    resp.raise_for_status()
    metric = (resp.json() or {}).get("metric") or {}
    out: dict[str, float] = {}
    for field, (key, scale) in _FIELD_MAP.items():
        val = metric.get(key)
        if isinstance(val, (int, float)):
            out[field] = float(val) * scale
    # derived, matching the yfinance path
    fcf = metric.get("freeCashFlowTTM")  # $M when present
    if isinstance(fcf, (int, float)) and out.get("market_cap"):
        out["free_cash_flow_yield"] = float(fcf) * 1e6 / out["market_cap"]
    return out


def fetch_finnhub_fundamentals(symbols: list[str], api_key: str | None,
                               max_calls: int = 150, rate_per_min: int = 55,
                               timeout: float = 10.0) -> dict[str, dict[str, float]]:
    """Best-effort metrics for up to ``max_calls`` symbols.

    Paced through the process-wide Finnhub limiter shared with insider, earnings
    and news, so the free-tier quota is respected globally (no cross-module
    bursts). Per-symbol isolation; empty dict when the key is missing. With the
    7-day cache the bounded budget converges to full universe coverage over a few
    runs and then just rolls the weekly refresh.
    """
    if not api_key or not symbols:
        return {}
    import requests

    from ..utils.ratelimit import get_limiter

    limiter = get_limiter("finnhub", rate_per_min)
    session = requests.Session()
    out: dict[str, dict[str, float]] = {}
    for sym in symbols[:max_calls]:
        limiter.acquire()
        try:
            metrics = _one(session, sym, api_key, timeout)
            if metrics:
                out[sym] = metrics
        except Exception as exc:  # noqa: BLE001
            log.debug("finnhub metrics failed for %s: %s", sym, exc)
    log.info("Finnhub fundamentals: %d/%d symbols (budget %d)",
             len(out), min(len(symbols), max_calls), max_calls)
    return out
