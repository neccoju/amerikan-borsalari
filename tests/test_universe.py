import pandas as pd

from usbot.universe.build import (
    Universe,
    apply_marketcap_filter,
    apply_price_liquidity_filter,
    build_universe,
)


def _px(close, vol):
    idx = pd.date_range("2026-01-01", periods=25, freq="B")
    return pd.DataFrame({"close": [close] * 25, "volume": [vol] * 25}, index=idx)


def test_price_filter_drops_penny_and_illiquid_and_caps():
    uni = Universe(symbols=["BIG", "MID", "PENNY", "THIN"], watchlist=set())
    prices = {
        "BIG": _px(500.0, 1_000_000),     # $500M/day
        "MID": _px(100.0, 500_000),       # $50M/day
        "PENNY": _px(1.0, 10_000_000),    # price < min
        "THIN": _px(50.0, 100),           # illiquid
    }
    settings = {"universe": {"min_price": 3.0, "min_avg_dollar_volume": 5_000_000,
                             "max_names": 2}}
    out = apply_price_liquidity_filter(uni, prices, settings)
    assert "PENNY" in uni.dropped and uni.dropped["PENNY"].startswith("price<")
    assert uni.dropped["THIN"] == "illiquid"
    # capped to 2, most-liquid kept first
    assert out.symbols == ["BIG", "MID"]


def test_marketcap_filter_relaxes_watchlist():
    uni = Universe(symbols=["BIGCAP", "SMALL", "WATCH"], watchlist={"WATCH"})
    funds = {"BIGCAP": {"market_cap": 5e10}, "SMALL": {"market_cap": 1e8},
             "WATCH": {"market_cap": 1e8}}
    settings = {"universe": {"min_market_cap": 2e9, "watchlist_relax_filters": True}}
    out = apply_marketcap_filter(uni, funds, settings)
    assert "BIGCAP" in out.symbols
    assert "SMALL" not in out.symbols          # below cap, dropped
    assert "WATCH" in out.symbols              # watchlist relaxed


def test_marketcap_filter_keeps_missing_cap():
    uni = Universe(symbols=["NOINFO"], watchlist=set())
    out = apply_marketcap_filter(uni, {}, {"universe": {"min_market_cap": 2e9}})
    assert out.symbols == ["NOINFO"]           # missing cap -> not over-pruned


def test_build_universe_sp500_includes_watchlist():
    uni = build_universe({"universe": {"source": "sp500", "include_watchlist": True}})
    assert len(uni.symbols) > 50
    assert uni.watchlist  # watchlist names tracked for relaxed filtering
