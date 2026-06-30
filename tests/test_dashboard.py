"""Tests for the interactive dashboard: sector analytics, smart-money proxy,
RRG quadrants, sankey pairing, performance prep, and a full offline render."""
from __future__ import annotations

import numpy as np
import pandas as pd

from usbot.dashboard import sectors, perf, charts, build_dashboard
from usbot.dashboard.sectors import SectorRow
from usbot.reports.builder import PortfolioReport, ReportContext


# ---- synthetic price helpers ----------------------------------------------
def _ohlcv(closes: list[float], vol: float = 1e6) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "open": c * 0.99, "high": c * 1.01, "low": c * 0.98,
        "close": c, "adj_close": c, "volume": vol,
    }, index=idx)


def _trend(n: int, start: float, daily: float, vol: float = 1e6) -> pd.DataFrame:
    closes = [start * (1 + daily) ** i for i in range(n)]
    return _ohlcv(closes, vol)


# ---- sector analytics ------------------------------------------------------
def test_returns_multi_horizon():
    df = _trend(80, 100.0, 0.01)
    r = sectors._ret(df["close"], 1)
    assert abs(r - 0.01) < 1e-6


def test_quadrant_classification():
    assert sectors._classify_quadrant(101, 101) == "Leading"
    assert sectors._classify_quadrant(101, 99) == "Weakening"
    assert sectors._classify_quadrant(99, 99) == "Lagging"
    assert sectors._classify_quadrant(99, 101) == "Improving"


def test_compute_sector_rows_and_proxy_range():
    prices = {"SPY": _trend(120, 400, 0.001)}
    etfs = {"XLK": "Technology", "XLU": "Utilities", "XLE": "Energy"}
    prices["XLK"] = _trend(120, 100, 0.004, vol=2e6)   # strong leader
    prices["XLU"] = _trend(120, 50, -0.002, vol=5e5)    # laggard
    prices["XLE"] = _trend(120, 80, 0.001)
    rows = sectors.compute_sector_rows(prices, etfs, bench="SPY")
    assert len(rows) == 3
    for r in rows:
        assert -100.0 <= r.proxy <= 100.0
        assert r.direction in ("inflow", "outflow", "neutral")
        assert r.quadrant in ("Leading", "Weakening", "Lagging", "Improving")
    # the strong, high-volume sector should out-rank the declining one on RS
    names = [r.etf for r in rows]
    assert names.index("XLK") < names.index("XLU")


def test_compute_sector_rows_empty_without_benchmark():
    assert sectors.compute_sector_rows({}, {"XLK": "Technology"}, bench="SPY") == []


def test_smart_money_proxy_weights():
    """Proxy uses the documented 0.35/0.25/0.20/0.10/0.10 blend."""
    df = _trend(120, 100, 0.003, vol=2e6)
    row = SectorRow(etf="XLK", name="Technology")
    row.ret = {"1M": 0.05}
    row.momentum = 4.0
    raw = sectors._smart_money_proxy(df, df["close"], df["close"], row)
    assert isinstance(raw, float)
    # RS-change term dominates a flat-volume series
    assert raw == 0.35 * row.momentum + 0.25 * (sectors._zscore_last(df["volume"].astype(float)) * 10.0) \
        + 0.20 * (sectors.mfi(df).iloc[-1] - sectors.mfi(df).iloc[-6]) \
        + 0.10 * (np.sign(sectors.obv(df).tail(10).iloc[-1] - sectors.obv(df).tail(10).iloc[0]) * 10.0) \
        + 0.10 * (row.ret["1M"] * 100.0)


def test_sankey_pairing_conserves_flow():
    rows = [
        SectorRow(etf="XLK", name="Technology", proxy=60.0, direction="inflow"),
        SectorRow(etf="XLY", name="Consumer Disc", proxy=20.0, direction="inflow"),
        SectorRow(etf="XLU", name="Utilities", proxy=-50.0, direction="outflow"),
        SectorRow(etf="XLP", name="Staples", proxy=-15.0, direction="outflow"),
        SectorRow(etf="XLE", name="Energy", proxy=2.0, direction="neutral"),
    ]
    pairs = sectors.sankey_pairs(rows, threshold=8.0)
    assert pairs, "expected at least one rotation flow"
    # strongest outflow paired with strongest inflow first
    assert pairs[0][0].startswith("Utilities") and pairs[0][1].startswith("Technology")
    # total routed flow cannot exceed total outflow magnitude
    total = sum(v for _, _, v in pairs)
    assert total <= 50.0 + 15.0 + 1e-6
    # neutral sector never appears
    joined = " ".join(s + d for s, d, _ in pairs)
    assert "Energy" not in joined


def test_rotation_summary_text():
    rows = [SectorRow(etf="XLK", name="Technology", proxy=40, direction="inflow"),
            SectorRow(etf="XLU", name="Utilities", proxy=-40, direction="outflow")]
    txt = sectors.rotation_summary(rows)
    assert "Utilities" in txt and "Technology" in txt
    assert sectors.rotation_summary([]) == "Sector data unavailable."


# ---- performance prep ------------------------------------------------------
def test_benchmark_equity_normalized_to_100():
    prices = {"SPY": _trend(60, 400, 0.001)}
    eq = perf.benchmark_equity(prices, "SPY")
    assert eq is not None and abs(eq.iloc[0] - 100.0) < 1e-9
    assert eq.iloc[-1] > 100.0


def test_portfolio_equity_needs_two_points():
    assert perf.portfolio_equity([{"date": "2024-01-01", "total_value": 1000}]) is None
    eq = perf.portfolio_equity([
        {"date": "2024-01-01", "total_value": 1000},
        {"date": "2024-01-02", "total_value": 1010}])
    assert eq is not None and len(eq) == 2 and abs(eq.iloc[0] - 100.0) < 1e-9


def test_perf_table_alpha_vs_benchmarks():
    prices = {"SPY": _trend(260, 400, 0.0005), "QQQ": _trend(260, 350, 0.0008)}
    series = [
        perf.PerfSeries("Growth", perf.benchmark_equity(prices, "QQQ"), is_portfolio=True),
        perf.PerfSeries("SPY", perf.benchmark_equity(prices, "SPY")),
    ]
    table = perf.build_perf_table(series, prices)
    growth = next(r for r in table if r.name == "Growth")
    # Growth tracks QQQ which outperformed SPY -> positive alpha vs SPY
    assert growth.alpha_spy > 0


# ---- chart graceful placeholders -------------------------------------------
def test_charts_graceful_on_empty():
    assert "placeholder" in charts.treemap([])
    assert "placeholder" in charts.perf_lines([])
    assert "placeholder" in charts.sector_bar([])
    assert "placeholder" in charts.rrg_scatter([])
    assert "placeholder" in charts.sankey([])


def test_treemap_renders_div():
    rows = [{"symbol": "AAPL", "sector": "Technology", "size": 3e12, "color": 0.01, "label": "+1.0%"}]
    out = charts.treemap(rows)
    assert "plotly" in out.lower() or "<div" in out


# ---- full offline render ---------------------------------------------------
class _Scores:
    def __init__(self):
        idx = ["AAPL", "MSFT", "NVDA"]
        self.composite = {"balanced": pd.Series([90.0, 80.0, 70.0], index=idx)}
        self.factor_scores = {
            "technical": pd.Series([60, 70, 80], index=idx),
            "fundamental": pd.Series([55, 65, 75], index=idx),
            "macro": pd.Series([50, 50, 50], index=idx),
            "news": pd.Series([52, 58, 61], index=idx),
        }
        self.enabled_factors = ["technical", "fundamental", "macro", "news"]


class _Store:
    def __init__(self, data):
        self._data = data

    def _all(self):
        return self._data


def test_full_dashboard_render(tmp_path):
    ctx = ReportContext(date="2026-06-30", market_status="closed")
    ctx.regime_label = "Risk-On"
    ctx.regime_score = 72.0
    ctx.portfolios = [PortfolioReport(
        name="Growth", total_value=1080.0, cash=80.0, daily_pl=12.5, total_pl=80.0,
        holdings=[{"symbol": "AAPL", "shares": 2.5, "price": 200.0, "weight": 0.5, "pl_pct": 0.08}],
        actions=["BUY AAPL 2.5 @ 200.00"])]
    ctx.news_highlights = [{"symbol": "AAPL", "headline": "Apple beats", "label": "positive", "sentiment": 0.4}]

    prices = {
        "SPY": _trend(260, 400, 0.0005), "QQQ": _trend(260, 350, 0.0008),
        "AAPL": _trend(260, 180, 0.001), "MSFT": _trend(260, 300, 0.0009),
        "NVDA": _trend(260, 500, 0.002),
        "XLK": _trend(260, 200, 0.001), "XLU": _trend(260, 60, -0.0005),
    }
    fundamentals = {
        "AAPL": {"sector": "Technology", "market_cap": 3e12},
        "MSFT": {"sector": "Technology", "market_cap": 2.8e12},
        "NVDA": {"sector": "Technology", "market_cap": 2.5e12},
    }
    store = _Store({"portfolios": {"Growth": {"history": [
        {"date": "2026-06-27", "total_value": 1000.0},
        {"date": "2026-06-28", "total_value": 1020.0},
        {"date": "2026-06-30", "total_value": 1080.0}]}}})

    out = tmp_path / "index.html"
    # LLM unavailable -> must not crash and must show the unavailable message
    p = build_dashboard(ctx, prices, fundamentals, _Scores(), store,
                        llm_review="", llm_available=False, out_path=out)
    html = p.read_text()
    assert p.exists() and len(html) > 2000
    for section in ("Executive Overview", "Portfolio vs Benchmarks", "Market Heatmap",
                    "Sector Rotation", "Smart Money Rotation Proxy", "Holdings and Signals",
                    "Monthly AI Portfolio Review", "Data Quality"):
        assert section in html, f"missing section: {section}"
    assert "LLM review unavailable" in html
    assert "NOT actual dollar flow" in html
    assert "AAPL" in html


def test_dashboard_render_with_llm(tmp_path):
    ctx = ReportContext(date="2026-06-30", market_status="closed")
    out = tmp_path / "index.html"
    p = build_dashboard(ctx, {}, {}, None, None,
                        llm_review="Growth: keep. Risks: concentration.",
                        llm_available=True, out_path=out)
    html = p.read_text()
    assert "Growth: keep" in html
    assert "LLM review unavailable" not in html
