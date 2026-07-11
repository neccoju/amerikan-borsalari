"""Tests for the data-quality batch: Stooq fallback, Finnhub fundamentals +
cache, price quality gates, and the academic momentum upgrades."""
from __future__ import annotations

import numpy as np
import pandas as pd

from usbot.data.quality import validate_prices
from usbot.indicators.technical import momentum_12_1, risk_adjusted_momentum


# ---- Stooq parser -----------------------------------------------------------
def test_stooq_parser_maps_schema(monkeypatch):
    import usbot.data.stooq as stooq

    csv = ("Date,Open,High,Low,Close,Volume\n"
           "2026-06-26,100,102,99,101,1000000\n"
           "2026-06-29,101,104,100,103,1200000\n"
           "2026-06-30,103,105,102,104,900000\n")

    class _Resp:
        text = csv
        def raise_for_status(self):  # noqa: D401
            pass

    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    df = stooq.fetch_stooq_daily("AAPL")
    assert df is not None and list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3 and float(df["close"].iloc[-1]) == 104.0


def test_stooq_rejects_html_and_empty(monkeypatch):
    import usbot.data.stooq as stooq

    class _Resp:
        text = "<html>blocked</html>"
        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    assert stooq.fetch_stooq_daily("AAPL") is None


def test_fetch_prices_uses_stooq_for_batch_misses(monkeypatch, tmp_path):
    import usbot.data.prices as prices_mod

    def fake_download(symbols, period_days):
        # batch only returns AAA; BBB is missing
        cols = pd.MultiIndex.from_product([["AAA"], ["Close"]])
        return pd.DataFrame([[10.0], [11.0]],
                            index=pd.date_range("2026-06-29", periods=2, freq="B"),
                            columns=cols)

    fallback_frame = pd.DataFrame(
        {"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [5.0, 6.0],
         "volume": [10, 10]}, index=pd.date_range("2026-06-29", periods=2, freq="B"))

    monkeypatch.setattr(prices_mod, "_download", fake_download)
    monkeypatch.setattr("usbot.data.stooq.fetch_stooq_daily",
                        lambda sym, period_days=420: fallback_frame if sym == "BBB" else None)
    out = prices_mod.fetch_prices(["AAA", "BBB", "CCC"])
    assert "AAA" in out.history
    assert "BBB" in out.history and out.fallback_used == ["BBB"]
    assert any("CCC" in e for e in out.errors)          # both sources missed CCC


# ---- Finnhub fundamentals mapping ---------------------------------------------
def test_finnhub_units_normalized_to_yfinance_conventions(monkeypatch):
    import usbot.data.fundamentals_finnhub as fh

    payload = {"metric": {
        "revenueGrowthTTMYoy": 12.5,            # percent -> 0.125
        "netProfitMarginTTM": 20.0,             # percent -> 0.20
        "roeTTM": 30.0,                         # percent -> 0.30
        "totalDebt/totalEquityQuarterly": 1.5,  # ratio  -> 150 (yf percent-style)
        "peTTM": 25.0,
        "marketCapitalization": 1000.0,         # $M -> 1e9
        "beta": 1.2,
        "freeCashFlowTTM": 50.0,                # $M -> fcf yield 0.05
    }}

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return payload

    class _Session:
        def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr("requests.Session", lambda: _Session())
    out = fh.fetch_finnhub_fundamentals(["AAPL"], api_key="k", max_calls=1)
    m = out["AAPL"]
    assert abs(m["revenue_growth"] - 0.125) < 1e-9
    assert abs(m["profit_margin"] - 0.20) < 1e-9
    assert abs(m["return_on_equity"] - 0.30) < 1e-9
    assert abs(m["debt_to_equity"] - 150.0) < 1e-9
    assert abs(m["market_cap"] - 1e9) < 1e-3
    assert abs(m["free_cash_flow_yield"] - 0.05) < 1e-9


def test_finnhub_noop_without_key():
    from usbot.data.fundamentals_finnhub import fetch_finnhub_fundamentals

    assert fetch_finnhub_fundamentals(["AAPL"], api_key=None) == {}


# ---- fundamentals cache round-trip ---------------------------------------------
def test_fundamentals_cache_ttl(tmp_path):
    from usbot.db.repository import Repository

    with Repository(tmp_path / "f.db") as repo:
        repo.save_fundamentals_cache("AAPL", "finnhub", {"return_on_equity": 0.3})
        repo.commit()
        fresh = repo.load_fundamentals_cache(max_age_days=7)
        assert fresh["AAPL"]["return_on_equity"] == 0.3
        # zero-day TTL -> everything is stale
        assert repo.load_fundamentals_cache(max_age_days=0) == {}


# ---- price quality gates --------------------------------------------------------
def _frame(closes, end="2026-06-30", splits=None):
    idx = pd.date_range(end=end, periods=len(closes), freq="B")
    df = pd.DataFrame({"close": closes}, index=idx)
    if splits is not None:
        df["splits"] = splits
    return df


def test_quality_flags_stale_suspect_corrupt():
    today = pd.Timestamp("2026-06-30").date()
    history = {
        "OK": _frame([100, 101, 102]),
        "STALE": _frame([50, 51, 52], end="2026-06-10"),
        "JUMP": _frame([100, 100, 250]),                       # +150%, no split
        "SPLIT": _frame([100, 100, 25], splits=[0, 0, 4.0]),   # -75% BUT split day
        "BAD": _frame([100, 100, -5]),
    }
    rep = validate_prices(history, today=today)
    assert "STALE" in rep.stale
    assert "JUMP" in rep.suspects
    assert "SPLIT" not in rep.suspects        # split explains the move
    assert "BAD" in rep.corrupt
    assert "OK" not in rep.stale + rep.suspects + rep.corrupt
    assert rep.checked == 5


# ---- academic momentum ------------------------------------------------------------
def _series(n, daily, crash_last_month=0.0):
    vals = [100.0]
    for i in range(n - 1):
        r = daily
        if crash_last_month and i >= n - 22:
            r = crash_last_month
        vals.append(vals[-1] * (1 + r))
    return pd.Series(vals)


def test_momentum_12_1_skips_recent_month():
    """A last-month crash hits plain 12M momentum but not 12-1 (short-term
    reversal is exactly what the skip removes — Jegadeesh & Titman 1993)."""
    from usbot.indicators.technical import momentum

    s = _series(300, 0.002, crash_last_month=-0.02)
    plain = momentum(s, 252)
    skip = momentum_12_1(s, 252, 21)
    assert skip > plain                     # crash excluded from 12-1
    assert skip > 0.3                       # the underlying trend is intact

    # identical drift without a crash: 12-1 ≈ return of months 2..13
    s2 = _series(300, 0.002)
    assert abs(momentum_12_1(s2, 252, 21) - (s2.iloc[-22] / s2.iloc[-253] - 1.0)) < 1e-12


def test_risk_adjusted_momentum_prefers_smooth_trends():
    """Same total return, wildly different volatility -> the smooth trend wins
    (Barroso & Santa-Clara 2015 volatility scaling)."""
    rng = np.random.default_rng(3)
    smooth_rets = rng.normal(0.002, 0.005, 299)   # low-vol trend
    noisy_rets = rng.normal(0.002, 0.04, 299)     # same drift, 8x the volatility
    smooth = pd.Series(100.0 * np.cumprod(np.r_[1, 1 + smooth_rets]))
    noisy = pd.Series(100.0 * np.cumprod(np.r_[1, 1 + noisy_rets]))
    # force identical endpoint (same raw momentum)
    noisy = noisy * (smooth.iloc[-1] / noisy.iloc[-1])
    ram_smooth = risk_adjusted_momentum(smooth)
    ram_noisy = risk_adjusted_momentum(noisy)
    assert ram_smooth > ram_noisy


def test_technical_score_uses_12_1_and_risk_adj():
    """A stock with a last-month crash but intact 12-1 trend must outscore one
    with the same plain-12M momentum but a broken risk-adjusted profile."""
    from usbot.scoring.technical_score import technical_scores

    base = {"above_sma50": 1.0, "above_sma200": 1.0, "golden_cross": 1.0,
            "rsi": 60.0, "macd_hist": 1.0, "drawdown_52w": -0.05, "vol_breakout": 1.0}
    ind = {
        "SMOOTH": {**base, "mom_21": 0.02, "mom_63": 0.06, "mom_126": 0.12,
                   "mom_252": 0.25, "mom_12_1": 0.25, "risk_adj_mom": 2.0},
        "CHOPPY": {**base, "mom_21": 0.02, "mom_63": 0.06, "mom_126": 0.12,
                   "mom_252": 0.25, "mom_12_1": 0.25, "risk_adj_mom": 0.3},
    }
    scores = technical_scores(ind, {})
    assert scores["SMOOTH"] > scores["CHOPPY"]