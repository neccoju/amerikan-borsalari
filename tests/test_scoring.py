import numpy as np
import pandas as pd

from usbot.indicators.technical import compute_indicators, rsi, sma
from usbot.scoring.fundamental_score import fundamental_scores
from usbot.scoring.normalize import percentile_rank
from usbot.scoring.technical_score import technical_scores


def _trending_df(n=300, start=100.0, drift=0.5):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = start + np.arange(n) * drift
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close, "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def test_sma_and_rsi_basic():
    df = _trending_df()
    s = sma(df["close"], 50)
    assert s.iloc[-1] < df["close"].iloc[-1]  # uptrend: price above its MA
    r = rsi(df["close"], 14)
    assert r.iloc[-1] > 50  # steady uptrend -> RSI elevated


def test_percentile_rank_direction():
    vals = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    hi = percentile_rank(vals, higher_better=True)
    lo = percentile_rank(vals, higher_better=False)
    assert hi["c"] > hi["a"]
    assert lo["a"] > lo["c"]


def test_percentile_rank_nan_is_neutral():
    vals = pd.Series({"a": 1.0, "b": np.nan, "c": 3.0})
    out = percentile_rank(vals)
    assert out["b"] == 50.0


def test_technical_score_uptrend_beats_downtrend():
    cfg = {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
           "atr_period": 14, "momentum_lookbacks": [21, 63, 126, 252],
           "momentum_weights": [0.15, 0.25, 0.30, 0.30]}
    up = compute_indicators(_trending_df(drift=0.5), cfg)
    down = compute_indicators(_trending_df(drift=-0.3, start=300.0), cfg)
    scores = technical_scores({"UP": up, "DOWN": down}, cfg)
    assert scores["UP"] > scores["DOWN"]


def test_fundamental_score_ranges():
    funds = {
        "GOOD": {"revenue_growth": 0.3, "earnings_growth": 0.3, "profit_margin": 0.25,
                 "return_on_equity": 0.3, "debt_to_equity": 10, "valuation_pe": 15},
        "BAD": {"revenue_growth": -0.1, "earnings_growth": -0.2, "profit_margin": 0.01,
                "return_on_equity": 0.02, "debt_to_equity": 300, "valuation_pe": 90},
    }
    cfg = {"weights": {"revenue_growth": 0.2, "earnings_growth": 0.2, "profit_margin": 0.15,
                       "return_on_equity": 0.15, "free_cash_flow_yield": 0.1,
                       "debt_to_equity": 0.1, "valuation_pe": 0.1}}
    s = fundamental_scores(funds, cfg)
    assert 0 <= s["BAD"] <= 100 and 0 <= s["GOOD"] <= 100
    assert s["GOOD"] > s["BAD"]
