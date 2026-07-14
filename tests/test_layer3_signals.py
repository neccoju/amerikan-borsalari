"""Layer-3 tests: short-interest factor + Deflated/Probabilistic Sharpe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from usbot.backtest.metrics import (
    deflated_sharpe_from_equity,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from usbot.scoring.short_interest import short_interest_highlights, short_interest_scores


# ---- short-interest factor ---------------------------------------------------
def test_low_short_interest_scores_above_high():
    fund = {
        "LOW":  {"short_percent_float": 0.01, "shares_short": 100, "shares_short_prior": 100},
        "HIGH": {"short_percent_float": 0.30, "shares_short": 100, "shares_short_prior": 100},
    }
    s = short_interest_scores(fund, ["LOW", "HIGH"])
    assert s["LOW"] > s["HIGH"]                     # crowded short -> bearish -> lower score


def test_rising_short_interest_penalized_vs_covering():
    # equal LEVEL, so only the CHANGE leg differentiates them
    fund = {
        "RISING":  {"short_percent_float": 0.10, "shares_short": 150, "shares_short_prior": 100},
        "COVERING": {"short_percent_float": 0.10, "shares_short": 60, "shares_short_prior": 100},
    }
    s = short_interest_scores(fund, ["RISING", "COVERING"], level_weight=0.5)
    assert s["COVERING"] > s["RISING"]              # short covering is bullish


def test_missing_short_data_is_neutral():
    fund = {"A": {"profit_margin": 0.2}, "B": {}}   # no short fields at all
    s = short_interest_scores(fund, ["A", "B"])
    assert abs(s["A"] - 50.0) < 1e-9 and abs(s["B"] - 50.0) < 1e-9


def test_short_scores_reindex_to_universe():
    fund = {"A": {"short_percent_float": 0.2}}
    s = short_interest_scores(fund, ["A", "B", "C"])
    assert list(s.index) == ["A", "B", "C"]
    assert s["B"] == 50.0 and s["C"] == 50.0        # names without data stay neutral


def test_highlights_sorted_by_pct_float():
    fund = {
        "X": {"short_percent_float": 0.05, "shares_short": 10, "shares_short_prior": 8},
        "Y": {"short_percent_float": 0.25},
        "Z": {"profit_margin": 0.1},                # no short data -> excluded
    }
    hl = short_interest_highlights(fund, ["X", "Y", "Z"])
    assert [r["symbol"] for r in hl] == ["Y", "X"]
    assert abs(hl[1]["change"] - 0.25) < 1e-9       # (10-8)/8


# ---- Probabilistic / Deflated Sharpe ----------------------------------------
def test_psr_increases_with_sample_size():
    # same observed Sharpe, more observations -> more confidence it beats 0
    lo = probabilistic_sharpe_ratio(0.1, n=50, skew=0.0, kurtosis=3.0)
    hi = probabilistic_sharpe_ratio(0.1, n=500, skew=0.0, kurtosis=3.0)
    assert 0.0 <= lo < hi <= 1.0


def test_negative_skew_and_fat_tails_lower_psr():
    base = probabilistic_sharpe_ratio(0.1, n=250, skew=0.0, kurtosis=3.0)
    worse = probabilistic_sharpe_ratio(0.1, n=250, skew=-1.0, kurtosis=8.0)
    assert worse < base


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(2, sharpe_variance=0.01)
    b = expected_max_sharpe(100, sharpe_variance=0.01)
    assert b > a > 0.0


def test_deflated_sharpe_below_psr_when_many_trials():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    rets = rng.normal(0.0006, 0.01, len(dates))     # a mildly positive Sharpe
    equity = pd.Series(10000 * np.cumprod(1 + rets), index=dates)
    one = deflated_sharpe_from_equity(equity, n_trials=1)
    many = deflated_sharpe_from_equity(equity, n_trials=200)
    assert one.benchmark_sharpe == 0.0
    assert many.benchmark_sharpe > 0.0
    assert many.dsr < one.psr_vs_zero               # deflation raises the bar
    assert 0.0 <= many.dsr <= 1.0


def test_deflated_sharpe_handles_degenerate_curve():
    flat = pd.Series([100.0, 100.0, 100.0, 100.0])
    ds = deflated_sharpe_from_equity(flat, n_trials=10)
    assert ds.dsr == 0.0 and ds.sharpe_period == 0.0
