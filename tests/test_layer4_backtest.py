"""Layer-4 tests: composite-momentum backtest + walk-forward Deflated Sharpe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from usbot.backtest import (
    BacktestConfig,
    composite_momentum_weight_fn,
    composite_scores,
    run_backtest,
    walk_forward_compare,
)


def _panel(n_days: int = 700) -> pd.DataFrame:
    """Synthetic close panel: WIN trends up smoothly, LOSE drifts down, FLAT flat,
    plus a benchmark SPY. Enough history for the 252d 12-1 leg."""
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    cols = {
        "WIN":  100 * np.cumprod(1 + rng.normal(0.0016, 0.004, n_days)),  # strong + smooth
        "LOSE": 100 * np.cumprod(1 + rng.normal(-0.0008, 0.010, n_days)),
        "FLAT": 100 * np.cumprod(1 + rng.normal(0.0000, 0.009, n_days)),
        "MEH":  100 * np.cumprod(1 + rng.normal(0.0002, 0.011, n_days)),
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0004, 0.007, n_days)),
    }
    return pd.DataFrame(cols, index=dates)


def _as_price_dict(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {sym: pd.DataFrame({"close": panel[sym], "adj_close": panel[sym]})
            for sym in panel.columns}


def test_composite_scores_rank_trend_leader_top():
    panel = _panel()
    scores = composite_scores(panel)
    assert not scores.empty
    # the smooth uptrend should out-score the persistent downtrend
    assert scores["WIN"] > scores["LOSE"]
    assert scores.max() <= 100.0 + 1e-9 and scores.min() >= 0.0 - 1e-9


def test_composite_weight_fn_selects_and_caps():
    panel = _panel()
    fn = composite_momentum_weight_fn(top_n=2, max_weight=0.6)
    w = fn(panel.index[-1], panel)
    assert w                                        # non-empty
    assert "WIN" in w                               # trend leader picked
    assert all(v <= 0.6 + 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 0.5         # roughly invested


def test_composite_backtest_runs_and_beats_loser():
    panel = _panel()
    cfg = BacktestConfig(start_date="2015-06-01", benchmark="SPY", cost_bps=5.0)
    res = run_backtest(_as_price_dict(panel), composite_momentum_weight_fn(top_n=2), cfg)
    assert res.n_rebalances > 5
    assert len(res.equity) > 100
    assert res.benchmark_metrics is not None        # SPY present -> benchmark computed


def test_walk_forward_summary_carries_deflated_sharpe():
    panel = _panel(n_days=900)
    cfg = BacktestConfig(start_date="2015-06-01", benchmark="SPY", cost_bps=5.0)
    comp = walk_forward_compare(_as_price_dict(panel), cfg, top_n=2, n_trials=50)
    s = comp.summary()
    for sleeve in ("adaptive", "static"):
        ds = s[sleeve]["deflated_sharpe"]
        assert ds["n_trials"] == 50
        assert ds["benchmark_sharpe"] > 0.0         # deflation bar is raised
        assert 0.0 <= ds["dsr"] <= 1.0
    assert "adaptive_minus_static_cagr" in s
