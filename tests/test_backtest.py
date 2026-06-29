import numpy as np
import pandas as pd

from usbot.backtest import (
    BacktestConfig,
    compute_metrics,
    momentum_weight_fn,
    run_backtest,
    walk_forward_windows,
)


def _series_prices(n=600, seed=1):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)

    def mk(drift, vol, start=100.0):
        rets = rng.normal(drift, vol, n)
        return pd.DataFrame({"adj_close": start * np.cumprod(1 + rets),
                             "close": start * np.cumprod(1 + rets)}, index=idx)

    return {
        "WIN": mk(0.0012, 0.012),
        "MID": mk(0.0004, 0.013),
        "LOSE": mk(-0.0005, 0.015, start=200.0),
        "SPY": mk(0.0006, 0.009, start=300.0),
    }


def test_metrics_basic_uptrend():
    idx = pd.date_range("2015-01-01", periods=252, freq="B")
    equity = pd.Series(np.linspace(100, 130, 252), index=idx)
    m = compute_metrics(equity)
    assert m.total_return > 0.25
    assert m.max_drawdown <= 0.0
    assert 0.0 <= m.hit_rate <= 1.0


def test_metrics_drawdown_detected():
    idx = pd.date_range("2015-01-01", periods=10, freq="B")
    equity = pd.Series([100, 110, 120, 90, 80, 85, 95, 100, 105, 110], index=idx)
    m = compute_metrics(equity)
    # peak 120 -> trough 80 => -33%
    assert abs(m.max_drawdown - (80 / 120 - 1.0)) < 1e-9


def test_backtest_runs_and_compares_benchmark():
    prices = _series_prices()
    cfg = BacktestConfig(start_date="2015-02-01", benchmark="SPY", cost_bps=10.0)
    res = run_backtest(prices, momentum_weight_fn(top_n=2, lookback=63), cfg)
    assert res.n_rebalances > 0
    assert res.equity.iloc[-1] > 0
    assert res.benchmark_metrics is not None
    assert "excess_cagr" in res.summary()
    # costs must drag (turnover > 0 over the run)
    assert res.cost_drag >= 0.0


def test_no_lookahead_weight_fn_only_sees_past():
    """weight_fn must never receive future data relative to the asof date."""
    prices = _series_prices(n=400)
    seen = {}

    def spy_fn(asof, history):
        # history must end at or before asof
        assert history.index.max() <= asof
        seen["max"] = history.index.max()
        return {"WIN": 1.0}

    cfg = BacktestConfig(start_date="2015-02-01", benchmark="SPY")
    run_backtest(prices, spy_fn, cfg)
    assert "max" in seen


def test_walk_forward_windows_are_ordered():
    idx = pd.date_range("2015-01-01", periods=252 * 6, freq="B")
    wins = list(walk_forward_windows(idx, train_years=2, test_years=1))
    assert wins
    for tr_s, tr_e, te_s, te_e in wins:
        assert tr_s < tr_e <= te_s < te_e
