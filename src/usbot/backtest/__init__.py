"""Backtesting (Phase 2). Look-ahead-safe engine + metrics."""
from .engine import (
    BacktestConfig,
    BacktestResult,
    momentum_weight_fn,
    run_backtest,
    walk_forward_windows,
)
from .metrics import (
    DeflatedSharpe,
    Metrics,
    compute_metrics,
    deflated_sharpe_from_equity,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from .walkforward import WalkForwardComparison, walk_forward_compare

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "momentum_weight_fn",
    "walk_forward_windows",
    "Metrics",
    "compute_metrics",
    "DeflatedSharpe",
    "deflated_sharpe_from_equity",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "WalkForwardComparison",
    "walk_forward_compare",
]
