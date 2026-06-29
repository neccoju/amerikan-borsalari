"""Backtesting (Phase 2). Look-ahead-safe engine + metrics."""
from .engine import (
    BacktestConfig,
    BacktestResult,
    momentum_weight_fn,
    run_backtest,
    walk_forward_windows,
)
from .metrics import Metrics, compute_metrics
from .walkforward import WalkForwardComparison, walk_forward_compare

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "momentum_weight_fn",
    "walk_forward_windows",
    "Metrics",
    "compute_metrics",
    "WalkForwardComparison",
    "walk_forward_compare",
]
