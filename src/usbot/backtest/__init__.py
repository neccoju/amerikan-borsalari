"""Backtesting (Phase 2). Look-ahead-safe engine + metrics."""
from .engine import (
    BacktestConfig,
    BacktestResult,
    momentum_weight_fn,
    run_backtest,
    walk_forward_windows,
)
from .metrics import Metrics, compute_metrics

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "momentum_weight_fn",
    "walk_forward_windows",
    "Metrics",
    "compute_metrics",
]
