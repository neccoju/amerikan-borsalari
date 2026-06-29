"""Look-ahead-safe backtesting engine (Phase 2).

Contract enforced here:
- **Point-in-time signals:** at each rebalance date T the ``weight_fn`` is given
  price history strictly up to and including T (``prices.loc[:T]``). It cannot
  see the future.
- **T+1 execution:** target weights decided at T are applied at the NEXT trading
  day's price, so the close that generated the signal is never the fill price.
- **Transaction costs:** charged on turnover at each rebalance (in basis points).
- **Walk-forward:** ``walk_forward_windows`` splits the timeline into rolling
  train/test segments for out-of-sample evaluation.

Data note: only price-derived signals (momentum/technical) are truly point-in-time
from free history. Fundamental/news factors lack free point-in-time history, so
fundamental/news backtesting is intentionally out of scope here (see
docs/research_notes.md, deferred to Phase 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .metrics import Metrics, compute_metrics

log = get_logger(__name__)

# weight_fn(asof_date, history_df) -> {symbol: target_weight}
WeightFn = Callable[[pd.Timestamp, pd.DataFrame], dict]


@dataclass
class BacktestConfig:
    start_date: str = "2015-01-01"
    end_date: str | None = None
    benchmark: str = "SPY"
    cost_bps: float = 10.0           # round-trip-ish per-trade cost in basis points
    rebalance: str = "ME"            # pandas offset alias: 'ME' month-end, 'W-FRI', etc.
    initial_capital: float = 10000.0


@dataclass
class BacktestResult:
    equity: pd.Series
    metrics: Metrics
    benchmark_equity: pd.Series | None
    benchmark_metrics: Metrics | None
    turnover: float
    cost_drag: float
    n_rebalances: int

    def summary(self) -> dict:
        out = {"strategy": self.metrics.as_dict(), "turnover": self.turnover,
               "cost_drag": self.cost_drag, "n_rebalances": self.n_rebalances}
        if self.benchmark_metrics:
            out["benchmark"] = self.benchmark_metrics.as_dict()
            out["excess_cagr"] = self.metrics.cagr - self.benchmark_metrics.cagr
        return out


def _close_panel(prices: dict[str, pd.DataFrame], field_pref=("adj_close", "close")) -> pd.DataFrame:
    """Build a wide close-price panel (columns=symbols) from per-symbol frames."""
    cols = {}
    for sym, df in prices.items():
        if df is None or df.empty:
            continue
        col = next((c for c in field_pref if c in df.columns), None)
        if col is None:
            continue
        cols[sym] = df[col].astype(float)
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def run_backtest(prices: dict[str, pd.DataFrame], weight_fn: WeightFn,
                 config: BacktestConfig) -> BacktestResult:
    """Run a monthly (or configured) rebalanced backtest with T+1 execution."""
    panel = _close_panel(prices)
    if panel.empty:
        raise ValueError("no usable price data for backtest")

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date) if config.end_date else panel.index[-1]
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)]
    if len(panel) < 30:
        raise ValueError("insufficient history in selected window")

    daily_ret = panel.pct_change().fillna(0.0)
    rebal_dates = pd.Series(panel.index, index=panel.index).resample(config.rebalance).last().dropna()
    rebal_set = set(pd.to_datetime(rebal_dates.values))

    dates = panel.index
    weights = pd.Series(dtype=float)          # current holdings weights
    pending: dict | None = None               # weights decided at T, to apply at T+1
    equity_val = config.initial_capital
    equity_curve = []
    total_cost = 0.0
    turnovers = []
    n_rebal = 0

    for i, today in enumerate(dates):
        # Apply yesterday's decision at today's price (T+1 execution).
        if pending is not None:
            new_w = pd.Series(pending, dtype=float)
            turnover = float((new_w.subtract(weights, fill_value=0.0)).abs().sum())
            cost = equity_val * turnover * (config.cost_bps / 10000.0)
            equity_val -= cost
            total_cost += cost
            turnovers.append(turnover)
            weights = new_w
            pending = None

        # Grow holdings by today's returns.
        if not weights.empty:
            port_ret = float((weights * daily_ret.loc[today].reindex(weights.index).fillna(0.0)).sum())
            equity_val *= (1.0 + port_ret)
        equity_curve.append((today, equity_val))

        # Decide new target weights at rebalance dates (point-in-time history).
        if today in rebal_set:
            history = panel.loc[:today]
            try:
                target = weight_fn(today, history) or {}
            except Exception as exc:  # noqa: BLE001
                log.warning("weight_fn failed at %s: %s", today.date(), exc)
                target = {}
            # normalize/clip to <=1 gross
            target = {k: float(v) for k, v in target.items() if v and v > 0}
            gross = sum(target.values())
            if gross > 1.0:
                target = {k: v / gross for k, v in target.items()}
            pending = target
            n_rebal += 1

    equity = pd.Series(dict(equity_curve)).sort_index()
    metrics = compute_metrics(equity)

    bench_equity = bench_metrics = None
    if config.benchmark in panel.columns:
        b = panel[config.benchmark].dropna()
        bench_equity = (b / b.iloc[0]) * config.initial_capital
        bench_metrics = compute_metrics(bench_equity)

    return BacktestResult(
        equity=equity, metrics=metrics,
        benchmark_equity=bench_equity, benchmark_metrics=bench_metrics,
        turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        cost_drag=float(total_cost / config.initial_capital),
        n_rebalances=n_rebal,
    )


def walk_forward_windows(index: pd.DatetimeIndex, train_years: int = 3,
                         test_years: int = 1):
    """Yield (train_start, train_end, test_start, test_end) rolling windows."""
    if len(index) == 0:
        return
    start = index[0]
    end = index[-1]
    cur = start
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_start >= end:
            break
        yield (train_start, min(train_end, end), test_start, min(test_end, end))
        cur = cur + pd.DateOffset(years=test_years)


def momentum_weight_fn(top_n: int = 10, lookback: int = 126, max_weight: float = 0.15):
    """Default point-in-time signal: top-N by trailing total return, equal-ish weight."""
    def _fn(asof: pd.Timestamp, history: pd.DataFrame) -> dict:
        if len(history) <= lookback:
            return {}
        window = history.iloc[-lookback - 1:]
        mom = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
        top = mom.sort_values(ascending=False).head(top_n)
        top = top[top > 0]
        if top.empty:
            return {}
        w = 1.0 / len(top)
        return {sym: min(w, max_weight) for sym in top.index}
    return _fn
