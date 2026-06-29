"""Backtesting engine skeleton (Phase 2).

Design contract (enforced when fully implemented in Phase 2):
- Point-in-time: signals computed on data up to and including day T; execution at
  the next valid price (T+1 open or close) -> NO look-ahead.
- Transaction costs applied on every simulated trade.
- Realistic rebalance timing (month-end for model sleeves; daily for active).
- Walk-forward / train-test split for any adaptive component.
- Metrics vs SPY/QQQ: CAGR, vol, Sharpe, Sortino, max drawdown, hit rate,
  turnover, avg holding period, cost drag, benchmark-relative return.

Phase 1 ships the config + a guard that refuses to run silently, so callers
get a clear "not implemented yet" rather than a misleading empty result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    start_date: str = "2015-01-01"
    end_date: str | None = None
    benchmark: str = "SPY"
    txn_cost: float = 1.5
    rebalance: str = "month_end"


def run_backtest(config: BacktestConfig):  # pragma: no cover - Phase 2
    raise NotImplementedError(
        "Backtesting lands in Phase 2. The look-ahead-safe contract is documented "
        "in this module and docs/research_notes.md."
    )
