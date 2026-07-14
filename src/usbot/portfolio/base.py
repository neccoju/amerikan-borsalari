"""Shared portfolio data structures and valuation helpers (paper trading)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Holding:
    symbol: str
    shares: float
    avg_cost: float
    # Highest price seen while held (persisted); powers the trailing stop.
    # 0.0 = not yet observed -> treated as max(avg_cost, current price).
    high_water: float = 0.0

    def market_value(self, price: float) -> float:
        return self.shares * price


@dataclass
class PortfolioState:
    """Simulated portfolio book. Fractional shares allowed."""

    name: str
    ptype: str
    cash: float
    starting_capital: float
    txn_cost: float = 0.0
    paper_only: bool = True
    holdings: dict[str, Holding] = field(default_factory=dict)
    # Entry orders decided at one close, awaiting fill at the NEXT session's open
    # (T+1 execution — removes the look-ahead of filling at an observed close).
    # Each: {symbol, notional, score, reason, decided_date}. Empty for close-fill.
    pending_orders: list[dict] = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        equity = sum(
            h.market_value(prices.get(sym, h.avg_cost)) for sym, h in self.holdings.items()
        )
        return self.cash + equity

    def equity_value(self, prices: dict[str, float]) -> float:
        return sum(
            h.market_value(prices.get(sym, h.avg_cost)) for sym, h in self.holdings.items()
        )

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        tv = self.total_value(prices)
        if tv <= 0:
            return {}
        return {
            sym: h.market_value(prices.get(sym, h.avg_cost)) / tv
            for sym, h in self.holdings.items()
        }
