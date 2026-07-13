"""Active Entry portfolio ($1600), daily, transaction-cost aware.

Key rules (Phase 1):
- Every buy/sell costs a flat fee (default $1.5). A trade is only executed if its
  expected benefit clears the cost + a configurable edge threshold.
- Staged entry: deploy gradually (default 25% first), scale in on confirmation,
  hold cash when the macro regime is weak.
- Exits: score deterioration, technical breakdown (below 50DMA), regime risk-off
  de-risking, and a per-position trailing/stop drawdown.
- Anti-overtrading: minimum position notional so the $1.5 fee is not material.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.logging import get_logger
from .base import Holding, PortfolioState

log = get_logger(__name__)


@dataclass
class TradePlan:
    symbol: str
    side: str          # "buy" | "sell"
    shares: float
    price: float
    cost: float
    reason: str


@dataclass
class ActiveDecision:
    state: PortfolioState
    trades: list[TradePlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ActivePortfolio:
    ENTRY_SCORE = 62.0          # composite threshold to consider a buy
    EXIT_SCORE = 48.0           # composite below this -> exit
    STOP_DRAWDOWN = -0.12       # hard stop: per-position drawdown from avg cost
    MIN_POSITION_NOTIONAL = 80.0  # avoid tiny positions where $1.5 fee bites
    # Expected per-trade benefit proxy: how far score is above entry, mapped to $.
    EDGE_PER_SCORE_POINT = 0.004  # 0.4% expected edge per point above threshold

    def __init__(self, risk_cfg: dict, txn_cost: float, min_cash_buffer_pct: float,
                 initial_deploy_pct: float) -> None:
        self.max_position = risk_cfg.get("max_position", 0.15)
        self.max_daily_turnover = risk_cfg.get("max_daily_turnover", 0.25)
        self.min_edge_after_cost = risk_cfg.get("min_expected_edge_after_cost", 0.0)
        # Trailing stop: exit when price falls this far from the position's
        # high-water mark. Protects accrued gains, which the avg-cost hard stop
        # cannot — a +30% winner can round-trip all the way to -12% before
        # STOP_DRAWDOWN reacts.
        self.trail_drawdown = float(risk_cfg.get("trail_drawdown", -0.15))
        self.txn_cost = txn_cost
        self.min_cash_buffer_pct = min_cash_buffer_pct
        self.initial_deploy_pct = initial_deploy_pct

    def _target_deploy_fraction(self, regime_label: str, invested_frac: float) -> float:
        """How much of capital should be deployed given regime + current state."""
        ceiling = {"risk_on": 0.95, "neutral": 0.65, "risk_off": 0.30}.get(regime_label, 0.5)
        if invested_frac <= 0.0:
            return min(self.initial_deploy_pct, ceiling)  # gradual first entry
        # scale in toward the ceiling
        return ceiling

    def _expected_benefit(self, score: float, notional: float) -> float:
        edge = max(0.0, score - self.ENTRY_SCORE) * self.EDGE_PER_SCORE_POINT
        return edge * notional

    def decide(self, state: PortfolioState, scores: pd.Series,
               prices: dict[str, float], indicators: dict[str, dict],
               regime_label: str, blackout: set | None = None) -> ActiveDecision:
        dec = ActiveDecision(state=state)
        blackout = blackout or set()
        total_value = state.total_value(prices)
        if total_value <= 0:
            dec.notes.append("non-positive portfolio value; no action")
            return dec

        turnover_budget = self.max_daily_turnover * total_value
        turnover_used = 0.0

        # ---- 1. EXITS ----
        for sym in list(state.holdings.keys()):
            h = state.holdings[sym]
            price = prices.get(sym)
            if not price or price <= 0:
                continue
            # ratchet the high-water mark (persisted) before the exit checks
            h.high_water = max(h.high_water, h.avg_cost, float(price))
            score = float(scores.get(sym, 0.0))
            ret = price / h.avg_cost - 1.0 if h.avg_cost else 0.0
            from_high = price / h.high_water - 1.0 if h.high_water else 0.0
            below_ma = indicators.get(sym, {}).get("above_sma50") == 0.0
            reason = None
            if regime_label == "risk_off":
                reason = "regime_risk_off_derisk"
            elif score < self.EXIT_SCORE:
                reason = f"score_decay({score:.0f})"
            elif ret <= self.STOP_DRAWDOWN:
                reason = f"stop_loss({ret:.1%})"
            elif from_high <= self.trail_drawdown:
                reason = f"trailing_stop({from_high:.1%} from high)"
            elif below_ma:
                reason = "technical_breakdown(<50DMA)"
            if reason:
                notional = h.market_value(price)
                state.cash += notional - self.txn_cost
                dec.trades.append(TradePlan(sym, "sell", h.shares, price,
                                            self.txn_cost, reason))
                turnover_used += notional
                del state.holdings[sym]

        # ---- 2. ENTRIES (staged, cost-aware) ----
        invested_frac = state.equity_value(prices) / total_value if total_value else 0.0
        target_frac = self._target_deploy_fraction(regime_label, invested_frac)
        min_cash = self.min_cash_buffer_pct * total_value
        deployable = max(0.0, target_frac * total_value - state.equity_value(prices))
        deployable = min(deployable, state.cash - min_cash)

        if deployable <= self.MIN_POSITION_NOTIONAL:
            dec.notes.append(f"no deployment (regime={regime_label}, deployable={deployable:.0f})")
            return dec

        candidates = scores.sort_values(ascending=False)
        per_name_cap = self.max_position * total_value
        for sym, score in candidates.items():
            if deployable <= self.MIN_POSITION_NOTIONAL or turnover_used >= turnover_budget:
                break
            if score < self.ENTRY_SCORE or sym in state.holdings:
                continue
            if sym in blackout:
                continue  # reports within days -> don't open a binary earnings bet
            price = prices.get(sym)
            if not price or price <= 0:
                continue
            if indicators.get(sym, {}).get("above_sma50") == 0.0:
                continue  # require technical confirmation
            # conviction- & vol-aware sizing
            vol = indicators.get(sym, {}).get("realized_vol") or 0.3
            conviction = (score - self.ENTRY_SCORE) / 40.0
            size = min(per_name_cap, deployable, per_name_cap * max(0.3, min(1.0, conviction)) / max(0.2, vol))
            size = min(size, deployable)
            if size < self.MIN_POSITION_NOTIONAL:
                continue
            # cost gate: only trade if expected benefit beats cost + threshold
            benefit = self._expected_benefit(score, size)
            if benefit < self.txn_cost + self.min_edge_after_cost:
                continue
            shares = size / price
            state.holdings[sym] = Holding(symbol=sym, shares=shares, avg_cost=price,
                                          high_water=price)
            state.cash -= size + self.txn_cost
            deployable -= size
            turnover_used += size
            dec.trades.append(TradePlan(sym, "buy", shares, price, self.txn_cost,
                                        f"entry(score={score:.0f})"))

        dec.notes.append(
            f"regime={regime_label}, target_deploy={target_frac:.0%}, "
            f"trades={len(dec.trades)}, cash={state.cash:.2f}"
        )
        return dec
