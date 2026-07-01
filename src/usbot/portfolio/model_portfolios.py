"""Growth / Defensive / Balanced model portfolios ($1000 each).

Rebalance only on the last trading day of the month. Between rebalances the book
is held; daily runs just revalue it. Defensive tilts toward low-beta/defensive
names and can hold cash-like ETFs in risk-off regimes.
"""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from .base import Holding, PortfolioState
from .risk import target_weights_from_scores

log = get_logger(__name__)


def _defensive_filter(scores: pd.Series, fundamentals: dict, max_beta: float) -> pd.Series:
    """Keep names with beta <= max_beta (unknown beta treated as neutral, kept)."""
    keep = {}
    for sym, sc in scores.items():
        beta = fundamentals.get(sym, {}).get("beta")
        if beta is None or beta <= max_beta:
            keep[sym] = sc
    return pd.Series(keep)


def compute_model_targets(ptype: str, scores: pd.Series, sectors: dict[str, str],
                          fundamentals: dict, risk_cfg: dict) -> dict[str, float]:
    """Score-proportional target weights for a model sleeve, caps applied."""
    scores = scores.dropna()
    if ptype == "defensive":
        scores = _defensive_filter(scores, fundamentals, risk_cfg.get("max_beta", 0.80))
    return target_weights_from_scores(
        scores, n=int(risk_cfg.get("max_names", 15)),
        max_position=risk_cfg.get("max_position", 0.12),
        sectors=sectors,
        max_sector=risk_cfg.get("max_sector", 0.30),
    )


def rebalance_to_targets(state: PortfolioState, target_weights: dict[str, float],
                         prices: dict[str, float], txn_cost: float = 0.0) -> list[dict]:
    """Rebalance ``state`` in place to ``target_weights`` at current prices.

    Liquidates current holdings to cash (at current prices), then buys the target
    weights using the *current total value* so gains compound. Fractional shares;
    fill price recorded as the current price.

    Returns an itemised trade list ``[{side, symbol, shares, price, cost}]``: a
    BUY at the fill price for every new target position, and a SELL at the current
    price for every name dropped from the book (a genuine exit).
    """
    total = state.total_value(prices)
    old = dict(state.holdings)                 # snapshot before liquidation
    n_trades = len(state.holdings)
    state.holdings.clear()
    # apply sell-side costs (0 for model sleeves)
    state.cash = total - n_trades * txn_cost
    invest = state.cash
    trades: list[dict] = []

    if not target_weights:
        log.warning("[%s] no eligible names; holding 100%% cash", state.name)
        for sym, h in old.items():
            trades.append({"side": "sell", "symbol": sym, "shares": h.shares,
                           "price": float(prices.get(sym, h.avg_cost)), "cost": txn_cost})
        return trades

    for sym, w in target_weights.items():
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        alloc = invest * w
        shares = alloc / price
        if shares <= 0:
            continue
        state.holdings[sym] = Holding(symbol=sym, shares=shares, avg_cost=price)
        state.cash -= alloc + txn_cost
        trades.append({"side": "buy", "symbol": sym, "shares": shares,
                       "price": float(price), "cost": txn_cost})
    # record exits (held before, absent from the new book) at the current price
    for sym, h in old.items():
        if sym not in state.holdings:
            trades.append({"side": "sell", "symbol": sym, "shares": h.shares,
                           "price": float(prices.get(sym, h.avg_cost)), "cost": txn_cost})
    log.info("[%s] rebalanced into %d names at real prices, cash=%.2f (%d trades)",
             state.name, len(state.holdings), state.cash, len(trades))
    return trades


def build_model_portfolio(name: str, ptype: str, scores: pd.Series,
                          prices: dict[str, float], sectors: dict[str, str],
                          fundamentals: dict, risk_cfg: dict, capital: float,
                          regime_label: str = "neutral") -> tuple[PortfolioState, dict]:
    """Build a fresh model portfolio at real prices (one-shot; no persistence).

    Retained for convenience/back-compat; the orchestrator uses
    compute_model_targets + rebalance_to_targets against persisted state.
    """
    weights = compute_model_targets(ptype, scores, sectors, fundamentals, risk_cfg)
    state = PortfolioState(name=name, ptype=ptype, cash=capital,
                           starting_capital=capital, txn_cost=0.0)
    rebalance_to_targets(state, weights, prices, txn_cost=0.0)
    return state, weights
