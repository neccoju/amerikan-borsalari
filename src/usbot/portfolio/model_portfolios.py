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


def build_model_portfolio(name: str, ptype: str, scores: pd.Series,
                          prices: dict[str, float], sectors: dict[str, str],
                          fundamentals: dict, risk_cfg: dict, capital: float,
                          regime_label: str = "neutral") -> tuple[PortfolioState, dict]:
    """Construct target holdings for a model portfolio at a rebalance date.

    Returns (state, target_weights). Fractional shares; no per-trade cost on the
    model sleeves (they rebalance monthly and are pure allocation studies).
    """
    scores = scores.dropna()

    if ptype == "defensive":
        scores = _defensive_filter(scores, fundamentals, risk_cfg.get("max_beta", 0.80))

    n = int(risk_cfg.get("max_names", 15))
    weights = target_weights_from_scores(
        scores, n=n,
        max_position=risk_cfg.get("max_position", 0.12),
        sectors=sectors,
        max_sector=risk_cfg.get("max_sector", 0.30),
    )

    state = PortfolioState(name=name, ptype=ptype, cash=capital,
                           starting_capital=capital, txn_cost=0.0)

    if not weights:
        log.warning("[%s] no eligible names; holding 100%% cash", name)
        return state, {}

    invest = capital
    for sym, w in weights.items():
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        alloc = invest * w
        shares = alloc / price
        state.holdings[sym] = Holding(symbol=sym, shares=shares, avg_cost=price)
        state.cash -= alloc
    log.info("[%s] rebalanced into %d names, cash=%.2f", name, len(state.holdings), state.cash)
    return state, weights
