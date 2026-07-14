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
                          fundamentals: dict, risk_cfg: dict,
                          vols: dict[str, float] | None = None) -> dict[str, float]:
    """Target weights for a model sleeve, caps applied, optional inverse-vol tilt.

    ``inv_vol_weight`` (per-sleeve risk config) blends score-proportional sizing
    toward inverse-volatility; Defensive defaults to a stronger tilt.
    """
    scores = scores.dropna()
    if ptype == "defensive":
        scores = _defensive_filter(scores, fundamentals, risk_cfg.get("max_beta", 0.80))
    return target_weights_from_scores(
        scores, n=int(risk_cfg.get("max_names", 15)),
        max_position=risk_cfg.get("max_position", 0.12),
        sectors=sectors,
        max_sector=risk_cfg.get("max_sector", 0.30),
        vols=vols,
        inv_vol_weight=float(risk_cfg.get("inv_vol_weight", 0.0)),
    )


def rebalance_to_targets(state: PortfolioState, target_weights: dict[str, float],
                         prices: dict[str, float], txn_cost: float = 0.0,
                         min_trade_value: float = 1.0, band: float = 0.0) -> list[dict]:
    """Rebalance ``state`` in place to ``target_weights`` by trading DELTAS.

    Only the difference between current and target shares is traded, at current
    prices. Carried names keep their cost basis (weighted-average ``avg_cost`` on
    adds, unchanged on trims), so per-holding P/L stays meaningful across
    rebalances instead of resetting monthly.

    A NO-TRADE BAND suppresses churn: a name is only traded when its delta exceeds
    ``max(min_trade_value, band * total_value)`` dollars, so tiny drifts (a name
    a few tenths of a percent off target) are left alone. This cuts turnover and
    cost materially versus rebalancing every position to the exact target.

    Returns an itemised trade list ``[{side, symbol, shares, price, cost}]``
    where ``shares`` is the traded delta (buys positive, sells the amount sold).
    """
    total = state.total_value(prices)
    trades: list[dict] = []
    if total <= 0:
        return trades

    threshold = max(min_trade_value, band * total)

    if not target_weights:
        log.warning("[%s] no eligible names; holding 100%% cash", state.name)

    # target dollar allocation per symbol (missing/invalid price -> untradable)
    target_alloc = {sym: total * w for sym, w in target_weights.items()
                    if prices.get(sym) and prices[sym] > 0}

    # ---- sells first (exits + trims) so their cash funds the buys ----
    for sym in list(state.holdings.keys()):
        h = state.holdings[sym]
        price = float(prices.get(sym, h.avg_cost))
        target_shares = target_alloc.get(sym, 0.0) / price if price > 0 else 0.0
        delta = h.shares - target_shares
        if delta * price < threshold:
            continue
        state.cash += delta * price - txn_cost
        trades.append({"side": "sell", "symbol": sym, "shares": delta,
                       "price": price, "cost": txn_cost})
        if target_shares <= 0:
            del state.holdings[sym]           # genuine exit
        else:
            h.shares = target_shares          # trim; avg_cost unchanged

    # ---- buys (new positions + adds) ----
    for sym, alloc in target_alloc.items():
        price = float(prices[sym])
        held = state.holdings.get(sym)
        cur_shares = held.shares if held else 0.0
        delta = alloc / price - cur_shares
        if delta * price < threshold:
            continue
        state.cash -= delta * price + txn_cost
        if held:
            # weighted-average cost basis on adds
            new_shares = cur_shares + delta
            held.avg_cost = (cur_shares * held.avg_cost + delta * price) / new_shares
            held.shares = new_shares
        else:
            state.holdings[sym] = Holding(symbol=sym, shares=delta, avg_cost=price,
                                          high_water=price)
        trades.append({"side": "buy", "symbol": sym, "shares": delta,
                       "price": price, "cost": txn_cost})

    log.info("[%s] delta-rebalanced to %d names, cash=%.2f (%d trades)",
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
