"""Self-Learning paper portfolio (Phase 1 skeleton).

PAPER ONLY. Never trades live. V1 scope: adapt factor weights monthly via a
simple Bayesian/online update based on which factors recently correlated with
forward returns, and compare against the static weights. Phase 4 adds proper
walk-forward validation and (optionally) RL/FinRL.

In Phase 1 this provides the structure and a conservative default: it starts
from the static 'balanced' weights and exposes an ``update_weights`` hook that
is intentionally a no-op until return history accumulates (avoids acting on
look-ahead-biased or sparse data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.logging import get_logger
from .base import Holding, PortfolioState
from .risk import target_weights_from_scores

log = get_logger(__name__)


@dataclass
class SelfLearningState:
    factor_weights: dict[str, float] = field(default_factory=dict)
    method: str = "static_balanced_seed"
    note: str = "Paper only. Adaptive update deferred until sufficient history (Phase 4)."


class SelfLearningPortfolio:
    paper_only = True

    def __init__(self, base_weights: dict[str, float], capital: float) -> None:
        self.state = SelfLearningState(factor_weights=dict(base_weights))
        self.capital = capital

    def update_weights(self, factor_returns: pd.DataFrame | None) -> dict[str, float]:
        """Bayesian/online weight update hook.

        No-op in Phase 1: with <1 month of return history any update would be
        noise / look-ahead biased. Returns current weights unchanged and logs
        the decision for auditability.
        """
        if factor_returns is None or len(factor_returns) < 21:
            log.info("[self-learning] insufficient history; keeping seed weights (paper)")
            return self.state.factor_weights
        # Placeholder for Phase 4 adaptive logic (kept explicit & inert for safety).
        log.info("[self-learning] adaptive update deferred to Phase 4")
        return self.state.factor_weights

    def build(self, composite_scores: pd.Series, prices: dict[str, float],
              sectors: dict[str, str]) -> PortfolioState:
        weights = target_weights_from_scores(
            composite_scores.dropna(), n=15, max_position=0.12,
            sectors=sectors, max_sector=0.30,
        )
        state = PortfolioState(name="self_learning", ptype="self_learning",
                               cash=self.capital, starting_capital=self.capital,
                               paper_only=True)
        for sym, w in weights.items():
            price = prices.get(sym)
            if not price or price <= 0:
                continue
            alloc = self.capital * w
            state.holdings[sym] = Holding(sym, alloc / price, price)
            state.cash -= alloc
        return state
