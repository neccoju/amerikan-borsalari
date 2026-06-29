"""Composite scoring: combine sub-scores per portfolio with re-normalized weights.

Only factors listed in scoring.enabled_factors contribute. Their weights are
re-normalized so that disabling news/institutional/congress/llm (not yet live)
never silently shrinks the composite range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.logging import get_logger
from .fundamental_score import fundamental_scores
from .macro_score import MacroRegime, compute_macro_regime
from .technical_score import technical_scores

log = get_logger(__name__)

PORTFOLIO_KEYS = ["growth", "defensive", "balanced", "active"]


@dataclass
class ScoreResult:
    technical: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fundamental: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    macro: MacroRegime | None = None
    # composite[portfolio] -> Series(symbol -> score)
    composite: dict[str, pd.Series] = field(default_factory=dict)

    def top(self, portfolio: str, n: int = 10) -> pd.Series:
        s = self.composite.get(portfolio, pd.Series(dtype=float))
        return s.sort_values(ascending=False).head(n)


def _effective_weights(factor_weights: dict, enabled: list[str]) -> dict[str, float]:
    """Restrict to enabled factors and renormalize to sum 1."""
    sub = {f: factor_weights.get(f, 0.0) for f in enabled}
    total = sum(sub.values())
    if total <= 0:
        # equal-weight fallback
        return {f: 1.0 / len(enabled) for f in enabled}
    return {f: w / total for f, w in sub.items()}


def score_universe(indicators: dict[str, dict], fundamentals: dict[str, dict],
                   macro: dict[str, pd.DataFrame], scoring_cfg: dict,
                   news_score: pd.Series | None = None) -> ScoreResult:
    """Run all live sub-scores and assemble per-portfolio composites.

    ``news_score`` (optional, 0..100 per symbol) is folded in only when provided
    and non-empty; in that case the 'news' factor is enabled for this run and its
    configured weight participates in the renormalization.
    """
    tech = technical_scores(indicators, scoring_cfg.get("technical", {}))
    fund = fundamental_scores(fundamentals, scoring_cfg.get("fundamental", {}))
    regime = compute_macro_regime(macro, scoring_cfg.get("macro", {}))

    enabled = list(scoring_cfg.get("enabled_factors", ["macro", "fundamental", "technical"]))
    news_active = news_score is not None and len(news_score) > 0
    if news_active and "news" not in enabled:
        enabled.append("news")
    factor_table = scoring_cfg.get("factors", {})

    # Macro is market-wide; applied as a per-name constant component.
    macro_component = regime.score

    symbols = sorted(set(tech.index) | set(fund.index))
    sub_values = {
        "technical": tech,
        "fundamental": fund,
        "macro": pd.Series(macro_component, index=symbols),
        # News folded in when provided; otherwise neutral & excluded from enabled.
        "news": (news_score.reindex(symbols).fillna(50.0) if news_active
                 else pd.Series(50.0, index=symbols)),
        "institutional": pd.Series(50.0, index=symbols),
        "congress": pd.Series(50.0, index=symbols),
        "llm": pd.Series(50.0, index=symbols),
    }

    result = ScoreResult(technical=tech, fundamental=fund, macro=regime)
    for pf in PORTFOLIO_KEYS:
        weights = _effective_weights(factor_table.get(pf, {}), enabled)
        composite = pd.Series(0.0, index=symbols)
        for factor, w in weights.items():
            composite = composite.add(sub_values[factor].reindex(symbols).fillna(50.0) * w,
                                      fill_value=0.0)
        result.composite[pf] = composite.sort_values(ascending=False)
        log.info("Composite[%s]: %d symbols, top=%.1f", pf, len(composite),
                 composite.max() if len(composite) else float("nan"))
    return result
