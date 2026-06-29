"""Monthly LLM review: qualitative commentary on portfolios, macro, risks.

Runs ONLY on month-end (or when forced) to control cost, and is purely
explanatory — its output is logged/persisted and shown in the report. It does
NOT alter trades in Phase 1 (the bounded +/- adjustment factor lands in Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger
from .provider import LLMProvider

log = get_logger(__name__)

SYSTEM = (
    "You are a cautious quant analyst. Provide concise, balanced commentary. "
    "You are a decision-support layer only and must not issue direct buy/sell "
    "orders. Highlight risks and uncertainties. Output is for a research tool."
)


@dataclass
class LLMReview:
    ran: bool
    provider: str
    model: str
    text: str
    note: str


def run_monthly_review(provider: LLMProvider, context: dict, *, force: bool = False,
                       monthly_only: bool = True, is_month_end: bool = False) -> LLMReview:
    """Generate a monthly review if conditions are met; else return a skip note."""
    if monthly_only and not is_month_end and not force:
        return LLMReview(False, provider.provider, provider.model, "",
                         "LLM review skipped: not month-end (monthly_only mode)")

    if not provider.available:
        return LLMReview(False, provider.provider, provider.model, "",
                         f"LLM skipped: {provider.reason}")

    prompt = _build_prompt(context)
    text = provider.complete(SYSTEM, prompt)
    return LLMReview(True, provider.provider, provider.model, text, "ok")


def _build_prompt(ctx: dict) -> str:
    lines = [
        "Review the following US-market snapshot and comment on regime, top-ranked",
        "names, portfolio posture, key risks, and potential catalysts. Be brief.",
        "",
        f"Macro regime: {ctx.get('regime_label')} (score {ctx.get('regime_score')})",
        f"Top growth names: {ctx.get('top_growth')}",
        f"Top defensive names: {ctx.get('top_defensive')}",
        f"Active portfolio cash%: {ctx.get('active_cash_pct')}",
        "",
        "Remember: do not give direct trade orders; surface considerations only.",
    ]
    return "\n".join(lines)
