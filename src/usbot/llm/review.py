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

import re

SYSTEM = (
    "You are a cautious quant analyst. Provide concise, balanced commentary. "
    "You are a decision-support layer only and must not issue direct buy/sell "
    "orders. Highlight risks and uncertainties. Output is for a research tool. "
    "End your reply with a single line 'ADJUSTMENTS: TICKER:+N, TICKER:-N' giving "
    "small conviction nudges in points (integers, small), or 'ADJUSTMENTS: none'."
)


def parse_adjustments(text: str, max_points: float) -> dict[str, float]:
    """Extract bounded per-ticker score nudges from an LLM reply.

    Looks for an 'ADJUSTMENTS:' line of 'TICKER:+N' pairs and clamps each to
    +/- max_points. Returns {} when absent/none — the LLM can only *nudge*, never
    decide. Defensive: malformed entries are ignored.
    """
    if not text:
        return {}
    m = re.search(r"ADJUSTMENTS:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return {}
    body = m.group(1).strip()
    if body.lower().startswith("none"):
        return {}
    out: dict[str, float] = {}
    for token in re.findall(r"([A-Z][A-Z.\-]{0,6})\s*:\s*([+-]?\d+(?:\.\d+)?)", body):
        sym, val = token[0].upper(), float(token[1])
        out[sym] = max(-max_points, min(max_points, val))
    return out


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
