"""Build the daily report in HTML (email) and plain-text/markdown (log/archive)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils.logging import get_logger

log = get_logger(__name__)
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class PortfolioReport:
    name: str
    total_value: float
    cash: float
    daily_pl: float
    total_pl: float
    holdings: list[dict] = field(default_factory=list)  # {symbol, weight, value}
    actions: list[str] = field(default_factory=list)


@dataclass
class ReportContext:
    date: str
    market_status: str
    regime_label: str = "n/a"
    regime_score: float = 0.0
    regime_detail: dict = field(default_factory=dict)
    top_scores: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    portfolios: list[PortfolioReport] = field(default_factory=list)
    # News highlights: list of {symbol, headline, label, category, sentiment}
    news_highlights: list[dict] = field(default_factory=list)
    news_note: str = ""
    llm_note: str = ""
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def build_report(ctx: ReportContext) -> tuple[str, str]:
    """Return (html, text). Text is a markdown-ish plain version for logs/archive."""
    html = _env().get_template("daily_report.html.j2").render(ctx=ctx)
    text = _build_text(ctx)
    return html, text


def _build_text(ctx: ReportContext) -> str:
    lines = [
        f"usbot daily report — {ctx.date}",
        f"Market: {ctx.market_status} | Regime: {ctx.regime_label} ({ctx.regime_score:.0f})",
        "",
    ]
    for pf in ctx.portfolios:
        lines.append(
            f"[{pf.name}] value=${pf.total_value:,.2f} cash=${pf.cash:,.2f} "
            f"daily={pf.daily_pl:+.2f} total={pf.total_pl:+.2f}"
        )
        for h in pf.holdings[:10]:
            lines.append(f"    {h['symbol']:<6} {h['weight']*100:5.1f}%  ${h['value']:,.0f}")
        for a in pf.actions:
            lines.append(f"    action: {a}")
        lines.append("")
    if ctx.top_scores:
        lines.append("Top scores:")
        for pf_name, items in ctx.top_scores.items():
            top = ", ".join(f"{s}({v:.0f})" for s, v in items[:5])
            lines.append(f"    {pf_name}: {top}")
        lines.append("")
    if ctx.news_highlights:
        lines.append("Important news:")
        for n in ctx.news_highlights[:8]:
            lines.append(f"    [{n['label']}/{n['category']}] {n['symbol']}: {n['headline'][:90]}")
        lines.append("")
    elif ctx.news_note:
        lines.append(f"News: {ctx.news_note}")
    if ctx.llm_note:
        lines.append(f"LLM: {ctx.llm_note}")
    if ctx.skipped:
        lines.append("Skipped: " + "; ".join(ctx.skipped))
    if ctx.errors:
        lines.append("Data/API errors: " + "; ".join(ctx.errors[:10]))
    lines.append("")
    lines.append("Research/monitoring tool. NOT financial advice. Paper trading only.")
    return "\n".join(lines)


def save_report(text: str, html: str, out_dir: str | Path = "reports",
                date: str | None = None) -> Path:
    date = date or dt.date.today().isoformat()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date}.md").write_text(text, encoding="utf-8")
    (d / f"{date}.html").write_text(html, encoding="utf-8")
    return d / f"{date}.md"
