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
    equity: float = 0.0                                 # value held in shares (= total - cash)
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
    # Institutional (13F) updates: list of {symbol, fund, change_type}
    institutional_updates: list[dict] = field(default_factory=list)
    institutional_note: str = ""
    # Congressional updates: list of {symbol, politician, txn_type, amount_range, chamber}
    congress_updates: list[dict] = field(default_factory=list)
    congress_note: str = ""
    llm_note: str = ""
    # Dashboard link surfaced in the email. ``dashboard_url`` comes from the
    # DASHBOARD_URL secret (empty if unset); ``dashboard_generated`` is True once
    # site/index.html was written, so the email can note "generated but no URL".
    dashboard_url: str = ""
    dashboard_generated: bool = False
    # Whether this run is a month-end rebalance day (model/self-learning rebuild).
    is_month_end: bool = False
    # Transaction journal: today's entries across all sleeves + cost totals.
    ledger_today: list[dict] = field(default_factory=list)
    cost_today: float = 0.0
    cost_total: float = 0.0
    # Symbols yfinance returned no data for (delisted/renamed/illiquid/hiccup);
    # tracked separately from real errors so they don't look alarming.
    data_gaps: list[str] = field(default_factory=list)
    # Coverage + quality stats: price_total/ok/fallback, fund_total/ok/cache/
    # yf/finnhub, quality_flags (stale/suspect/corrupt series).
    data_quality: dict = field(default_factory=dict)
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
        f"Month-end rebalance day: {'YES' if ctx.is_month_end else 'no'}",
        "",
    ]
    if ctx.dashboard_url:
        lines.append("Dashboard:")
        lines.append(f"    Open Dashboard: {ctx.dashboard_url}")
        lines.append("")
    elif ctx.dashboard_generated:
        lines.append("Dashboard generated, but DASHBOARD_URL is not configured yet.")
        lines.append("")
    # ---- transaction journal (today) ----
    if ctx.ledger_today:
        lines.append(f"Transaction ledger — {ctx.date} "
                     f"(today's cost ${ctx.cost_today:,.2f} · lifetime ${ctx.cost_total:,.2f}):")
        for e in ctx.ledger_today:
            typ = e.get("type", "")
            if typ in ("buy", "sell"):
                lines.append(
                    f"    [{e['sleeve']}] {typ.upper():4} {e.get('symbol',''):<6} "
                    f"{e.get('shares',0):.4f} @ ${e.get('price',0):,.2f}  "
                    f"fee ${e.get('cost',0):.2f}  ({e.get('reason','')})")
            else:
                lines.append(
                    f"    [{e['sleeve']}] {typ.upper()}: {e.get('reason','')}"
                    + (f"  fee ${e.get('cost',0):.2f}" if e.get('cost') else ""))
        lines.append("")
    elif ctx.cost_total:
        lines.append(f"Transaction ledger: no trades today "
                     f"(lifetime cost ${ctx.cost_total:,.2f}).")
        lines.append("")
    for pf in ctx.portfolios:
        lines.append(
            f"[{pf.name}] total=${pf.total_value:,.2f} "
            f"(in shares ${pf.equity:,.2f} + cash ${pf.cash:,.2f}) "
            f"daily={pf.daily_pl:+.2f} total_pl={pf.total_pl:+.2f}"
        )
        for h in pf.holdings[:12]:
            shares = h.get("shares", 0.0)
            avg = h.get("avg_cost", 0.0)
            price = h.get("price", avg)
            pl = h.get("pl_pct", 0.0) * 100
            lines.append(
                f"    {h['symbol']:<6} {h.get('weight', 0)*100:5.1f}%  "
                f"{shares:8.4f} sh @ ${avg:,.2f} → ${price:,.2f}  "
                f"= ${h.get('value', 0):,.2f}  ({pl:+.1f}%)"
            )
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
    if ctx.institutional_updates:
        lines.append("Institutional (13F) moves:")
        for u in ctx.institutional_updates[:8]:
            lines.append(f"    {u['symbol']}: {u['change_type']} by {u['fund']}")
        lines.append("")
    elif ctx.institutional_note:
        lines.append(f"Institutional: {ctx.institutional_note}")
    if ctx.congress_updates:
        lines.append("Congressional trades:")
        for u in ctx.congress_updates[:8]:
            lines.append(f"    {u['symbol']}: {u['txn_type']} {u.get('amount_range','')} "
                         f"by {u['politician']} ({u['chamber']})")
        lines.append("")
    elif ctx.congress_note:
        lines.append(f"Congress: {ctx.congress_note}")
    if ctx.llm_note:
        lines.append(f"LLM: {ctx.llm_note}")
    if ctx.skipped:
        lines.append("Skipped: " + "; ".join(ctx.skipped))
    if ctx.data_quality:
        q = ctx.data_quality
        fb = q.get("price_fallback", [])
        lines.append(
            f"Data quality: prices {q.get('price_ok', 0)}/{q.get('price_total', 0)}"
            + (f" ({len(fb)} via Stooq fallback: {', '.join(fb[:8])})" if fb else "")
            + f" | fundamentals {q.get('fund_ok', 0)}/{q.get('fund_total', 0)}"
            f" (cache {q.get('fund_cache', 0)}, yfinance {q.get('fund_yf', 0)},"
            f" finnhub {q.get('fund_finnhub', 0)})")
        flags = q.get("quality_flags", [])
        if flags:
            lines.append("Quality flags: " + "; ".join(flags[:10]))
    if ctx.data_gaps:
        preview = ", ".join(ctx.data_gaps[:15])
        more = f" (+{len(ctx.data_gaps) - 15} more)" if len(ctx.data_gaps) > 15 else ""
        lines.append(f"Data gaps: {len(ctx.data_gaps)} symbols had no price data, "
                     f"skipped (normal — delisted/renamed/illiquid or a data hiccup): "
                     f"{preview}{more}")
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
