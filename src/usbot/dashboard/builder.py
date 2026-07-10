"""Assemble the dashboard context and render site/index.html.

Pulls together the report context, price history, scores, sector analytics and
the persisted LLM review into a single dark-theme page. Every section degrades
to a labelled placeholder when its data is missing, and the whole build is
wrapped by the caller so it can never break the email report.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..universe.watchlist import get_benchmark_etfs, get_sector_etfs
from ..utils.logging import get_logger
from . import charts, perf
from .sectors import SectorRow, compute_sector_rows, rotation_summary, sankey_pairs

log = get_logger(__name__)
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _daily_ret(df: pd.DataFrame, n: int = 1) -> float:
    if df is None or df.empty or "close" not in df or len(df) <= n:
        return float("nan")
    c = df["close"].astype(float)
    p0, p1 = float(c.iloc[-n - 1]), float(c.iloc[-1])
    return (p1 / p0 - 1.0) if p0 else float("nan")


@dataclass
class DashboardData:
    date: str = ""
    market_status: str = ""
    regime_label: str = "n/a"
    regime_score: float = 0.0
    kpis: dict = field(default_factory=dict)
    benchmark_cards: list[dict] = field(default_factory=list)
    alpha_cards: list[dict] = field(default_factory=list)
    perf_table: list = field(default_factory=list)
    portfolios: list = field(default_factory=list)
    holdings_signals: list[dict] = field(default_factory=list)
    sector_rows: list[SectorRow] = field(default_factory=list)
    rotation_text: str = ""
    proxy_rows: list[dict] = field(default_factory=list)
    news_highlights: list[dict] = field(default_factory=list)
    news_note: str = ""
    is_month_end: bool = False
    ledger_today: list[dict] = field(default_factory=list)
    cost_today: float = 0.0
    cost_total: float = 0.0
    data_gaps: list[str] = field(default_factory=list)
    llm_review: str = ""
    llm_available: bool = False
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    generated_at: str = ""
    email_status: str = ""
    # rendered chart divs
    chart_treemap: str = ""
    chart_perf: str = ""
    chart_drawdown: str = ""
    chart_sector_bar: str = ""
    chart_rrg: str = ""
    chart_sankey: str = ""


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)),
                       autoescape=select_autoescape(["html"]))


def build_dashboard(ctx, prices: dict, fundamentals: dict, scores, store,
                    *, llm_review: str = "", llm_available: bool = False,
                    email_status: str = "", out_path: str | Path = "site/index.html") -> Path:
    """Build the dashboard and write it to ``out_path``. Returns the path."""
    d = DashboardData(
        date=ctx.date, market_status=ctx.market_status,
        regime_label=ctx.regime_label, regime_score=ctx.regime_score,
        portfolios=ctx.portfolios, news_highlights=ctx.news_highlights,
        news_note=ctx.news_note, skipped=ctx.skipped, errors=ctx.errors,
        is_month_end=getattr(ctx, "is_month_end", False),
        ledger_today=getattr(ctx, "ledger_today", []),
        cost_today=getattr(ctx, "cost_today", 0.0),
        cost_total=getattr(ctx, "cost_total", 0.0),
        data_gaps=getattr(ctx, "data_gaps", []),
        llm_review=llm_review, llm_available=llm_available, email_status=email_status,
        generated_at=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    # ---- KPI cards ----
    total_val = sum(p.total_value for p in ctx.portfolios)
    daily_pl = sum(p.daily_pl for p in ctx.portfolios)
    total_pl = sum(p.total_pl for p in ctx.portfolios)
    cash = sum(p.cash for p in ctx.portfolios)
    best, worst = _best_worst(prices, scores)
    d.kpis = {
        "run_date": ctx.date, "regime": ctx.regime_label, "regime_score": ctx.regime_score,
        "total_value": total_val, "daily_pl": daily_pl, "total_pl": total_pl,
        "cash_pct": (cash / total_val if total_val else 0.0),
        "best": best, "worst": worst,
        "warnings": len(ctx.errors), "skipped": len(ctx.skipped),
    }

    # ---- benchmark snapshot cards ----
    for sym, name in get_benchmark_etfs().items():
        df = prices.get(sym)
        if df is not None and not df.empty and "close" in df:
            d.benchmark_cards.append({
                "symbol": sym, "name": name,
                "price": float(df["close"].iloc[-1]), "ret": _daily_ret(df)})

    # ---- portfolio vs benchmarks ----
    series = _perf_series(ctx, store, prices)
    d.chart_perf = charts.perf_lines(series)
    d.chart_drawdown = charts.drawdown_lines(series)
    d.perf_table = perf.build_perf_table(series, prices)
    d.alpha_cards = [
        {"name": r.name, "alpha_spy": r.alpha_spy, "alpha_qqq": r.alpha_qqq, "cagr": r.cagr}
        for r in d.perf_table if r.name not in ("SPY", "QQQ")
    ]

    # ---- market heatmap (treemap) ----
    d.chart_treemap = charts.treemap(_treemap_rows(ctx, prices, fundamentals, scores),
                                     color_label="Daily return")

    # ---- sector rotation + RRG ----
    d.sector_rows = compute_sector_rows(prices, get_sector_etfs(), bench="SPY")
    d.rotation_text = rotation_summary(d.sector_rows)
    d.chart_sector_bar = charts.sector_bar(d.sector_rows)
    d.chart_rrg = charts.rrg_scatter(d.sector_rows)

    # ---- smart money rotation proxy ----
    d.proxy_rows = [
        {"name": r.name, "etf": r.etf, "proxy": r.proxy, "direction": r.direction,
         "explanation": _proxy_explanation(r)}
        for r in sorted(d.sector_rows, key=lambda x: x.proxy, reverse=True)
    ]
    d.chart_sankey = charts.sankey(sankey_pairs(d.sector_rows))

    # ---- holdings & signals ----
    d.holdings_signals = _holdings_signals(scores)

    html = _env().get_template("dashboard.html.j2").render(d=d)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Dashboard written to %s (%d bytes)", out, len(html))
    return out


def _best_worst(prices: dict, scores) -> tuple[dict | None, dict | None]:
    rets = {}
    syms = list(scores.composite.get("balanced", pd.Series(dtype=float)).index) if scores else []
    for s in syms:
        r = _daily_ret(prices.get(s))
        if not np.isnan(r):
            rets[s] = r
    if not rets:
        return None, None
    best = max(rets, key=rets.get)
    worst = min(rets, key=rets.get)
    return ({"symbol": best, "ret": rets[best]},
            {"symbol": worst, "ret": rets[worst]})


def _perf_series(ctx, store, prices) -> list:
    out = []
    # portfolios (from saved equity history)
    try:
        loaded = store.all_portfolios() if store else {}
    except Exception:  # noqa: BLE001
        loaded = {}
    for name, p in loaded.items():
        eq = perf.portfolio_equity(p.get("history", []))
        if eq is not None:
            out.append(perf.PerfSeries(name=name, equity=eq, is_portfolio=True))
    # benchmarks
    for sym in ("SPY", "QQQ", "DIA", "IWM", "VTI", "GLD", "TLT", "SGOV"):
        eq = perf.benchmark_equity(prices, sym)
        if eq is not None:
            out.append(perf.PerfSeries(name=sym, equity=eq, is_portfolio=False))
    return out


def _treemap_rows(ctx, prices, fundamentals, scores, max_names: int = 140) -> list[dict]:
    held = {h["symbol"] for p in ctx.portfolios for h in p.holdings}
    comp = scores.composite.get("balanced", pd.Series(dtype=float)) if scores else pd.Series(dtype=float)
    by_cap = sorted(fundamentals.items(),
                    key=lambda kv: kv[1].get("market_cap", 0) or 0, reverse=True)
    candidates = list(held) + [s for s, _ in by_cap]
    seen, rows = set(), []
    for sym in candidates:
        if sym in seen or len(rows) >= max_names:
            continue
        seen.add(sym)
        meta = fundamentals.get(sym, {})
        sector = meta.get("sector") or "Unknown"
        size = meta.get("market_cap") or 0
        if not size:
            size = 1e9 if sym in held else 5e8
        ret = _daily_ret(prices.get(sym))
        sc = float(comp.get(sym, float("nan")))
        rows.append({
            "symbol": sym, "sector": sector, "size": float(size),
            "color": 0.0 if np.isnan(ret) else float(ret),
            "label": (f"{ret*100:+.1f}%" if not np.isnan(ret) else "")
                     + (f" · score {sc:.0f}" if not np.isnan(sc) else ""),
        })
    return rows


def _holdings_signals(scores, top_n: int = 25) -> list[dict]:
    if not scores:
        return []
    comp = scores.composite.get("balanced", pd.Series(dtype=float)).sort_values(ascending=False)
    fs = scores.factor_scores or {}
    out = []
    for sym in list(comp.index)[:top_n]:
        out.append({
            "symbol": sym, "score": float(comp.get(sym, float("nan"))),
            "technical": _fv(fs.get("technical"), sym),
            "fundamental": _fv(fs.get("fundamental"), sym),
            "macro": _fv(fs.get("macro"), sym),
            "news": _fv(fs.get("news"), sym),
        })
    return out


def _fv(series, sym) -> float:
    if series is None:
        return float("nan")
    try:
        return float(series.get(sym, float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")


def _proxy_explanation(r: SectorRow) -> str:
    bits = []
    if not np.isnan(r.rsi):
        bits.append(f"RSI {r.rsi:.0f}")
    bits.append(f"RS {r.rs_vs_spy:+.1f}% vs SPY")
    bits.append(f"mom {r.momentum:+.1f}")
    return ", ".join(bits)
