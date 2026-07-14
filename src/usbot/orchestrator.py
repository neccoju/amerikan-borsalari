"""Daily orchestration: ties data -> scoring -> portfolios -> report -> email.

Fail-soft throughout: a failure in any single stage is recorded and the run
continues to produce a (possibly degraded) report. Honors the trading-day guard
and emits a "market closed, skipped" report on non-trading days.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import get_secrets, load_settings
from .config.secrets import Secrets
from .config.settings import Settings
from .data import fetch_fundamentals, fetch_macro_series, fetch_prices
from .data.cache import Cache
from .indicators import compute_indicators
from .llm.provider import get_provider
from .llm.review import run_monthly_review
from .mailer import send_report
from .news import fetch_news, news_scores, score_sentiment
# Portfolio classes/helpers are imported locally in the sleeve functions to keep
# this module's import surface small and avoid unused-import churn.
from .reports import ReportContext, build_report
from .reports.builder import PortfolioReport, save_report
from .scoring import score_universe
from .universe import build_universe
from .universe.build import apply_marketcap_filter, apply_price_liquidity_filter
from .utils.dates import is_last_trading_day_of_month, is_trading_day, market_status
from .utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def run_daily(force: bool = False, dry_run: bool | None = None,
              config_dir: str | None = None) -> ReportContext:
    settings = load_settings(config_dir)
    run_cfg = settings.get("run", {})
    setup_logging(run_cfg.get("log_level", "INFO"))
    secrets = get_secrets()

    today = dt.date.today()
    date_str = today.isoformat()
    status = market_status(today)
    dry = run_cfg.get("dry_run", False) if dry_run is None else dry_run

    ctx = ReportContext(date=date_str, market_status=status)

    # ---- trading-day guard ----
    if not is_trading_day(today) and not force:
        log.info("Market closed (%s). Skipping run.", status)
        ctx.skipped.append(f"market_closed ({status})")
        ctx.llm_note = "Run skipped: US market closed."
        _emit(secrets, settings, ctx, dry)
        return ctx

    is_month_end = is_last_trading_day_of_month(today) or force

    try:
        # persist=True only on real (non-dry) runs so previews never mutate the book.
        _run_pipeline(settings, secrets, ctx, is_month_end, force=force, persist=not dry)
    except Exception as exc:  # noqa: BLE001 - never let the whole run crash
        log.exception("pipeline error")
        ctx.errors.append(f"pipeline: {exc}")

    _emit(secrets, settings, ctx, dry)
    return ctx


def _run_pipeline(settings: Settings, secrets: Secrets, ctx: ReportContext,
                  is_month_end: bool, *, force: bool = False, persist: bool = False) -> None:
    scfg = settings.get("data", {})
    cache = Cache(scfg.get("cache_dir", "data/raw"), scfg.get("cache_ttl_hours", 12))
    history_days = int(scfg.get("history_days", 420))

    # ---- universe ----
    universe = build_universe(settings.settings)
    log.info("Universe candidates: %d equities + %d ETFs",
             len(universe.symbols), len(universe.etfs))

    # ---- prices (yfinance batch + Stooq fallback for whatever it misses) ----
    pdata = fetch_prices(universe.all_symbols, period_days=history_days, cache=cache,
                         max_fallback=int(scfg.get("price_fallback_max", 60)))
    # "no price data: SYM" is a benign per-symbol gap (delisted/renamed/illiquid or
    # a transient yfinance miss) — track those separately so they don't look like
    # real failures. Everything else stays in errors.
    for e in pdata.errors:
        if e.startswith("no price data:"):
            ctx.data_gaps.append(e.split(":", 1)[1].strip())
        else:
            ctx.errors.append(e)

    # ---- price quality gates (stale series / suspect prints / corrupt rows) ----
    from .data.quality import validate_prices

    quality = validate_prices(pdata.history)
    ctx.data_quality = {
        "price_total": len(universe.all_symbols),
        "price_ok": len(pdata.history),
        "price_fallback": list(pdata.fallback_used),
        "quality_flags": quality.flags,
    }

    # ---- price/volume pre-filter + cap (bounds the later fundamentals fetch) ----
    universe = apply_price_liquidity_filter(universe, pdata, settings.settings)

    # ---- fundamentals: 7-day cache -> yfinance -> Finnhub (bounded) ----
    fundamentals = _fetch_fundamentals_layered(universe.symbols, settings, secrets, ctx)
    # Sector: Wikipedia GICS (full coverage, consistent naming) wins; yfinance
    # .info fills only what the constituent tables don't cover (e.g. watchlist).
    sectors = {s: m["sector"] for s, m in fundamentals.items() if m.get("sector")}
    sectors.update(universe.sectors)
    # Beta: realized beta vs SPY from price history covers every symbol; yfinance
    # beta (when present) is kept. This makes the defensive low-beta filter real.
    from .indicators.technical import realized_betas

    for sym, b in realized_betas(pdata.history, bench="SPY").items():
        fundamentals.setdefault(sym, {}).setdefault("beta", b)
    # Backfill merged sector into fundamentals so downstream consumers (dashboard
    # treemap, defensive filter) see one consistent naming scheme.
    for sym, sec in sectors.items():
        fundamentals.setdefault(sym, {})["sector"] = sec
    universe = apply_marketcap_filter(universe, fundamentals, settings.settings)

    # ---- indicators ----
    indicators: dict[str, dict] = {}
    tcfg = settings.scoring.get("technical", {})
    for sym in universe.symbols:
        if sym in pdata:
            try:
                indicators[sym] = compute_indicators(pdata[sym], tcfg)
            except Exception as exc:  # noqa: BLE001
                ctx.errors.append(f"indicators {sym}: {exc}")

    # ---- macro ----
    macro = fetch_macro_series(scfg.get("macro_tickers", {}), period_days=history_days,
                               cache=cache)

    # ---- preliminary score (no news) to pick which names to pull news for ----
    prelim = score_universe(indicators, fundamentals, macro, settings.scoring)
    news_top_n = int(settings.get("news", {}).get("top_n", 50))
    news_targets = _news_targets(prelim, news_top_n)

    # ---- alternative-data signals (all graceful-skip) ----
    extra_scores: dict = {}
    news_series = _run_news(settings, secrets, news_targets, ctx)
    if news_series is not None:
        extra_scores["news"] = news_series.reindex(
            sorted(set(indicators))).fillna(50.0) if len(news_series) else news_series
    inst_series = _run_institutional(settings, secrets, universe.symbols, ctx)
    if inst_series is not None:
        extra_scores["institutional"] = inst_series
    congress_series = _run_congress(settings, secrets, universe.symbols, ctx)
    if congress_series is not None:
        extra_scores["congress"] = congress_series
    # Insider (Form 4) + earnings/PEAD share the news top-N targeting budget.
    insider_series = _run_insider(settings, secrets, news_targets, ctx)
    if insider_series is not None:
        extra_scores["insider"] = insider_series.reindex(sorted(set(indicators))).fillna(50.0)
    earn_series, earnings_blackout = _run_earnings(settings, secrets, news_targets, ctx)
    if earn_series is not None:
        extra_scores["earnings"] = earn_series.reindex(sorted(set(indicators))).fillna(50.0)
    # Short interest (keyless): derived from the fundamentals already fetched.
    si_series = _run_short_interest(settings, fundamentals, sorted(set(indicators)), ctx)
    if si_series is not None:
        extra_scores["short_interest"] = si_series.reindex(sorted(set(indicators))).fillna(50.0)

    # ---- final scoring (with alternative-data factors) ----
    scores = score_universe(indicators, fundamentals, macro, settings.scoring,
                            extra_factor_scores=extra_scores)
    if scores.macro:
        ctx.regime_label = scores.macro.label
        ctx.regime_score = scores.macro.score
        ctx.regime_detail = {k: v for k, v in scores.macro.detail.items() if v is not None}

    for pf in ("growth", "defensive", "balanced", "active"):
        comp = scores.composite.get(pf, pd.Series(dtype=float))
        ctx.top_scores[pf] = [(s, float(v)) for s, v in comp.head(5).items()]

    # last close price map
    prices = {s: float(pdata[s]["close"].iloc[-1]) for s in pdata.symbols
              if not pdata[s].empty}

    # ---- portfolios (persisted: real prices, real positions, real P/L) ----
    from .portfolio import PortfolioStore

    pcfg = settings.get("portfolios", {})
    rcfg = settings.get("risk", {})
    store = PortfolioStore(pcfg.get("state_path", "state/portfolios.json"))

    # annualized realized vol per name -> inverse-vol sizing + risk overlays
    vols = {s: indicators[s]["realized_vol"] for s in indicators
            if indicators[s].get("realized_vol") == indicators[s].get("realized_vol")}

    _build_model_sleeves(ctx, store, scores, prices, sectors, fundamentals, pcfg, rcfg,
                         ctx.date, is_month_end, price_history=pdata.history, vols=vols)
    _run_active(ctx, store, scores, prices, indicators, pcfg, rcfg, ctx.date,
                force=force, price_history=pdata.history, blackout=earnings_blackout)
    _build_self_learning(ctx, store, scores, prices, sectors, settings, pcfg, ctx.date,
                         is_month_end, price_history=pdata.history, vols=vols)

    # ---- transaction journal (today's activity + cost totals) ----
    ctx.is_month_end = is_month_end
    _collect_ledger(ctx, store, ctx.date)

    if persist:
        store.commit()
        # SQLite archive: full queryable history of trades + daily snapshots
        # (the JSON ledger keeps only the most recent rows). Idempotent per day;
        # best-effort — the archive must never break the run.
        try:
            _archive_to_db(ctx, store, settings, ctx.date)
        except Exception as exc:  # noqa: BLE001
            log.warning("SQLite archive skipped: %s", exc)

    # ---- LLM monthly review ----
    provider = get_provider(secrets)
    review = run_monthly_review(
        provider,
        context={
            "regime_label": ctx.regime_label,
            "regime_score": f"{ctx.regime_score:.0f}",
            "top_growth": [s for s, _ in ctx.top_scores.get("growth", [])],
            "top_defensive": [s for s, _ in ctx.top_scores.get("defensive", [])],
            "active_cash_pct": next((f"{p.cash/p.total_value:.0%}" for p in ctx.portfolios
                                     if p.name == "Active Entry" and p.total_value), "n/a"),
        },
        monthly_only=settings.get("llm", {}).get("monthly_only", True),
        is_month_end=is_month_end,
    )
    if review.ran:
        from .llm.review import parse_adjustments

        max_pts = float(settings.get("llm", {}).get("max_adjustment_points", 5.0))
        adj = parse_adjustments(review.text, max_pts)
        ctx.llm_note = review.text
        if adj:
            nudges = ", ".join(f"{k}{v:+g}" for k, v in list(adj.items())[:12])
            ctx.llm_note += f"\n[bounded nudges (±{max_pts:g}): {nudges}]"
    else:
        ctx.llm_note = review.note
        ctx.skipped.append(review.note)

    # ---- persist the LLM review to SQLite (best-effort) ----
    if persist:
        try:
            from .db.repository import Repository

            with Repository(_db_path(settings)) as repo:
                repo.save_llm_review(
                    ctx.date, "monthly", provider.provider, provider.model,
                    review.text if review.ran else "", review.note)
                repo.commit()
        except Exception as exc:  # noqa: BLE001 - persistence must never break the run
            log.warning("LLM review persistence skipped: %s", exc)

    # ---- interactive dashboard (site/index.html); never breaks the email ----
    try:
        from .dashboard import build_dashboard

        dcfg = settings.get("dashboard", {})
        if dcfg.get("enabled", True):
            build_dashboard(
                ctx, pdata.history, fundamentals, scores, store,
                llm_review=review.text if review.ran else "",
                llm_available=provider.available,
                out_path=dcfg.get("out_path", "site/index.html"))
            ctx.dashboard_generated = True
    except Exception as exc:  # noqa: BLE001 - dashboard is optional, must never crash the run
        log.warning("Dashboard build skipped: %s", exc)
        ctx.skipped.append(f"dashboard: {exc}")


def _collect_ledger(ctx: ReportContext, store, date: str) -> None:
    """Gather today's transaction-journal entries (all sleeves) + cost totals."""
    from .portfolio import entries_on, total_cost

    today: list[dict] = []
    lifetime = 0.0
    try:
        sleeves = store.all_portfolios()
    except Exception:  # noqa: BLE001
        sleeves = {}
    for p in sleeves.values():
        led = p.get("ledger", [])
        today.extend(entries_on(led, date))
        lifetime += total_cost(led)
    # buys/sells first, summaries (rebalance/rebuild) after; stable within group
    order = {"buy": 0, "sell": 1, "rebalance": 2, "rebuild": 3}
    today.sort(key=lambda e: order.get(e.get("type", ""), 9))
    ctx.ledger_today = today
    ctx.cost_today = round(sum(float(e.get("cost", 0.0)) for e in today), 2)
    ctx.cost_total = round(lifetime, 2)


def _db_path(settings: Settings) -> str:
    """Archive DB location. Defaults under state/ so the workflow's commit-back
    step persists it across ephemeral CI runners, like the JSON book."""
    return settings.get("database", {}).get("path", "state/usbot.db")


def _fetch_fundamentals_layered(symbols: list[str], settings: Settings,
                                secrets: Secrets, ctx: ReportContext) -> dict:
    """Fundamentals with layered sourcing + a rolling cache.

    Order: (1) SQLite cache entries fresher than ``fundamentals_ttl_days`` —
    fundamentals are quarterly data, a weekly refresh loses nothing; (2) yfinance
    for stale/missing names (rate-limited from CI, covers what it can); (3) a
    bounded Finnhub budget for what yfinance missed. Because the cache persists
    across runs (committed back with state/), coverage converges to the full
    universe within a few runs and then just rolls the weekly refresh.
    """
    fcfg = settings.get("data", {})
    ttl_days = int(fcfg.get("fundamentals_ttl_days", 7))
    fh_budget = int(fcfg.get("fundamentals_finnhub_max", 150))

    cached: dict[str, dict] = {}
    repo = None
    try:
        from .db.repository import Repository

        repo = Repository(_db_path(settings))
        cached = repo.load_fundamentals_cache(max_age_days=ttl_days)
    except Exception as exc:  # noqa: BLE001
        log.warning("fundamentals cache unavailable: %s", exc)

    want = set(symbols)
    fundamentals = {s: dict(m) for s, m in cached.items() if s in want}
    stale = [s for s in symbols if s not in fundamentals]

    yf_new = fetch_fundamentals(stale) if stale else {}
    fundamentals.update(yf_new)

    still = [s for s in stale if s not in fundamentals]
    fh_new = {}
    if still:
        from .data.fundamentals_finnhub import fetch_finnhub_fundamentals

        fh_new = fetch_finnhub_fundamentals(still, secrets.get("FINNHUB_API_KEY"),
                                            max_calls=fh_budget)
        fundamentals.update(fh_new)

    if repo is not None:
        try:
            for sym, m in yf_new.items():
                repo.save_fundamentals_cache(sym, "yfinance", m)
            for sym, m in fh_new.items():
                repo.save_fundamentals_cache(sym, "finnhub", m)
            repo.commit()
            repo.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals cache write failed: %s", exc)

    ctx.data_quality.update({
        "fund_total": len(symbols),
        "fund_ok": sum(1 for s in symbols if s in fundamentals),
        "fund_cache": sum(1 for s in symbols if s in cached),
        "fund_yf": len(yf_new),
        "fund_finnhub": len(fh_new),
    })
    log.info("Fundamentals coverage: %d/%d (cache %d, yfinance %d, finnhub %d)",
             ctx.data_quality["fund_ok"], len(symbols),
             ctx.data_quality["fund_cache"], len(yf_new), len(fh_new))
    return fundamentals


def _archive_to_db(ctx: ReportContext, store, settings: Settings, date: str) -> None:
    """Archive today's journal rows + a daily snapshot per sleeve into SQLite.

    The JSON ledger is capped to recent rows; SQLite keeps the FULL queryable
    history (trades, dividends, splits, rebalances, equity snapshots).
    Idempotent per (sleeve, day): journal rows are replaced, snapshots upserted.
    """
    from .db.repository import Repository
    from .portfolio import entries_on

    reports = {p.name: p for p in ctx.portfolios}
    with Repository(_db_path(settings)) as repo:
        for name, rec in store.all_portfolios().items():
            pid = repo.ensure_portfolio(
                name, rec.get("ptype", ""), float(rec.get("starting_capital", 0.0)),
                txn_cost=0.0, paper_only=True)
            repo.replace_trades_for_day(pid, date, entries_on(rec.get("ledger", []), date))
            rep = reports.get(name)
            if rep is not None:
                history = rec.get("history", [])
                peak = max([float(h.get("total_value", 0.0)) for h in history]
                           + [rep.total_value]) or 1.0
                repo.save_snapshot(pid, date, rep.cash, rep.equity, rep.total_value,
                                   rep.daily_pl, rep.total_pl,
                                   drawdown=rep.total_value / peak - 1.0)
        repo.commit()
    log.info("SQLite archive updated (%s)", _db_path(settings))


def _news_targets(prelim, top_n: int) -> list[str]:
    """Union of the top-n names across portfolios from the prelim (news-free) score.

    Bounds per-symbol news fetching so a broad universe doesn't blow API limits.
    """
    targets: set[str] = set()
    for pf in ("growth", "defensive", "balanced", "active"):
        comp = prelim.composite.get(pf, pd.Series(dtype=float))
        targets.update(comp.sort_values(ascending=False).head(top_n).index)
    return sorted(targets)


def _run_news(settings: Settings, secrets: Secrets, symbols: list[str],
              ctx: ReportContext) -> "pd.Series | None":
    """Fetch + score news; returns a per-symbol 0..100 news score, or None if skipped.

    Always fail-soft: any error degrades to 'no news' and the report notes it.
    """
    ncfg = settings.get("news", {})
    if not ncfg.get("enabled", True):
        ctx.news_note = "News disabled in settings"
        return None
    try:
        result = fetch_news(symbols, secrets, days=int(ncfg.get("lookback_days", 3)),
                            max_per_symbol=int(ncfg.get("max_per_symbol", 10)))
    except Exception as exc:  # noqa: BLE001
        ctx.news_note = f"News skipped: {exc}"
        ctx.skipped.append(ctx.news_note)
        return None

    if not result.enabled or result.total == 0:
        ctx.news_note = result.skip_reason or "No news available"
        if result.skip_reason:
            ctx.skipped.append(f"news: {result.skip_reason}")
        ctx.errors.extend(result.errors)
        return None

    # Sentiment-annotate every item, then build per-symbol scores + highlights.
    all_items = [it for items in result.items.values() for it in items]
    sent = score_sentiment(all_items, model=ncfg.get("sentiment_model"))
    series = news_scores(result.items, symbols)

    # Highlights: strongest |sentiment| non-neutral items first.
    ranked = sorted(all_items, key=lambda it: abs(it.sentiment), reverse=True)
    ctx.news_highlights = [
        {"symbol": it.symbol, "headline": it.headline, "label": it.label,
         "category": it.category, "sentiment": round(it.sentiment, 2)}
        for it in ranked if it.label != "neutral"
    ][:12]
    ctx.news_note = f"{result.total} items via {result.provider} (sentiment={sent.model})"
    log.info("News: %s", ctx.news_note)
    return series


def _run_institutional(settings: Settings, secrets: Secrets, symbols: list[str],
                       ctx: ReportContext) -> "pd.Series | None":
    """Fetch 13F changes for tracked funds and score them. Graceful skip."""
    icfg = settings.get("institutional", {})
    if not icfg.get("enabled", True):
        ctx.institutional_note = "Institutional disabled in settings"
        return None
    try:
        from .institutional import fetch_institutional_changes, institutional_scores

        result = fetch_institutional_changes(symbols)
    except Exception as exc:  # noqa: BLE001
        ctx.institutional_note = f"Institutional skipped: {exc}"
        ctx.skipped.append(ctx.institutional_note)
        return None

    if not result.enabled or not result.changes:
        ctx.institutional_note = result.skip_reason or "No 13F changes"
        ctx.skipped.append(f"institutional: {ctx.institutional_note}")
        ctx.errors.extend(result.errors[:5])
        return None

    series = institutional_scores(result.changes, symbols)
    notable = sorted(result.changes, key=lambda c: abs(c.signed_weight), reverse=True)
    ctx.institutional_updates = [
        {"symbol": c.symbol, "fund": c.fund, "change_type": c.change_type}
        for c in notable if c.change_type in ("new", "exited", "increased", "decreased")
    ][:12]
    ctx.institutional_note = (f"{len(result.changes)} changes across "
                              f"{len(result.funds_seen)} funds")
    log.info("Institutional: %s", ctx.institutional_note)
    return series


def _run_congress(settings: Settings, secrets: Secrets, symbols: list[str],
                  ctx: ReportContext) -> "pd.Series | None":
    """Fetch recent congressional trades and score them. Graceful skip."""
    ccfg = settings.get("congress", {})
    if not ccfg.get("enabled", True):
        ctx.congress_note = "Congress disabled in settings"
        return None
    try:
        from .congress import congress_scores, fetch_congress_trades

        result = fetch_congress_trades(symbols, lookback_days=int(ccfg.get("lookback_days", 90)),
                                       secrets=secrets)
    except Exception as exc:  # noqa: BLE001
        ctx.congress_note = f"Congress skipped: {exc}"
        ctx.skipped.append(ctx.congress_note)
        return None

    if not result.enabled or not result.trades:
        ctx.congress_note = result.skip_reason or "No congressional trades in window"
        if result.skip_reason:
            ctx.skipped.append(f"congress: {result.skip_reason}")
        ctx.errors.extend(result.errors[:5])
        return None

    series = congress_scores(result.trades, symbols)
    recent = sorted(result.trades, key=lambda t: (t.traded_date or dt.date.min), reverse=True)
    ctx.congress_updates = [
        {"symbol": t.symbol, "politician": t.politician, "txn_type": t.txn_type,
         "amount_range": t.amount_range, "chamber": t.chamber}
        for t in recent
    ][:12]
    ctx.congress_note = f"{len(result.trades)} trades via {'+'.join(result.sources)}"
    log.info("Congress: %s", ctx.congress_note)
    return series


def _run_insider(settings: Settings, secrets: Secrets, symbols: list[str],
                 ctx: ReportContext) -> "pd.Series | None":
    """Fetch recent Form 4 trades (Finnhub) and score opportunistic cluster buys."""
    icfg = settings.get("insider", {})
    if not icfg.get("enabled", True):
        return None
    try:
        from .insider import fetch_insider_trades, insider_scores

        result = fetch_insider_trades(
            symbols, secrets.get("FINNHUB_API_KEY"),
            lookback_days=int(icfg.get("lookback_days", 90)),
            max_symbols=int(icfg.get("max_symbols", 120)))
    except Exception as exc:  # noqa: BLE001
        ctx.insider_note = f"Insider skipped: {exc}"
        ctx.skipped.append(ctx.insider_note)
        return None
    if not result.enabled or not result.trades:
        ctx.insider_note = result.skip_reason or "No insider (Form 4) trades in window"
        if result.skip_reason:
            ctx.skipped.append(f"insider: {result.skip_reason}")
        return None

    series = insider_scores(result.trades, symbols)
    buys = [t for t in result.trades if t.is_open_market_buy]
    top = sorted(buys, key=lambda t: t.value, reverse=True)
    ctx.insider_updates = [
        {"symbol": t.symbol, "insider": t.insider, "title": t.title,
         "value": t.value, "shares": t.shares} for t in top][:12]
    ctx.insider_note = (f"{len(buys)} open-market buys / {len(result.trades)} Form-4 "
                        f"trades across {result.symbols_seen} names")
    log.info("Insider: %s", ctx.insider_note)
    return series


def _run_earnings(settings: Settings, secrets: Secrets, symbols: list[str],
                  ctx: ReportContext) -> "tuple[pd.Series | None, set[str]]":
    """PEAD score from recent surprises + the upcoming-earnings blackout set."""
    ecfg = settings.get("earnings", {})
    if not ecfg.get("enabled", True):
        return None, set()
    try:
        from .earnings import earnings_blackout, fetch_earnings, pead_scores

        result = fetch_earnings(
            symbols, secrets.get("FINNHUB_API_KEY"),
            lookback_days=int(ecfg.get("lookback_days", 90)),
            days_ahead=int(ecfg.get("blackout_days", 5)) + 5,
            max_symbols=int(ecfg.get("max_symbols", 120)))
    except Exception as exc:  # noqa: BLE001
        ctx.earnings_note = f"Earnings skipped: {exc}"
        ctx.skipped.append(ctx.earnings_note)
        return None, set()
    if not result.enabled:
        ctx.earnings_note = result.skip_reason or "Earnings disabled"
        if result.skip_reason:
            ctx.skipped.append(f"earnings: {result.skip_reason}")
        return None, set()

    blackout = earnings_blackout(result.upcoming, days_ahead=int(ecfg.get("blackout_days", 5)))
    series = pead_scores(result.surprises, symbols) if result.surprises else None
    beats = [s for s in result.surprises if s.surprise_pct > 0]
    ctx.earnings_note = (f"{len(result.surprises)} surprises ({len(beats)} beats), "
                         f"{len(result.upcoming)} upcoming, {len(blackout)} in blackout")
    ctx.earnings_upcoming = [
        {"symbol": u.symbol, "date": u.date.isoformat(), "hour": u.hour}
        for u in sorted(result.upcoming, key=lambda u: u.date)][:12]
    log.info("Earnings: %s", ctx.earnings_note)
    return series, blackout


def _run_short_interest(settings: Settings, fundamentals: dict[str, dict],
                        symbols: list[str], ctx: ReportContext) -> "pd.Series | None":
    """Score short-interest level + change from the yfinance fundamentals we
    already have (no extra network call). Bearish high/rising short interest."""
    sicfg = settings.get("short_interest", {})
    if not sicfg.get("enabled", True):
        return None
    try:
        from .scoring.short_interest import short_interest_highlights, short_interest_scores

        series = short_interest_scores(
            fundamentals, symbols, level_weight=float(sicfg.get("level_weight", 0.6)))
    except Exception as exc:  # noqa: BLE001
        ctx.short_interest_note = f"Short interest skipped: {exc}"
        ctx.skipped.append(ctx.short_interest_note)
        return None
    covered = sum(1 for s in symbols
                  if isinstance(fundamentals.get(s, {}).get("short_percent_float"), (int, float)))
    if not covered:
        ctx.short_interest_note = "No short-interest data available this run"
        return None
    ctx.short_interest_updates = short_interest_highlights(fundamentals, symbols)
    ctx.short_interest_note = f"short-interest data for {covered}/{len(symbols)} names"
    log.info("Short interest: %s", ctx.short_interest_note)
    return series


def _holding_rows(state, prices: dict[str, float]) -> list[dict]:
    """Detailed holding rows: shares, fill price, live price, value, weight, P/L%."""
    w = state.weights(prices)
    rows = []
    for sym, h in state.holdings.items():
        price = float(prices.get(sym, h.avg_cost))
        value = h.shares * price
        pl_pct = (price / h.avg_cost - 1.0) if h.avg_cost else 0.0
        rows.append({"symbol": sym, "shares": h.shares, "avg_cost": h.avg_cost,
                     "price": price, "value": value, "weight": w.get(sym, 0.0),
                     "pl_pct": pl_pct})
    rows.sort(key=lambda r: -r["value"])
    return rows


def _build_model_sleeves(ctx, store, scores, prices, sectors, fundamentals, pcfg, rcfg,
                         date: str, is_month_end: bool, price_history: dict | None = None,
                         vols: dict | None = None) -> None:
    """Model sleeves hold real positions; rebalance at month-end, hold otherwise."""
    from .portfolio import (apply_corporate_actions, compute_model_targets,
                            performance_from_history, rebalance_to_targets, trade_row)
    from .portfolio.risk import circuit_breaker_trim

    capital = float(pcfg.get("model_capital", 1000.0))
    band = float(pcfg.get("rebalance_band", 0.0))
    breaker = float(rcfg.get("circuit_breaker", 0.0))
    breaker_cash = float(rcfg.get("circuit_breaker_cash", 0.5))
    for display, key in [("Growth", "growth"), ("Defensive", "defensive"),
                         ("Balanced", "balanced")]:
        loaded = store.load(display, capital, txn_cost=0.0)
        state = loaded.state
        comp = scores.composite.get(key, pd.Series(dtype=float))
        ledger = list(loaded.ledger)

        # dividends / splits since the last processed date (before any rebalance,
        # so credited cash is reinvested at month-end)
        last_date = loaded.history[-1]["date"] if loaded.history else None
        ca_rows, ca_notes = apply_corporate_actions(state, price_history or {},
                                                    last_date, display, date)
        ledger.extend(ca_rows)

        # Rebalance on month-end, on the first ever run, or if somehow empty.
        # Idempotent: a second run on the same day (manual re-run, force) must
        # not re-fill the book at different prices or duplicate ledger rows.
        already_rebalanced_today = loaded.last_rebalance_date == date
        do_rebalance = ((is_month_end and not already_rebalanced_today)
                        or not loaded.existed or not state.holdings)
        extra_notes: list[str] = []
        if do_rebalance:
            targets = compute_model_targets(key, comp, sectors, fundamentals, rcfg.get(key, {}),
                                            vols=vols)
            # Regime scales gross exposure (risk_on 100% / neutral 80% / risk_off
            # 50% invested); the remainder stays in cash until conditions improve.
            mult = scores.macro.exposure_multiplier if scores.macro else 1.0
            if mult < 1.0:
                targets = {s: w * mult for s, w in targets.items()}
            trades = rebalance_to_targets(state, targets, prices, txn_cost=0.0, band=band)
            reason = "Initial allocation" if not loaded.existed else "Month-end rebalance"
            action = (f"{reason} into {len(state.holdings)} names at live prices "
                      f"(exposure {mult:.0%}, {ctx.regime_label})")
            for t in trades:
                ledger.append(trade_row(date, display, t["side"], t["symbol"], t["shares"],
                                        t["price"], t["cost"], reason))
        else:
            action = "Hold (rebalance only at month-end)"
            # Between rebalances: drawdown circuit breaker de-risks a sinking book.
            if breaker > 0:
                cb_trades, dd, fired = circuit_breaker_trim(
                    state, prices, loaded.history, breaker, breaker_cash)
                if fired:
                    note = (f"CIRCUIT BREAKER: {display} drawdown {dd:.1%} ≤ -{breaker:.0%} "
                            f"→ trimmed to {breaker_cash:.0%} cash")
                    extra_notes.append(note)
                    ctx.risk_alerts.append(note)
                    for t in cb_trades:
                        ledger.append(trade_row(date, display, t["side"], t["symbol"],
                                                t["shares"], t["price"], t["cost"],
                                                "circuit breaker de-risk"))

        history, tv = store.stage(state, prices, date, loaded.history, ptype=key,
                                  last_rebalance_date=date if do_rebalance
                                  else loaded.last_rebalance_date, ledger=ledger)
        perf = performance_from_history(history, tv, capital)
        ctx.portfolios.append(PortfolioReport(
            name=display, total_value=tv, cash=state.cash, equity=tv - state.cash,
            daily_pl=perf["daily_pl"], total_pl=perf["total_pl"],
            holdings=_holding_rows(state, prices), actions=[action] + extra_notes + ca_notes,
        ))


def _run_active(ctx, store, scores, prices, indicators, pcfg, rcfg, date: str,
                *, force: bool = False, price_history: dict | None = None,
                blackout: set | None = None) -> None:
    """Active sleeve: loads prior book, decides once/day, accumulates over time."""
    from .portfolio import (ActivePortfolio, apply_corporate_actions,
                            performance_from_history, trade_row)

    capital = float(pcfg.get("active_capital", 1600.0))
    txn_cost = float(pcfg.get("active_txn_cost", 1.5))
    loaded = store.load("Active Entry", capital, txn_cost)
    state = loaded.state
    ledger = list(loaded.ledger)
    last_date = loaded.history[-1]["date"] if loaded.history else None
    ca_rows, ca_notes = apply_corporate_actions(state, price_history or {},
                                                last_date, "Active Entry", date)
    ledger.extend(ca_rows)

    active = ActivePortfolio(
        risk_cfg=rcfg.get("active", {}),
        txn_cost=txn_cost,
        min_cash_buffer_pct=float(pcfg.get("min_cash_buffer_pct", 0.05)),
        initial_deploy_pct=float(pcfg.get("active_initial_deploy_pct", 0.25)),
    )
    comp = scores.composite.get("active", pd.Series(dtype=float))

    already_decided_today = (loaded.last_decision_date == date) and not force
    actions: list[str] = list(ca_notes)
    if already_decided_today:
        actions.append("Already decided today — revalue only (no new trades)")
    else:
        decision = active.decide(state, comp, prices, indicators, ctx.regime_label,
                                 blackout=blackout)
        if blackout:
            actions.append(f"Earnings blackout: {len(blackout)} names excluded from entry")
        actions += [f"{t.side.upper()} {t.symbol} ({t.reason})" for t in decision.trades[:10]]
        actions += decision.notes
        total_cost = sum(t.cost for t in decision.trades)
        if total_cost:
            actions.append(f"Transaction costs paid: ${total_cost:.2f}")
        # itemise each real buy/sell (with its fee) into the transaction ledger
        for t in decision.trades:
            ledger.append(trade_row(date, "Active Entry", t.side, t.symbol, t.shares,
                                    t.price, t.cost, t.reason))

    history, tv = store.stage(state, prices, date, loaded.history, ptype="active",
                              last_decision_date=date if not already_decided_today
                              else loaded.last_decision_date, ledger=ledger)
    perf = performance_from_history(history, tv, state.starting_capital)
    ctx.portfolios.append(PortfolioReport(
        name="Active Entry", total_value=tv, cash=state.cash, equity=tv - state.cash,
        daily_pl=perf["daily_pl"], total_pl=perf["total_pl"],
        holdings=_holding_rows(state, prices), actions=actions,
    ))


def _build_self_learning(ctx, store, scores, prices, sectors, settings, pcfg, date: str,
                         is_month_end: bool, price_history: dict | None = None,
                         vols: dict | None = None) -> None:
    """Adaptive Self-Learning paper sleeve (Phase 4).

    Monthly it (1) scores how each factor's *previous* scores predicted the
    realized returns since then (information coefficient), (2) nudges its factor
    weights toward what worked (online exponential-gradient update, bounded),
    then (3) rebuilds holdings from a composite using the LEARNED weights. It
    stores the current factor scores for next month's IC. Paper-only, no
    look-ahead (IC uses only past scores vs. later returns). Logs adaptive vs
    static for comparison.
    """
    from .learning import compute_factor_ic, realized_returns, update_weights
    from .portfolio import (apply_corporate_actions, performance_from_history,
                            rebalance_to_targets, trade_row)
    from .portfolio.risk import target_weights_from_scores

    name = "Self-Learning (paper)"
    capital = float(pcfg.get("self_learning_capital", 1000.0))
    lcfg = settings.get("learning", {})
    loaded = store.load(name, capital, txn_cost=0.0)
    state = loaded.state
    meta = loaded.meta or {}
    ledger = list(loaded.ledger)
    last_date = loaded.history[-1]["date"] if loaded.history else None
    ca_rows, ca_notes = apply_corporate_actions(state, price_history or {},
                                                last_date, name, date)
    ledger.extend(ca_rows)

    factor_scores = scores.factor_scores or {}
    enabled = [f for f in (scores.enabled_factors or []) if f in factor_scores]
    static_weights = {f: float(scores_cfg_weight(settings, "balanced", f)) for f in enabled}
    static_weights = _renorm(static_weights)

    # Current learned weights (seeded from static on first run).
    weights = {f: float(meta.get("factor_weights", {}).get(f, static_weights.get(f, 0.0)))
               for f in enabled}
    weights = _renorm(weights) if weights else dict(static_weights)

    # Idempotent like the model sleeves: never re-learn/re-fill twice in one day.
    do_rebalance = ((is_month_end and loaded.last_rebalance_date != date)
                    or not loaded.existed or not state.holdings)
    actions: list[str] = list(ca_notes)
    weight_history = list(meta.get("weight_history", []))

    # ---- monthly learning step (only with a prior snapshot + history) ----
    if do_rebalance and meta.get("last_scores") and price_history and meta.get("last_scored_date"):
        prev_scores = {f: pd.Series(v, dtype=float) for f, v in meta["last_scores"].items()
                       if f in enabled}
        rets = realized_returns(price_history, meta["last_scored_date"],
                                list({s for v in prev_scores.values() for s in v.index}))
        if len(rets) >= 5 and prev_scores:
            ic = compute_factor_ic(prev_scores, rets)
            # Update on an EMA of the stored IC history + this month, not on a
            # single noisy monthly observation.
            from .learning import smooth_ic

            past_ics = [h.get("ic", {}) for h in weight_history[-5:]]
            smoothed = smooth_ic(past_ics + [ic],
                                 alpha=float(lcfg.get("ic_ema_alpha", 0.4)))
            weights = update_weights(weights, smoothed,
                                     lr=float(lcfg.get("learning_rate", 0.25)),
                                     min_w=float(lcfg.get("min_weight", 0.05)),
                                     max_w=float(lcfg.get("max_weight", 0.50)))
            top_ic = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
            actions.append("Learned from smoothed IC: " +
                           ", ".join(f"{f}={v:+.2f}" for f, v in top_ic[:4]))
            weight_history.append({"date": date, "weights": {k: round(v, 4)
                                   for k, v in weights.items()},
                                   "ic": {k: round(v, 3) for k, v in ic.items()}})

    # ---- build composite with LEARNED weights, then rebalance (monthly) ----
    if do_rebalance:
        symbols = sorted({s for v in factor_scores.values() for s in v.index})
        comp = pd.Series(0.0, index=symbols)
        for f, w in weights.items():
            comp = comp.add(factor_scores[f].reindex(symbols).fillna(50.0) * w, fill_value=0.0)
        targets = target_weights_from_scores(
            comp.dropna(), n=15, max_position=0.12, sectors=sectors, max_sector=0.30,
            vols=vols, inv_vol_weight=float(settings.get("risk", {})
                                            .get("self_learning", {}).get("inv_vol_weight", 0.0)))
        mult = scores.macro.exposure_multiplier if scores.macro else 1.0
        if mult < 1.0:
            targets = {s: w * mult for s, w in targets.items()}
        trades = rebalance_to_targets(state, targets, prices, txn_cost=0.0,
                                      band=float(pcfg.get("rebalance_band", 0.0)))
        wtxt = ", ".join(f"{f}:{w:.2f}" for f, w in sorted(weights.items(),
                                                           key=lambda kv: -kv[1])[:4])
        actions.append(f"PAPER — rebuilt with learned weights ({wtxt})")
        reason = f"Paper rebuild (learned weights: {wtxt})"
        for t in trades:
            ledger.append(trade_row(date, name, t["side"], t["symbol"], t["shares"],
                                    t["price"], t["cost"], reason))
        # snapshot current scores for next month's IC
        meta["last_scores"] = {f: {s: round(float(v), 3) for s, v in
                                   factor_scores[f].dropna().items()} for f in enabled}
        meta["last_scored_date"] = date
    else:
        actions.append("PAPER — hold (adaptive rebuild monthly)")

    meta["factor_weights"] = {k: round(v, 6) for k, v in weights.items()}
    meta["static_weights"] = {k: round(v, 6) for k, v in static_weights.items()}
    meta["weight_history"] = weight_history[-36:]

    history, tv = store.stage(state, prices, date, loaded.history, ptype="self_learning",
                              last_rebalance_date=date if do_rebalance
                              else loaded.last_rebalance_date, meta=meta, ledger=ledger)
    perf = performance_from_history(history, tv, capital)
    ctx.portfolios.append(PortfolioReport(
        name=name, total_value=tv, cash=state.cash, equity=tv - state.cash,
        daily_pl=perf["daily_pl"], total_pl=perf["total_pl"],
        holdings=_holding_rows(state, prices), actions=actions,
    ))


def _renorm(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def scores_cfg_weight(settings: Settings, portfolio: str, factor: str) -> float:
    return settings.scoring.get("factors", {}).get(portfolio, {}).get(factor, 0.0)


def _emit(secrets: Secrets, settings: Settings, ctx: ReportContext, dry: bool) -> None:
    # Surface the published-dashboard link in the email (never hard-coded; comes
    # from the optional DASHBOARD_URL secret). Empty -> the report shows a note.
    ctx.dashboard_url = secrets.get("DASHBOARD_URL", "") or ""
    html, text = build_report(ctx)
    save_report(text, html, out_dir="reports", date=ctx.date)
    log.info("\n%s", text)
    ecfg = settings.get("email", {})
    subject = f"{ecfg.get('subject_prefix', '[usbot]')} {ctx.date} — {ctx.regime_label}"
    result = send_report(secrets, subject, html, text,
                         enabled=ecfg.get("enabled", True), dry_run=dry)
    log.info("[email] channel=%s sent=%s (%s)", result.channel, result.sent, result.note)
    if not result.sent and "missing" in result.note:
        ctx.skipped.append(f"email: {result.note}")
