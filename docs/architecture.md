# Architecture

```
GitHub Actions (workflow_dispatch / repository_dispatch)
        │   (external trigger: cron-job.org → GitHub API)
        ▼
usbot.cli  ──►  usbot.orchestrator.run_daily()
        │
        ├─ trading-day guard (utils.dates)         → skip + "market closed" report
        ├─ universe.build (S&P500 seed + watchlist) → liquidity/cap filters
        ├─ data.prices (yfinance, cached, fail-soft)
        ├─ data.fetch_fundamentals (best-effort)
        ├─ data.macro (SPY/QQQ/IWM/VIX/^TNX proxies)
        ├─ indicators.compute_indicators (pure pandas)
        ├─ scoring.score_universe
        │     ├─ technical_score   (live)
        │     ├─ fundamental_score (live)
        │     ├─ macro_score       (live, regime + exposure multiplier)
        │     └─ news/inst/congress/llm (Phase 2-3, weight=0 until live)
        ├─ portfolio
        │     ├─ model: Growth / Defensive / Balanced  (month-end rebalance)
        │     ├─ active: Active Entry $1600 (daily, $1.5 cost-aware, staged)
        │     └─ self_learning: paper-only seed (adaptive deferred to Phase 4)
        ├─ db.Repository (SQLite, Postgres-portable)
        ├─ llm (monthly review; graceful skip if no key)
        ├─ reports.build_report (HTML + text)
        └─ mailer.send_report (SMTP; logs to disk if no secrets)
```

## Module map

| Package | Responsibility |
|---|---|
| `config/` | YAML settings + graceful secret detection |
| `utils/` | logging, retry/backoff, trading-day calendar |
| `data/` | price/fundamental/macro fetch + on-disk cache |
| `universe/` | S&P 500 seed, watchlist, ETFs, liquidity filters |
| `indicators/` | RSI/MACD/MA/ATR/momentum/volatility (pure pandas) |
| `scoring/` | sub-scores + per-portfolio composite with renormalized weights |
| `portfolio/` | model sleeves, active engine, self-learning, risk caps |
| `db/` | SQLite schema + repository (only place with SQL) |
| `reports/` | HTML/text report builder + Jinja template |
| `mailer/` | SMTP sender with log fallback |
| `news/` | news ingestion (Finnhub/AV, graceful skip), dedup, VADER/FinBERT sentiment, per-symbol news score |
| `llm/` | configurable provider (anthropic/openai/ollama/none) + monthly review |
| `backtest/` | look-ahead-safe engine (T+1 execution, costs, walk-forward) + metrics |

## Design guarantees

- **Keyless by default:** runs end-to-end with zero secrets (yfinance + calendars).
- **Fail-soft:** per-ticker isolation; pipeline-level try/except; degraded runs
  still emit a report listing skips and errors.
- **Auditable:** scores, trades, positions, snapshots, runs persisted to SQLite.
- **Cost-aware:** Active sleeve refuses trades whose expected benefit < fee + edge.
- **Stateful Active sleeve:** the Active portfolio persists cash/holdings/history
  to `state/active_portfolio.json`, which the workflow commits back after each
  real run (Actions runners are ephemeral). It accumulates day-to-day, scales in
  gradually, and a same-day guard prevents duplicate-trigger double-trading.
  Dry-run is a read-only preview (never persists).
- **Safety:** LLM never trades; Self-Learning is paper-only.
