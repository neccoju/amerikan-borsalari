# usbot — American Stock Market Analysis & Portfolio Bot

> **Research and monitoring tool — NOT financial advice.**
> All portfolios are **simulated (paper trading)**. Nothing here places real orders.

A GitHub-hosted Python bot that, once per US trading day after the close,
scores US stocks with a multi-factor model, manages five simulated portfolios,
and produces a daily report (emailed if SMTP is configured, otherwise logged).

This repository currently implements **Phase 1 (MVP)**. Later phases add news,
sentiment, SEC 13F, congressional data, richer LLM review, backtesting, and an
adaptive self-learning sleeve. See [`docs/`](docs/) for the full design.

---

## Highlights

- **Runs with zero API keys.** Uses keyless `yfinance` + `pandas-market-calendars`.
  Every key-dependent module *skips gracefully* and the report notes the skip.
- **Layered, self-healing data:** prices fall back to Stooq (keyless) when the
  yfinance batch misses a symbol; fundamentals flow through a 7-day SQLite cache
  → yfinance → Finnhub (free tier), converging to full universe coverage across
  runs. Every run screens prices for stale series / suspect prints / corrupt
  rows and reports source coverage in the email + dashboard Data Quality panel.
- **Academically-grounded momentum:** the 12M trend leg uses the classic 12-1
  specification (skip the most recent month — Jegadeesh & Titman 1993) and a
  volatility-scaled momentum rank (Barroso & Santa-Clara 2015) prefers smooth
  trends over crash-prone ones.
- **Alt-data factors:** insider (SEC Form 4) opportunistic *cluster buys*
  (Cohen–Malloy–Pomorski 2012), post-earnings-announcement drift (PEAD;
  Bernard–Thomas) via Finnhub's free tier, and a keyless **short-interest**
  factor (high/rising short interest is bearish — Boehmer–Jones–Zhang 2008,
  Rapach–Ringgenberg–Zhou 2016; derived from the yfinance fundamentals already
  fetched) all feed the composite. The Active sleeve also enforces an **earnings
  blackout** — it won't open a new position within a few days of a scheduled
  report (no accidental binary earnings bets).
- **Multiple-testing-aware backtest:** `usbot backtest` reports the Probabilistic
  and **Deflated Sharpe Ratio** (López de Prado 2014) — pass `--trials N` for the
  number of strategy variants explored and the Sharpe is deflated by the return
  you'd expect from luck alone after that many trials, so a strategy that merely
  won a big search stops looking significant. `--composite` backtests the actual
  shipped technical-momentum specification (the 12-1 blend + risk-adjusted
  momentum + drawdown legs, reconstructed point-in-time), not a toy proxy; the
  `--walk-forward` adaptive-vs-static comparison carries the Deflated Sharpe too.
- **Sentiment engine is swappable:** VADER by default (keyless, fast); set
  `news.sentiment_model: finbert` + install the `[finbert]` extra to use the
  finance-domain FinBERT model (heavier — adds torch; best with a persistent
  model cache rather than the daily ephemeral runner).
- **Five portfolios:** Growth ($1000), Defensive ($1000), Balanced ($1000),
  Active Entry ($1600, $1.5/trade cost-aware), Self-Learning (paper).
- **Transparent scoring:** technical + fundamental + macro-regime composite,
  with per-portfolio weights in `config/scoring.yaml`. Optional **sector-
  neutralization** scores each name on how much it beats its *own* sector, so the
  top ranks don't pile into one hot sector (configurable per sleeve; Defensive
  stays the most diversified).
- **Interactive dashboard:** a dark-theme `site/index.html` with KPI cards, a
  Finviz-style sector→ticker treemap, portfolio-vs-benchmark curves + drawdowns,
  TradingView-style sector rotation, a StockCharts-style RRG, an estimated
  smart-money rotation proxy (Sankey), holdings/signals, and the monthly LLM
  review. Every chart degrades to a labelled placeholder when data is thin.
- **Risk-aware portfolio construction:** inverse-volatility weighting (risk-parity
  tilt, strongest on Defensive), a no-trade rebalance band that suppresses churn,
  a per-position trailing stop, and a **drawdown circuit breaker** that de-risks a
  sinking sleeve to cash between rebalances (flagged in red at the top of the email).
- **Trading-day aware:** weekends/US holidays produce a "market closed" report.
- **External trigger ready:** `workflow_dispatch` + `repository_dispatch` for
  cron-job.org (see [`docs/cron_job_org_setup.md`](docs/cron_job_org_setup.md)).
- **Fail-soft & auditable:** per-ticker isolation, SQLite persistence.

---

## Quick start

```bash
# Python 3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run today's pipeline (logs the report; no email unless SMTP secrets set)
python -m usbot run --dry-run

# Force a run on a non-trading day and treat it as a month-end rebalance
python -m usbot run --force --dry-run

# Just check market status
python -m usbot status

# Run a look-ahead-safe momentum backtest vs a benchmark (needs price history)
# --trials N deflates the Sharpe for multiple testing (Deflated Sharpe Ratio)
python -m usbot backtest --start 2015-01-01 --benchmark SPY --top-n 10 --cost-bps 10 --trials 20

# Backtest the LIVE technical-momentum composite (12-1 blend + risk-adj momentum
# + drawdown) — the honest point-in-time subset of the production score
python -m usbot backtest --composite --start 2015-01-01 --benchmark SPY --trials 20
```

Run the tests:

```bash
pip install -e ".[dev]"
pytest
```

---

## Configuration

- `config/settings.yaml` — universe filters, capitals, risk caps, schedule knobs.
- `config/scoring.yaml` — factor weights per portfolio + indicator parameters.
- `.env` (local) or **GitHub Secrets** (CI) — all optional. See `.env.example`.

### Secrets (all optional)

| Name | Used for | Missing → |
|---|---|---|
| `FRED_API_KEY` | macro enrichment (Phase 2) | yfinance macro proxies only |
| `FINNHUB_API_KEY` / `ALPHA_VANTAGE_API_KEY` | news/fundamentals (Phase 2) | skipped |
| `LLM_PROVIDER` (`anthropic`/`openai`/`ollama`/`none`) | LLM review | `none` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | LLM provider | LLM skipped |
| `SMTP_USERNAME` / `SMTP_APP_PASSWORD` | email report | report logged to disk |
| `EMAIL_TO` / `EMAIL_FROM` | email addresses | defaults to SMTP user |
| `CRON_SECRET_TOKEN` | verify external trigger | check skipped |
| `DASHBOARD_URL` | "Open Dashboard" link in the email | email notes it's unset |

> Never commit real keys. `.env` is git-ignored; `.env.example` lists names only.

---

## Dashboard

Each run also writes an interactive HTML dashboard to `site/index.html` (in
addition to the email/log report — the dashboard build is wrapped so it can
**never** break the email). It's a self-contained page (Plotly loaded from a
CDN) inspired by the UX of Finviz, Portfolio Visualizer, Koyfin, TradingView,
StockCharts and Yahoo Finance — built entirely from this bot's own data.

Ten sections: **1)** Executive Overview (KPI cards) · **2)** Portfolio vs
Benchmarks (cumulative return =100, drawdowns, rolling metrics, alpha vs
SPY/QQQ) · **3)** Portfolio Dashboard · **4)** Market Heatmap (treemap) ·
**5)** Sector Rotation (RS bar + RRG quadrants + table) · **6)** Smart Money
Rotation Proxy (Sankey; *estimate, NOT actual dollar flow*) · **7)** Holdings &
Signals · **8)** News & LLM Insights · **9)** Monthly AI Portfolio Review
(decision-support only — the LLM never trades; shows "review unavailable" if no
key) · **10)** Data Quality & System Health.

**Open it locally:**

```bash
python -m usbot run --dry-run     # writes site/index.html
open site/index.html              # macOS;  xdg-open on Linux;  start on Windows
```

Configure via the `dashboard:` block in `config/settings.yaml`
(`enabled`, `out_path`). `site/` is git-ignored.

**Publish to GitHub Pages (optional):** in **Settings → Pages**, set
**Source = "GitHub Actions"**. The daily workflow then deploys the dashboard
automatically on each real run; it's served at
`https://<owner>.github.io/<repo>/`. Until Pages is enabled the deploy step is a
soft no-op — the dashboard is always available from the run's **`dashboard`
artifact** regardless.

### Dashboard link in daily email

The daily email can carry an **"Open Dashboard"** button straight to your
published dashboard. The link is never hard-coded — it comes from the optional
`DASHBOARD_URL` secret/env var.

1. **Enable GitHub Pages** (see above) so the dashboard gets a public URL,
   typically `https://<github_username>.github.io/<repo_name>/`.
2. **Add the `DASHBOARD_URL` secret** with that URL:
   - GitHub: **Settings → Secrets and variables → Actions → New repository
     secret** → name `DASHBOARD_URL`, value e.g.
     `https://neccoju.github.io/amerikan-borsalari/`.
   - Local dev: put `DASHBOARD_URL=...` in your `.env`.

How it shows up in the email:

- **`DASHBOARD_URL` set** → HTML email shows a green **"📊 Open Dashboard"**
  button (plus a copyable link); the plain-text part shows
  `Open Dashboard: <url>`.
- **`DASHBOARD_URL` not set** → the email still sends normally and simply notes:
  *"Dashboard generated, but DASHBOARD_URL is not configured yet."*

The dashboard is always built **before** the email is sent, so the link points
at a freshly generated page. The build is wrapped so a dashboard failure can
never block the email.

---

## Automation

The daily cadence is driven **externally** by cron-job.org calling the GitHub API
after the US close, not by GitHub's native cron. The workflow also supports
manual runs. Full setup (token, permissions, URL, headers, payload, testing) is
in [`docs/cron_job_org_setup.md`](docs/cron_job_org_setup.md).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Universe, prices, technical+fundamental+macro scoring, 5 portfolios, SQLite, report, email/LLM graceful skip, workflow, tests | ✅ done |
| 2 | News ingestion + sentiment (VADER/FinBERT) wired into scoring, look-ahead-safe backtesting engine, richer report | ✅ done |
| 3 | Congressional trades (keyless) + SEC 13F (experimental) + monthly LLM review with bounded nudges, all wired into scoring & report | ✅ done |
| 4 | Adaptive self-learning sleeve (online factor-weight learning via IC) + walk-forward adaptive-vs-static validation + broad S&P 1500 universe | ✅ done |
| 5 | Interactive dashboard (`site/index.html`): treemap heatmap, portfolio-vs-benchmark analytics, sector rotation + RRG, smart-money rotation proxy, monthly LLM review, GitHub Pages publish | ✅ done |

---

## Disclaimer

This software is for research and educational purposes only. It does not provide
financial advice and does not execute real trades. Markets involve risk; do your
own research.
