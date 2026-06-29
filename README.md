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
- **Five portfolios:** Growth ($1000), Defensive ($1000), Balanced ($1000),
  Active Entry ($1600, $1.5/trade cost-aware), Self-Learning (paper).
- **Transparent scoring:** technical + fundamental + macro-regime composite,
  with per-portfolio weights in `config/scoring.yaml`.
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
python -m usbot backtest --start 2015-01-01 --benchmark SPY --top-n 10 --cost-bps 10
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

> Never commit real keys. `.env` is git-ignored; `.env.example` lists names only.

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
| 3 | SEC 13F, congressional trades, monthly LLM adjustment | planned |
| 4 | Adaptive self-learning (Bayesian/online), walk-forward, broad universe | planned |

---

## Disclaimer

This software is for research and educational purposes only. It does not provide
financial advice and does not execute real trades. Markets involve risk; do your
own research.
