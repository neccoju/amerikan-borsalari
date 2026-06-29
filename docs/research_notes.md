# Research Notes & Factor → Paper Mapping

> Literature-informed design rationale for usbot. This is a **research and
> monitoring tool, not financial advice**. All portfolios are simulated.

## Factor → academic basis (mapping table)

| Factor | Core idea | Representative literature (concept) | Our proxy in v1 | Known caveats |
|---|---|---|---|---|
| Momentum | Past 3–12M winners keep winning short-term | Jegadeesh & Titman (1993); Asness et al. | Blended 1/3/6/12M returns, vol-adjusted, cross-sectional rank | Crashes in sharp reversals; crowded |
| Value | Cheap valuations earn premium | Fama & French (1992); HML | P/E, P/S percentile (lower better) | Value traps; sector distortions |
| Quality | Profitable, low-leverage firms outperform | Novy-Marx (2013); QMJ (Asness) | Margins, ROE, debt/equity, FCF | Definitions vary; data gaps |
| Low volatility | Low-vol stocks earn higher risk-adjusted returns | Baker, Bradley & Wurgler (2011) | Realized vol, beta filter (defensive) | Rate-sensitive; can lag in rallies |
| PEAD | Drift after earnings surprises | Bernard & Thomas (1989) | (Phase 2) earnings calendar + surprise | Needs estimates; short window |
| Analyst revisions | Estimate revisions predict returns | Various | (Phase 2) revision trend | Free data sparse |
| News sentiment | Tone predicts short-horizon returns | Tetlock (2007); FinBERT | (Phase 2) FinBERT/VADER on headlines | Noisy; headline dedup needed |
| Institutional (13F) | Smart-money holdings signal | Various 13F studies | (Phase 3) SEC EDGAR 13F deltas | 45-day delay; long-only snapshot |
| Congressional trades | Disclosed politician trades | STOCK Act disclosure studies | (Phase 3) disclosure feeds | Delayed, noisy; weak signal |
| Macro regime | Trend/vol regimes change factor payoffs | Ang & Bekaert; regime-switching lit. | SPY/QQQ/IWM vs 200DMA + VIX | Regimes shift abruptly |
| Portfolio construction | MV/risk-parity/Black-Litterman | Markowitz; Black-Litterman | Score-proportional + caps (v1) | Estimation error; turnover |

## Strategy assumptions (v1)

- Daily close-based signals; no intraday.
- Fractional shares allowed in simulation.
- Model sleeves rebalance monthly (last trading day); Active trades daily.
- Active trade cost is a flat $1.5; model sleeves are allocation studies (no fee).
- Cross-sectional percentile normalization; missing data → neutral (50), never
  treated as worst-in-class.

## Known limitations

- **Survivorship bias:** the current S&P 500 seed is used for both live and
  backtests. Rigorous point-in-time universe reconstruction is **deferred to
  Phase 4** (documented and accepted).
- **Backtest scope (Phase 2):** only price-derived signals (momentum/technical)
  are truly point-in-time from free history. Fundamental/news factors lack free
  point-in-time history, so they are excluded from backtests (deferred to Phase 4).
  The live daily pipeline still uses them; only historical replay omits them.
- **News sentiment:** default model is VADER (lexicon-based), which mis-reads
  some financial jargon (e.g. "crushes earnings" → negative). FinBERT is the
  optional `[finbert]` upgrade for finance-aware sentiment.
- **Congressional trades (Phase 3):** sourced from free community datasets
  (house/senate stock-watcher). Delayed (up to ~45 days) and noisy; used as a
  weak-to-moderate signal with a small weight, never a direct trigger. Source
  availability is not guaranteed — the module skips gracefully if unreachable.
- **Institutional 13F (Phase 3):** via SEC EDGAR (edgartools). 13F reports by
  CUSIP/issuer, so ticker resolution is imperfect and the signal is experimental;
  it is delayed quarterly and used only as slow-moving confirmation. Skips
  gracefully when edgartools or ticker resolution is unavailable.
- **LLM nudges (Phase 3):** the monthly review may emit bounded per-ticker
  adjustments (±N points). They are surfaced in the report; feeding them back
  into the live composite is a documented follow-up. The LLM never trades.
- yfinance fundamentals are best-effort and occasionally stale/missing.
- Macro relies on keyless proxies (VIX/^TNX) until FRED is enabled.
- No transaction-cost modeling on model sleeves (intentional); the Active sleeve
  and the backtester both model costs.

## Implementation-safe rules

- LLM is decision-support only; it can never place a trade and is bounded to a
  ±N-point adjustment (Phase 3).
- Self-Learning sleeve is **paper-only** and never overwrites rule-based sleeves.
- One ticker / API failure must never abort the run.
- Backtests must enforce: point-in-time data, T+1 execution, transaction costs,
  and walk-forward for adaptive components (Phase 2 engine contract).
