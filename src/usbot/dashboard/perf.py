"""Portfolio-vs-benchmark performance prep (equity curves, drawdowns, metrics).

Builds normalized (=100) equity series for each portfolio (from its saved
equity history) and each benchmark ETF (from price history), then computes
comparison metrics. Graceful with short history.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.metrics import compute_metrics


@dataclass
class PerfSeries:
    name: str
    equity: pd.Series              # normalized to 100 at the window start
    is_portfolio: bool = False


@dataclass
class PerfRow:
    name: str
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    alpha_spy: float
    alpha_qqq: float


def _normalize(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty or s.iloc[0] == 0:
        return s
    return s / s.iloc[0] * 100.0


def benchmark_equity(prices: dict[str, pd.DataFrame], symbol: str,
                     window_days: int = 252) -> pd.Series | None:
    df = prices.get(symbol)
    if df is None or df.empty:
        return None
    col = "adj_close" if "adj_close" in df.columns else "close"
    s = df[col].astype(float).tail(window_days)
    s.index = pd.to_datetime(s.index)
    return _normalize(s) if len(s) >= 2 else None


def portfolio_equity(history: list[dict]) -> pd.Series | None:
    """Normalized equity from a portfolio's saved history list."""
    pts = [(h.get("date"), h.get("total_value")) for h in (history or [])
           if h.get("date") and h.get("total_value")]
    if len(pts) < 2:
        return None
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in pts}).sort_index()
    return _normalize(s)


def drawdown(equity: pd.Series) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    return equity / equity.cummax() - 1.0


def build_perf_table(series: list[PerfSeries], prices: dict) -> list[PerfRow]:
    """Per-series metrics + alpha vs SPY/QQQ (CAGR difference)."""
    def cagr_of(sym: str) -> float:
        eq = benchmark_equity(prices, sym)
        return compute_metrics(eq).cagr if eq is not None and len(eq) > 2 else 0.0

    spy_cagr, qqq_cagr = cagr_of("SPY"), cagr_of("QQQ")
    rows: list[PerfRow] = []
    for ps in series:
        if ps.equity is None or len(ps.equity) < 2:
            continue
        m = compute_metrics(ps.equity)
        rows.append(PerfRow(
            name=ps.name, total_return=m.total_return, cagr=m.cagr, ann_vol=m.ann_vol,
            sharpe=m.sharpe, max_drawdown=m.max_drawdown,
            alpha_spy=m.cagr - spy_cagr, alpha_qqq=m.cagr - qqq_cagr,
        ))
    return rows
