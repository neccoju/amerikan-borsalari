"""Corporate actions for the paper books: dividend cash credits + split adjustments.

Benchmarks are compared on adjusted (total-return) prices, so the paper books must
also receive dividends or their alpha is systematically understated — especially
the Defensive sleeve. Splits must adjust share counts or a 10:1 split would show
as a -90% position loss.

Events are read from the price frames (yfinance ``actions=True`` adds
``dividends`` and ``splits`` columns) strictly AFTER the book's last processed
date, so same-day re-runs never double-credit. Dividend cash uses the shares
held now — with daily runs the gap to the ex-date is at most one session, so
this matches the ex-date position in practice.
"""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from .base import PortfolioState

log = get_logger(__name__)


def apply_corporate_actions(state: PortfolioState, price_history: dict,
                            since_date: str | None, sleeve: str,
                            date: str) -> tuple[list[dict], list[str]]:
    """Credit dividends / apply splits for events in ``(since_date, today]``.

    Mutates ``state`` (cash and share counts). Returns (ledger_rows, action_notes).
    First-ever run (``since_date`` falsy) is a no-op: no retroactive windfalls.
    """
    rows: list[dict] = []
    notes: list[str] = []
    if not since_date or not price_history:
        return rows, notes
    cutoff = pd.Timestamp(since_date)
    total_div = 0.0

    for sym, h in list(state.holdings.items()):
        df = price_history.get(sym)
        if df is None or getattr(df, "empty", True):
            continue
        idx = pd.to_datetime(df.index)

        if "splits" in df.columns:
            sp = pd.Series(df["splits"].astype(float).values, index=idx)
            for ts, ratio in sp[(sp.index > cutoff) & (sp > 0) & (sp != 1.0)].items():
                h.shares *= ratio
                h.avg_cost /= ratio
                rows.append({"date": date, "sleeve": sleeve, "type": "split",
                             "side": "split", "symbol": sym, "shares": h.shares,
                             "price": 0.0, "cost": 0.0,
                             "reason": f"{ratio:g}:1 split — shares x{ratio:g}, "
                                       f"cost basis adjusted ({ts.date()})"})
                log.info("[%s] %s split %g:1 applied", sleeve, sym, ratio)

        if "dividends" in df.columns:
            div = pd.Series(df["dividends"].astype(float).values, index=idx)
            for ts, amt in div[(div.index > cutoff) & (div > 0)].items():
                cash = h.shares * float(amt)
                state.cash += cash
                total_div += cash
                rows.append({"date": date, "sleeve": sleeve, "type": "dividend",
                             "side": "dividend", "symbol": sym,
                             "shares": round(h.shares, 8), "price": float(amt),
                             "cost": 0.0,
                             "reason": f"dividend ${amt:.4f}/sh (ex {ts.date()})"})

    if total_div > 0:
        notes.append(f"Dividends credited: ${total_div:.2f}")
    return rows, notes
