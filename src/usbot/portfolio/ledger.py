"""Transaction journal ("defter") helpers.

A ledger is a per-portfolio list of dated entries recording what the bot DID on
each run: individual buys/sells (Active sleeve, with the real per-trade cost) and
month-end rebalances / model rebuilds (model + self-learning sleeves). Entries are
plain dicts so they persist verbatim in the JSON state and render in both the
email and the dashboard. Costs are always recorded so they can be totalled.

Entry schema::

    {date, sleeve, type, symbol, side, shares, price, cost, reason}

``type`` is one of: "buy", "sell", "rebalance", "rebuild".
"""
from __future__ import annotations


def trade_row(date: str, sleeve: str, side: str, symbol: str, shares: float,
              price: float, cost: float, reason: str = "") -> dict:
    """One itemised buy/sell (e.g. an Active-sleeve trade carrying a fee)."""
    return {
        "date": date, "sleeve": sleeve, "type": side, "side": side,
        "symbol": symbol, "shares": round(float(shares), 8),
        "price": round(float(price), 6), "cost": round(float(cost), 4),
        "reason": reason,
    }


def rebalance_row(date: str, sleeve: str, n_names: int, cost: float = 0.0,
                  reason: str = "", kind: str = "rebalance") -> dict:
    """A model/self-learning month-end rebalance or rebuild (summary row)."""
    return {
        "date": date, "sleeve": sleeve, "type": kind, "side": kind,
        "symbol": f"{n_names} names", "shares": 0.0, "price": 0.0,
        "cost": round(float(cost), 4), "reason": reason,
    }


def total_cost(ledger: list[dict] | None) -> float:
    """Sum of all transaction costs in a ledger (lifetime if given the full list)."""
    return round(sum(float(e.get("cost", 0.0)) for e in (ledger or [])), 4)


def entries_on(ledger: list[dict] | None, date: str) -> list[dict]:
    """The subset of entries recorded on ``date`` (today's activity)."""
    return [e for e in (ledger or []) if e.get("date") == date]
