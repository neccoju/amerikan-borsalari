"""Persistent JSON state for the Active portfolio.

GitHub Actions runners are ephemeral, so day-to-day continuity requires storing
state somewhere durable. We use a small, diff-friendly JSON file that the
workflow commits back to the repo after each real run. This lets the Active
sleeve accumulate positions, cash and a daily equity history across runs.

Only the Active sleeve is persisted (the model sleeves are monthly allocation
studies that rebuild from scratch each run by design).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging import get_logger
from .base import Holding, PortfolioState

log = get_logger(__name__)


@dataclass
class LoadedState:
    state: PortfolioState
    history: list[dict] = field(default_factory=list)   # [{date, total_value, cash}]
    last_decision_date: str | None = None
    existed: bool = False


class ActivePortfolioStore:
    def __init__(self, path: str | Path = "state/active_portfolio.json") -> None:
        self.path = Path(path)

    # ---- load -------------------------------------------------------------
    def load(self, name: str, starting_capital: float, txn_cost: float) -> LoadedState:
        if not self.path.exists():
            log.info("No prior active state at %s; starting fresh at $%.2f",
                     self.path, starting_capital)
            return LoadedState(
                state=PortfolioState(name=name, ptype="active", cash=starting_capital,
                                     starting_capital=starting_capital, txn_cost=txn_cost),
                existed=False,
            )
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to read active state (%s); starting fresh", exc)
            return LoadedState(
                state=PortfolioState(name=name, ptype="active", cash=starting_capital,
                                     starting_capital=starting_capital, txn_cost=txn_cost),
                existed=False,
            )
        state = PortfolioState(
            name=name, ptype="active",
            cash=float(data.get("cash", starting_capital)),
            starting_capital=float(data.get("starting_capital", starting_capital)),
            txn_cost=txn_cost,
        )
        for h in data.get("holdings", []):
            state.holdings[h["symbol"]] = Holding(
                symbol=h["symbol"], shares=float(h["shares"]), avg_cost=float(h["avg_cost"]))
        log.info("Loaded active state: cash=$%.2f, %d holdings, last_decision=%s",
                 state.cash, len(state.holdings), data.get("last_decision_date"))
        return LoadedState(
            state=state,
            history=list(data.get("history", [])),
            last_decision_date=data.get("last_decision_date"),
            existed=True,
        )

    # ---- save -------------------------------------------------------------
    def save(self, state: PortfolioState, prices: dict[str, float], date: str,
             history: list[dict], decided_today: bool,
             max_history: int = 750) -> tuple[list[dict], float]:
        """Persist state + append today's equity point. Returns (history, total_value)."""
        total_value = state.total_value(prices)
        history = [h for h in history if h.get("date") != date]  # dedupe today
        history.append({"date": date, "total_value": round(total_value, 2),
                        "cash": round(state.cash, 2)})
        history = history[-max_history:]
        data = {
            "name": state.name,
            "starting_capital": state.starting_capital,
            "cash": round(state.cash, 6),
            "holdings": [
                {"symbol": s, "shares": round(h.shares, 8), "avg_cost": round(h.avg_cost, 6)}
                for s, h in sorted(state.holdings.items())
            ],
            "history": history,
            "last_decision_date": date if decided_today else None,
            "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        log.info("Saved active state: total=$%.2f, cash=$%.2f, %d holdings",
                 total_value, state.cash, len(state.holdings))
        return history, total_value


def performance_from_history(history: list[dict], total_value: float,
                             starting_capital: float) -> dict:
    """Compute daily P/L, total P/L and drawdown from the equity history.

    ``history`` includes today's point as the last element.
    """
    prev_total = starting_capital
    if len(history) >= 2:
        prev_total = float(history[-2].get("total_value", starting_capital))
    peak = max([float(h.get("total_value", 0.0)) for h in history] + [total_value])
    drawdown = (total_value / peak - 1.0) if peak > 0 else 0.0
    return {
        "daily_pl": total_value - prev_total,
        "total_pl": total_value - starting_capital,
        "drawdown": drawdown,
    }
