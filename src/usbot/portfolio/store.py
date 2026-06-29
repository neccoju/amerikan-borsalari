"""Persistent JSON state for ALL simulated portfolios.

GitHub Actions runners are ephemeral, so day-to-day continuity requires storing
state somewhere durable. We keep a single, diff-friendly JSON file
(`state/portfolios.json`) holding every sleeve's cash, holdings and equity
history. The workflow commits it back after each real run, so the portfolios
buy at real prices, hold real (fractional) positions, and accumulate true P/L.

Each holding records the actual fill price (``avg_cost``), so the report can show
real share counts and prices rather than abstract weight allocations.
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
    history: list[dict] = field(default_factory=list)       # [{date, total_value, cash}]
    last_decision_date: str | None = None                   # active sleeve
    last_rebalance_date: str | None = None                  # model/self-learning sleeves
    meta: dict = field(default_factory=dict)                # sleeve-specific extras (e.g. learned weights)
    existed: bool = False


class PortfolioStore:
    """Single-file store keyed by portfolio name. Batches writes: stage many,
    then commit once."""

    def __init__(self, path: str | Path = "state/portfolios.json") -> None:
        self.path = Path(path)
        self._data: dict | None = None

    # ---- internal ---------------------------------------------------------
    def _all(self) -> dict:
        if self._data is not None:
            return self._data
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to read %s (%s); starting fresh", self.path, exc)
                self._data = {"portfolios": {}}
        else:
            self._data = {"portfolios": {}}
            self._migrate_legacy_active()
        self._data.setdefault("portfolios", {})
        return self._data

    def _migrate_legacy_active(self) -> None:
        """Import the old per-active file (state/active_portfolio.json) once."""
        legacy = self.path.parent / "active_portfolio.json"
        if not legacy.exists():
            return
        try:
            old = json.loads(legacy.read_text(encoding="utf-8"))
            name = old.get("name", "Active Entry")
            self._data["portfolios"][name] = old
            log.info("Migrated legacy active state for '%s'", name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not migrate legacy active state: %s", exc)

    # ---- load -------------------------------------------------------------
    def load(self, name: str, starting_capital: float, txn_cost: float = 0.0) -> LoadedState:
        p = self._all()["portfolios"].get(name)
        if not p:
            return LoadedState(
                state=PortfolioState(name=name, ptype="", cash=starting_capital,
                                     starting_capital=starting_capital, txn_cost=txn_cost),
                existed=False,
            )
        state = PortfolioState(
            name=name, ptype=p.get("ptype", ""),
            cash=float(p.get("cash", starting_capital)),
            starting_capital=float(p.get("starting_capital", starting_capital)),
            txn_cost=txn_cost,
        )
        for h in p.get("holdings", []):
            state.holdings[h["symbol"]] = Holding(
                symbol=h["symbol"], shares=float(h["shares"]), avg_cost=float(h["avg_cost"]))
        return LoadedState(
            state=state,
            history=list(p.get("history", [])),
            last_decision_date=p.get("last_decision_date"),
            last_rebalance_date=p.get("last_rebalance_date"),
            meta=dict(p.get("meta", {})),
            existed=True,
        )

    # ---- stage (in-memory) ------------------------------------------------
    def stage(self, state: PortfolioState, prices: dict[str, float], date: str,
              history: list[dict], *, ptype: str = "",
              last_decision_date: str | None = None,
              last_rebalance_date: str | None = None,
              meta: dict | None = None,
              max_history: int = 750) -> tuple[list[dict], float]:
        total_value = state.total_value(prices)
        history = [h for h in history if h.get("date") != date]
        history.append({"date": date, "total_value": round(total_value, 2),
                        "cash": round(state.cash, 2)})
        history = history[-max_history:]
        self._all()["portfolios"][state.name] = {
            "ptype": ptype or state.ptype,
            "starting_capital": state.starting_capital,
            "cash": round(state.cash, 6),
            "holdings": [
                {"symbol": s, "shares": round(h.shares, 8), "avg_cost": round(h.avg_cost, 6)}
                for s, h in sorted(state.holdings.items())
            ],
            "history": history,
            "last_decision_date": last_decision_date,
            "last_rebalance_date": last_rebalance_date,
            "meta": meta or {},
            "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        return history, total_value

    # ---- commit (write once) ---------------------------------------------
    def commit(self) -> None:
        if self._data is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n", encoding="utf-8")
        log.info("Committed portfolio state -> %s (%d sleeves)",
                 self.path, len(self._data.get("portfolios", {})))


def performance_from_history(history: list[dict], total_value: float,
                             starting_capital: float) -> dict:
    """Daily P/L, total P/L and drawdown from the equity history (incl. today)."""
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
