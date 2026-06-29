"""Thin SQLite repository. The ONLY place raw SQL lives.

Keeps a single connection; all writes are upserts so reruns on the same day are
idempotent. Designed so swapping to Postgres means changing this file only.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging import get_logger
from .schema import SCHEMA

log = get_logger(__name__)


class Repository:
    def __init__(self, db_path: str | Path = "data/usbot.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- securities -------------------------------------------------------
    def upsert_security(self, sym: str, **fields: Any) -> None:
        today = dt.date.today().isoformat()
        self.conn.execute(
            """INSERT INTO securities (symbol, name, exchange, sector, industry, is_etf,
                                        active, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,1,?,?)
               ON CONFLICT(symbol) DO UPDATE SET
                   name=excluded.name, exchange=excluded.exchange, sector=excluded.sector,
                   industry=excluded.industry, is_etf=excluded.is_etf, last_seen=excluded.last_seen""",
            (
                sym,
                fields.get("name"),
                fields.get("exchange"),
                fields.get("sector"),
                fields.get("industry"),
                int(fields.get("is_etf", 0)),
                today,
                today,
            ),
        )

    # ---- prices -----------------------------------------------------------
    def save_prices(self, sym: str, rows: Iterable[tuple]) -> None:
        self.conn.executemany(
            """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, date) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low,
                   close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume""",
            [(sym, *r) for r in rows],
        )

    # ---- generic score / snapshot writers --------------------------------
    def save_score(self, sym: str, date: str, factor: str, raw: float, norm: float) -> None:
        self.conn.execute(
            """INSERT INTO scores (symbol, date, factor, raw, normalized) VALUES (?,?,?,?,?)
               ON CONFLICT(symbol, date, factor) DO UPDATE SET raw=excluded.raw,
                   normalized=excluded.normalized""",
            (sym, date, factor, raw, norm),
        )

    def save_composite(self, sym: str, date: str, portfolio: str, score: float, rank: int) -> None:
        self.conn.execute(
            """INSERT INTO composite_scores (symbol, date, portfolio, score, rank)
               VALUES (?,?,?,?,?)
               ON CONFLICT(symbol, date, portfolio) DO UPDATE SET score=excluded.score,
                   rank=excluded.rank""",
            (sym, date, portfolio, score, rank),
        )

    def ensure_portfolio(self, name: str, ptype: str, capital: float, txn_cost: float,
                          paper_only: bool = True) -> int:
        cur = self.conn.execute("SELECT id FROM portfolios WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            """INSERT INTO portfolios (name, type, starting_capital, txn_cost, paper_only)
               VALUES (?,?,?,?,?)""",
            (name, ptype, capital, txn_cost, int(paper_only)),
        )
        return int(cur.lastrowid)

    def save_trade(self, portfolio_id: int, date: str, sym: str, side: str, shares: float,
                   price: float, cost: float, reason: str) -> None:
        self.conn.execute(
            """INSERT INTO trades (portfolio_id, date, symbol, side, shares, price, cost, reason)
               VALUES (?,?,?,?,?,?,?,?)""",
            (portfolio_id, date, sym, side, shares, price, cost, reason),
        )

    def save_position(self, portfolio_id: int, date: str, sym: str, shares: float,
                      avg_cost: float, weight: float) -> None:
        self.conn.execute(
            """INSERT INTO positions (portfolio_id, symbol, date, shares, avg_cost, weight)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(portfolio_id, symbol, date) DO UPDATE SET shares=excluded.shares,
                   avg_cost=excluded.avg_cost, weight=excluded.weight""",
            (portfolio_id, sym, date, shares, avg_cost, weight),
        )

    def save_snapshot(self, portfolio_id: int, date: str, cash: float, equity: float,
                      total_value: float, daily_pl: float, total_pl: float,
                      drawdown: float) -> None:
        self.conn.execute(
            """INSERT INTO portfolio_snapshots
                   (portfolio_id, date, cash, equity, total_value, daily_pl, total_pl, drawdown)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(portfolio_id, date) DO UPDATE SET cash=excluded.cash,
                   equity=excluded.equity, total_value=excluded.total_value,
                   daily_pl=excluded.daily_pl, total_pl=excluded.total_pl,
                   drawdown=excluded.drawdown""",
            (portfolio_id, date, cash, equity, total_value, daily_pl, total_pl, drawdown),
        )

    def latest_snapshot(self, portfolio_id: int) -> sqlite3.Row | None:
        cur = self.conn.execute(
            """SELECT * FROM portfolio_snapshots WHERE portfolio_id=?
               ORDER BY date DESC LIMIT 1""",
            (portfolio_id,),
        )
        return cur.fetchone()

    def save_llm_review(self, date: str, scope: str, provider: str, model: str,
                        output: str, note: str) -> None:
        self.conn.execute(
            """INSERT INTO llm_reviews (date, scope, provider, model, output, note)
               VALUES (?,?,?,?,?,?)""",
            (date, scope, provider, model, output, note),
        )

    def save_run(self, run_date: str, status: str, degraded_reason: str | None,
                 api_errors: list[str] | None) -> None:
        self.conn.execute(
            """INSERT INTO runs (run_date, status, degraded_reason, api_errors)
               VALUES (?,?,?,?)""",
            (run_date, status, degraded_reason, json.dumps(api_errors or [])),
        )

    def commit(self) -> None:
        self.conn.commit()
