"""SQLite schema. Written to be Postgres-portable (no SQLite-only types).

The repository layer is the only place that touches SQL, so migrating to
Postgres later is a driver/SQL-dialect change rather than a rewrite.
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS securities (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    exchange    TEXT,
    sector      TEXT,
    industry    TEXT,
    is_etf      INTEGER DEFAULT 0,
    active      INTEGER DEFAULT 1,
    first_seen  TEXT,
    last_seen   TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    symbol     TEXT,
    date       TEXT,
    open       REAL, high REAL, low REAL, close REAL, adj_close REAL,
    volume     REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol     TEXT,
    asof_date  TEXT,
    metric     TEXT,
    value      REAL,
    source     TEXT,
    PRIMARY KEY (symbol, asof_date, metric)
);

CREATE TABLE IF NOT EXISTS macro (
    series_id  TEXT,
    date       TEXT,
    value      REAL,
    source     TEXT,
    PRIMARY KEY (series_id, date)
);

CREATE TABLE IF NOT EXISTS scores (
    symbol     TEXT,
    date       TEXT,
    factor     TEXT,
    raw        REAL,
    normalized REAL,
    PRIMARY KEY (symbol, date, factor)
);

CREATE TABLE IF NOT EXISTS composite_scores (
    symbol     TEXT,
    date       TEXT,
    portfolio  TEXT,
    score      REAL,
    rank       INTEGER,
    PRIMARY KEY (symbol, date, portfolio)
);

CREATE TABLE IF NOT EXISTS portfolios (
    id               INTEGER PRIMARY KEY,
    name             TEXT UNIQUE,
    type             TEXT,
    starting_capital REAL,
    txn_cost         REAL DEFAULT 0,
    paper_only       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS positions (
    portfolio_id INTEGER,
    symbol       TEXT,
    date         TEXT,
    shares       REAL,
    avg_cost     REAL,
    weight       REAL,
    PRIMARY KEY (portfolio_id, symbol, date)
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER,
    date         TEXT,
    symbol       TEXT,
    side         TEXT,
    shares       REAL,
    price        REAL,
    cost         REAL,
    reason       TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    portfolio_id INTEGER,
    date         TEXT,
    cash         REAL,
    equity       REAL,
    total_value  REAL,
    daily_pl     REAL,
    total_pl     REAL,
    drawdown     REAL,
    PRIMARY KEY (portfolio_id, date)
);

CREATE TABLE IF NOT EXISTS llm_reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT,
    scope       TEXT,
    provider    TEXT,
    model       TEXT,
    output      TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT,
    status          TEXT,
    degraded_reason TEXT,
    api_errors      TEXT
);

CREATE TABLE IF NOT EXISTS factor_weights (
    portfolio  TEXT,
    asof_date  TEXT,
    factor     TEXT,
    weight     REAL,
    source     TEXT,
    PRIMARY KEY (portfolio, asof_date, factor, source)
);
"""
