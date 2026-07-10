"""Tests for the follow-up fixes: trailing stop, SQLite archive, close-aware cache."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from usbot.portfolio import ActivePortfolio, PortfolioState, PortfolioStore
from usbot.portfolio.base import Holding
from usbot.utils.dates import close_crossed


# ---- trailing stop -----------------------------------------------------------
def _active() -> ActivePortfolio:
    return ActivePortfolio(risk_cfg={"trail_drawdown": -0.15}, txn_cost=1.5,
                           min_cash_buffer_pct=0.05, initial_deploy_pct=0.25)


def _healthy_indicators(sym: str) -> dict:
    return {sym: {"above_sma50": 1.0, "realized_vol": 0.3}}


def test_trailing_stop_protects_gains():
    """+30% winner falls 15%+ from its high -> trailing stop exits, even though
    the position is still up vs cost (the avg-cost hard stop would never fire)."""
    act = _active()
    st = PortfolioState(name="Active Entry", ptype="active", cash=0.0,
                        starting_capital=1600.0)
    st.holdings["AMD"] = Holding("AMD", 2.0, 100.0)
    scores = pd.Series({"AMD": 75.0})                    # healthy score, no decay exit

    # day 1: price runs to 130 -> high-water ratchets, no exit
    dec = act.decide(st, scores, {"AMD": 130.0}, _healthy_indicators("AMD"), "risk_on")
    assert "AMD" in st.holdings and not [t for t in dec.trades if t.side == "sell"]
    assert abs(st.holdings["AMD"].high_water - 130.0) < 1e-9

    # day 2: price 110 -> +10% vs cost (hard stop silent) but -15.4% from high
    dec2 = act.decide(st, scores, {"AMD": 110.0}, _healthy_indicators("AMD"), "risk_on")
    sells = [t for t in dec2.trades if t.side == "sell"]
    assert sells and sells[0].symbol == "AMD"
    assert "trailing_stop" in sells[0].reason
    assert "AMD" not in st.holdings


def test_trailing_stop_not_triggered_within_band():
    act = _active()
    st = PortfolioState(name="Active Entry", ptype="active", cash=0.0,
                        starting_capital=1600.0)
    st.holdings["AMD"] = Holding("AMD", 2.0, 100.0)
    scores = pd.Series({"AMD": 75.0})
    act.decide(st, scores, {"AMD": 130.0}, _healthy_indicators("AMD"), "risk_on")
    # -10% from high: inside the -15% band -> hold
    dec = act.decide(st, scores, {"AMD": 117.0}, _healthy_indicators("AMD"), "risk_on")
    assert "AMD" in st.holdings and not [t for t in dec.trades if t.side == "sell"]


def test_high_water_persists_via_store(tmp_path):
    store = PortfolioStore(tmp_path / "pf.json")
    st = PortfolioState(name="Active Entry", ptype="active", cash=0.0,
                        starting_capital=1600.0)
    st.holdings["AMD"] = Holding("AMD", 2.0, 100.0, high_water=130.0)
    store.stage(st, {"AMD": 120.0}, "2026-06-30", [], ptype="active")
    store.commit()
    loaded = PortfolioStore(tmp_path / "pf.json").load("Active Entry", 1600.0)
    assert abs(loaded.state.holdings["AMD"].high_water - 130.0) < 1e-9


# ---- SQLite archive ------------------------------------------------------------
def test_archive_to_db_idempotent(tmp_path):
    from usbot.config.settings import Settings
    from usbot.db.repository import Repository
    from usbot.orchestrator import _archive_to_db
    from usbot.portfolio import trade_row
    from usbot.reports.builder import PortfolioReport, ReportContext

    store = PortfolioStore(tmp_path / "pf.json")
    st = PortfolioState(name="Active Entry", ptype="active", cash=100.0,
                        starting_capital=1600.0)
    st.holdings["AMD"] = Holding("AMD", 2.0, 100.0)
    ledger = [trade_row("2026-06-30", "Active Entry", "buy", "AMD", 2.0, 100.0, 1.5, "entry"),
              trade_row("2026-06-29", "Active Entry", "buy", "C", 1.0, 140.0, 1.5, "entry")]
    store.stage(st, {"AMD": 110.0}, "2026-06-30", [], ptype="active", ledger=ledger)

    ctx = ReportContext(date="2026-06-30", market_status="open")
    ctx.portfolios = [PortfolioReport(name="Active Entry", total_value=320.0, cash=100.0,
                                      equity=220.0, daily_pl=5.0, total_pl=20.0)]
    settings = Settings(settings={"database": {"path": str(tmp_path / "arc.db")}},
                        scoring={})

    _archive_to_db(ctx, store, settings, "2026-06-30")
    _archive_to_db(ctx, store, settings, "2026-06-30")   # rerun must not duplicate

    with Repository(tmp_path / "arc.db") as repo:
        rows = repo.conn.execute(
            "SELECT date, symbol, side, price FROM trades ORDER BY date").fetchall()
        snaps = repo.conn.execute("SELECT date, total_value FROM portfolio_snapshots").fetchall()
    # only TODAY's row archived by this run (yesterday's was archived that day),
    # and the same-day rerun replaced rather than duplicated
    today_rows = [r for r in rows if r["date"] == "2026-06-30"]
    assert len(today_rows) == 1 and today_rows[0]["symbol"] == "AMD"
    assert len(snaps) == 1 and abs(snaps[0]["total_value"] - 320.0) < 1e-9


# ---- close-aware cache ----------------------------------------------------------
def _epoch(y, m, d, hh, mm, tz="America/New_York") -> float:
    return dt.datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz)).timestamp()


def test_close_crossed_detects_session_close_between():
    # Tue 2026-06-30: cached 10:00 ET (intraday), read 20:30 ET -> 16:00 close between
    assert close_crossed(_epoch(2026, 6, 30, 10, 0), _epoch(2026, 6, 30, 20, 30)) is True
    # cached 17:00 ET (post-close), read 20:30 ET same day -> no close between
    assert close_crossed(_epoch(2026, 6, 30, 17, 0), _epoch(2026, 6, 30, 20, 30)) is False
    # Fri post-close -> Saturday: weekend, no close between
    assert close_crossed(_epoch(2026, 6, 26, 17, 0), _epoch(2026, 6, 27, 12, 0)) is False
    # Fri pre-close -> Saturday: Friday's close is in between
    assert close_crossed(_epoch(2026, 6, 26, 15, 0), _epoch(2026, 6, 27, 12, 0)) is True
    # degenerate: t1 <= t0
    assert close_crossed(_epoch(2026, 6, 30, 12, 0), _epoch(2026, 6, 30, 12, 0)) is False


def test_cached_prices_refetched_after_close(tmp_path, monkeypatch):
    """A frame cached before the close must be re-downloaded after the close,
    even though the 12h TTL has not expired."""
    import usbot.data.prices as prices_mod
    from usbot.data.cache import Cache

    cache = Cache(tmp_path, ttl_hours=12)
    idx = pd.date_range("2026-06-26", periods=3, freq="B")
    cache.save("px_AAPL", pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx))

    downloads: list[list[str]] = []

    def fake_download(symbols, period_days):
        downloads.append(list(symbols))
        cols = pd.MultiIndex.from_product([symbols, ["Close"]])
        return pd.DataFrame([[10.0] * len(symbols)] * 2,
                            index=pd.date_range("2026-06-29", periods=2, freq="B"),
                            columns=cols)

    monkeypatch.setattr(prices_mod, "_download", fake_download)

    # pretend a market close happened since the cache was written
    monkeypatch.setattr("usbot.utils.dates.close_crossed", lambda t0, t1: True)
    out = prices_mod.fetch_prices(["AAPL"], cache=cache)
    assert downloads and "AAPL" in downloads[0], "must refetch after a close"

    # and with no close in between, the cache is served (no new download)
    downloads.clear()
    monkeypatch.setattr("usbot.utils.dates.close_crossed", lambda t0, t1: False)
    out = prices_mod.fetch_prices(["AAPL"], cache=cache)
    assert not downloads
    assert "AAPL" in out.history
