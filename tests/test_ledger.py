"""Transaction journal: helpers, store persistence, and surfacing in email + dashboard."""
from __future__ import annotations

from usbot.portfolio import (PortfolioState, PortfolioStore, entries_on, rebalance_row,
                             total_cost, trade_row)
from usbot.reports.builder import ReportContext, build_report


# ---- helpers ---------------------------------------------------------------
def test_trade_and_rebalance_rows():
    t = trade_row("2026-06-30", "Active Entry", "buy", "VICR", 0.34, 366.79, 1.5, "entry")
    assert t["type"] == "buy" and t["symbol"] == "VICR" and t["cost"] == 1.5
    r = rebalance_row("2026-06-30", "Growth", 15, cost=0.0, reason="Month-end rebalance")
    assert r["type"] == "rebalance" and r["symbol"] == "15 names" and r["cost"] == 0.0


def test_total_cost_and_entries_on():
    led = [
        trade_row("2026-06-29", "Active Entry", "buy", "AMD", 1, 500, 1.5, "x"),
        trade_row("2026-06-30", "Active Entry", "sell", "AMD", 1, 540, 1.5, "y"),
        rebalance_row("2026-06-30", "Growth", 15, cost=0.0),
    ]
    assert total_cost(led) == 3.0
    assert total_cost([]) == 0.0
    today = entries_on(led, "2026-06-30")
    assert len(today) == 2


# ---- store persistence -----------------------------------------------------
def test_store_persists_and_appends_ledger(tmp_path):
    path = tmp_path / "pf.json"
    store = PortfolioStore(path)
    st = PortfolioState(name="Active Entry", ptype="active", cash=1000.0,
                        starting_capital=1000.0)
    led = [trade_row("2026-06-30", "Active Entry", "buy", "AMD", 0.5, 500.0, 1.5, "entry")]
    store.stage(st, {}, "2026-06-30", [], ptype="active", ledger=led)
    store.commit()

    # reload: ledger round-trips, then append a second day's entry
    store2 = PortfolioStore(path)
    loaded = store2.load("Active Entry", 1000.0)
    assert len(loaded.ledger) == 1 and loaded.ledger[0]["symbol"] == "AMD"
    led2 = list(loaded.ledger) + [
        trade_row("2026-07-01", "Active Entry", "sell", "AMD", 0.5, 520.0, 1.5, "exit")]
    store2.stage(loaded.state, {}, "2026-07-01", loaded.history, ptype="active", ledger=led2)
    store2.commit()

    loaded3 = PortfolioStore(path).load("Active Entry", 1000.0)
    assert len(loaded3.ledger) == 2
    assert total_cost(loaded3.ledger) == 3.0


def test_store_preserves_ledger_when_not_passed(tmp_path):
    """Staging without a ledger arg must not wipe an existing journal."""
    path = tmp_path / "pf.json"
    store = PortfolioStore(path)
    st = PortfolioState(name="Growth", ptype="growth", cash=1000.0, starting_capital=1000.0)
    store.stage(st, {}, "2026-06-30", [], ptype="growth",
                ledger=[rebalance_row("2026-06-30", "Growth", 15)])
    store.commit()
    # stage again with no ledger kwarg
    store.stage(st, {}, "2026-07-01", [], ptype="growth")
    store.commit()
    assert len(PortfolioStore(path).load("Growth", 1000.0).ledger) == 1


# ---- email surfacing -------------------------------------------------------
def _ctx(**kw) -> ReportContext:
    ctx = ReportContext(date="2026-06-30", market_status="open")
    ctx.regime_label = "risk_on"
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_email_shows_ledger_and_costs():
    ledger = [
        trade_row("2026-06-30", "Active Entry", "buy", "VICR", 0.34, 366.79, 1.5, "entry(score=81)"),
        rebalance_row("2026-06-30", "Growth", 15, cost=0.0, reason="Month-end rebalance into 15 names"),
    ]
    html, text = build_report(_ctx(is_month_end=True, ledger_today=ledger,
                                   cost_today=1.5, cost_total=12.0))
    # HTML
    assert "Transaction Ledger" in html
    assert "VICR" in html and "BUY" in html
    assert "Month-end rebalance: <b>YES</b>" in html
    assert "1.50" in html and "12.00" in html
    # text
    assert "Transaction ledger" in text
    assert "Month-end rebalance day: YES" in text
    assert "VICR" in text


def test_email_handles_no_trades():
    html, text = build_report(_ctx(is_month_end=False, ledger_today=[],
                                   cost_today=0.0, cost_total=9.0))
    assert "No trades today" in html
    assert "Month-end rebalance: <b>no</b>" in html
    assert "no trades today" in text.lower()
