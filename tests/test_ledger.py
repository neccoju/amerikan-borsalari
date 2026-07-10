"""Transaction journal: helpers, store persistence, and surfacing in email + dashboard."""
from __future__ import annotations

from usbot.portfolio import (Holding, PortfolioState, PortfolioStore, entries_on, rebalance_row,
                             rebalance_to_targets, total_cost, trade_row)
from usbot.reports.builder import PortfolioReport, ReportContext, build_report


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


# ---- itemised delta-rebalance trades (all sleeves, with prices) ------------
def test_rebalance_trades_deltas_and_preserves_avg_cost():
    st = PortfolioState(name="Growth", ptype="growth", cash=0.0, starting_capital=1000.0)
    st.holdings["AAA"] = Holding("AAA", 1.0, 100.0)   # dropped -> exit sell
    st.holdings["BBB"] = Holding("BBB", 2.0, 50.0)    # target == current -> NO trade
    prices = {"AAA": 110.0, "BBB": 55.0, "CCC": 10.0}
    # total = 110 + 110 = 220; targets: BBB 50% (=110 -> 2.0 sh, unchanged), CCC 50%
    trades = rebalance_to_targets(st, {"BBB": 0.5, "CCC": 0.5}, prices, txn_cost=0.0)
    buys = {t["symbol"]: t for t in trades if t["side"] == "buy"}
    sells = {t["symbol"]: t for t in trades if t["side"] == "sell"}
    assert set(buys) == {"CCC"}                        # only the actual delta is traded
    assert buys["CCC"]["price"] == 10.0                # fill price recorded
    assert set(sells) == {"AAA"}                       # genuine exit at current price
    assert sells["AAA"]["price"] == 110.0
    # carried name keeps its original cost basis -> P/L stays meaningful
    assert st.holdings["BBB"].avg_cost == 50.0
    assert abs(st.holdings["BBB"].shares - 2.0) < 1e-9
    # book is fully invested per targets; cash ~0
    assert abs(st.cash) < 1e-6


def test_rebalance_add_uses_weighted_average_cost():
    st = PortfolioState(name="Balanced", ptype="balanced", cash=100.0, starting_capital=200.0)
    st.holdings["MU"] = Holding("MU", 1.0, 100.0)      # 1 sh @ $100
    prices = {"MU": 100.0}
    # total = 200 -> target 100% MU = 2 sh -> buy 1 more at $100... use a different
    # price to see the averaging: reprice MU to 120 (total 220, target 220/120)
    prices = {"MU": 120.0}
    trades = rebalance_to_targets(st, {"MU": 1.0}, prices, txn_cost=0.0)
    assert len(trades) == 1 and trades[0]["side"] == "buy"
    h = st.holdings["MU"]
    # bought (220/120 - 1) ≈ 0.8333 sh @120; avg = (1*100 + 0.8333*120)/1.8333 ≈ 109.09
    assert 100.0 < h.avg_cost < 120.0


def test_rebalance_all_names_have_prices_for_ledger():
    st = PortfolioState(name="Balanced", ptype="balanced", cash=1000.0, starting_capital=1000.0)
    trades = rebalance_to_targets(st, {"MU": 0.6, "LLY": 0.4},
                                  {"MU": 100.0, "LLY": 200.0}, txn_cost=0.0)
    assert all(t["price"] > 0 and t["shares"] > 0 for t in trades)
    assert {t["symbol"] for t in trades} == {"MU", "LLY"}


# ---- portfolio value breakdown (total / in-shares / cash) ------------------
def test_email_shows_value_breakdown():
    pf = PortfolioReport(name="Growth", total_value=1011.28, cash=250.0, equity=761.28,
                         daily_pl=1.4, total_pl=11.28)
    html, text = build_report(_ctx(portfolios=[pf]))
    assert "In shares:" in html and "761.28" in html and "250.00" in html
    assert "in shares $761.28" in text and "cash $250.00" in text


# ---- data gaps separated from real errors ----------------------------------
def test_email_data_gaps_summarised_not_errors():
    gaps = [f"SYM{i}" for i in range(12)]
    html, text = build_report(_ctx(data_gaps=gaps, errors=["fred: real failure"]))
    # gaps are a benign summary, not in the error list
    assert "Data Gaps" in html and "12 symbol" in html
    assert "12 symbols had no price data" in text
    # a genuine error still shows in its own section
    assert "real failure" in html and "real failure" in text
