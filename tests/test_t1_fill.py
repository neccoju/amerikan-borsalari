"""T+1 open-fill for the Active sleeve: entries decided at close fill at the
next session's open; risk exits stay immediate; same-day re-runs are idempotent."""
from __future__ import annotations

import pandas as pd

from usbot.portfolio.active import ActivePortfolio
from usbot.portfolio.base import Holding, PortfolioState


def _active(fill_timing="t1_open"):
    return ActivePortfolio(
        risk_cfg={"max_position": 0.5, "max_daily_turnover": 1.0},
        txn_cost=1.5, min_cash_buffer_pct=0.05, initial_deploy_pct=0.95,
        fill_timing=fill_timing,
    )


def _state(cash=1600.0):
    return PortfolioState(name="Active", ptype="active", cash=cash, starting_capital=1600.0)


def test_entry_is_queued_not_filled_on_decision_day():
    ap = _active()
    st = _state()
    scores = pd.Series({"AAA": 90.0})
    prices = {"AAA": 100.0}
    ind = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    dec = ap.decide(st, scores, prices, ind, "risk_on", opens={"AAA": 100.0}, date="2026-07-14")
    assert [t for t in dec.trades if t.side == "buy"] == []   # nothing filled today
    assert st.cash == 1600.0                                  # cash untouched
    assert len(st.pending_orders) == 1 and st.pending_orders[0]["symbol"] == "AAA"


def test_pending_fills_next_day_at_open():
    ap = _active()
    st = _state()
    scores = pd.Series({"AAA": 90.0})
    ind = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    # day 1: decide at close 100 -> queue
    ap.decide(st, scores, {"AAA": 100.0}, ind, "risk_on", opens={"AAA": 100.0}, date="2026-07-14")
    notional = st.pending_orders[0]["notional"]
    # day 2: opens at 110 -> fill there (not at day-2 close of 108). Score 55
    # holds the position (above exit 48, below entry 62 -> no new order).
    dec2 = ap.decide(st, pd.Series({"AAA": 55.0}), {"AAA": 108.0}, {"AAA": {"above_sma50": 1.0}},
                     "risk_on", opens={"AAA": 110.0}, date="2026-07-15")
    fills = [t for t in dec2.trades if t.side == "buy"]
    assert len(fills) == 1
    assert fills[0].price == 110.0                            # filled at the OPEN
    assert abs(fills[0].shares - notional / 110.0) < 1e-9
    assert "AAA" in st.holdings and st.holdings["AAA"].avg_cost == 110.0
    assert abs(st.cash - (1600.0 - notional - 1.5)) < 1e-6    # cash debited at fill


def test_same_day_rerun_is_idempotent():
    ap = _active()
    st = _state()
    scores = pd.Series({"AAA": 90.0})
    prices = {"AAA": 100.0}
    ind = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    ap.decide(st, scores, prices, ind, "risk_on", opens={"AAA": 100.0}, date="2026-07-14")
    ap.decide(st, scores, prices, ind, "risk_on", opens={"AAA": 100.0}, date="2026-07-14")
    # a re-run on the SAME day must not fill today's pending nor duplicate it
    assert len(st.pending_orders) == 1
    assert st.cash == 1600.0 and not st.holdings


def test_risk_exit_is_immediate_even_in_t1_mode():
    ap = _active()
    st = _state(cash=0.0)
    st.holdings["AAA"] = Holding("AAA", shares=5.0, avg_cost=100.0)
    dec = ap.decide(st, pd.Series({"AAA": 90.0}), {"AAA": 105.0}, {"AAA": {"above_sma50": 1.0}},
                    "risk_off", opens={"AAA": 106.0}, date="2026-07-15")
    sells = [t for t in dec.trades if t.side == "sell"]
    assert sells and sells[0].price == 105.0                 # sold now, at the close
    assert "AAA" not in st.holdings


def test_pending_falls_back_to_close_when_open_missing():
    ap = _active()
    st = _state()
    ind = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    ap.decide(st, pd.Series({"AAA": 90.0}), {"AAA": 100.0}, ind, "risk_on",
              opens={"AAA": 100.0}, date="2026-07-14")
    # next day: no open available -> fill at the close as a graceful fallback
    dec2 = ap.decide(st, pd.Series({"AAA": 90.0}), {"AAA": 104.0}, ind, "risk_on",
                     opens={}, date="2026-07-15")
    fills = [t for t in dec2.trades if t.side == "buy"]
    assert len(fills) == 1 and fills[0].price == 104.0


def test_close_mode_still_fills_immediately():
    ap = _active(fill_timing="close")
    st = _state()
    dec = ap.decide(st, pd.Series({"AAA": 90.0}), {"AAA": 100.0},
                    {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}, "risk_on",
                    date="2026-07-14")
    assert [t for t in dec.trades if t.side == "buy"]        # legacy path unchanged
    assert not st.pending_orders and "AAA" in st.holdings
