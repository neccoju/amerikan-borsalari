import pandas as pd

from usbot.portfolio.active import ActivePortfolio
from usbot.portfolio.base import Holding, PortfolioState


def _active(min_edge=0.0):
    return ActivePortfolio(
        risk_cfg={"max_position": 0.15, "max_daily_turnover": 1.0,
                  "min_expected_edge_after_cost": min_edge},
        txn_cost=1.5, min_cash_buffer_pct=0.05, initial_deploy_pct=0.25,
        fill_timing="close",   # these tests assert immediate-fill accounting
    )


def _state(cash=1600.0):
    return PortfolioState(name="Active", ptype="active", cash=cash,
                          starting_capital=1600.0, txn_cost=1.5)


def test_buys_charge_transaction_cost():
    ap = _active()
    state = _state()
    scores = pd.Series({"AAA": 90.0})
    prices = {"AAA": 100.0}
    indicators = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    dec = ap.decide(state, scores, prices, indicators, regime_label="risk_on")
    buys = [t for t in dec.trades if t.side == "buy"]
    assert buys, "expected at least one buy on a high score in risk_on"
    assert all(t.cost == 1.5 for t in buys)
    # cash reduced by notional + fee
    spent = sum(t.shares * t.price + t.cost for t in buys)
    assert abs((1600.0 - spent) - state.cash) < 1e-6


def test_no_buy_when_below_entry_threshold():
    ap = _active()
    state = _state()
    scores = pd.Series({"AAA": 55.0})  # below ENTRY_SCORE
    prices = {"AAA": 100.0}
    indicators = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.25}}
    dec = ap.decide(state, scores, prices, indicators, regime_label="risk_on")
    assert not [t for t in dec.trades if t.side == "buy"]


def test_risk_off_exits_holdings():
    ap = _active()
    state = _state(cash=0.0)
    state.holdings["AAA"] = Holding("AAA", shares=5.0, avg_cost=100.0)
    scores = pd.Series({"AAA": 90.0})
    prices = {"AAA": 105.0}
    indicators = {"AAA": {"above_sma50": 1.0}}
    dec = ap.decide(state, scores, prices, indicators, regime_label="risk_off")
    sells = [t for t in dec.trades if t.side == "sell"]
    assert sells and sells[0].reason == "regime_risk_off_derisk"
    assert "AAA" not in state.holdings


def test_stop_loss_triggers_exit():
    ap = _active()
    state = _state(cash=0.0)
    state.holdings["AAA"] = Holding("AAA", shares=5.0, avg_cost=100.0)
    scores = pd.Series({"AAA": 90.0})
    prices = {"AAA": 80.0}  # -20% < stop
    indicators = {"AAA": {"above_sma50": 1.0}}
    dec = ap.decide(state, scores, prices, indicators, regime_label="risk_on")
    sells = [t for t in dec.trades if t.side == "sell"]
    assert sells and "stop_loss" in sells[0].reason
