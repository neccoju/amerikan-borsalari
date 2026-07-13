"""Tests for Layer-1 signals: insider (Form 4) score, earnings/PEAD score +
blackout, the Finnhub parsers, and the Active-sleeve earnings blackout."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from usbot.earnings.ingest import _parse_surprises
from usbot.earnings.model import EarningsSurprise, UpcomingEarnings
from usbot.earnings.score import earnings_blackout, pead_scores
from usbot.insider.ingest import _parse_rows
from usbot.insider.model import InsiderTrade, classify
from usbot.insider.score import insider_scores
from usbot.portfolio import ActivePortfolio, PortfolioState


# ---- insider classification + score -----------------------------------------
def test_classify_open_market_only():
    assert classify("P", is_planned=False) == "opportunistic_buy"
    assert classify("P", is_planned=True) == "other"      # 10b5-1 -> routine
    assert classify("S", is_planned=False) == "sell"
    assert classify("A", is_planned=False) == "other"     # grant, not a signal


def _buy(sym, insider, shares=1000, price=50, title="Director"):
    return InsiderTrade(sym, insider, title, "P", shares, price, dt.date(2026, 6, 1))


def test_cluster_buy_scores_above_single_buy():
    single = insider_scores([_buy("AAA", "Jane")], ["AAA", "BBB"])
    cluster = insider_scores(
        [_buy("AAA", "Jane"), _buy("AAA", "John"), _buy("AAA", "Ann")], ["AAA"])
    assert single["AAA"] > 50.0
    assert cluster["AAA"] > single["AAA"]                 # more distinct buyers = stronger
    assert single["BBB"] == 50.0                          # no activity -> neutral


def test_senior_buyer_bonus_and_sell_penalty():
    junior_buy = insider_scores([_buy("AAA", "J", title="Manager")], ["AAA"])["AAA"]
    ceo_buy = insider_scores([_buy("AAA", "J", title="Chief Executive Officer")], ["AAA"])["AAA"]
    assert ceo_buy > junior_buy               # senior officer conviction bonus
    sell = InsiderTrade("BBB", "K", "Director", "S", 1000, 50, dt.date(2026, 6, 1))
    assert insider_scores([sell], ["BBB"])["BBB"] < 50.0


def test_insider_parse_rows_filters_codes_and_window():
    rows = [
        {"transactionCode": "P", "share": 1000, "transactionPrice": 50,
         "transactionDate": "2026-06-01", "name": "Jane"},
        {"transactionCode": "A", "share": 500, "transactionPrice": 40,          # grant -> skip
         "transactionDate": "2026-06-01", "name": "Grant"},
        {"transactionCode": "P", "share": 100, "transactionPrice": 10,          # too old -> skip
         "transactionDate": "2020-01-01", "name": "Old"},
        {"transactionCode": "S", "share": 0, "transactionPrice": 50,            # zero shares -> skip
         "transactionDate": "2026-06-01", "name": "Zero"},
    ]
    out = _parse_rows("AAA", rows, since=dt.date(2026, 3, 1))
    assert len(out) == 1 and out[0].code == "P" and out[0].insider == "Jane"


# ---- earnings PEAD + blackout ------------------------------------------------
def _sur(sym, days_ago, pct, today):
    return EarningsSurprise(sym, today - dt.timedelta(days=days_ago), 1.0, 1.0, pct)


def test_pead_positive_surprise_decays_to_neutral():
    today = dt.date(2026, 7, 1)
    fresh = pead_scores([_sur("AAA", 2, 0.20, today)], ["AAA"], today=today)["AAA"]
    old = pead_scores([_sur("AAA", 60, 0.20, today)], ["AAA"], today=today)["AAA"]
    stale = pead_scores([_sur("AAA", 200, 0.20, today)], ["AAA"], today=today)["AAA"]
    assert fresh > 60.0                       # strong recent beat -> boosted
    assert 50.0 < old < fresh                 # decayed but still positive
    assert stale == 50.0                      # outside the drift window -> neutral


def test_pead_negative_surprise_below_neutral_and_uses_latest():
    today = dt.date(2026, 7, 1)
    miss = pead_scores([_sur("AAA", 3, -0.20, today)], ["AAA"], today=today)["AAA"]
    assert miss < 50.0
    # two reports: the most recent (a beat) wins over an older miss
    latest = pead_scores([_sur("AAA", 80, -0.30, today), _sur("AAA", 3, 0.20, today)],
                         ["AAA"], today=today)["AAA"]
    assert latest > 50.0


def test_earnings_blackout_window():
    today = dt.date(2026, 7, 1)
    up = [UpcomingEarnings("AAA", dt.date(2026, 7, 3)),      # in 2 days -> blackout
          UpcomingEarnings("BBB", dt.date(2026, 7, 20)),     # far out -> ok
          UpcomingEarnings("CCC", dt.date(2026, 6, 25))]     # past -> ok
    bl = earnings_blackout(up, today=today, days_ahead=5)
    assert bl == {"AAA"}


def test_earnings_parse_surprises_prefers_surprise_percent():
    rows = [{"period": "2026-06-30", "actual": 1.2, "estimate": 1.0, "surprisePercent": 20.0},
            {"period": "bad-date", "actual": 1, "estimate": 1},                # skip
            {"period": "2026-03-31", "actual": 1.0, "estimate": 0.0}]          # est 0 -> skip
    out = _parse_surprises("AAA", rows, since=dt.date(2026, 1, 1))
    assert len(out) == 1 and abs(out[0].surprise_pct - 0.20) < 1e-9


# ---- Active-sleeve earnings blackout integration -----------------------------
def test_active_skips_entry_for_blackout_names():
    act = ActivePortfolio(risk_cfg={}, txn_cost=1.5, min_cash_buffer_pct=0.05,
                          initial_deploy_pct=0.95)
    st = PortfolioState(name="Active Entry", ptype="active", cash=1600.0,
                        starting_capital=1600.0)
    scores = pd.Series({"AAA": 85.0, "BBB": 84.0})
    prices = {"AAA": 100.0, "BBB": 100.0}
    ind = {"AAA": {"above_sma50": 1.0, "realized_vol": 0.3},
           "BBB": {"above_sma50": 1.0, "realized_vol": 0.3}}
    dec = act.decide(st, scores, prices, ind, "risk_on", blackout={"AAA"})
    bought = {t.symbol for t in dec.trades if t.side == "buy"}
    assert "AAA" not in bought          # reporting soon -> not entered
    assert "BBB" in bought              # clear name still entered
