"""Tests for Phase 3 alternative-data signals (congress, 13F, LLM nudges)."""
import datetime as dt

from usbot.congress.model import CongressTrade, normalize_txn_type
from usbot.congress.score import congress_scores
from usbot.institutional.model import HoldingChange
from usbot.institutional.score import institutional_scores
from usbot.llm.review import parse_adjustments


# ---- congress --------------------------------------------------------------
def _trade(sym, ttype, who, amt="$15,001 - $50,000"):
    return CongressTrade(symbol=sym, chamber="house", politician=who, txn_type=ttype,
                         traded_date=dt.date(2026, 6, 1), filed_date=None, amount_range=amt)


def test_normalize_txn_type():
    assert normalize_txn_type("Purchase") == "buy"
    assert normalize_txn_type("Sale (Full)") == "sell"
    assert normalize_txn_type("exchange") is None


def test_congress_buy_beats_sell_and_neutral_default():
    trades = [
        _trade("AAA", "buy", "Rep X"), _trade("AAA", "buy", "Rep Y"),
        _trade("BBB", "sell", "Rep X"), _trade("BBB", "sell", "Rep Z"),
    ]
    s = congress_scores(trades, ["AAA", "BBB", "CCC"])
    assert s["AAA"] > 50 > s["BBB"]
    assert s["CCC"] == 50.0           # no trades -> neutral
    assert 0 <= s["BBB"] <= 100


def test_congress_consensus_buyers_strengthen_score():
    one = congress_scores([_trade("AAA", "buy", "Rep X")], ["AAA"])
    many = congress_scores(
        [_trade("AAA", "buy", f"Rep {i}") for i in range(5)], ["AAA"])
    assert many["AAA"] > one["AAA"]


def test_amount_midpoint_and_signed():
    t = _trade("AAA", "sell", "Rep X", amt="$1,000,001 - $5,000,000")
    assert t.amount_mid == 3000000
    assert t.signed_amount < 0


# ---- institutional (13F) ---------------------------------------------------
def test_institutional_new_and_increase_raise_score():
    changes = [
        HoldingChange("AAA", "Fund1", "new"),
        HoldingChange("AAA", "Fund2", "increased"),
        HoldingChange("BBB", "Fund1", "exited"),
    ]
    s = institutional_scores(changes, ["AAA", "BBB", "CCC"])
    assert s["AAA"] > 50 > s["BBB"]
    assert s["CCC"] == 50.0
    assert 0 <= s["AAA"] <= 100


def test_institutional_consensus_clamped():
    changes = [HoldingChange("AAA", f"F{i}", "new") for i in range(50)]
    s = institutional_scores(changes, ["AAA"])
    assert s["AAA"] <= 100.0   # clamped despite huge consensus


# ---- LLM bounded nudges ----------------------------------------------------
def test_parse_adjustments_clamps_and_filters():
    text = ("...commentary...\n"
            "ADJUSTMENTS: AAPL:+3, MSFT:-2, NVDA:+99, GARBAGE")
    adj = parse_adjustments(text, max_points=5.0)
    assert adj["AAPL"] == 3.0
    assert adj["MSFT"] == -2.0
    assert adj["NVDA"] == 5.0          # clamped to max
    assert "GARBAGE" not in adj or adj.get("GARBAGE") is None


def test_parse_adjustments_none_and_missing():
    assert parse_adjustments("text\nADJUSTMENTS: none", 5.0) == {}
    assert parse_adjustments("no adjustments line here", 5.0) == {}
    assert parse_adjustments("", 5.0) == {}
