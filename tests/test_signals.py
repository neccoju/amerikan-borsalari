"""Tests for Phase 3 alternative-data signals (congress, 13F, LLM nudges)."""
import datetime as dt

from usbot.congress.model import CongressTrade, normalize_txn_type
from usbot.congress.score import congress_scores
from usbot.congress.ingest import _parse_quiver_rows, _parse_capitoltrades_rows
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


def test_parse_quiver_rows_filters_and_maps():
    rows = [
        {"Ticker": "NVDA", "Transaction": "Purchase", "Representative": "Rep A",
         "Range": "$1,000,001 - $5,000,000", "TransactionDate": "2026-06-10",
         "ReportDate": "2026-06-20", "House": "Representatives", "Party": "D"},
        {"Ticker": "AAPL", "Transaction": "Sale", "Senator": "Sen B",
         "Range": "$15,001 - $50,000", "TransactionDate": "2026-06-12", "House": "Senate"},
        {"Ticker": "ZZZZ", "Transaction": "Purchase", "TransactionDate": "2026-06-12"},  # not in uni
        {"Ticker": "NVDA", "Transaction": "Exchange", "TransactionDate": "2026-06-12"},  # not a trade
        {"Ticker": "NVDA", "Transaction": "Purchase", "TransactionDate": "2020-01-01"},  # too old
    ]
    out = _parse_quiver_rows(rows, {"NVDA", "AAPL"}, dt.date(2026, 4, 1))
    syms = {(t.symbol, t.txn_type, t.chamber) for t in out}
    assert ("NVDA", "buy", "house") in syms
    assert ("AAPL", "sell", "senate") in syms
    assert len(out) == 2  # ZZZZ filtered, Exchange skipped, old skipped


def test_parse_capitoltrades_rows():
    rows = [
        {"txType": "buy", "txDate": "2026-06-10", "pubDate": "2026-06-20", "value": 250000,
         "size": "$100,001 - $250,000",
         "issuer": {"issuerTicker": "NVDA:US"},
         "politician": {"firstName": "Nancy", "lastName": "P", "party": "D", "chamber": "house"}},
        {"txType": "sell", "txDate": "2026-06-11",
         "asset": {"assetTicker": "AAPL:US"},
         "politician": {"firstName": "Tom", "lastName": "T", "chamber": "senate"}},
        {"txType": "exchange", "txDate": "2026-06-11", "issuer": {"issuerTicker": "NVDA:US"}},
        {"txType": "buy", "txDate": "2019-01-01", "issuer": {"issuerTicker": "NVDA:US"}},  # old
    ]
    out = _parse_capitoltrades_rows(rows, {"NVDA", "AAPL"}, dt.date(2026, 4, 1))
    assert len(out) == 2
    nvda = [t for t in out if t.symbol == "NVDA"][0]
    assert nvda.txn_type == "buy" and nvda.chamber == "house"
    assert nvda.amount_value == 250000          # numeric value used as amount_mid
    assert nvda.amount_mid == 250000
    aapl = [t for t in out if t.symbol == "AAPL"][0]
    assert aapl.txn_type == "sell" and aapl.chamber == "senate"


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
