import json

from usbot.portfolio import ActivePortfolioStore, performance_from_history
from usbot.portfolio.base import Holding


def test_fresh_load_when_no_file(tmp_path):
    store = ActivePortfolioStore(tmp_path / "active.json")
    loaded = store.load("Active Entry", 1600.0, 1.5)
    assert loaded.existed is False
    assert loaded.state.cash == 1600.0
    assert loaded.state.holdings == {}


def test_save_then_load_round_trip(tmp_path):
    store = ActivePortfolioStore(tmp_path / "active.json")
    loaded = store.load("Active Entry", 1600.0, 1.5)
    state = loaded.state
    # simulate a buy: spend 300 + 1.5 fee on 3 shares @ 100
    state.holdings["MU"] = Holding("MU", shares=3.0, avg_cost=100.0)
    state.cash = 1600.0 - 300.0 - 1.5
    prices = {"MU": 100.0}
    history, tv = store.save(state, prices, "2026-06-29", loaded.history, decided_today=True)
    assert abs(tv - (state.cash + 300.0)) < 1e-6

    # reload — holdings and cash must persist
    re = store.load("Active Entry", 1600.0, 1.5)
    assert re.existed is True
    assert re.last_decision_date == "2026-06-29"
    assert "MU" in re.state.holdings
    assert re.state.holdings["MU"].shares == 3.0
    assert abs(re.state.cash - (1600.0 - 301.5)) < 1e-6


def test_history_dedupes_same_day(tmp_path):
    store = ActivePortfolioStore(tmp_path / "active.json")
    state = store.load("Active Entry", 1600.0, 1.5).state
    prices = {}
    h, _ = store.save(state, prices, "2026-06-29", [], decided_today=True)
    h, _ = store.save(state, prices, "2026-06-29", h, decided_today=False)  # same day again
    assert len([x for x in h if x["date"] == "2026-06-29"]) == 1


def test_performance_metrics_from_history():
    history = [
        {"date": "2026-06-25", "total_value": 1600.0},
        {"date": "2026-06-26", "total_value": 1650.0},
        {"date": "2026-06-29", "total_value": 1620.0},  # today
    ]
    perf = performance_from_history(history, 1620.0, 1600.0)
    assert abs(perf["daily_pl"] - (1620.0 - 1650.0)) < 1e-9   # vs previous point
    assert abs(perf["total_pl"] - (1620.0 - 1600.0)) < 1e-9   # vs starting capital
    # peak was 1650 -> drawdown = 1620/1650 - 1
    assert abs(perf["drawdown"] - (1620.0 / 1650.0 - 1.0)) < 1e-9


def test_saved_json_is_valid_and_diff_friendly(tmp_path):
    store = ActivePortfolioStore(tmp_path / "active.json")
    state = store.load("Active Entry", 1600.0, 1.5).state
    state.holdings["AAA"] = Holding("AAA", 1.0, 50.0)
    store.save(state, {"AAA": 55.0}, "2026-06-29", [], decided_today=True)
    data = json.loads((tmp_path / "active.json").read_text())
    assert data["name"] == "Active Entry"
    assert data["holdings"][0]["symbol"] == "AAA"
    assert data["last_decision_date"] == "2026-06-29"
