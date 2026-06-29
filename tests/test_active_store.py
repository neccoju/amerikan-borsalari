import json

from usbot.portfolio import PortfolioStore, performance_from_history
from usbot.portfolio.base import Holding


def test_fresh_load_when_no_entry(tmp_path):
    store = PortfolioStore(tmp_path / "portfolios.json")
    loaded = store.load("Active Entry", 1600.0, 1.5)
    assert loaded.existed is False
    assert loaded.state.cash == 1600.0
    assert loaded.state.holdings == {}


def test_stage_commit_then_load_round_trip(tmp_path):
    path = tmp_path / "portfolios.json"
    store = PortfolioStore(path)
    loaded = store.load("Active Entry", 1600.0, 1.5)
    state = loaded.state
    state.holdings["MU"] = Holding("MU", shares=3.0, avg_cost=100.0)
    state.cash = 1600.0 - 300.0 - 1.5
    prices = {"MU": 100.0}
    history, tv = store.stage(state, prices, "2026-06-29", loaded.history,
                              ptype="active", last_decision_date="2026-06-29")
    assert abs(tv - (state.cash + 300.0)) < 1e-6
    store.commit()

    # reload from a brand-new store instance
    re = PortfolioStore(path).load("Active Entry", 1600.0, 1.5)
    assert re.existed is True
    assert re.last_decision_date == "2026-06-29"
    assert "MU" in re.state.holdings
    assert re.state.holdings["MU"].shares == 3.0
    assert abs(re.state.cash - (1600.0 - 301.5)) < 1e-6


def test_multiple_sleeves_in_one_file(tmp_path):
    path = tmp_path / "portfolios.json"
    store = PortfolioStore(path)
    for name, cap in [("Growth", 1000.0), ("Active Entry", 1600.0)]:
        st = store.load(name, cap).state
        st.holdings["AAA"] = Holding("AAA", 1.0, 50.0)
        store.stage(st, {"AAA": 55.0}, "2026-06-29", [], ptype="x")
    store.commit()
    data = json.loads(path.read_text())
    assert set(data["portfolios"].keys()) == {"Growth", "Active Entry"}


def test_history_dedupes_same_day(tmp_path):
    store = PortfolioStore(tmp_path / "p.json")
    state = store.load("Active Entry", 1600.0, 1.5).state
    h, _ = store.stage(state, {}, "2026-06-29", [], ptype="active")
    h, _ = store.stage(state, {}, "2026-06-29", h, ptype="active")
    assert len([x for x in h if x["date"] == "2026-06-29"]) == 1


def test_performance_metrics_from_history():
    history = [
        {"date": "2026-06-25", "total_value": 1600.0},
        {"date": "2026-06-26", "total_value": 1650.0},
        {"date": "2026-06-29", "total_value": 1620.0},
    ]
    perf = performance_from_history(history, 1620.0, 1600.0)
    assert abs(perf["daily_pl"] - (1620.0 - 1650.0)) < 1e-9
    assert abs(perf["total_pl"] - (1620.0 - 1600.0)) < 1e-9
    assert abs(perf["drawdown"] - (1620.0 / 1650.0 - 1.0)) < 1e-9


def test_legacy_active_file_migrates(tmp_path):
    # Old per-active file present, new unified file absent -> should import it.
    legacy = tmp_path / "active_portfolio.json"
    legacy.write_text(json.dumps({
        "name": "Active Entry", "starting_capital": 1600.0, "cash": 1200.0,
        "holdings": [{"symbol": "LLY", "shares": 0.2, "avg_cost": 1200.0}],
        "history": [{"date": "2026-06-29", "total_value": 1597.0}],
        "last_decision_date": "2026-06-29",
    }))
    store = PortfolioStore(tmp_path / "portfolios.json")
    loaded = store.load("Active Entry", 1600.0, 1.5)
    assert loaded.existed is True
    assert "LLY" in loaded.state.holdings
    assert abs(loaded.state.cash - 1200.0) < 1e-9
