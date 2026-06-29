"""Model sleeves hold real positions at real prices and track real P/L."""
from usbot.portfolio import PortfolioStore, compute_model_targets, rebalance_to_targets
from usbot.portfolio.base import Holding, PortfolioState
import pandas as pd


def test_rebalance_uses_real_prices_and_fractional_shares():
    state = PortfolioState("Growth", "growth", cash=1000.0, starting_capital=1000.0)
    targets = {"LLY": 0.5, "MU": 0.5}
    prices = {"LLY": 1223.34, "MU": 100.0}
    rebalance_to_targets(state, targets, prices, txn_cost=0.0)
    # fractional shares at real prices
    assert abs(state.holdings["LLY"].shares - 500.0 / 1223.34) < 1e-6
    assert abs(state.holdings["MU"].shares - 500.0 / 100.0) < 1e-6
    # fill price recorded as the live price
    assert state.holdings["LLY"].avg_cost == 1223.34
    # total value preserved (no cost)
    assert abs(state.total_value(prices) - 1000.0) < 1e-6


def test_hold_then_price_move_changes_value():
    state = PortfolioState("Growth", "growth", cash=1000.0, starting_capital=1000.0)
    rebalance_to_targets(state, {"MU": 1.0}, {"MU": 100.0})
    # 10 shares @ 100 = 1000
    assert abs(state.holdings["MU"].shares - 10.0) < 1e-9
    # price rises 10% -> value 1100
    assert abs(state.total_value({"MU": 110.0}) - 1100.0) < 1e-6


def test_compute_targets_respects_caps():
    scores = pd.Series({f"s{i}": 90 - i for i in range(20)})
    sectors = {f"s{i}": ("Tech" if i % 2 else "Health") for i in range(20)}
    w = compute_model_targets("growth", scores, sectors, {},
                              {"max_names": 10, "max_position": 0.12, "max_sector": 0.30})
    assert len(w) == 10
    assert max(w.values()) <= 0.12 + 1e-9


def test_model_sleeve_persists_and_revalues(tmp_path):
    path = tmp_path / "portfolios.json"
    # Day 1: allocate
    store = PortfolioStore(path)
    st = store.load("Growth", 1000.0).state
    rebalance_to_targets(st, {"MU": 0.6, "LLY": 0.4}, {"MU": 100.0, "LLY": 1000.0})
    store.stage(st, {"MU": 100.0, "LLY": 1000.0}, "2026-06-29", store.load("Growth", 1000.0).history,
                ptype="growth", last_rebalance_date="2026-06-29")
    store.commit()

    # Day 2: reload, prices up, just revalue (no rebalance)
    store2 = PortfolioStore(path)
    loaded = store2.load("Growth", 1000.0)
    assert loaded.existed and "MU" in loaded.state.holdings
    tv = loaded.state.total_value({"MU": 110.0, "LLY": 1050.0})
    # MU 6 sh -> 660, LLY 0.4 sh -> 420 ; original 600/400
    assert tv > 1000.0  # gained
