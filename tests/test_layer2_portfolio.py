"""Tests for Layer-2 portfolio engineering: rebalance bands, inverse-vol
weighting, drawdown circuit breaker, and the shared Finnhub rate limiter."""
from __future__ import annotations

import time

import pandas as pd

from usbot.portfolio.base import Holding, PortfolioState
from usbot.portfolio.model_portfolios import rebalance_to_targets
from usbot.portfolio.risk import circuit_breaker_trim, target_weights_from_scores
from usbot.utils.ratelimit import RateLimiter, get_limiter


# ---- rebalance no-trade band -------------------------------------------------
def test_rebalance_band_suppresses_small_drifts():
    st = PortfolioState(name="Growth", ptype="growth", cash=0.0, starting_capital=1000.0)
    # total = 1000: A slightly over target, B on target, plus a new name C
    st.holdings["A"] = Holding("A", 5.10, 100.0)   # $510 (target 500 -> 1% drift)
    st.holdings["B"] = Holding("B", 5.00, 100.0)   # $500 exactly on target
    prices = {"A": 100.0, "B": 100.0, "C": 100.0}
    targets = {"A": 0.5, "B": 0.5}                  # C not targeted

    # band 1.5% of $1010 ≈ $15 -> A's $10 drift is inside the band -> no trade
    trades = rebalance_to_targets(st, targets, prices, band=0.015)
    assert trades == []
    assert abs(st.holdings["A"].shares - 5.10) < 1e-9   # untouched

    # a large drift still trades
    st.holdings["A"].shares = 8.0                        # $800 vs $500 target
    trades2 = rebalance_to_targets(st, {"A": 0.5, "B": 0.5}, prices, band=0.015)
    assert any(t["symbol"] == "A" and t["side"] == "sell" for t in trades2)


# ---- inverse-vol weighting ---------------------------------------------------
def test_inverse_vol_tilts_toward_low_vol_names():
    scores = pd.Series({"HI": 80.0, "LO": 80.0})    # equal score
    sectors = {"HI": "Tech", "LO": "Tech"}
    vols = {"HI": 0.60, "LO": 0.15}                 # HI 4x more volatile
    # pure score-weighting: equal
    even = target_weights_from_scores(scores, 2, 0.9, sectors, 0.9, inv_vol_weight=0.0)
    assert abs(even["HI"] - even["LO"]) < 1e-9
    # inverse-vol tilt: the calmer name gets more weight
    tilt = target_weights_from_scores(scores, 2, 0.9, sectors, 0.9, vols=vols,
                                      inv_vol_weight=0.6)
    assert tilt["LO"] > tilt["HI"]


def test_inverse_vol_respects_position_cap():
    scores = pd.Series({s: 80.0 - i for i, s in enumerate("ABCDE")})
    sectors = {s: "Tech" for s in "ABCDE"}
    vols = {"A": 0.05, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5}   # A ultra-low vol
    w = target_weights_from_scores(scores, 5, 0.25, sectors, 0.9, vols=vols,
                                   inv_vol_weight=0.9)
    assert max(w.values()) <= 0.25 + 1e-9           # cap still holds despite the tilt


# ---- circuit breaker ---------------------------------------------------------
def _sleeve(equity_frac=1.0, total=1000.0):
    st = PortfolioState(name="Growth", ptype="growth", cash=total * (1 - equity_frac),
                        starting_capital=1000.0)
    eq = total * equity_frac
    # two equal holdings making up the equity
    st.holdings["A"] = Holding("A", (eq / 2) / 100.0, 120.0)
    st.holdings["B"] = Holding("B", (eq / 2) / 100.0, 120.0)
    return st


def test_circuit_breaker_trims_on_deep_drawdown():
    st = _sleeve(equity_frac=1.0, total=800.0)      # fully invested, now $800
    prices = {"A": 100.0, "B": 100.0}
    history = [{"date": "2026-06-01", "total_value": 1000.0}]   # peak was $1000
    trades, dd, fired = circuit_breaker_trim(st, prices, history, threshold=0.18,
                                             breaker_cash=0.5)
    assert fired and dd < -0.18
    # trimmed to ~50% cash
    assert abs(st.cash - 400.0) < 1.0
    assert abs(st.equity_value(prices) - 400.0) < 1.0
    assert all(t["side"] == "sell" for t in trades)


def test_circuit_breaker_silent_when_shallow_or_already_derisked():
    prices = {"A": 100.0, "B": 100.0}
    # shallow drawdown -> no action
    st = _sleeve(1.0, 950.0)
    _, dd, fired = circuit_breaker_trim(st, prices, [{"date": "d", "total_value": 1000.0}],
                                        threshold=0.18, breaker_cash=0.5)
    assert not fired and dd > -0.18
    # deep drawdown but already at target cash -> no further trimming
    st2 = _sleeve(equity_frac=0.5, total=700.0)
    _, _, fired2 = circuit_breaker_trim(st2, prices, [{"date": "d", "total_value": 1000.0}],
                                        threshold=0.18, breaker_cash=0.5)
    assert not fired2


def test_circuit_breaker_preserves_cost_basis():
    st = _sleeve(1.0, 800.0)
    st.holdings["A"].avg_cost = 150.0
    prices = {"A": 100.0, "B": 100.0}
    circuit_breaker_trim(st, prices, [{"date": "d", "total_value": 1000.0}],
                         threshold=0.18, breaker_cash=0.5)
    assert st.holdings["A"].avg_cost == 150.0        # partial sell doesn't reset basis


# ---- shared rate limiter -----------------------------------------------------
def test_rate_limiter_enforces_min_interval():
    lim = RateLimiter(rate_per_min=600)              # 10/sec -> 0.1s min interval
    t0 = time.monotonic()
    for _ in range(4):
        lim.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.28                            # 3 gaps * 0.1s (first is free)


def test_get_limiter_is_shared_by_name():
    a = get_limiter("finnhub-test", 60)
    b = get_limiter("finnhub-test", 999)             # same name -> same instance
    assert a is b
