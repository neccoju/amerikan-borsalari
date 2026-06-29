import numpy as np
import pandas as pd

from usbot.learning import compute_factor_ic, normalize_weights, realized_returns, update_weights


def test_normalize_weights_sums_to_one_and_clamps():
    w = normalize_weights({"a": 0.9, "b": 0.05, "c": 0.05}, min_w=0.1, max_w=0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.5 + 1e-9
    assert min(w.values()) >= 0.1 - 1e-9


def test_normalize_empty_and_zero():
    assert normalize_weights({}) == {}
    eq = normalize_weights({"a": 0.0, "b": 0.0})
    assert abs(eq["a"] - 0.5) < 1e-9


def test_update_weights_rewards_positive_ic():
    w0 = {"technical": 0.34, "fundamental": 0.33, "macro": 0.33}
    w1 = update_weights(w0, {"technical": 0.8, "fundamental": -0.2}, lr=0.5,
                        min_w=0.02, max_w=0.9)
    assert w1["technical"] > w0["technical"]      # positive IC -> more weight
    assert w1["fundamental"] < w0["fundamental"]
    assert abs(sum(w1.values()) - 1.0) < 1e-9


def test_update_weights_no_reward_is_stable():
    w0 = {"a": 0.3, "b": 0.3, "c": 0.4}
    w1 = update_weights(w0, {}, lr=0.5)
    for k in w0:
        assert abs(w1[k] - w0[k]) < 1e-6


def test_update_weights_respects_bounds_over_iterations():
    w = {"a": 0.34, "b": 0.33, "c": 0.33}
    for _ in range(40):
        w = update_weights(w, {"a": 1.0, "b": -1.0, "c": 0.0}, lr=0.5, min_w=0.05, max_w=0.6)
    assert max(w.values()) <= 0.6 + 1e-9
    assert min(w.values()) >= 0.05 - 1e-9
    assert w["a"] > w["c"] > w["b"]


def test_realized_returns_from_history():
    idx = pd.date_range("2026-05-01", periods=40, freq="B")
    def mk(p0, p1):
        vals = np.linspace(p0, p1, 40)
        return pd.DataFrame({"adj_close": vals, "close": vals}, index=idx)
    hist = {"UP": mk(100, 120), "DOWN": mk(200, 180)}
    r = realized_returns(hist, since_date=idx[0].isoformat(), symbols=["UP", "DOWN", "MISSING"])
    assert r["UP"] > 0 > r["DOWN"]
    assert "MISSING" not in r


def test_factor_ic_detects_predictive_factor():
    # 'good' factor scores perfectly rank realized returns; 'bad' is reversed.
    syms = [f"S{i}" for i in range(10)]
    rets = pd.Series({s: i / 10 for i, s in enumerate(syms)})
    good = {s: i for i, s in enumerate(syms)}
    bad = {s: -i for i, s in enumerate(syms)}
    ic = compute_factor_ic({"good": good, "bad": bad}, rets)
    assert ic["good"] > 0.9
    assert ic["bad"] < -0.9


def test_factor_ic_insufficient_data_is_zero():
    ic = compute_factor_ic({"f": {"A": 1.0}}, pd.Series({"A": 0.1}))
    assert ic["f"] == 0.0
