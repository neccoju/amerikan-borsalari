import pandas as pd

from usbot.portfolio.risk import apply_caps, apply_sector_cap, target_weights_from_scores


def test_apply_caps_water_fills_without_violating_cap():
    # Water-filling caps each name at 0.4 while preserving the total (feasible: 3*0.4>=1).
    w = {"a": 0.6, "b": 0.3, "c": 0.1}
    capped = apply_caps(w, max_position=0.4)
    assert max(capped.values()) <= 0.4 + 1e-9
    assert abs(sum(capped.values()) - 1.0) < 1e-9


def test_sector_cap_redistributes_to_other_sector():
    # Tech (0.8) capped to 0.5; freed weight flows to Health which has capacity.
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    sectors = {"a": "Tech", "b": "Tech", "c": "Health"}
    out = apply_sector_cap(w, sectors, max_sector=0.5)
    assert out["a"] + out["b"] <= 0.5 + 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-6  # capacity existed -> fully reinvested


def test_target_weights_single_sector_leaves_cash():
    # All names in one sector: sector cap (0.30) is a hard limit; rest is cash.
    scores = pd.Series({f"s{i}": 100 - i for i in range(30)})
    sectors = {f"s{i}": "Tech" for i in range(30)}
    w = target_weights_from_scores(scores, n=10, max_position=0.15,
                                   sectors=sectors, max_sector=0.30)
    assert len(w) == 10
    assert max(w.values()) <= 0.15 + 1e-9       # position cap respected
    assert sum(w.values()) <= 0.30 + 1e-6       # sector cap respected (leftover=cash)


def test_empty_scores_yield_no_weights():
    assert target_weights_from_scores(pd.Series(dtype=float), 10, 0.1, {}, 0.3) == {}
