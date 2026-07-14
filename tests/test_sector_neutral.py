"""Tests for sector-neutralization of composite scores."""
from __future__ import annotations

import pandas as pd

from usbot.scoring.sector_neutral import sector_neutralize


def _scores():
    # Tech sector runs hot (80/78), Utilities cold (40/38); one lone Energy name.
    s = pd.Series({"AAA": 80.0, "BBB": 78.0, "CCC": 40.0, "DDD": 38.0, "EEE": 60.0})
    sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Util", "DDD": "Util", "EEE": "Energy"}
    return s, sectors


def test_strength_zero_is_identity():
    s, sectors = _scores()
    out = sector_neutralize(s, sectors, strength=0.0)
    assert out.equals(s)


def test_hot_sector_pulled_down_cold_lifted():
    s, sectors = _scores()
    out = sector_neutralize(s, sectors, strength=1.0)
    # Tech names lose their sector tailwind, Utilities lose the headwind
    assert out["AAA"] < s["AAA"] and out["BBB"] < s["BBB"]
    assert out["CCC"] > s["CCC"] and out["DDD"] > s["DDD"]


def test_within_sector_ordering_preserved():
    s, sectors = _scores()
    out = sector_neutralize(s, sectors, strength=0.6)
    assert out["AAA"] > out["BBB"]        # Tech order kept
    assert out["CCC"] > out["DDD"]        # Util order kept


def test_cross_sectional_mean_preserved():
    s, sectors = _scores()
    out = sector_neutralize(s, sectors, strength=1.0)
    assert abs(out.mean() - s.mean()) < 1e-9


def test_unknown_sector_unchanged():
    s = pd.Series({"AAA": 80.0, "BBB": 78.0, "ZZZ": 90.0})
    sectors = {"AAA": "Tech", "BBB": "Tech"}      # ZZZ has no sector
    out = sector_neutralize(s, sectors, strength=1.0)
    assert out["ZZZ"] == 90.0


def test_reduces_top_sector_concentration():
    # Tech dominates the top; neutralization should let a strong non-Tech name in.
    s = pd.Series({"T1": 90, "T2": 88, "T3": 86, "T4": 84, "U1": 82.0})
    sectors = {"T1": "Tech", "T2": "Tech", "T3": "Tech", "T4": "Tech", "U1": "Util"}
    raw_top3 = set(s.sort_values(ascending=False).head(3).index)
    out = sector_neutralize(s, sectors, strength=1.0)
    neu_top3 = set(out.sort_values(ascending=False).head(3).index)
    assert "U1" in neu_top3 and "U1" not in raw_top3   # the lone Util name breaks in


def test_empty_or_no_sectors_safe():
    s = pd.Series({"AAA": 50.0})
    assert sector_neutralize(s, {}, 0.5).equals(s)
    assert sector_neutralize(pd.Series(dtype=float), {"AAA": "Tech"}, 0.5).empty
