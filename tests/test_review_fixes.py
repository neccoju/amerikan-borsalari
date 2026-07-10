"""Tests for the review fixes: joint cap enforcement, Wikipedia sectors,
realized beta, rebalance idempotency, corporate actions, macro-as-exposure,
and IC smoothing — plus a mini end-to-end sleeve wiring test."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from usbot.indicators.technical import realized_betas
from usbot.learning import smooth_ic
from usbot.portfolio import PortfolioState, PortfolioStore, apply_corporate_actions
from usbot.portfolio.base import Holding
from usbot.portfolio.risk import apply_sector_cap, target_weights_from_scores
from usbot.reports.builder import ReportContext
from usbot.scoring.composite import score_universe


# ---- 1. joint sector + position caps ---------------------------------------
def test_sector_redistribution_respects_position_cap():
    """Regression: freed sector weight must not push a name past max_position.

    Reproduces the real Defensive book failure (SNDK/LLY/V at 25% each with a
    10% position cap) caused by most names having Unknown sector."""
    scores = pd.Series({f"S{i}": 80 - i * 0.5 for i in range(15)})
    sectors = {"S0": "Technology", "S1": "Health Care", "S2": "Financials"}
    w = target_weights_from_scores(scores, n=15, max_position=0.10,
                                   sectors=sectors, max_sector=0.25)
    assert max(w.values()) <= 0.10 + 1e-9
    by_sec: dict[str, float] = {}
    for s, v in w.items():
        by_sec[sectors.get(s, "Unknown")] = by_sec.get(sectors.get(s, "Unknown"), 0) + v
    assert all(v <= 0.25 + 1e-9 for v in by_sec.values())


def test_sector_cap_without_position_cap_still_works():
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy"}
    out = apply_sector_cap(w, sectors, max_sector=0.4)
    tech = out["A"] + out["B"]
    assert tech <= 0.4 + 1e-9


# ---- 2. Wikipedia sectors + realized beta -----------------------------------
def test_sp500_constituents_parse_sectors(monkeypatch):
    tbl = pd.DataFrame({
        "Symbol": [f"T{i}" for i in range(450)],
        "GICS Sector": (["Information Technology", "Financials", "Energy"] * 150),
    })
    import usbot.universe.sp500 as sp500

    monkeypatch.setattr(sp500, "_CACHE", None)
    monkeypatch.setattr("usbot.universe.wiki.read_wikipedia_tables", lambda url: [tbl])
    syms, secs = sp500.get_sp500_constituents(dynamic=True)
    assert len(syms) == 450
    assert secs["T0"] == "Information Technology"
    assert secs["T1"] == "Financials"


def test_realized_beta_of_levered_clone_is_two():
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    rng = np.random.default_rng(7)
    r = rng.normal(0.0004, 0.01, 300)
    mk = lambda c: pd.DataFrame({"close": c}, index=idx)  # noqa: E731
    betas = realized_betas({
        "SPY": mk(400 * np.cumprod(1 + r)),
        "LEV2": mk(100 * np.cumprod(1 + 2 * r)),
    })
    assert abs(betas["LEV2"] - 2.0) < 0.05


# ---- 5. corporate actions ----------------------------------------------------
def _px(closes, dividends=None, splits=None):
    idx = pd.date_range("2026-06-24", periods=len(closes), freq="B")
    df = pd.DataFrame({"close": closes}, index=idx)
    df["dividends"] = dividends if dividends is not None else 0.0
    df["splits"] = splits if splits is not None else 0.0
    return df


def test_dividend_credited_once_and_not_retroactively():
    st = PortfolioState(name="Growth", ptype="growth", cash=0.0, starting_capital=1000.0)
    st.holdings["JNJ"] = Holding("JNJ", 10.0, 150.0)
    hist = {"JNJ": _px([150] * 5, dividends=[0, 0, 0, 1.25, 0])}  # ex-date 2026-06-29

    # first run since 2026-06-26 -> credit 10 * 1.25
    rows, notes = apply_corporate_actions(st, hist, "2026-06-26", "Growth", "2026-06-30")
    assert abs(st.cash - 12.5) < 1e-9
    assert rows and rows[0]["type"] == "dividend" and "1.25" in rows[0]["reason"]
    assert notes and "12.50" in notes[0]

    # same-day rerun (since = today) -> nothing double-credited
    rows2, _ = apply_corporate_actions(st, hist, "2026-06-30", "Growth", "2026-06-30")
    assert not rows2 and abs(st.cash - 12.5) < 1e-9

    # first-ever run (no since_date) -> no retroactive windfall
    st2 = PortfolioState(name="G2", ptype="growth", cash=0.0, starting_capital=1000.0)
    st2.holdings["JNJ"] = Holding("JNJ", 10.0, 150.0)
    rows3, _ = apply_corporate_actions(st2, hist, None, "G2", "2026-06-30")
    assert not rows3 and st2.cash == 0.0


def test_split_adjusts_shares_and_cost_basis():
    st = PortfolioState(name="Growth", ptype="growth", cash=0.0, starting_capital=1000.0)
    st.holdings["NVDA"] = Holding("NVDA", 2.0, 1000.0)
    hist = {"NVDA": _px([1000, 1000, 100, 100, 100], splits=[0, 0, 10.0, 0, 0])}
    rows, _ = apply_corporate_actions(st, hist, "2026-06-24", "Growth", "2026-06-30")
    h = st.holdings["NVDA"]
    assert abs(h.shares - 20.0) < 1e-9          # 2 sh -> 20 sh
    assert abs(h.avg_cost - 100.0) < 1e-9       # $1000 -> $100 basis
    assert rows and rows[0]["type"] == "split"
    # position value unchanged by the split
    assert abs(h.shares * 100.0 - 2000.0) < 1e-6


# ---- 6. macro excluded from the cross-sectional composite --------------------
def test_macro_does_not_enter_composite_ranking():
    indicators = {"AAA": {"mom_21": 0.10, "above_sma50": 1.0, "rsi": 60.0},
                  "BBB": {"mom_21": -0.10, "above_sma50": 0.0, "rsi": 45.0}}
    cfg = {"enabled_factors": ["macro", "fundamental", "technical"],
           "factors": {"balanced": {"macro": 0.9, "technical": 0.05, "fundamental": 0.05}}}
    res = score_universe(indicators, {}, {}, cfg)
    assert "macro" not in res.enabled_factors
    comp = res.composite["balanced"]
    # with macro (a constant) excluded, the 0.9 weight renormalizes over the
    # real cross-sectional factors -> AAA must outrank BBB decisively
    assert comp["AAA"] > comp["BBB"]
    assert res.macro is not None  # regime is still computed for exposure/report


# ---- 7. IC smoothing ----------------------------------------------------------
def test_smooth_ic_dampens_single_month_noise():
    history = [{"technical": 0.05, "fundamental": 0.02}] * 4
    spike = {"technical": 0.9, "fundamental": -0.8}     # one wild month
    sm = smooth_ic(history + [spike], alpha=0.4)
    assert sm["technical"] < 0.5                        # far below the raw spike
    assert sm["fundamental"] > -0.5
    # empty history -> passthrough of the single observation
    assert smooth_ic([spike])["technical"] == 0.9


# ---- mini end-to-end sleeve wiring (idempotency + ledger + exposure) ---------
class _Scores:
    def __init__(self, mult=0.8):
        idx = [f"S{i}" for i in range(20)]
        vals = pd.Series(np.linspace(85, 55, 20), index=idx)
        self.composite = {k: vals for k in ("growth", "defensive", "balanced", "active")}
        self.factor_scores = {"technical": vals, "fundamental": vals}
        self.enabled_factors = ["technical", "fundamental"]
        self.macro = types.SimpleNamespace(score=55.0, label="neutral",
                                           exposure_multiplier=mult, detail={})


def test_model_sleeve_end_to_end_idempotent_and_exposure(tmp_path):
    from usbot.orchestrator import _build_model_sleeves

    store = PortfolioStore(tmp_path / "pf.json")
    scores = _Scores(mult=0.8)                      # neutral -> 80% invested
    prices = {f"S{i}": 50.0 + i for i in range(20)}
    sectors = {f"S{i}": ["Tech", "Health", "Energy", "Fin"][i % 4] for i in range(20)}
    ctx = ReportContext(date="2026-06-30", market_status="open")
    ctx.regime_label = "neutral"

    _build_model_sleeves(ctx, store, scores, prices, sectors, {}, {}, {},
                         "2026-06-30", True, price_history={})
    growth = next(p for p in ctx.portfolios if p.name == "Growth")
    # exposure multiplier -> roughly 20% cash held back
    assert growth.cash > 0.15 * growth.total_value
    led1 = store.load("Growth", 1000.0).ledger
    assert led1, "rebalance must journal itemised trades"
    n1 = len(led1)

    # same-day rerun (e.g. manual re-trigger) must be a no-op revalue
    ctx2 = ReportContext(date="2026-06-30", market_status="open")
    _build_model_sleeves(ctx2, store, scores, prices, sectors, {}, {}, {},
                         "2026-06-30", True, price_history={})
    led2 = store.load("Growth", 1000.0).ledger
    assert len(led2) == n1, "second run on the same day must not duplicate the ledger"
    g2 = next(p for p in ctx2.portfolios if p.name == "Growth")
    assert "Hold" in g2.actions[0]
