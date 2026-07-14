"""Performance metrics computed directly from an equity curve (no heavy deps)."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


@dataclass
class Metrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float
    n_days: int

    def as_dict(self) -> dict:
        return asdict(self)


def _ann_factor(n_days: int, equity_len: int) -> float:
    return TRADING_DAYS / max(1, equity_len)


def compute_metrics(equity: pd.Series, rf: float = 0.0) -> Metrics:
    """Compute standard metrics from a daily equity curve (index = dates)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return Metrics(0, 0, 0, 0, 0, 0, 0, len(equity))

    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # CAGR using actual elapsed calendar time when index is datetime; else periods.
    if isinstance(equity.index, pd.DatetimeIndex):
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    else:
        years = max(len(equity) / TRADING_DAYS, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

    ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))
    excess = rets - rf / TRADING_DAYS
    sharpe = float(excess.mean() / rets.std(ddof=0) * np.sqrt(TRADING_DAYS)) if rets.std(ddof=0) > 0 else 0.0

    downside = rets[rets < 0]
    dd_std = downside.std(ddof=0)
    sortino = float(excess.mean() / dd_std * np.sqrt(TRADING_DAYS)) if dd_std > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    hit_rate = float((rets > 0).mean())

    return Metrics(
        total_return=total_return, cagr=cagr, ann_vol=ann_vol, sharpe=sharpe,
        sortino=sortino, max_drawdown=max_drawdown, hit_rate=hit_rate, n_days=len(equity),
    )


# --------------------------------------------------------------------------- #
# Deflated / Probabilistic Sharpe (López de Prado 2014, "The Deflated Sharpe
# Ratio"). A high in-sample Sharpe is easy to manufacture by trying many
# strategy variants (selection bias / multiple testing). The PSR asks "given
# skew, fat tails and only n observations, how confident are we the TRUE Sharpe
# beats a benchmark?"; the DSR sets that benchmark to the Sharpe you'd expect to
# get by luck alone after N independent trials, so a strategy that merely won a
# large search no longer looks significant.
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    p = min(1 - 1e-12, max(1e-12, p))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def probabilistic_sharpe_ratio(sharpe: float, n: int, skew: float = 0.0,
                               kurtosis: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``). ``sharpe``/``sr_benchmark`` are
    *per-period* (NOT annualized); ``kurtosis`` is Pearson (normal = 3)."""
    if n < 2:
        return 0.0
    var = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    denom = math.sqrt(max(1e-12, var))
    z = (sharpe - sr_benchmark) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe under the null across ``n_trials``
    independent strategies whose Sharpes have variance ``sharpe_variance``."""
    n_t = max(2, int(n_trials))
    sd = math.sqrt(max(0.0, sharpe_variance))
    return sd * ((1 - _EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_t)
                 + _EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_t * math.e)))


@dataclass
class DeflatedSharpe:
    sharpe_period: float        # per-period Sharpe of the selected strategy
    sharpe_annual: float        # annualized (for reading alongside Metrics.sharpe)
    n_trials: int
    benchmark_sharpe: float     # expected-max Sharpe under the null (per period)
    psr_vs_zero: float          # P(true SR > 0)
    dsr: float                  # P(true SR > benchmark) — the multiple-testing-aware number
    skew: float
    kurtosis: float

    def as_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe_from_equity(equity: pd.Series, n_trials: int = 1,
                                sharpe_variance: float | None = None) -> DeflatedSharpe:
    """Deflated Sharpe from a daily equity curve.

    ``n_trials`` is how many strategy configurations were explored to arrive at
    this one (the multiple-testing count). When ``sharpe_variance`` (the variance
    of the trial Sharpes) is unknown we fall back to the variance of the Sharpe
    *estimator* under the null — the standard approximation.
    """
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    n = len(rets)
    if n < 3 or rets.std(ddof=0) == 0:
        return DeflatedSharpe(0, 0, int(n_trials), 0, 0, 0, 0, 3)
    sr = float(rets.mean() / rets.std(ddof=0))                 # per-period
    skew = float(rets.skew())
    kurt = float(rets.kurt() + 3.0)                            # pandas kurt() is excess
    if sharpe_variance is None:
        # Var of the Sharpe estimator (Lo 2002 / Mertens), used as the null
        # spread of trial Sharpes when the empirical spread isn't recorded.
        sharpe_variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)
    sr0 = expected_max_sharpe(n_trials, sharpe_variance) if n_trials > 1 else 0.0
    return DeflatedSharpe(
        sharpe_period=sr,
        sharpe_annual=sr * math.sqrt(TRADING_DAYS),
        n_trials=int(n_trials),
        benchmark_sharpe=sr0,
        psr_vs_zero=probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0),
        dsr=probabilistic_sharpe_ratio(sr, n, skew, kurt, sr0),
        skew=skew, kurtosis=kurt,
    )
