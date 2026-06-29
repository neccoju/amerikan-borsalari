"""Performance metrics computed directly from an equity curve (no heavy deps)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


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
