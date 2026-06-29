"""Online factor-weight updates via exponential gradient (multiplicative weights).

Given the latest per-factor reward (information coefficient), nudge weights
toward factors that recently worked:

    w_f  <-  w_f * exp(lr * reward_f)

then renormalize to sum 1 and clamp each weight to [min_w, max_w] (with a second
normalization pass). Conservative by design: a small learning rate and tight
bounds keep the adaptive sleeve from chasing noise. Paper-only.
"""
from __future__ import annotations

import math


def normalize_weights(weights: dict[str, float], min_w: float = 0.02,
                      max_w: float = 0.50) -> dict[str, float]:
    """Project weights onto the simplex (sum 1) with box constraints [min_w, max_w].

    Iteratively fixes any weight that violates a bound at that bound and
    redistributes the remaining budget proportionally among the still-free
    weights — a converging capped-simplex projection.
    """
    if not weights:
        return {}
    n = len(weights)
    # Make the bounds feasible for n weights (n*min <= 1 <= n*max).
    min_w = min(min_w, 1.0 / n)
    max_w = max(max_w, 1.0 / n)

    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(w.values())
    w = {k: (v / total if total > 0 else 1.0 / n) for k, v in w.items()}

    fixed: dict[str, float] = {}
    free = set(w)
    for _ in range(4 * n + 5):
        budget = 1.0 - sum(fixed.values())
        free_sum = sum(w[k] for k in free) or 1.0
        for k in free:
            w[k] = w[k] / free_sum * budget
        # Resolve upper-bound violations first; only then lower-bound ones, so a
        # weight isn't pinned at min before the freed budget is redistributed.
        over = [k for k in free if w[k] > max_w + 1e-12]
        if over:
            for k in over:
                w[k] = max_w; fixed[k] = max_w; free.discard(k)
            continue
        under = [k for k in free if w[k] < min_w - 1e-12]
        if under:
            for k in under:
                w[k] = min_w; fixed[k] = min_w; free.discard(k)
            continue
        break  # no violations -> converged
    return w


def update_weights(weights: dict[str, float], reward: dict[str, float],
                   lr: float = 0.5, min_w: float = 0.02,
                   max_w: float = 0.50) -> dict[str, float]:
    """Exponential-gradient update of factor weights using per-factor reward.

    Only factors present in ``weights`` are updated; missing rewards are treated
    as 0 (no change before normalization). Returns normalized, clamped weights.
    """
    if not weights:
        return {}
    updated = {}
    for f, w in weights.items():
        r = float(reward.get(f, 0.0))
        # guard against extreme exponents
        exponent = max(-5.0, min(5.0, lr * r))
        updated[f] = max(1e-9, float(w)) * math.exp(exponent)
    return normalize_weights(updated, min_w=min_w, max_w=max_w)
