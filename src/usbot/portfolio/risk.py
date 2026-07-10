"""Risk control helpers: position caps, sector caps, weight normalization.

Contract: the returned weights ALWAYS respect the requested caps. Caps are hard
constraints; when honoring them leaves capital unallocated (e.g. a single sector
hitting its cap), the leftover is intentionally NOT force-redistributed — it is
treated as cash. This keeps risk limits truthful rather than cosmetically met.
"""
from __future__ import annotations

import pandas as pd


def apply_caps(weights: dict[str, float], max_position: float) -> dict[str, float]:
    """Iterative water-filling: cap each name at ``max_position`` while preserving
    the original total (when feasible), redistributing excess to uncapped names."""
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)

    # Work in weight space normalized to the original total.
    remaining = dict(weights)
    capped: dict[str, float] = {}
    budget = total

    while remaining:
        # proportionally allocate the current budget among remaining names
        sub_total = sum(remaining.values())
        if sub_total <= 0:
            break
        alloc = {s: budget * (w / sub_total) for s, w in remaining.items()}
        over = {s: a for s, a in alloc.items() if a > max_position + 1e-12}
        if not over:
            capped.update(alloc)
            break
        # fix over-cap names at the cap, subtract from budget, iterate on the rest
        for s in over:
            capped[s] = max_position
            budget -= max_position
            del remaining[s]
    return capped


def apply_sector_cap(weights: dict[str, float], sectors: dict[str, str],
                     max_sector: float, max_position: float | None = None) -> dict[str, float]:
    """Ensure no sector exceeds ``max_sector`` of total weight.

    Over-cap sectors are scaled to the cap; freed weight is redistributed to
    under-cap sectors. Redistribution respects BOTH constraints: a name never
    receives more than its ``max_position`` headroom and a sector never exceeds
    ``max_sector``. If no capacity remains, the freed weight becomes cash (sum
    may be < 1). Caps are never violated.
    """
    if not weights:
        return {}

    by_sector: dict[str, list[str]] = {}
    sector_total: dict[str, float] = {}
    for sym, w in weights.items():
        sec = sectors.get(sym, "Unknown")
        by_sector.setdefault(sec, []).append(sym)
        sector_total[sec] = sector_total.get(sec, 0.0) + w

    adjusted = dict(weights)
    freed = 0.0
    for sec, total in sector_total.items():
        if total > max_sector and total > 0:
            scale = max_sector / total
            for sym in by_sector[sec]:
                freed += adjusted[sym] * (1 - scale)
                adjusted[sym] *= scale

    def _headroom(sym: str) -> float:
        if max_position is None:
            return max_sector  # effectively unbounded within a sector's room
        return max(0.0, max_position - adjusted[sym])

    # Redistribute freed weight to capacity that violates neither cap.
    for _ in range(10):  # a few passes converge
        if freed <= 1e-9:
            break
        capacity: dict[str, float] = {}
        for sec, syms in by_sector.items():
            used = sum(adjusted[s] for s in syms)
            sec_room = max_sector - used
            name_room = sum(_headroom(s) for s in syms)
            room = min(sec_room, name_room)
            if room > 1e-9:
                capacity[sec] = room
        total_cap = sum(capacity.values())
        if total_cap <= 1e-9:
            break  # no room: remainder stays as cash
        give = min(freed, total_cap)
        for sec, room in capacity.items():
            sec_share = give * (room / total_cap)
            syms = by_sector[sec]
            rooms = {s: _headroom(s) for s in syms}
            tot_room = sum(rooms.values()) or 1.0
            for s in syms:
                # proportional to headroom and hard-capped by it
                adjusted[s] += min(rooms[s], sec_share * (rooms[s] / tot_room))
        freed -= give
    return adjusted


def target_weights_from_scores(scores: pd.Series, n: int, max_position: float,
                               sectors: dict[str, str], max_sector: float) -> dict[str, float]:
    """Score-proportional target weights for the top-n names, caps enforced."""
    top = scores.sort_values(ascending=False).head(n)
    if top.empty or top.sum() <= 0:
        return {}
    total = float(top.sum())
    weights = {sym: float(sc) / total for sym, sc in top.items()}
    weights = apply_caps(weights, max_position)
    weights = apply_sector_cap(weights, sectors, max_sector, max_position=max_position)
    return weights
