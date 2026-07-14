"""Process-wide, thread-safe rate limiter (shared across modules).

Several modules hit the SAME Finnhub per-minute quota — fundamentals, insider,
earnings and news. Each pacing itself independently still lets them collide when
one finishes and the next starts a burst (the live 429s came from exactly that).
A single named limiter serializes every Finnhub call through one min-interval
gate, so the quota is respected globally regardless of which module is calling.
"""
from __future__ import annotations

import threading
import time

_LIMITERS: dict[str, "RateLimiter"] = {}
_LOCK = threading.Lock()


class RateLimiter:
    """Minimum-interval gate. ``acquire()`` blocks until the next call is allowed."""

    def __init__(self, rate_per_min: float) -> None:
        self.min_interval = 60.0 / max(1e-9, rate_per_min)
        self._next = 0.0
        self._lk = threading.Lock()

    def acquire(self) -> None:
        with self._lk:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = now + self.min_interval

    def set_rate(self, rate_per_min: float) -> None:
        with self._lk:
            self.min_interval = 60.0 / max(1e-9, rate_per_min)


def get_limiter(name: str, rate_per_min: float = 55.0) -> RateLimiter:
    """Return the shared limiter for ``name`` (created once, rate set on first use)."""
    with _LOCK:
        lim = _LIMITERS.get(name)
        if lim is None:
            lim = RateLimiter(rate_per_min)
            _LIMITERS[name] = lim
        return lim
