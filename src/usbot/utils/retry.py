"""Simple retry decorator with exponential backoff for flaky network calls."""
from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from .logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")


def with_retry(
    attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a callable up to ``attempts`` times with exponential backoff.

    The final exception is re-raised so callers can decide how to degrade.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exc: Exception | None = None
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad catch
                    last_exc = exc
                    if i < attempts:
                        log.warning("%s failed (attempt %d/%d): %s", fn.__name__, i, attempts, exc)
                        time.sleep(delay)
                        delay *= 2
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
