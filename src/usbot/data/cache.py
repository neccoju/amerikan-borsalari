"""On-disk parquet/CSV cache to reduce API calls and survive flaky sources."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


class Cache:
    def __init__(self, cache_dir: str | Path, ttl_hours: float = 12.0) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("^", "_")
        return self.dir / f"{safe}.csv"

    def fresh(self, key: str) -> bool:
        p = self._path(key)
        return p.exists() and (time.time() - p.stat().st_mtime) < self.ttl_seconds

    def mtime(self, key: str) -> float | None:
        """Epoch mtime of the cached entry, or None if absent. Lets callers apply
        content-aware validity rules (e.g. 'no market close since caching')."""
        p = self._path(key)
        return p.stat().st_mtime if p.exists() else None

    def load(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache read failed for %s: %s", key, exc)
            return None

    def save(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._path(key))
        except Exception as exc:  # noqa: BLE001
            log.warning("cache write failed for %s: %s", key, exc)
