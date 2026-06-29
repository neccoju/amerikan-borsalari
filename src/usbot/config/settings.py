"""YAML settings loader. Returns a light dict-backed object with attribute access."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@dataclass
class Settings:
    settings: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    config_dir: Path = _CONFIG_DIR

    def __getitem__(self, key: str) -> Any:
        return self.settings[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_dir: str | Path | None = None) -> Settings:
    cfg_dir = Path(config_dir) if config_dir else _CONFIG_DIR
    return Settings(
        settings=_read_yaml(cfg_dir / "settings.yaml"),
        scoring=_read_yaml(cfg_dir / "scoring.yaml"),
        config_dir=cfg_dir,
    )
