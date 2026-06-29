"""Graceful secret detection.

Reads secrets from environment variables (and an optional local .env for dev).
NOTHING here ever raises on a missing key: callers ask ``has(...)`` and modules
that need a key skip gracefully when it is absent. This is what guarantees the
bot runs end-to-end with zero API keys configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)

# All recognized secret names. Keep in sync with .env.example.
KNOWN_SECRETS = [
    "FRED_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "QUIVER_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "SMTP_USERNAME",
    "SMTP_APP_PASSWORD",
    "SMTP_HOST",
    "SMTP_PORT",
    "EMAIL_TO",
    "EMAIL_FROM",
    "CRON_SECRET_TOKEN",
]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency). Only sets vars not already set."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class Secrets:
    """Read-only view over environment secrets with safe accessors."""

    _values: dict[str, str]

    def get(self, name: str, default: str | None = None) -> str | None:
        val = self._values.get(name)
        return val if val else default

    def has(self, *names: str) -> bool:
        """True only if every requested secret is present and non-empty."""
        return all(bool(self._values.get(n)) for n in names)

    def missing(self, *names: str) -> list[str]:
        return [n for n in names if not self._values.get(n)]


def get_secrets(dotenv_path: str | Path = ".env") -> Secrets:
    _load_dotenv(Path(dotenv_path))
    values = {name: os.environ.get(name, "").strip() for name in KNOWN_SECRETS}
    present = [k for k, v in values.items() if v]
    log.info("Secrets detected: %s", ", ".join(present) if present else "(none — running keyless)")
    return Secrets(_values=values)
