"""Configurable LLM provider abstraction with a no-op fallback.

Providers: anthropic | openai | ollama | none. If the selected provider's key
or package is missing, ``available`` is False and callers skip cleanly. The LLM
is strictly a decision-support / explainability layer; it never decides trades.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config.secrets import Secrets
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMProvider:
    provider: str
    available: bool
    model: str
    reason: str  # why unavailable, if applicable

    def complete(self, system: str, prompt: str, max_tokens: int = 1200) -> str:
        if not self.available:
            return f"[LLM skipped: {self.reason}]"
        try:
            if self.provider == "anthropic":
                return self._anthropic(system, prompt, max_tokens)
            if self.provider == "openai":
                return self._openai(system, prompt, max_tokens)
            if self.provider == "ollama":
                return self._ollama(system, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM call failed: %s", exc)
            return f"[LLM error: {exc}]"
        return "[LLM skipped: unknown provider]"

    def _anthropic(self, system: str, prompt: str, max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    def _openai(self, system: str, prompt: str, max_tokens: int) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def _ollama(self, system: str, prompt: str, max_tokens: int) -> str:
        import requests

        from os import environ

        base = environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.post(
            f"{base}/api/generate",
            json={"model": self.model, "system": system, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def get_provider(secrets: Secrets) -> LLMProvider:
    """Resolve provider from config/secrets, degrading to no-op when unavailable."""
    provider = (secrets.get("LLM_PROVIDER") or "none").lower()

    if provider in ("none", ""):
        return LLMProvider("none", False, "", "provider=none")

    # Model is configurable via LLM_MODEL so the exact string your account
    # supports can be set without code changes; sensible defaults otherwise.
    model_override = secrets.get("LLM_MODEL")

    if provider == "anthropic":
        if not secrets.has("ANTHROPIC_API_KEY"):
            return LLMProvider("anthropic", False, "", "missing ANTHROPIC_API_KEY")
        return LLMProvider("anthropic", True,
                           model_override or "claude-3-5-sonnet-latest", "")

    if provider == "openai":
        if not secrets.has("OPENAI_API_KEY"):
            return LLMProvider("openai", False, "", "missing OPENAI_API_KEY")
        return LLMProvider("openai", True, model_override or "gpt-4o-mini", "")

    if provider == "ollama":
        model = secrets.get("OLLAMA_MODEL", "llama3.1")
        return LLMProvider("ollama", True, model, "")

    return LLMProvider(provider, False, "", f"unknown provider '{provider}'")
