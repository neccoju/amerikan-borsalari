"""Smoke tests: keyless modules import and degrade gracefully."""
from usbot.config import get_secrets, load_settings
from usbot.llm.provider import get_provider
from usbot.reports import ReportContext, build_report


def test_settings_load():
    s = load_settings()
    assert s.get("portfolios", {}).get("active_capital") == 1600.0
    assert "growth" in s.scoring.get("factors", {})


def test_secrets_keyless_does_not_crash(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SMTP_USERNAME", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    secrets = get_secrets(dotenv_path="/nonexistent.env")
    assert secrets.has("ANTHROPIC_API_KEY") is False
    prov = get_provider(secrets)
    assert prov.available is False  # no key -> graceful skip
    assert "skip" in prov.complete("sys", "hi").lower()


def test_report_builds_with_minimal_context():
    ctx = ReportContext(date="2024-01-03", market_status="open", regime_label="neutral",
                        regime_score=55.0)
    html, text = build_report(ctx)
    assert "usbot" in html.lower()
    assert "not financial advice" in text.lower()
