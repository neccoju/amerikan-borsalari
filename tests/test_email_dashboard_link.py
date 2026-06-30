"""The daily email surfaces the dashboard link (HTML button + text link) when
DASHBOARD_URL is set, and degrades to a note (without breaking) when it isn't."""
from __future__ import annotations

from usbot.reports.builder import ReportContext, build_report


def _ctx(**kw) -> ReportContext:
    ctx = ReportContext(date="2026-06-30", market_status="closed")
    ctx.regime_label = "neutral"
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_email_shows_dashboard_button_when_url_set():
    url = "https://neccoju.github.io/amerikan-borsalari/"
    html, text = build_report(_ctx(dashboard_url=url, dashboard_generated=True))
    # HTML: a clickable "Open Dashboard" button pointing at the URL
    assert "Open Dashboard" in html
    assert url in html
    assert f'href="{url}"' in html
    # plain text: a normal link line
    assert f"Open Dashboard: {url}" in text


def test_email_notes_missing_url_without_breaking():
    html, text = build_report(_ctx(dashboard_url="", dashboard_generated=True))
    msg = "Dashboard generated, but DASHBOARD_URL is not configured yet."
    assert msg in html
    assert msg in text
    # no broken/empty anchor when there's no URL
    assert "Open Dashboard</a>" not in html


def test_email_has_no_dashboard_block_when_not_generated():
    html, text = build_report(_ctx(dashboard_url="", dashboard_generated=False))
    assert "Open Dashboard" not in html
    assert "DASHBOARD_URL is not configured" not in html
    assert "Open Dashboard" not in text
    # report still renders fine
    assert "usbot" in html and "usbot daily report" in text


def test_url_takes_precedence_over_fallback_note():
    url = "https://example.github.io/repo/"
    html, _ = build_report(_ctx(dashboard_url=url, dashboard_generated=True))
    assert "is not configured yet" not in html
    assert url in html
