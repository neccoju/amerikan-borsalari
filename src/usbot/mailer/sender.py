"""Email sending via SMTP, with a console/log fallback when secrets are absent.

If SMTP_USERNAME / SMTP_APP_PASSWORD are missing (or email disabled / dry-run),
the report is logged and written to disk instead of emailed. Never raises.
"""
from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ..config.secrets import Secrets
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SendResult:
    sent: bool
    channel: str   # "smtp" | "log"
    note: str


def send_report(secrets: Secrets, subject: str, html_body: str, text_body: str,
                *, enabled: bool = True, dry_run: bool = False) -> SendResult:
    if dry_run:
        log.info("[email] dry-run: not sending. Report logged only.")
        return SendResult(False, "log", "dry_run")

    if not enabled:
        return SendResult(False, "log", "email disabled in settings")

    if not secrets.has("SMTP_USERNAME", "SMTP_APP_PASSWORD"):
        missing = secrets.missing("SMTP_USERNAME", "SMTP_APP_PASSWORD")
        log.warning("[email] SMTP secrets missing (%s); logging report instead",
                    ", ".join(missing))
        return SendResult(False, "log", f"missing {missing}")

    host = secrets.get("SMTP_HOST", "smtp.gmail.com")
    port = int(secrets.get("SMTP_PORT", "587"))
    user = secrets.get("SMTP_USERNAME")
    pwd = secrets.get("SMTP_APP_PASSWORD")
    to_addr = secrets.get("EMAIL_TO", user)
    from_addr = secrets.get("EMAIL_FROM", user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, pwd)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        log.info("[email] sent to %s", to_addr)
        return SendResult(True, "smtp", "ok")
    except Exception as exc:  # noqa: BLE001
        log.error("[email] send failed: %s; report logged instead", exc)
        return SendResult(False, "log", f"smtp error: {exc}")
