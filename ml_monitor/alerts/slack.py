"""Slack alert formatting + delivery via incoming webhook."""
from __future__ import annotations

from .webhook import send_webhook


def format_drift_table(alerts: list[dict]) -> str:
    """Build a simple Slack mrkdwn table of drifted features."""
    if not alerts:
        return "No drift detected."
    lines = ["*Drift Alert*", "```", f"{'feature':<20}{'severity':<12}{'drift_score':<12}"]
    for a in alerts:
        lines.append(f"{a.get('feature', '-')!s:<20}{a.get('severity', '-')!s:<12}{a.get('drift_score', 0):<12.4f}")
    lines.append("```")
    return "\n".join(lines)


def send_slack_alert(webhook_url: str, alerts: list[dict]) -> bool:
    text = format_drift_table(alerts)
    return send_webhook(webhook_url, {"text": text})
