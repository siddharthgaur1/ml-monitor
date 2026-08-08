"""Structured JSON alert logger. Always active regardless of AlertConfig."""
from __future__ import annotations

import json
import logging
import time

_logger = logging.getLogger("ml_platform.monitoring.alerts")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def log_alert(severity: str, message: str, drift_type: str, feature: str | None = None, **extra) -> dict:
    record = {
        "timestamp": time.time(),
        "severity": severity,
        "drift_type": drift_type,
        "feature": feature,
        "message": message,
        **extra,
    }
    _logger.info(json.dumps(record, default=str))
    return record
