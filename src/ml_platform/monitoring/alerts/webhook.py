"""Generic webhook POST for drift alerts."""
from __future__ import annotations

import json
import urllib.request


def send_webhook(url: str, payload: dict, timeout: float = 5.0) -> bool:
    """POST a JSON payload to an arbitrary webhook URL. Returns True on 2xx,
    False (never raises) on any failure -- alerting must not crash the monitor.
    """
    try:
        data = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001 -- alerting must not crash the monitor, see docstring
        return False
