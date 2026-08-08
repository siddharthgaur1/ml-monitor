"""Suppress re-alerting on the same feature within a cooldown window (default 1hr)."""
from __future__ import annotations

import time


class Deduplicator:
    def __init__(self, cooldown_seconds: float = 3600.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: dict[tuple[str, str], float] = {}

    def should_alert(self, drift_type: str, feature: str | None, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        key = (drift_type, feature or "")
        last = self._last_fired.get(key)
        if last is not None and (now - last) < self.cooldown_seconds:
            return False
        self._last_fired[key] = now
        return True

    def reset(self):
        self._last_fired.clear()
