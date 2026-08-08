"""Redis-backed online store for low-latency production serving.

Key pattern: ml_platform.features:{feature_group}:{entity_id} -> msgpack-encoded
dict of {feature_name: value}. Batch reads use a Redis pipeline. Per-group
TTL is applied via Redis EXPIRE.

Optional extra: pip install ml-platform[redis]. Importing this module
never crashes the package if redis isn't installed or a server isn't
running -- instantiating RedisOnlineStore raises a clear error instead.
"""
from __future__ import annotations

from typing import Any

import msgpack

try:
    import redis
except ImportError:  # pragma: no cover - exercised only without extra installed
    redis = None

from ml_platform.features.stores.online.base import OnlineStore


def _key(group_name: str, entity_id) -> str:
    return f"ml_platform.features:{group_name}:{entity_id}"


class RedisOnlineStore(OnlineStore):
    def __init__(self, host: str = "localhost", port: int = 6379, ttl_hours: float = 24, **kwargs):
        if redis is None:
            raise ImportError(
                "redis is not installed. Install with: pip install ml-platform[redis]"
            )
        self.default_ttl_hours = ttl_hours
        self.client = redis.Redis(host=host, port=port, **kwargs)
        # Fail fast with a clear error rather than a confusing traceback later.
        self.client.ping()

    def write_batch(
        self,
        group_name: str,
        rows: list[dict[str, Any]],
        ttl_hours: float | None = None,
    ) -> int:
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        ttl_seconds = int(ttl * 3600) if ttl else None
        pipe = self.client.pipeline()
        for row in rows:
            key = _key(group_name, row["entity_id"])
            pipe.set(key, msgpack.packb(row["features"], use_bin_type=True))
            if ttl_seconds:
                pipe.expire(key, ttl_seconds)
        pipe.execute()
        return len(rows)

    def read_batch(self, group_name: str, entity_ids: list) -> dict[Any, dict]:
        keys = [_key(group_name, eid) for eid in entity_ids]
        pipe = self.client.pipeline()
        for key in keys:
            pipe.get(key)
        values = pipe.execute()

        out: dict[Any, dict] = {}
        for entity_id, raw in zip(entity_ids, values):
            if raw is not None:
                out[str(entity_id)] = msgpack.unpackb(raw, raw=False)
        return out
