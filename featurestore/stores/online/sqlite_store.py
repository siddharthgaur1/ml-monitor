"""Zero-dependency local-dev online store, backed by stdlib sqlite3.

This is the default/demo online store: no Redis required to run the
quickstart or the test suite.

Schema: single table (feature_group, entity_id, feature_name, value_json,
expires_at) with a lookup index on (feature_group, entity_id).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from featurestore.stores.online.base import OnlineStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS online_features (
    feature_group TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value TEXT,
    expires_at REAL,
    PRIMARY KEY (feature_group, entity_id, feature_name)
);
CREATE INDEX IF NOT EXISTS idx_group_entity
    ON online_features (feature_group, entity_id);
"""


class SQLiteOnlineStore(OnlineStore):
    def __init__(self, path: str = "./data/online.db", ttl_hours: float = 24):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl_hours = ttl_hours
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def write_batch(
        self,
        group_name: str,
        rows: list[dict[str, Any]],
        ttl_hours: float | None = None,
    ) -> int:
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        expires_at = time.time() + ttl * 3600 if ttl else None
        n = 0
        for row in rows:
            entity_id = str(row["entity_id"])
            for name, value in row["features"].items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO online_features "
                    "(feature_group, entity_id, feature_name, value, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (group_name, entity_id, name, json.dumps(value), expires_at),
                )
                n += 1
        self.conn.commit()
        return n

    def read_batch(self, group_name: str, entity_ids: list) -> dict[Any, dict]:
        ids = [str(e) for e in entity_ids]
        placeholders = ",".join("?" * len(ids))
        now = time.time()
        rows = self.conn.execute(
            f"SELECT entity_id, feature_name, value, expires_at FROM online_features "
            f"WHERE feature_group = ? AND entity_id IN ({placeholders})",
            [group_name, *ids],
        ).fetchall()

        out: dict[str, dict] = {}
        for entity_id, feature_name, value, expires_at in rows:
            if expires_at is not None and expires_at < now:
                continue
            out.setdefault(entity_id, {})[feature_name] = json.loads(value)
        return out
