"""SQLite-backed feature registry: CRUD for entities, feature groups,
features, and an ingestion log. This is the "catalog" that makes features
discoverable and lets ingest-time validation check against a known schema.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    dtype TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS feature_groups (
    name TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,
    ttl_hours REAL,
    online INTEGER NOT NULL DEFAULT 1,
    offline INTEGER NOT NULL DEFAULT 1,
    tags TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS features (
    feature_group TEXT NOT NULL,
    name TEXT NOT NULL,
    dtype TEXT NOT NULL,
    description TEXT,
    PRIMARY KEY (feature_group, name)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_group TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    ingested_at REAL NOT NULL,
    stats_json TEXT
);
"""


class FeatureRegistry:
    def __init__(self, path: str = "./data/registry.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- entities ----------------------------------------------------
    def register_entity(self, name: str, dtype: str, description: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO entities (name, dtype, description) VALUES (?, ?, ?)",
            (name, dtype, description),
        )
        self.conn.commit()

    # -- feature groups ------------------------------------------------
    def register_feature_group(
        self,
        name: str,
        entity_name: str,
        features: list[dict],
        ttl_hours: float | None = None,
        online: bool = True,
        offline: bool = True,
        tags: list[str] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_groups "
            "(name, entity_name, ttl_hours, online, offline, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, entity_name, ttl_hours, int(online), int(offline),
             json.dumps(tags or []), time.time()),
        )
        self.conn.execute("DELETE FROM features WHERE feature_group = ?", (name,))
        for f in features:
            self.conn.execute(
                "INSERT INTO features (feature_group, name, dtype, description) VALUES (?, ?, ?, ?)",
                (name, f["name"], f["dtype"], f.get("description", "")),
            )
        self.conn.commit()

    def get_feature_group(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM feature_groups WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        group = dict(row)
        group["tags"] = json.loads(group["tags"] or "[]")
        group["online"] = bool(group["online"])
        group["offline"] = bool(group["offline"])
        group["features"] = self.get_features(name)
        return group

    def get_features(self, group_name: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, dtype, description FROM features WHERE feature_group = ?", (group_name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_feature_groups(self) -> list[dict]:
        rows = self.conn.execute("SELECT name FROM feature_groups ORDER BY name").fetchall()
        return [self.get_feature_group(r["name"]) for r in rows]

    def search_features(self, keyword: str) -> list[dict]:
        keyword = f"%{keyword.lower()}%"
        rows = self.conn.execute(
            "SELECT f.feature_group, f.name, f.dtype, f.description "
            "FROM features f "
            "WHERE lower(f.name) LIKE ? OR lower(f.description) LIKE ? "
            "OR f.feature_group IN ("
            "  SELECT name FROM feature_groups WHERE lower(tags) LIKE ?)",
            (keyword, keyword, keyword),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- ingestion log -------------------------------------------------
    def log_ingestion(self, group_name: str, row_count: int, stats: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO ingestion_log (feature_group, row_count, ingested_at, stats_json) "
            "VALUES (?, ?, ?, ?)",
            (group_name, row_count, time.time(), json.dumps(stats or {})),
        )
        self.conn.commit()

    def get_latest_stats(self, group_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT stats_json FROM ingestion_log WHERE feature_group = ? "
            "ORDER BY ingested_at DESC LIMIT 1",
            (group_name,),
        ).fetchone()
        return json.loads(row["stats_json"]) if row else None
