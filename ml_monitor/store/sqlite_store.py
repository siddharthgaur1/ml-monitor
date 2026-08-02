"""SQLite-backed persistent store for every logged prediction."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    features TEXT NOT NULL,
    prediction TEXT NOT NULL,
    label TEXT,
    drift_scores TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    feature TEXT,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    drift_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
"""


class SQLiteStore:
    """Thread-safe-ish SQLite store (one connection guarded by a lock — fine
    for the write volumes a monitoring sidecar sees; swap for a pool if this
    ever needs to serve heavy concurrent writers).
    """

    def __init__(self, db_path: str = "ml_monitor.db", retention_days: int = 30):
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: Any,
        label: Any | None = None,
        drift_scores: dict | None = None,
        timestamp: float | None = None,
    ) -> int:
        timestamp = timestamp if timestamp is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO predictions (timestamp, features, prediction, label, drift_scores) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp,
                    json.dumps(features, default=str),
                    json.dumps(prediction, default=str),
                    json.dumps(label, default=str) if label is not None else None,
                    json.dumps(drift_scores, default=str) if drift_scores is not None else None,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def log_alert(self, severity: str, message: str, drift_type: str, feature: str | None = None,
                   timestamp: float | None = None) -> int:
        timestamp = timestamp if timestamp is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO alerts (timestamp, feature, severity, message, drift_type) VALUES (?, ?, ?, ?, ?)",
                (timestamp, feature, severity, message, drift_type),
            )
            self._conn.commit()
            return cur.lastrowid

    def recent_predictions(self, limit: int = 1000, since: float | None = None) -> list[dict]:
        query = "SELECT id, timestamp, features, prediction, label, drift_scores FROM predictions"
        params: list = []
        if since is not None:
            query += " WHERE timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows][::-1]

    def recent_alerts(self, limit: int = 200, since: float | None = None) -> list[dict]:
        query = "SELECT id, timestamp, feature, severity, message, drift_type FROM alerts"
        params: list = []
        if since is not None:
            query += " WHERE timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        cols = ["id", "timestamp", "feature", "severity", "message", "drift_type"]
        return [dict(zip(cols, r)) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def purge_old(self, retention_days: int | None = None) -> int:
        retention_days = retention_days if retention_days is not None else self.retention_days
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
            self._conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def close(self):
        self._conn.close()

    @staticmethod
    def _row_to_dict(row) -> dict:
        id_, ts, features, prediction, label, drift_scores = row
        return {
            "id": id_,
            "timestamp": ts,
            "features": json.loads(features),
            "prediction": json.loads(prediction),
            "label": json.loads(label) if label is not None else None,
            "drift_scores": json.loads(drift_scores) if drift_scores is not None else None,
        }
