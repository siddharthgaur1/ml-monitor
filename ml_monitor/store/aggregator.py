"""Hourly/daily aggregations over the SQLite prediction log."""
from __future__ import annotations

import time
from collections import defaultdict

from .sqlite_store import SQLiteStore

_BUCKET_SECONDS = {"hourly": 3600, "daily": 86400}


def aggregate(store: SQLiteStore, granularity: str = "hourly", hours: int = 24) -> list[dict]:
    """Bucket predictions by hour/day, returning count, label coverage, and
    (if drift_scores were logged) the mean drift score per bucket.
    """
    if granularity not in _BUCKET_SECONDS:
        raise ValueError("granularity must be 'hourly' or 'daily'")
    bucket_size = _BUCKET_SECONDS[granularity]
    since = time.time() - hours * 3600
    rows = store.recent_predictions(limit=1_000_000, since=since)

    buckets = defaultdict(lambda: {"count": 0, "labeled": 0, "drift_scores": []})
    for row in rows:
        bucket_key = int(row["timestamp"] // bucket_size) * bucket_size
        b = buckets[bucket_key]
        b["count"] += 1
        if row["label"] is not None:
            b["labeled"] += 1
        if row["drift_scores"]:
            scores = [v for v in row["drift_scores"].values() if isinstance(v, (int, float))]
            b["drift_scores"].extend(scores)

    result = []
    for bucket_start in sorted(buckets):
        b = buckets[bucket_start]
        mean_drift = sum(b["drift_scores"]) / len(b["drift_scores"]) if b["drift_scores"] else None
        result.append({
            "bucket_start": bucket_start,
            "count": b["count"],
            "labeled": b["labeled"],
            "mean_drift_score": mean_drift,
        })
    return result
