"""Materialize offline -> online: push the latest feature value per entity
into the online store, in batches.
"""
from __future__ import annotations

from ml_platform.features.core.feature_store import FeatureStore, _latest_per_entity


def materialize(fs: FeatureStore, feature_group: str, batch_size: int = 500) -> int:
    """Reads all offline history for `feature_group`, keeps the latest row
    per entity, and writes it to the online store in batches of
    `batch_size`. Returns the number of entities materialized.
    """
    group = fs.registry.get_feature_group(feature_group)
    if group is None:
        raise ValueError(f"feature group not registered: {feature_group}")
    if not group["offline"]:
        raise ValueError(f"'{feature_group}' has offline=False, nothing to materialize from")

    history = fs.offline_store.read(feature_group)
    if history.empty:
        return 0

    latest = _latest_per_entity(history)
    feature_names = [f["name"] for f in group["features"]]

    written = 0
    records = latest.to_dict("records")
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        rows = [
            {"entity_id": r["entity_id"], "features": {n: r[n] for n in feature_names}}
            for r in chunk
        ]
        fs.online_store.write_batch(feature_group, rows, ttl_hours=group["ttl_hours"])
        written += len(rows)

    return written
