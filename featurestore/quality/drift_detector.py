"""Stub: feature drift detection.

Not implemented in this project. Intended integration point: a sibling
portfolio project ("ml-monitor") is planned to own drift detection
(population stability index / KS-test between a reference window and a
current window) and would call into this feature store's registry.stats
history (ingestion_log.stats_json per feature_group) as its data source.

When ml-monitor exists, this function should be replaced by a call out to
it, or implemented here using the stats already stored per ingest in
FeatureRegistry.get_latest_stats() compared across ingestion_log rows.
"""
from __future__ import annotations


def check_drift(feature_group: str, reference_stats: dict, current_stats: dict) -> dict:
    """Not implemented. See module docstring for the intended integration
    with the ml-monitor sibling project.
    """
    raise NotImplementedError(
        "Drift detection is out of scope for lite-featurestore; "
        "intended to be provided by the ml-monitor sibling project."
    )
