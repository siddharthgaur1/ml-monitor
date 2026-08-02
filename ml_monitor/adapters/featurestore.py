"""Adapter for the sibling "lite-featurestore" project.

Builds a Monitor whose reference_data comes straight from a registered
FeatureGroup's offline history, instead of the caller manually slicing
X_train. Duck-typed against `featurestore.core.feature_store.FeatureStore`
(offline_store.read(group) -> DataFrame, registry.get_feature_group(name) ->
dict) so this module does NOT import lite-featurestore -- no hard dependency
between the two packages, either one can be installed alone.
"""
from __future__ import annotations

from typing import Any

from ..core.monitor import AlertConfig, DriftConfig, Monitor


def _feature_columns(feature_store: Any, feature_group: str) -> list[str]:
    group = feature_store.registry.get_feature_group(feature_group)
    if group is None:
        raise ValueError(f"feature group not registered in feature store: {feature_group}")
    return [f["name"] for f in group["features"]]


def reference_data_from_feature_store(feature_store: Any, feature_group: str):
    """Pull that feature group's full offline history as a reference DataFrame."""
    feature_cols = _feature_columns(feature_store, feature_group)
    df = feature_store.offline_store.read(feature_group)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature store has no offline data for columns: {missing}")
    return df[feature_cols]


def monitor_from_feature_store(
    feature_store: Any,
    feature_group: str,
    model: Any = None,
    reference_predictions=None,
    config: DriftConfig | None = None,
    alerts: AlertConfig | None = None,
    db_path: str = "ml_monitor.db",
) -> Monitor:
    """Construct a Monitor whose reference_data is the given feature group's
    ingested history, read live from the feature store's offline store.

    Usage:
        from featurestore import FeatureStore
        from ml_monitor.adapters.featurestore import monitor_from_feature_store

        fs = FeatureStore(config="featurestore.yaml")
        monitor = monitor_from_feature_store(fs, "user_transaction_features", model=my_model)
        monitor.log(features=live_row, prediction=pred)
    """
    reference_data = reference_data_from_feature_store(feature_store, feature_group)
    return Monitor(
        model=model,
        reference_data=reference_data,
        reference_predictions=reference_predictions,
        config=config,
        alerts=alerts,
        db_path=db_path,
    )
