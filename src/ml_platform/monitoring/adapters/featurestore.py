"""Build a Monitor whose reference data comes from a registered FeatureGroup.

Instead of the caller manually slicing X_train, reference data is read from a
feature group's offline history, so the distribution the monitor compares
against is the same one the model was trained on.

`FeatureStore` is imported for typing only. When these were separate repos this
module was duck-typed on purpose -- neither package could depend on the other,
so it matched `offline_store.read(group)` and `registry.get_feature_group(name)`
structurally and documented that it "does NOT import" the feature store. In one
package that indirection buys nothing and costs type checking, so the contract
is now explicit. The runtime parameter stays `Any`-friendly: anything exposing
those two attributes still works, including test doubles.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.monitor import AlertConfig, DriftConfig, Monitor

if TYPE_CHECKING:
    from ml_platform.features.core.feature_store import FeatureStore


def _feature_columns(feature_store: FeatureStore | Any, feature_group: str) -> list[str]:
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
    db_path: str = "ml_platform.db",
) -> Monitor:
    """Construct a Monitor whose reference_data is the given feature group's
    ingested history, read live from the feature store's offline store.

    Usage:
        from ml_platform.features import FeatureStore
        from ml_platform.monitoring.adapters.featurestore import monitor_from_feature_store

        fs = FeatureStore(config="ml_platform.yaml")
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
