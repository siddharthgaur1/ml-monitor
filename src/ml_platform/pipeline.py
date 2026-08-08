"""The one path that crosses all three subsystems.

    features (register + ingest)
        -> features (point-in-time correct read)
        -> monitoring (reference distribution built from that feature group)
        -> monitoring (drift verdict on live traffic)

using `ml_platform.fraud`'s real feature schema as the payload, so this is the
platform running its own worked example rather than a toy.

Why this function exists at all: before the merge these were three repos that
could not import each other, so nothing could express this path. Each end held
up its side with duck typing and optional imports, and no test anywhere
exercised the join. `run_reference_pipeline` is that join, and
`tests/test_pipeline_end_to_end.py` is the test that would fail if any of the
three subsystems broke its contract with the others.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_platform.features.core.feature_store import (
    Entity,
    Feature,
    FeatureGroup,
    FeatureStore,
)
from ml_platform.fraud.common.features import FEATURE_NAMES
from ml_platform.monitoring.adapters.featurestore import monitor_from_feature_store
from ml_platform.monitoring.core.monitor import DriftConfig

FRAUD_FEATURE_GROUP = "user_transaction_features"


def register_fraud_feature_group(
    store: FeatureStore, name: str = FRAUD_FEATURE_GROUP
) -> FeatureGroup:
    """Register the fraud subsystem's feature schema in the feature store.

    `FEATURE_NAMES` is imported from `ml_platform.fraud`, not restated here.
    That is the point: one definition of what a transaction feature vector is,
    shared by the thing that computes it and the thing that stores it. Two
    repos could not do this without copying the list.
    """
    group = FeatureGroup(
        name=name,
        entity=Entity(name="user_id", dtype="int64", description="transacting user"),
        features=[
            Feature(name=f, dtype="float32", description=f"fraud model input: {f}")
            for f in FEATURE_NAMES
        ],
        tags=["fraud", "streaming"],
    )
    store.register(group)
    return group


def run_reference_pipeline(
    store: FeatureStore,
    history: pd.DataFrame,
    live: pd.DataFrame,
    feature_group: str = FRAUD_FEATURE_GROUP,
    db_path: str = "ml_platform.db",
) -> dict[str, Any]:
    """Ingest history, build a monitor from it, score `live` against it.

    `history` and `live` both need `entity_id`, `event_timestamp` and the
    columns in `FEATURE_NAMES` -- `entity_id` is the feature store's ingest
    contract, and it holds the user id.

    Returns the drift verdict plus the provenance needed to interpret it: which
    feature group the reference came from and how many rows it held. A drift
    number without its reference is not interpretable, and keeping the two
    together is the whole argument for the feature store owning the reference
    distribution rather than the caller slicing X_train by hand.
    """
    register_fraud_feature_group(store, feature_group)
    # The registry validates strictly: any column that is not `entity_id`,
    # `event_timestamp` or a registered feature is rejected. That is the right
    # behaviour -- an unregistered column reaching the offline store is how a
    # feature nobody declared ends up in a training set -- so the caller's extra
    # columns (a model's own score, say) are dropped here rather than the
    # validator being loosened.
    ingest_cols = ["entity_id", "event_timestamp", *FEATURE_NAMES]
    store.ingest(feature_group, history[ingest_cols])

    monitor = monitor_from_feature_store(
        store,
        feature_group,
        config=DriftConfig(window_size=max(len(live), 1)),
        db_path=db_path,
    )

    # `prediction` is required by Monitor.log. There is no trained model in this
    # path on purpose: the point being tested is that the feature store's
    # reference distribution and the monitor's live window line up on the same
    # schema, which is independent of what any model would score.
    for _, row in live.iterrows():
        monitor.log(
            features={f: float(row[f]) for f in FEATURE_NAMES},
            prediction=float(row.get("fraud_score", 0.0)),
        )

    report = monitor.drift_report()
    drifted = report.drifted_features()
    return {
        "feature_group": feature_group,
        "reference_rows": len(history),
        "live_rows": len(live),
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "report": report,
    }
