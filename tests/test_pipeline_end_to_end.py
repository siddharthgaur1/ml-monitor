"""The end-to-end test that only exists because the three repos were merged.

It crosses every subsystem boundary in one call: the fraud subsystem's feature
schema, ingested through the feature store, read back as a monitor's reference
distribution, and compared against live traffic. Before the merge no repo could
import the other two, so nothing could assert this path held together.

If any subsystem changes its contract with another -- the ingest column names,
the feature list, the adapter's read path, the report shape -- this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.features.core.config import FeatureStoreConfig
from ml_platform.features.core.feature_store import FeatureStore
from ml_platform.fraud.common.features import FEATURE_NAMES
from ml_platform.pipeline import FRAUD_FEATURE_GROUP, run_reference_pipeline


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    data = {
        "entity_id": rng.integers(1, 50, size=n),
        "event_timestamp": [base + pd.Timedelta(minutes=int(i)) for i in range(n)],
    }
    for f in FEATURE_NAMES:
        data[f] = rng.normal(loc=shift, scale=1.0, size=n).astype("float32")
    data["fraud_score"] = rng.random(n)
    return pd.DataFrame(data)


@pytest.fixture
def store(tmp_path):
    cfg = FeatureStoreConfig.from_dict(
        {
            "offline_store": {"type": "parquet", "path": str(tmp_path / "offline")},
            "online_store": {"type": "sqlite", "path": str(tmp_path / "online.db")},
            "registry": {"path": str(tmp_path / "registry.db")},
        }
    )
    return FeatureStore(config=cfg)


def test_features_flow_through_the_store_into_a_monitor(store, tmp_path):
    """The path runs, and the monitor's reference really came from the store."""
    result = run_reference_pipeline(
        store,
        history=_frame(300, seed=1),
        live=_frame(80, seed=2),
        db_path=str(tmp_path / "platform.db"),
    )
    assert result["feature_group"] == FRAUD_FEATURE_GROUP
    assert result["reference_rows"] == 300
    assert result["live_rows"] == 80

    # Deliberately NOT asserting zero drifted features. Each of the 11 features
    # is tested independently at alpha=0.05 with no multiple-comparison
    # correction, so a clean run trips at least one detector roughly 4 times in
    # 10 (1 - 0.95**11). Asserting `== []` here passes or fails on the seed, not
    # on the code. The real invariant is the contrast in the next test.
    assert set(result["drifted_features"]) <= set(FEATURE_NAMES)


def test_a_shifted_live_distribution_drifts_far_more_than_an_unshifted_one(
    store, tmp_path
):
    """The path is load-bearing: the same call separates shifted from unshifted.

    Without this contrast the first test would pass even if the monitor never
    read the reference distribution at all.
    """
    clean = run_reference_pipeline(
        store,
        history=_frame(300, seed=1),
        live=_frame(80, seed=2),
        db_path=str(tmp_path / "clean.db"),
    )
    shifted = run_reference_pipeline(
        store,
        history=_frame(300, seed=1),
        live=_frame(80, seed=2, shift=6.0),
        feature_group="user_transaction_features_shifted",
        db_path=str(tmp_path / "shifted.db"),
    )

    assert shifted["drift_detected"], "a 6-sigma shift must register as drift"
    assert len(shifted["drifted_features"]) > len(clean["drifted_features"]), (
        f"shifted={shifted['drifted_features']} should exceed "
        f"clean={clean['drifted_features']}"
    )


def test_the_feature_schema_has_one_definition(store):
    """The registered group's features come from the fraud subsystem's list.

    This is what two separate repos could not do without duplicating the
    feature names, and a duplicated schema is how training/serving skew starts.
    """
    from ml_platform.pipeline import register_fraud_feature_group

    group = register_fraud_feature_group(store)
    assert [f.name for f in group.features] == FEATURE_NAMES
