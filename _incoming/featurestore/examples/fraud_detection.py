"""Illustrative example: fraud detection feature group.

Shows the intended integration pattern for a sibling "fraud-detection"
portfolio project (which doesn't exist yet) -- it would import featurestore
and call fs.get_online_features() at inference time for real-time scoring,
and fs.get_historical_features() to build its training set. Runs here
against synthetic data only.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from featurestore import Entity, Feature, FeatureGroup, FeatureStore
from featurestore.core.config import FeatureStoreConfig

DEMO_DIR = Path(__file__).parent / "_fraud_demo_data"
shutil.rmtree(DEMO_DIR, ignore_errors=True)
fs = FeatureStore(config=FeatureStoreConfig.from_dict({
    "offline_store": {"type": "parquet", "path": str(DEMO_DIR / "offline")},
    "online_store": {"type": "sqlite", "path": str(DEMO_DIR / "online.db")},
    "registry": {"type": "sqlite", "path": str(DEMO_DIR / "registry.db")},
}))

card = Entity(name="card_id", dtype="int64", description="Payment card")
fraud_features = FeatureGroup(
    name="card_fraud_features",
    entity=card,
    features=[
        Feature("txn_velocity_1h", dtype="float32", description="Transactions in last hour"),
        Feature("amount_zscore", dtype="float32", description="Amount vs card's historical mean"),
        Feature("distinct_merchants_24h", dtype="int32", description="Distinct merchants in 24h"),
    ],
    ttl_hours=1,  # fraud features go stale fast
    online=True, offline=True,
    tags=["fraud", "payments", "realtime"],
)
fs.register(fraud_features)

rng = np.random.default_rng(7)
rows = [
    {
        "entity_id": card_id, "event_timestamp": ts,
        "txn_velocity_1h": float(rng.integers(0, 10)),
        "amount_zscore": round(float(rng.normal(0, 2)), 3),
        "distinct_merchants_24h": int(rng.integers(1, 8)),
    }
    for card_id in [5001, 5002]
    for ts in pd.date_range("2024-03-01", periods=3, freq="h")
]
fs.ingest(feature_group="card_fraud_features", data=pd.DataFrame(rows))

# A "fraud-detection" service would do this at inference time:
online = fs.get_online_features(feature_group="card_fraud_features", entity_ids=[5001, 5002])
for card_id, feats in online.items():
    risk = "HIGH" if feats["amount_zscore"] > 1.5 or feats["txn_velocity_1h"] > 5 else "low"
    print(f"card {card_id}: {feats} -> risk={risk}")
