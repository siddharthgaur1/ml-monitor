"""End-to-end demo: synthetic data -> register -> ingest -> serve online ->
generate a point-in-time correct training DataFrame.

Run standalone: python examples/quickstart.py
Uses the default parquet + sqlite stores, so no external services required.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml_platform.features import Entity, Feature, FeatureGroup, FeatureStore
from ml_platform.features.core.config import FeatureStoreConfig

DEMO_DIR = Path(__file__).parent / "_quickstart_data"
shutil.rmtree(DEMO_DIR, ignore_errors=True)

config = FeatureStoreConfig.from_dict(
    {
        "offline_store": {"type": "parquet", "path": str(DEMO_DIR / "offline")},
        "online_store": {"type": "sqlite", "path": str(DEMO_DIR / "online.db"), "ttl_hours": 24},
        "registry": {"type": "sqlite", "path": str(DEMO_DIR / "registry.db")},
    }
)
fs = FeatureStore(config=config)

# 1. Define and register a feature group.
user = Entity(name="user_id", dtype="int64", description="Platform user")
user_transactions = FeatureGroup(
    name="user_transaction_features",
    entity=user,
    features=[
        Feature("tx_count_7d", dtype="float32", description="Transaction count last 7 days"),
        Feature("tx_amount_avg_30d", dtype="float32", description="Avg transaction amount 30 days"),
    ],
    ttl_hours=24, online=True, offline=True,
    tags=["user", "transactions", "behavioral"],
)
fs.register(user_transactions)

# 2. Ingest synthetic feature snapshots for 3 users, 5 days each.
rng = np.random.default_rng(42)
rows = []
for user_id in [101, 102, 103]:
    for day in pd.date_range("2024-01-01", periods=5, freq="D"):
        rows.append({
            "entity_id": user_id, "event_timestamp": day,
            "tx_count_7d": float(rng.integers(0, 20)),
            "tx_amount_avg_30d": round(float(rng.uniform(10, 500)), 2),
        })
computed_features_df = pd.DataFrame(rows)
result = fs.ingest(feature_group="user_transaction_features", data=computed_features_df)
print(f"Ingested {result['rows_written']} rows. Stats: {result['stats']}\n")

# 3. Serve the latest features online (low latency lookup).
online = fs.get_online_features(feature_group="user_transaction_features", entity_ids=[101, 102, 103])
print("Online features (latest per entity):")
for entity_id, feats in online.items():
    print(f"  {entity_id}: {feats}")

# 4. Generate a point-in-time correct training DataFrame from historical labels.
labels_df = pd.DataFrame({
    "entity_id": [101, 102, 103],
    "event_timestamp": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-02"]),
    "label_churned": [0, 1, 0],
})
training_df = fs.get_historical_features(entity_df=labels_df, feature_groups=["user_transaction_features"])
print("\nPoint-in-time training DataFrame (no future leakage):")
print(training_df)

print("\nRegistered feature groups:", [g["name"] for g in fs.list_feature_groups()])
print("Search 'transaction':", [r["name"] for r in fs.search_features("transaction")])
