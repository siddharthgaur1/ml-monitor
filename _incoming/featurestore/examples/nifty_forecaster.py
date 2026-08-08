"""Illustrative example: time-series forecasting feature group.

Shows the intended integration pattern for a sibling "nifty-forecaster"
portfolio project (which doesn't exist yet) -- it would call
fs.get_historical_features() to build point-in-time correct training data
for a price-movement model. Runs here against synthetic data only.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from featurestore import Entity, Feature, FeatureGroup, FeatureStore
from featurestore.core.config import FeatureStoreConfig

DEMO_DIR = Path(__file__).parent / "_nifty_demo_data"
shutil.rmtree(DEMO_DIR, ignore_errors=True)
fs = FeatureStore(config=FeatureStoreConfig.from_dict({
    "offline_store": {"type": "parquet", "path": str(DEMO_DIR / "offline")},
    "online_store": {"type": "sqlite", "path": str(DEMO_DIR / "online.db")},
    "registry": {"type": "sqlite", "path": str(DEMO_DIR / "registry.db")},
}))

symbol = Entity(name="symbol_id", dtype="int64", description="Ticker symbol id")
market_features = FeatureGroup(
    name="daily_market_features",
    entity=symbol,
    features=[
        Feature("sma_5d", dtype="float32", description="5-day simple moving average"),
        Feature("volatility_10d", dtype="float32", description="10-day rolling volatility"),
    ],
    ttl_hours=24, online=False, offline=True,  # forecasting trains offline; no live serving needed
    tags=["market", "timeseries"],
)
fs.register(market_features)

rng = np.random.default_rng(3)
rows = [
    {
        "entity_id": 1, "event_timestamp": day,
        "sma_5d": round(float(rng.uniform(18000, 19000)), 2),
        "volatility_10d": round(float(rng.uniform(0.5, 3.0)), 3),
    }
    for day in pd.date_range("2024-01-01", periods=20, freq="D")
]
fs.ingest(feature_group="daily_market_features", data=pd.DataFrame(rows))

# A "nifty-forecaster" trainer would build labels (e.g. next-day direction)
# and pull point-in-time correct features for each label date:
labels_df = pd.DataFrame({
    "entity_id": [1, 1, 1],
    "event_timestamp": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-15"]),
    "next_day_up": [1, 0, 1],
})
training_df = fs.get_historical_features(entity_df=labels_df, feature_groups=["daily_market_features"])
print(training_df)
