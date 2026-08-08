
import pytest

from featurestore.core.config import FeatureStoreConfig
from featurestore.core.feature_store import Entity, Feature, FeatureGroup, FeatureStore


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    yield d


@pytest.fixture
def fs(tmp_data_dir):
    cfg = FeatureStoreConfig.from_dict(
        {
            "offline_store": {"type": "parquet", "path": str(tmp_data_dir / "offline")},
            "online_store": {"type": "sqlite", "path": str(tmp_data_dir / "online.db"), "ttl_hours": 24},
            "registry": {"type": "sqlite", "path": str(tmp_data_dir / "registry.db")},
        }
    )
    return FeatureStore(config=cfg)


@pytest.fixture
def user_entity():
    return Entity(name="user_id", dtype="int64", description="Platform user")


@pytest.fixture
def tx_group(user_entity):
    return FeatureGroup(
        name="user_transaction_features",
        entity=user_entity,
        features=[
            Feature("tx_count_7d", dtype="float32", description="Transaction count last 7 days"),
            Feature("tx_amount_avg_30d", dtype="float32", description="Avg transaction amount 30 days"),
        ],
        ttl_hours=24,
        online=True,
        offline=True,
        tags=["user", "transactions"],
    )
