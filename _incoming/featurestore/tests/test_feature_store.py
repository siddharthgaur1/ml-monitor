import pandas as pd
import pytest

from featurestore.quality.validator import DataQualityError
from featurestore.registry.validator import SchemaValidationError


def _sample_df():
    return pd.DataFrame(
        {
            "entity_id": [1, 1, 2, 2],
            "event_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-10", "2024-01-01", "2024-01-10"]
            ),
            "tx_count_7d": [1.0, 5.0, 2.0, 6.0],
            "tx_amount_avg_30d": [10.0, 12.0, 20.0, 22.0],
        }
    )


def test_register_and_list(fs, tx_group):
    fs.register(tx_group)
    groups = fs.list_feature_groups()
    assert len(groups) == 1
    assert groups[0]["name"] == "user_transaction_features"
    assert len(groups[0]["features"]) == 2


def test_ingest_and_get_online_features(fs, tx_group):
    fs.register(tx_group)
    fs.ingest(feature_group="user_transaction_features", data=_sample_df())

    features = fs.get_online_features(
        feature_group="user_transaction_features", entity_ids=[1, 2]
    )
    # online store keeps latest value per entity
    assert features["1"]["tx_count_7d"] == 5.0
    assert features["2"]["tx_count_7d"] == 6.0


def test_get_historical_features_no_leakage(fs, tx_group):
    fs.register(tx_group)
    fs.ingest(feature_group="user_transaction_features", data=_sample_df())

    labels = pd.DataFrame(
        {"entity_id": [1, 2], "event_timestamp": pd.to_datetime(["2024-01-05", "2024-01-05"])}
    )
    training_df = fs.get_historical_features(
        entity_df=labels, feature_groups=["user_transaction_features"]
    )
    # As of 2024-01-05, only the 01-01 row is known for each entity.
    assert training_df["user_transaction_features__tx_count_7d"].iloc[0] == 1.0
    assert training_df["user_transaction_features__tx_count_7d"].iloc[1] == 2.0


def test_ingest_rejects_unregistered_group(fs):
    with pytest.raises(ValueError):
        fs.ingest(feature_group="nope", data=_sample_df())


def test_ingest_rejects_bad_schema(fs, tx_group):
    fs.register(tx_group)
    bad = _sample_df().drop(columns=["tx_amount_avg_30d"])
    with pytest.raises(SchemaValidationError):
        fs.ingest(feature_group="user_transaction_features", data=bad)


def test_ingest_rejects_high_null_rate(fs, tx_group):
    fs.register(tx_group)
    df = _sample_df()
    df["tx_count_7d"] = float("nan")
    with pytest.raises(DataQualityError):
        fs.ingest(feature_group="user_transaction_features", data=df)


def test_search_features(fs, tx_group):
    fs.register(tx_group)
    results = fs.search_features("transaction")
    assert any(r["name"] == "tx_count_7d" for r in results)


def test_get_feature_stats(fs, tx_group):
    fs.register(tx_group)
    fs.ingest(feature_group="user_transaction_features", data=_sample_df())
    stats = fs.get_feature_stats("user_transaction_features")
    assert stats["tx_count_7d"]["count"] == 4
    assert stats["tx_count_7d"]["null_rate"] == 0.0
