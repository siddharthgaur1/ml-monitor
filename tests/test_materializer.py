import pandas as pd

from featurestore.materialization.materializer import materialize


def test_materialize_pushes_latest_value_per_entity(fs, tx_group):
    fs.register(tx_group)
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 2],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
            "tx_count_7d": [1.0, 9.0, 4.0],
            "tx_amount_avg_30d": [10.0, 90.0, 40.0],
        }
    )
    fs.ingest(feature_group="user_transaction_features", data=df)

    # wipe online store to prove materialize (re)writes it from offline history
    fs.online_store.conn.execute("DELETE FROM online_features")
    fs.online_store.conn.commit()

    n = materialize(fs, "user_transaction_features")
    assert n == 2

    features = fs.get_online_features("user_transaction_features", [1, 2])
    assert features["1"]["tx_count_7d"] == 9.0
    assert features["2"]["tx_count_7d"] == 4.0
