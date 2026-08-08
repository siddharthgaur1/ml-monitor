import pandas as pd
import pytest

from ml_platform.features.stores.offline.parquet_store import ParquetOfflineStore
from ml_platform.features.stores.online.sqlite_store import SQLiteOnlineStore


def test_parquet_store_write_read_roundtrip(tmp_path):
    store = ParquetOfflineStore(path=str(tmp_path / "offline"))
    df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "x": [1.0, 2.0],
        }
    )
    store.write("g", df)
    out = store.read("g")
    assert len(out) == 2
    assert set(out["entity_id"]) == {1, 2}


def test_parquet_store_read_missing_group_returns_empty(tmp_path):
    store = ParquetOfflineStore(path=str(tmp_path / "offline"))
    out = store.read("nope")
    assert out.empty


def test_parquet_store_missing_required_cols_raises(tmp_path):
    store = ParquetOfflineStore(path=str(tmp_path / "offline"))
    with pytest.raises(ValueError):
        store.write("g", pd.DataFrame({"x": [1]}))


def test_parquet_store_partitions_by_date(tmp_path):
    store = ParquetOfflineStore(path=str(tmp_path / "offline"))
    df = pd.DataFrame(
        {
            "entity_id": [1, 1],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "x": [1.0, 2.0],
        }
    )
    store.write("g", df)
    parts = list((tmp_path / "offline" / "g").glob("*.parquet"))
    assert len(parts) == 2


def test_sqlite_online_store_write_read(tmp_path):
    store = SQLiteOnlineStore(path=str(tmp_path / "online.db"), ttl_hours=24)
    store.write_batch("g", [{"entity_id": 1, "features": {"x": 5.0}}])
    result = store.read_batch("g", [1, 2])
    assert result["1"]["x"] == 5.0
    assert "2" not in result


def test_sqlite_online_store_expired_key_not_returned(tmp_path):
    store = SQLiteOnlineStore(path=str(tmp_path / "online.db"), ttl_hours=0.0000001)
    store.write_batch("g", [{"entity_id": 1, "features": {"x": 5.0}}])
    import time

    time.sleep(0.01)
    result = store.read_batch("g", [1])
    assert "1" not in result
