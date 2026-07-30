import os
import tempfile
import time

from ml_monitor.store.sqlite_store import SQLiteStore
from ml_monitor.store.aggregator import aggregate


def test_log_and_read_prediction():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(db_path=path)
    try:
        store.log_prediction(features={"a": 1.0}, prediction=0.5, label=1)
        rows = store.recent_predictions(limit=10)
        assert len(rows) == 1
        assert rows[0]["features"] == {"a": 1.0}
        assert rows[0]["label"] == 1
    finally:
        store.close()
        os.remove(path)


def test_purge_old_removes_stale_rows():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(db_path=path)
    try:
        old_ts = time.time() - 40 * 86400
        store.log_prediction(features={"a": 1}, prediction=0.1, timestamp=old_ts)
        store.log_prediction(features={"a": 2}, prediction=0.2)
        removed = store.purge_old(retention_days=30)
        assert removed == 1
        assert store.count() == 1
    finally:
        store.close()
        os.remove(path)


def test_aggregate_buckets_by_hour():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(db_path=path)
    try:
        for i in range(5):
            store.log_prediction(features={"a": i}, prediction=0.1)
        result = aggregate(store, granularity="hourly", hours=1)
        assert len(result) >= 1
        assert sum(b["count"] for b in result) == 5
    finally:
        store.close()
        os.remove(path)
