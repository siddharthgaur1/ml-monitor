import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml_platform.monitoring import DriftConfig, Monitor


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except PermissionError:
        pass  # sqlite connection held by the Monitor's store on Windows; harmless in a temp dir


def test_log_is_fast_and_does_not_compute_drift(tmp_db):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    monitor = Monitor(reference_data=reference, config=DriftConfig(window_size=50), db_path=tmp_db)

    for i in range(10):
        monitor.log(features={"a": float(i), "b": float(i)}, prediction=0.5)

    assert len(monitor.window) == 10
    assert monitor.store.count() == 10
    # no drift_report() called yet -- nothing computed
    assert monitor._last_report is None


def test_drift_report_detects_shifted_window(tmp_db):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"amount": rng.normal(50, 5, 500)})
    monitor = Monitor(reference_data=reference, config=DriftConfig(window_size=100), db_path=tmp_db)

    for v in rng.normal(500, 5, 100):  # heavily shifted
        monitor.log(features={"amount": float(v)}, prediction=0.9)

    report = monitor.drift_report()
    assert report.data_drift["amount"]["is_drifted"] is True
    assert "amount" in report.drifted_features()


def test_works_without_labels(tmp_db):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(size=100)})
    monitor = Monitor(reference_data=reference, db_path=tmp_db)
    for v in rng.normal(size=20):
        monitor.log(features={"a": float(v)}, prediction=0.1)  # no label
    report = monitor.drift_report()  # must not crash
    assert report.concept_drift is None


def test_report_to_html(tmp_db, tmp_path):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(size=100)})
    monitor = Monitor(reference_data=reference, db_path=tmp_db)
    for v in rng.normal(size=20):
        monitor.log(features={"a": float(v)}, prediction=0.1)
    report = monitor.drift_report()
    out = tmp_path / "report.html"
    report.to_html(str(out))
    assert out.exists()
    assert "ml-monitor drift report" in out.read_text()
