import numpy as np

from ml_monitor.detectors.prediction_drift import detect_prediction_drift


def test_no_drift_when_identical():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 1000)
    cur = rng.uniform(0, 1, 1000)
    result = detect_prediction_drift(ref, cur)
    assert result["is_drifted"] is False


def test_drift_detected_on_mean_shift():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 0.3, 1000)  # mostly low-risk scores
    cur = rng.uniform(0.6, 1.0, 1000)  # mostly high-risk scores
    result = detect_prediction_drift(ref, cur)
    assert result["is_drifted"] is True
    assert result["mean_shifted"] is True
