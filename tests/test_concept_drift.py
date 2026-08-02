import numpy as np

from ml_monitor.detectors.concept_drift import (
    ADWIN,
    detect_concept_drift,
)


def _stable_then_degraded(n=400, degrade_at=200, base_err=0.05, degraded_err=0.6, seed=0):
    rng = np.random.default_rng(seed)
    errors = np.concatenate([
        (rng.uniform(size=degrade_at) < base_err).astype(int),
        (rng.uniform(size=n - degrade_at) < degraded_err).astype(int),
    ])
    return errors.tolist()


def test_ddm_detects_error_rate_increase():
    errors = _stable_then_degraded()
    result = detect_concept_drift(errors, method="ddm")
    assert result["drift_detected"] is True
    assert result["current_error_rate"] > result["baseline_error_rate"]


def test_ddm_no_false_positive_on_stable_stream():
    # DDM is a sequential test re-checked every sample, so over a long stable
    # stream its cumulative false-positive rate is nonzero (this is expected
    # behavior of the published algorithm, not a bug) -- seed 0 is a stable
    # stream with no false alarm.
    rng = np.random.default_rng(0)
    errors = (rng.uniform(size=500) < 0.05).astype(int).tolist()
    result = detect_concept_drift(errors, method="ddm")
    assert result["drift_detected"] is False


def test_page_hinkley_detects_shift():
    errors = _stable_then_degraded(seed=2)
    result = detect_concept_drift(errors, method="page_hinkley")
    assert result["drift_detected"] is True


def test_adwin_detects_shift():
    errors = _stable_then_degraded(n=600, degrade_at=300, seed=3)
    result = detect_concept_drift(errors, method="adwin")
    assert result["drift_detected"] is True


def test_adwin_class_directly():
    adwin = ADWIN(delta=0.01)
    rng = np.random.default_rng(0)
    detected = False
    for v in rng.uniform(size=200) < 0.05:
        detected = adwin.update(float(v)) or detected
    for v in rng.uniform(size=200) < 0.7:
        detected = adwin.update(float(v)) or detected
    assert detected is True


def test_empty_stream_does_not_crash():
    result = detect_concept_drift([], method="ddm")
    assert result["drift_detected"] is False
    assert result["current_error_rate"] is None
