import numpy as np
import pandas as pd

from ml_platform.monitoring.detectors.data_drift import (
    chi2_test,
    detect_data_drift,
    detect_feature_drift,
    ks_test,
    psi,
    wasserstein,
)


def test_ks_no_drift_on_identical_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0, 1, 1000)
    result = ks_test(ref, cur)
    assert result["p_value"] > 0.05


def test_ks_detects_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)
    result = ks_test(ref, cur)
    assert result["p_value"] < 0.05
    assert result["drift_score"] > 0.5


def test_chi2_detects_categorical_drift():
    rng = np.random.default_rng(0)
    ref = rng.choice(["a", "b", "c"], size=1000, p=[0.8, 0.1, 0.1])
    cur = rng.choice(["a", "b", "c"], size=1000, p=[0.1, 0.1, 0.8])
    result = chi2_test(ref, cur)
    assert result["p_value"] < 0.05


def test_chi2_no_drift_same_distribution():
    rng = np.random.default_rng(1)
    ref = rng.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2])
    cur = rng.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2])
    result = chi2_test(ref, cur)
    assert result["p_value"] > 0.05


def test_wasserstein_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)
    result = wasserstein(ref, cur)
    assert result["drift_score"] > 1.0


def test_psi_flags_large_shift_and_not_identical():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(3, 1, 2000)
    assert psi(ref, same)["drift_score"] < 0.1
    assert psi(ref, shifted)["drift_score"] > 0.2


def test_detect_feature_drift_categorical_dispatch():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.choice(["x", "y"], size=500))
    cur = pd.Series(rng.choice(["x", "y"], size=500))
    result = detect_feature_drift(ref.to_numpy(), cur.to_numpy(), categorical_method="chi2")
    assert result["method"] == "chi2"


def test_detect_data_drift_end_to_end():
    rng = np.random.default_rng(0)
    ref_df = pd.DataFrame({
        "amount": rng.normal(50, 10, 1000),
        "category": rng.choice(["a", "b"], 1000),
    })
    cur_df = pd.DataFrame({
        "amount": rng.normal(200, 10, 500),  # heavily drifted
        "category": rng.choice(["a", "b"], 500),  # not drifted
    })
    results = detect_data_drift(ref_df, cur_df)
    assert results["amount"]["is_drifted"] is True
    assert results["amount"]["severity"] == "CRITICAL"
