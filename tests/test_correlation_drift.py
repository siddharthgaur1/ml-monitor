import numpy as np
import pandas as pd

from ml_platform.monitoring.detectors.correlation_drift import detect_correlation_drift


def test_flags_features_with_broken_correlation():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    ref_df = pd.DataFrame({"x": x, "y": x + rng.normal(scale=0.1, size=n)})  # strongly correlated

    x2 = rng.normal(size=n)
    cur_df = pd.DataFrame({"x": x2, "y": rng.normal(size=n)})  # correlation broken

    result = detect_correlation_drift(ref_df, cur_df, change_threshold=0.3)
    assert "x" in result["flagged_features"] or "y" in result["flagged_features"]


def test_no_flags_when_correlation_stable():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    ref_df = pd.DataFrame({"x": x, "y": x + rng.normal(scale=0.1, size=n)})
    x2 = rng.normal(size=n)
    cur_df = pd.DataFrame({"x": x2, "y": x2 + rng.normal(scale=0.1, size=n)})

    result = detect_correlation_drift(ref_df, cur_df, change_threshold=0.3)
    assert result["flagged_pairs"] == []
