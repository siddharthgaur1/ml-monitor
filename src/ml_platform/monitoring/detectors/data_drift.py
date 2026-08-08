"""Data drift detectors: KS test, PSI, Chi-squared, Wasserstein distance.

All functions are pure numpy/scipy/pandas — no optional deps.
Severity bands (applied to a normalized drift_score in ~[0, inf)):
    INFO:     0.1 <= score < 0.2
    WARNING:  0.2 <= score < 0.3
    CRITICAL: score >= 0.3
    (score < 0.1 -> None, not drifted)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEVERITY_BANDS = (
    (0.3, "CRITICAL"),
    (0.2, "WARNING"),
    (0.1, "INFO"),
)


def severity_for_score(score: float) -> str | None:
    for cutoff, label in SEVERITY_BANDS:
        if score >= cutoff:
            return label
    return None


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """Kolmogorov-Smirnov two-sample test for numerical features."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    statistic, p_value = stats.ks_2samp(reference, current)
    return {"statistic": float(statistic), "p_value": float(p_value), "drift_score": float(statistic)}


def chi2_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """Chi-squared test of independence for categorical features (binned into shared categories)."""
    reference = pd.Series(reference)
    current = pd.Series(current)
    categories = sorted(set(reference.unique()) | set(current.unique()), key=str)
    ref_counts = reference.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = current.value_counts().reindex(categories, fill_value=0).to_numpy()
    # Guard: chi2_contingency needs no all-zero rows/cols and >1 category
    if len(categories) < 2 or ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_score": 0.0}
    table = np.vstack([ref_counts, cur_counts]) + 1e-9  # avoid zero-cell issues
    statistic, p_value, _, _ = stats.chi2_contingency(table)
    drift_score = float(1.0 - p_value)
    return {"statistic": float(statistic), "p_value": float(p_value), "drift_score": drift_score}


def wasserstein(reference: np.ndarray, current: np.ndarray) -> dict:
    """Wasserstein (earth mover's) distance for numerical features, normalized by reference std."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    distance = stats.wasserstein_distance(reference, current)
    scale = np.std(reference) + 1e-9
    drift_score = float(distance / scale)
    return {"statistic": float(distance), "p_value": None, "drift_score": drift_score}


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict:
    """Population Stability Index. Works for numerical (quantile-binned on reference)
    and categorical (binned by category) features."""
    reference = pd.Series(reference)
    current = pd.Series(current)

    if _is_categorical(reference):
        categories = sorted(set(reference.unique()) | set(current.unique()), key=str)
        ref_counts = reference.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        cur_counts = current.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    else:
        ref_vals = reference.to_numpy(dtype=float)
        cur_vals = current.to_numpy(dtype=float)
        edges = np.unique(np.quantile(ref_vals, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(ref_vals.min(), ref_vals.max() + 1e-9, bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(ref_vals, bins=edges)
        cur_counts, _ = np.histogram(cur_vals, bins=edges)
        ref_counts = ref_counts.astype(float)
        cur_counts = cur_counts.astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + 1e-6
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + 1e-6
    value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return {"statistic": value, "p_value": None, "drift_score": value}


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    numerical_method: str = "ks",
    categorical_method: str = "chi2",
    drift_threshold: float = 0.05,
) -> dict:
    """Run the configured detector on one feature and return a normalized result dict:
    {statistic, p_value, drift_score, is_drifted, severity, method}
    """
    series = pd.Series(reference)
    categorical = _is_categorical(series)
    method = categorical_method if categorical else numerical_method

    if method == "ks":
        result = ks_test(reference, current)
    elif method == "chi2":
        result = chi2_test(reference, current)
    elif method == "wasserstein":
        result = wasserstein(reference, current)
    elif method == "psi":
        result = psi(reference, current)
    else:
        raise ValueError(f"Unknown drift method: {method}")

    if result["p_value"] is not None:
        is_drifted = result["p_value"] < drift_threshold
    else:
        is_drifted = result["drift_score"] >= 0.1

    result["is_drifted"] = bool(is_drifted)
    result["severity"] = severity_for_score(result["drift_score"])
    result["method"] = method
    return result


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_method: str = "ks",
    categorical_method: str = "chi2",
    drift_threshold: float = 0.05,
) -> dict:
    """Per-feature drift report across a reference and current dataframe."""
    results = {}
    for column in reference_df.columns:
        if column not in current_df.columns or len(current_df[column]) == 0:
            continue
        results[column] = detect_feature_drift(
            reference_df[column].to_numpy(),
            current_df[column].to_numpy(),
            numerical_method=numerical_method,
            categorical_method=categorical_method,
            drift_threshold=drift_threshold,
        )
    return results
