"""Prediction drift: PSI on prediction-score bins + mean-shift alerting.
Works without ground-truth labels.
"""
from __future__ import annotations

import numpy as np

from .data_drift import psi, severity_for_score


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    bins: int = 10,
    mean_shift_threshold: float = 0.1,
) -> dict:
    """
    mean_shift_threshold: fractional change in mean prediction (relative to
    reference mean, or absolute if reference mean ~ 0) that triggers an alert.
    """
    reference_predictions = np.asarray(reference_predictions, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    psi_result = psi(reference_predictions, current_predictions, bins=bins)
    drift_score = psi_result["drift_score"]

    ref_mean = float(np.mean(reference_predictions))
    cur_mean = float(np.mean(current_predictions))
    denom = abs(ref_mean) if abs(ref_mean) > 1e-9 else 1.0
    mean_shift_pct = (cur_mean - ref_mean) / denom

    mean_shifted = abs(mean_shift_pct) > mean_shift_threshold
    is_drifted = bool(drift_score >= 0.1 or mean_shifted)

    return {
        "statistic": psi_result["statistic"],
        "drift_score": drift_score,
        "is_drifted": is_drifted,
        "severity": severity_for_score(drift_score),
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "mean_shift_pct": float(mean_shift_pct),
        "mean_shifted": bool(mean_shifted),
        "method": "psi",
    }
