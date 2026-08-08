"""Correlation drift: compare Spearman correlation matrices between the
reference dataset and a rolling current window, flag features whose
correlation structure changed significantly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    if len(cols) == 2:
        # scipy.stats.spearmanr returns a bare scalar (not a matrix) for exactly 2 variables
        rho, _ = stats.spearmanr(df[cols[0]].to_numpy(), df[cols[1]].to_numpy())
        corr = np.array([[1.0, rho], [rho, 1.0]])
    else:
        corr, _ = stats.spearmanr(df[cols].to_numpy())
    return pd.DataFrame(corr, index=cols, columns=cols)


def detect_correlation_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    change_threshold: float = 0.3,
) -> dict:
    """Returns {reference_matrix, current_matrix, delta_matrix, flagged_pairs, flagged_features}."""
    ref_corr = spearman_matrix(reference_df)
    cur_corr = spearman_matrix(current_df)

    common = [c for c in ref_corr.columns if c in cur_corr.columns]
    ref_corr = ref_corr.loc[common, common]
    cur_corr = cur_corr.loc[common, common]
    delta = (cur_corr - ref_corr).abs()

    flagged_pairs = []
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            d = float(delta.loc[a, b]) if not pd.isna(delta.loc[a, b]) else 0.0
            if d >= change_threshold:
                flagged_pairs.append({"feature_a": a, "feature_b": b, "delta": d})

    flagged_features = sorted({p["feature_a"] for p in flagged_pairs} | {p["feature_b"] for p in flagged_pairs})

    return {
        "reference_matrix": ref_corr,
        "current_matrix": cur_corr,
        "delta_matrix": delta,
        "flagged_pairs": flagged_pairs,
        "flagged_features": flagged_features,
    }
