"""Feature drift detection, delegated to the sibling "ml-monitor" project.

lite-featurestore intentionally owns no drift math of its own (that's
ml-monitor's job) -- this module just slices a feature group's offline
history into a reference/current split and calls into
`ml_monitor.detectors.data_drift.detect_data_drift`. ml-monitor is an
optional dependency: importing this module never fails, only calling
`check_drift` without ml-monitor installed does, with a clear message.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def check_drift(
    feature_store: Any,
    feature_group: str,
    split_at: str | None = None,
    numerical_method: str = "ks",
    categorical_method: str = "chi2",
    drift_threshold: float = 0.05,
) -> dict:
    """Per-feature drift report for `feature_group`, reference vs current.

    `split_at`: ISO timestamp splitting the group's offline history into
    reference (before) and current (at/after). Defaults to the midpoint of
    the available history if omitted.

    Requires the optional "ml-monitor" package (pip install -e ../ml-monitor,
    it isn't published to PyPI): raises ImportError with that instruction if
    it isn't installed.
    """
    try:
        from ml_monitor.detectors.data_drift import detect_data_drift
    except ImportError as exc:
        raise ImportError(
            "check_drift requires the optional sibling package 'ml-monitor' "
            "(pip install -e ../ml-monitor)."
        ) from exc

    group = feature_store.registry.get_feature_group(feature_group)
    if group is None:
        raise ValueError(f"feature group not registered: {feature_group}")
    feature_cols = [f["name"] for f in group["features"]]

    df = feature_store.offline_store.read(feature_group)
    if df.empty:
        return {}
    df = df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df = df.sort_values("event_timestamp")

    if split_at is not None:
        split_ts = pd.Timestamp(split_at)
        reference_df = df[df["event_timestamp"] < split_ts]
        current_df = df[df["event_timestamp"] >= split_ts]
    else:
        midpoint = len(df) // 2
        reference_df, current_df = df.iloc[:midpoint], df.iloc[midpoint:]

    if reference_df.empty or current_df.empty:
        return {}

    return detect_data_drift(
        reference_df[feature_cols],
        current_df[feature_cols],
        numerical_method=numerical_method,
        categorical_method=categorical_method,
        drift_threshold=drift_threshold,
    )
