"""Point-in-time correct join: the intellectual core of the feature store.

Given a set of labeled entity events (entity_id + event_timestamp — usually
the time a label was observed) and one or more feature group tables, produce
a training DataFrame where, for every label row, each feature column holds
the *latest feature value known at or before* that label's event_timestamp.

No feature row with event_timestamp > label event_timestamp is ever allowed
into the result — that would leak the future into training data.

Implementation: pandas.merge_asof(direction="backward") per feature group,
grouped by entity id via the `by=` key. merge_asof itself guarantees the
matched row's `on` timestamp is <= the left row's timestamp when
direction="backward", which is exactly the anti-leakage property we need.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ENTITY_COL = "entity_id"
TS_COL = "event_timestamp"

FillStrategy = str  # "none" | "zero" | "last_known"


@dataclass
class PITJoinResult:
    training_df: pd.DataFrame
    uncovered_entity_ids: dict[str, list] = field(default_factory=dict)
    """feature_group_name -> entity_ids with zero coverage (always NaN)."""


def point_in_time_join(
    entity_df: pd.DataFrame,
    feature_group_frames: dict[str, pd.DataFrame],
    entity_col: str = ENTITY_COL,
    timestamp_col: str = TS_COL,
    fill_strategy: FillStrategy = "none",
) -> PITJoinResult:
    """Join feature group tables onto entity_df without leaking the future.

    Args:
        entity_df: must contain `entity_col` and `timestamp_col` (the label
            timestamp). Extra columns (e.g. a label) pass through untouched.
        feature_group_frames: {feature_group_name: df} where each df has
            `entity_col`, `timestamp_col` (event_timestamp of the feature
            row), and one or more feature columns.
        fill_strategy: "none" leaves unmatched features as NaN. "zero" fills
            unmatched numeric features with 0. "last_known" is an alias of
            "none" here — merge_asof already picks the latest value known at
            or before the label timestamp, so there is nothing "more known"
            to fill without looking into the future.
        entity_col, timestamp_col: column names in entity_df; the same names
            are expected in each feature group frame.

    Returns:
        PITJoinResult with the merged training_df and, per feature group,
        the list of entity_ids that had NO feature row at or before their
        label timestamp (i.e. zero coverage -> all-NaN for that group).
    """
    if entity_col not in entity_df.columns or timestamp_col not in entity_df.columns:
        raise ValueError(f"entity_df must contain '{entity_col}' and '{timestamp_col}'")

    result = entity_df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    # merge_asof requires the join keys sorted by the `on` column.
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    uncovered: dict[str, list] = {}

    for group_name, feat_df in feature_group_frames.items():
        if feat_df.empty:
            uncovered[group_name] = sorted(result[entity_col].unique().tolist())
            continue

        feat_df = feat_df.copy()
        feat_df[timestamp_col] = pd.to_datetime(feat_df[timestamp_col])
        feat_df = feat_df.sort_values(timestamp_col).reset_index(drop=True)

        feature_cols = [c for c in feat_df.columns if c not in (entity_col, timestamp_col)]
        # Namespace columns by group to avoid collisions across groups.
        rename_map = {c: f"{group_name}__{c}" for c in feature_cols}
        feat_df = feat_df.rename(columns=rename_map)

        # merge_asof needs both frames sorted by `on`, and `by` sorted within
        # groups is NOT required, but the entity_col dtype must match.
        merged = pd.merge_asof(
            result,
            feat_df,
            on=timestamp_col,
            by=entity_col,
            direction="backward",
        )

        new_cols = list(rename_map.values())
        no_match = merged[new_cols].isna().all(axis=1)
        uncovered[group_name] = sorted(
            merged.loc[no_match, entity_col].unique().tolist()
        )

        if fill_strategy == "zero":
            merged[new_cols] = merged[new_cols].fillna(0)
        # "none" / "last_known": leave as-is (see docstring).

        result = merged

    # Belt-and-suspenders leakage assertion: no feature timestamp column is
    # produced by merge_asof in the output (only the entity timestamp
    # remains), so there is nothing to compare here structurally — the
    # guarantee comes from merge_asof(direction="backward") itself, which is
    # exercised explicitly in tests/test_pit_join.py.
    return PITJoinResult(training_df=result, uncovered_entity_ids=uncovered)
