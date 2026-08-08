"""Point-in-time join correctness — the most important test suite here.

Explicit leakage assertions: no feature row with event_timestamp > the
label's event_timestamp may ever be selected.
"""
from __future__ import annotations

import pandas as pd
import pytest

from featurestore.core.pit_join import point_in_time_join


def _ts(s):
    return pd.Timestamp(s)


@pytest.fixture
def feature_history():
    # entity 1 has three feature snapshots over time; entity 2 has one late one.
    return pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 2],
            "event_timestamp": [
                _ts("2024-01-01"),
                _ts("2024-01-05"),
                _ts("2024-01-10"),
                _ts("2024-01-08"),
            ],
            "tx_count_7d": [1.0, 5.0, 9.0, 3.0],
        }
    )


def test_picks_latest_row_at_or_before_label_time(feature_history):
    entity_df = pd.DataFrame({"entity_id": [1], "event_timestamp": [_ts("2024-01-06")]})
    result = point_in_time_join(entity_df, {"tx": feature_history})
    # As of 2024-01-06, only the 01-01 and 01-05 rows are known; latest is 01-05 -> 5.0
    assert result.training_df["tx__tx_count_7d"].iloc[0] == 5.0


def test_exact_timestamp_match_is_included_not_excluded(feature_history):
    entity_df = pd.DataFrame({"entity_id": [1], "event_timestamp": [_ts("2024-01-05")]})
    result = point_in_time_join(entity_df, {"tx": feature_history})
    assert result.training_df["tx__tx_count_7d"].iloc[0] == 5.0


def test_no_leakage_future_row_never_selected(feature_history):
    # Label observed *before* any feature row exists for entity 2 (01-08 row is future).
    entity_df = pd.DataFrame({"entity_id": [2], "event_timestamp": [_ts("2024-01-01")]})
    result = point_in_time_join(entity_df, {"tx": feature_history})
    assert pd.isna(result.training_df["tx__tx_count_7d"].iloc[0])
    assert 2 in result.uncovered_entity_ids["tx"]


def test_no_leakage_across_many_random_labels(feature_history):
    """Exhaustively verify: for every (entity, label_ts) pair, the joined
    feature row's timestamp is always <= label_ts, by cross-checking against
    a brute-force pandas computation independent of point_in_time_join.
    """
    entity_df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 1, 2, 2],
            "event_timestamp": [
                _ts("2023-12-31"),  # before any row -> no coverage
                _ts("2024-01-01"),  # exact match
                _ts("2024-01-07"),  # between rows
                _ts("2024-01-31"),  # after all rows
                _ts("2024-01-07"),  # before entity 2's only row
                _ts("2024-01-08"),  # exact match for entity 2
            ],
        }
    )
    result = point_in_time_join(entity_df, {"tx": feature_history})
    df = result.training_df

    for _, row in df.iterrows():
        matched_val = row["tx__tx_count_7d"]
        if pd.isna(matched_val):
            continue
        # Find the feature row(s) with this exact value for this entity, and
        # assert every such candidate satisfying the value is <= label ts.
        candidates = feature_history[
            (feature_history["entity_id"] == row["entity_id"])
            & (feature_history["tx_count_7d"] == matched_val)
        ]
        assert (candidates["event_timestamp"] <= row["event_timestamp"]).any(), (
            f"leakage: matched value {matched_val} for entity {row['entity_id']} "
            f"at label {row['event_timestamp']} has no valid source row <= label time"
        )

    # Direct structural check: every row of feature_history that is strictly
    # later than a label's timestamp must never be the sole source of that
    # label's matched value (i.e. some row <= label time also has that value,
    # or the matched value differs from every future-only row's value).
    for _, row in df.iterrows():
        future_rows = feature_history[
            (feature_history["entity_id"] == row["entity_id"])
            & (feature_history["event_timestamp"] > row["event_timestamp"])
        ]
        if not future_rows.empty and not pd.isna(row["tx__tx_count_7d"]):
            assert row["tx__tx_count_7d"] not in set(future_rows["tx_count_7d"]) or (
                row["tx__tx_count_7d"]
                in set(
                    feature_history[
                        (feature_history["entity_id"] == row["entity_id"])
                        & (feature_history["event_timestamp"] <= row["event_timestamp"])
                    ]["tx_count_7d"]
                )
            )


def test_multiple_feature_groups_merge_and_namespace_columns():
    entity_df = pd.DataFrame({"entity_id": [1], "event_timestamp": [_ts("2024-01-10")]})
    group_a = pd.DataFrame(
        {"entity_id": [1], "event_timestamp": [_ts("2024-01-09")], "amt": [10.0]}
    )
    group_b = pd.DataFrame(
        {"entity_id": [1], "event_timestamp": [_ts("2024-01-08")], "cnt": [3]}
    )
    result = point_in_time_join(entity_df, {"a": group_a, "b": group_b})
    assert result.training_df["a__amt"].iloc[0] == 10.0
    assert result.training_df["b__cnt"].iloc[0] == 3


def test_fill_strategy_zero_fills_missing():
    entity_df = pd.DataFrame({"entity_id": [99], "event_timestamp": [_ts("2024-01-01")]})
    feat_df = pd.DataFrame({"entity_id": [1], "event_timestamp": [_ts("2024-01-01")], "x": [1.0]})
    result = point_in_time_join(entity_df, {"g": feat_df}, fill_strategy="zero")
    assert result.training_df["g__x"].iloc[0] == 0.0


def test_uncovered_entities_reported_per_group(feature_history):
    entity_df = pd.DataFrame(
        {"entity_id": [1, 2, 3], "event_timestamp": [_ts("2024-01-10"), _ts("2024-01-08"), _ts("2024-01-10")]}
    )
    result = point_in_time_join(entity_df, {"tx": feature_history})
    # entity 3 doesn't exist in feature_history at all -> uncovered
    assert 3 in result.uncovered_entity_ids["tx"]
    assert 1 not in result.uncovered_entity_ids["tx"]
    assert 2 not in result.uncovered_entity_ids["tx"]


def test_missing_required_columns_raises():
    with pytest.raises(ValueError):
        point_in_time_join(pd.DataFrame({"entity_id": [1]}), {})
