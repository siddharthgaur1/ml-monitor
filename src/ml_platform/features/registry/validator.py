"""Schema validation: does this DataFrame match what's registered for a
feature group? (Column presence/naming, not data quality — see
ml_platform.features.quality.validator for null-rate/dtype-quality checks.)
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLS = {"entity_id", "event_timestamp"}


class SchemaValidationError(ValueError):
    pass


def validate_schema(df: pd.DataFrame, feature_group: dict) -> None:
    """Raises SchemaValidationError if df doesn't match the registered group.

    feature_group: dict as returned by FeatureRegistry.get_feature_group().
    """
    missing_required = REQUIRED_COLS - set(df.columns)
    if missing_required:
        raise SchemaValidationError(f"data missing required columns: {missing_required}")

    registered_names = {f["name"] for f in feature_group["features"]}
    given_names = set(df.columns) - REQUIRED_COLS
    missing_features = registered_names - given_names
    unknown_features = given_names - registered_names

    errors = []
    if missing_features:
        errors.append(f"missing registered features: {sorted(missing_features)}")
    if unknown_features:
        errors.append(f"unregistered columns present: {sorted(unknown_features)}")
    if errors:
        raise SchemaValidationError("; ".join(errors))
