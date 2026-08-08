"""Data quality gate applied on every ingest: null rate and dtype checks.

Distinct from ml_platform.features.registry.validator, which checks structural
schema (right columns present). This module checks the *data itself*.
"""
from __future__ import annotations

import pandas as pd

NULL_RATE_THRESHOLD = 0.20

_DTYPE_CHECKS = {
    "int32": pd.api.types.is_integer_dtype,
    "int64": pd.api.types.is_integer_dtype,
    "float32": pd.api.types.is_float_dtype,
    "float64": pd.api.types.is_float_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "string": pd.api.types.is_object_dtype,
}


class DataQualityError(ValueError):
    pass


def check_quality(df: pd.DataFrame, features: list[dict]) -> None:
    """Raises DataQualityError if any feature has >20% nulls or a dtype
    mismatch against its registered dtype. `features`: list of
    {"name", "dtype"} as stored in the registry.
    """
    problems = []
    for f in features:
        name, expected_dtype = f["name"], f["dtype"]
        if name not in df.columns:
            continue  # missing-column is a schema concern, not a quality one
        col = df[name]

        null_rate = float(col.isna().mean()) if len(col) else 0.0
        if null_rate > NULL_RATE_THRESHOLD:
            problems.append(f"'{name}': null rate {null_rate:.0%} exceeds {NULL_RATE_THRESHOLD:.0%}")

        checker = _DTYPE_CHECKS.get(expected_dtype)
        non_null = col.dropna()
        if checker is not None and len(non_null) and not checker(non_null):
            problems.append(f"'{name}': dtype {col.dtype} does not match registered dtype {expected_dtype}")

    if problems:
        raise DataQualityError("; ".join(problems))
