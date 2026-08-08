"""Per-feature stats computed on every ingest and stored in the ingestion log."""
from __future__ import annotations

import pandas as pd


def compute_stats(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Returns {feature_name: {null_rate, mean, std, min, max, count}}.

    mean/std/min/max are omitted (None) for non-numeric features.
    """
    stats: dict = {}
    for name in feature_names:
        if name not in df.columns:
            continue
        col = df[name]
        entry = {
            "count": len(col),
            "null_rate": float(col.isna().mean()) if len(col) else 0.0,
        }
        is_statable = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
        if is_statable and col.notna().any():
            numeric = pd.to_numeric(col, errors="coerce")
            entry["mean"] = float(numeric.mean())
            entry["std"] = float(numeric.std()) if numeric.dropna().shape[0] > 1 else 0.0
            entry["min"] = float(numeric.min())
            entry["max"] = float(numeric.max())
        else:
            entry["mean"] = entry["std"] = entry["min"] = entry["max"] = None
        stats[name] = entry
    return stats
