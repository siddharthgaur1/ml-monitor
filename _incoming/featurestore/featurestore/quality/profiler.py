"""Thin wrapper around registry.stats used at ingest time.

Kept as a separate module (per spec) so quality concerns and registry
storage concerns stay decoupled, even though today profiling is just a
call-through to registry.stats.compute_stats.
"""
from __future__ import annotations

import pandas as pd

from featurestore.registry.stats import compute_stats


def profile(df: pd.DataFrame, feature_names: list[str]) -> dict:
    return compute_stats(df, feature_names)
