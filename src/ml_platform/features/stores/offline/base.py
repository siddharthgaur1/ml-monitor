"""Interface every offline store backend must implement.

To add a new backend: subclass OfflineStore and implement `write`, `read`,
and `delete_group`. `get_historical_features` has a default implementation
built on `read()` + core.pit_join, so most backends only need write/read/
delete_group. See CONTRIBUTING.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ml_platform.features.core.pit_join import PITJoinResult, point_in_time_join


class OfflineStore(ABC):
    @abstractmethod
    def write(self, group_name: str, df: pd.DataFrame) -> int:
        """Append rows for a feature group. Returns rows written."""

    @abstractmethod
    def read(self, group_name: str) -> pd.DataFrame:
        """Return all historical rows for a feature group."""

    @abstractmethod
    def delete_group(self, group_name: str) -> None:
        """Remove all stored data for a feature group."""

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: list[str],
        fill_strategy: str = "none",
    ) -> PITJoinResult:
        frames = {name: self.read(name) for name in feature_groups}
        return point_in_time_join(entity_df, frames, fill_strategy=fill_strategy)
