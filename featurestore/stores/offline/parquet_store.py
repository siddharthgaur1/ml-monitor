"""Offline store backed by partitioned parquet files.

Layout: <base_path>/<feature_group>/<YYYY-MM-DD>.parquet, partitioned by the
UTC date of event_timestamp. Point-in-time joins reuse the shared
core.pit_join implementation (pandas.merge_asof) via OfflineStore's default
get_historical_features.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from featurestore.stores.offline.base import OfflineStore

REQUIRED_COLS = {"entity_id", "event_timestamp"}


class ParquetOfflineStore(OfflineStore):
    def __init__(self, path: str = "./data/offline"):
        self.base_path = Path(path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _group_dir(self, group_name: str) -> Path:
        d = self.base_path / group_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, group_name: str, df: pd.DataFrame) -> int:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"data missing required columns: {missing}")

        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        group_dir = self._group_dir(group_name)

        for date, part in df.groupby(df["event_timestamp"].dt.date):
            part_path = group_dir / f"{date}.parquet"
            if part_path.exists():
                part = pd.concat([pd.read_parquet(part_path), part], ignore_index=True)
            part.to_parquet(part_path, index=False)

        return len(df)

    def read(self, group_name: str) -> pd.DataFrame:
        group_dir = self.base_path / group_name
        if not group_dir.exists():
            return pd.DataFrame(columns=["entity_id", "event_timestamp"])
        parts = sorted(group_dir.glob("*.parquet"))
        if not parts:
            return pd.DataFrame(columns=["entity_id", "event_timestamp"])
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    def delete_group(self, group_name: str) -> None:
        group_dir = self.base_path / group_name
        if group_dir.exists():
            shutil.rmtree(group_dir)
