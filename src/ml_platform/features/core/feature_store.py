"""The public SDK: Entity, Feature, FeatureGroup + FeatureStore facade.

FeatureStore wires together the registry, offline store, online store, and
quality checks behind the small API shown in the README quickstart.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from ml_platform.features.core.config import FeatureStoreConfig
from ml_platform.features.core.pit_join import PITJoinResult
from ml_platform.features.quality.profiler import profile
from ml_platform.features.quality.validator import check_quality
from ml_platform.features.registry.registry import FeatureRegistry
from ml_platform.features.registry.validator import validate_schema


@dataclasses.dataclass
class Entity:
    name: str
    dtype: str = "int64"
    description: str = ""


@dataclasses.dataclass
class Feature:
    name: str
    dtype: str = "float32"
    description: str = ""


@dataclasses.dataclass
class FeatureGroup:
    name: str
    entity: Entity
    features: list[Feature]
    ttl_hours: float | None = 24
    online: bool = True
    offline: bool = True
    tags: list[str] = dataclasses.field(default_factory=list)


def _build_offline_store(cfg):
    if cfg.type == "parquet":
        from ml_platform.features.stores.offline.parquet_store import ParquetOfflineStore
        return ParquetOfflineStore(path=cfg.get("path", "./data/offline"))
    if cfg.type == "duckdb":
        from ml_platform.features.stores.offline.duckdb_store import DuckDBOfflineStore
        return DuckDBOfflineStore(path=cfg.get("path", "./data/offline.duckdb"))
    raise ValueError(f"unknown offline_store.type: {cfg.type}")


def _build_online_store(cfg):
    if cfg.type == "sqlite":
        from ml_platform.features.stores.online.sqlite_store import SQLiteOnlineStore
        return SQLiteOnlineStore(path=cfg.get("path", "./data/online.db"), ttl_hours=cfg.get("ttl_hours", 24))
    if cfg.type == "redis":
        from ml_platform.features.stores.online.redis_store import RedisOnlineStore
        return RedisOnlineStore(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 6379),
            ttl_hours=cfg.get("ttl_hours", 24),
        )
    raise ValueError(f"unknown online_store.type: {cfg.type}")


class FeatureStore:
    def __init__(self, config: str | FeatureStoreConfig = "ml_platform.yaml"):
        self.config = config if isinstance(config, FeatureStoreConfig) else FeatureStoreConfig.load(config)
        self.registry = FeatureRegistry(path=self.config.registry.get("path", "./data/registry.db"))
        self.offline_store = _build_offline_store(self.config.offline_store)
        self.online_store = _build_online_store(self.config.online_store)

    # -- registration ---------------------------------------------------
    def register(self, feature_group: FeatureGroup) -> None:
        self.registry.register_entity(
            feature_group.entity.name, feature_group.entity.dtype, feature_group.entity.description
        )
        self.registry.register_feature_group(
            name=feature_group.name,
            entity_name=feature_group.entity.name,
            features=[dataclasses.asdict(f) for f in feature_group.features],
            ttl_hours=feature_group.ttl_hours,
            online=feature_group.online,
            offline=feature_group.offline,
            tags=feature_group.tags,
        )

    # -- ingestion --------------------------------------------------------
    def ingest(self, feature_group: str, data: pd.DataFrame) -> dict:
        """data: DataFrame with entity_id, event_timestamp, and feature columns."""
        group = self.registry.get_feature_group(feature_group)
        if group is None:
            raise ValueError(f"feature group not registered: {feature_group}")

        validate_schema(data, group)
        check_quality(data, group["features"])

        feature_names = [f["name"] for f in group["features"]]
        stats = profile(data, feature_names)

        rows_written = 0
        if group["offline"]:
            rows_written = self.offline_store.write(feature_group, data)

        if group["online"]:
            latest = _latest_per_entity(data)
            rows = [
                {"entity_id": row["entity_id"], "features": {n: row[n] for n in feature_names}}
                for row in latest.to_dict("records")
            ]
            self.online_store.write_batch(feature_group, rows, ttl_hours=group["ttl_hours"])

        self.registry.log_ingestion(feature_group, len(data), stats)
        return {"rows_written": rows_written or len(data), "stats": stats}

    # -- serving ----------------------------------------------------------
    def get_online_features(self, feature_group: str, entity_ids: list) -> dict[Any, dict]:
        if self.registry.get_feature_group(feature_group) is None:
            raise ValueError(f"feature group not registered: {feature_group}")
        return self.online_store.read_batch(feature_group, entity_ids)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: list[str],
        fill_strategy: str = "none",
    ) -> pd.DataFrame:
        for name in feature_groups:
            if self.registry.get_feature_group(name) is None:
                raise ValueError(f"feature group not registered: {name}")
        result: PITJoinResult = self.offline_store.get_historical_features(
            entity_df, feature_groups, fill_strategy=fill_strategy
        )
        for group_name, missing in result.uncovered_entity_ids.items():
            if missing:
                import warnings
                warnings.warn(
                    f"'{group_name}': no feature coverage for entity_ids {missing}", stacklevel=2
                )
        return result.training_df

    # -- discovery ----------------------------------------------------------
    def list_feature_groups(self) -> list[dict]:
        return self.registry.list_feature_groups()

    def search_features(self, keyword: str) -> list[dict]:
        return self.registry.search_features(keyword)

    def get_feature_stats(self, feature_group: str) -> dict | None:
        return self.registry.get_latest_stats(feature_group)


def _latest_per_entity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df.sort_values("event_timestamp").groupby("entity_id", as_index=False).tail(1)
