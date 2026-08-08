"""Loads featurestore.yaml into a plain, typed-enough config object."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = {
    "offline_store": {"type": "parquet", "path": "./data/offline"},
    "online_store": {"type": "sqlite", "path": "./data/online.db", "ttl_hours": 24},
    "registry": {"type": "sqlite", "path": "./data/registry.db"},
}


@dataclasses.dataclass
class StoreConfig:
    type: str
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)


@dataclasses.dataclass
class FeatureStoreConfig:
    offline_store: StoreConfig
    online_store: StoreConfig
    registry: StoreConfig

    @classmethod
    def from_dict(cls, raw: dict) -> FeatureStoreConfig:
        merged = {**DEFAULT_CONFIG, **raw}
        return cls(
            offline_store=_section(merged, "offline_store"),
            online_store=_section(merged, "online_store"),
            registry=_section(merged, "registry"),
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureStoreConfig:
        path = Path(path)
        if not path.exists():
            return cls.from_dict({})
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)


def _section(merged: dict, name: str) -> StoreConfig:
    section = {**DEFAULT_CONFIG[name], **merged.get(name, {})}
    store_type = section.pop("type")
    return StoreConfig(type=store_type, options=section)
