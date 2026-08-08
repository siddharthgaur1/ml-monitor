"""Interface every online store backend must implement.

To add a new backend: subclass OnlineStore and implement `write_batch` and
`read_batch`. See CONTRIBUTING.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OnlineStore(ABC):
    @abstractmethod
    def write_batch(
        self,
        group_name: str,
        rows: list[dict[str, Any]],
        ttl_hours: float | None = None,
    ) -> int:
        """rows: [{"entity_id": ..., "features": {name: value, ...}}, ...]."""

    @abstractmethod
    def read_batch(self, group_name: str, entity_ids: list) -> dict[Any, dict]:
        """Returns {entity_id: {feature_name: value, ...}} for found, non-expired rows."""
