"""Offline store backed by DuckDB, storing each feature group as a table.

Point-in-time join is expressed directly in SQL using DuckDB's Postgres-
compatible `SELECT DISTINCT ON (...) ... ORDER BY ...` construct: for each
(entity_id, label event_timestamp), pick the single latest feature row with
event_timestamp <= the label timestamp. Same anti-leakage guarantee as the
pandas merge_asof implementation, expressed as SQL instead.

Requires the optional `duckdb` dependency: pip install ml-platform[duckdb]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_platform.features.core.pit_join import PITJoinResult
from ml_platform.features.stores.offline.base import REQUIRED_COLS, OfflineStore

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without extra installed
    duckdb = None


class DuckDBOfflineStore(OfflineStore):
    def __init__(self, path: str = "./data/offline.duckdb"):
        if duckdb is None:
            raise ImportError(
                "duckdb is not installed. Install with: pip install ml-platform[duckdb]"
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(path)

    def _table(self, group_name: str) -> str:
        return f'"{group_name}"'

    def write(self, group_name: str, df: pd.DataFrame) -> int:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"data missing required columns: {missing}")
        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        table = self._table(group_name)
        self.conn.register("_incoming", df)
        exists = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [group_name],
        ).fetchone()[0]
        if exists:
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
        else:
            self.conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
        self.conn.unregister("_incoming")
        return len(df)

    def read(self, group_name: str) -> pd.DataFrame:
        table = self._table(group_name)
        exists = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [group_name],
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame(columns=["entity_id", "event_timestamp"])
        return self.conn.execute(f"SELECT * FROM {table}").fetchdf()

    def delete_group(self, group_name: str) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self._table(group_name)}")

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_groups: list[str],
        fill_strategy: str = "none",
    ) -> PITJoinResult:
        """SQL asof join per feature group, then merge results in pandas.

        For each group: SELECT DISTINCT ON (e.entity_id, e.event_timestamp)
        e.entity_id, e.event_timestamp, f.* FROM entity_df e LEFT JOIN
        <group> f ON f.entity_id = e.entity_id AND f.event_timestamp <=
        e.event_timestamp ORDER BY e.entity_id, e.event_timestamp,
        f.event_timestamp DESC -- picks latest known-at-label-time row.
        """
        entity_df = entity_df.copy()
        entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
        self.conn.register("_entity_df", entity_df)

        result = entity_df.sort_values("event_timestamp").reset_index(drop=True)
        uncovered: dict[str, list] = {}

        for group_name in feature_groups:
            table = self._table(group_name)
            feat_cols = [
                c for c in self.read(group_name).columns
                if c not in ("entity_id", "event_timestamp")
            ]
            select_feats = ", ".join(f'f."{c}" AS "{group_name}__{c}"' for c in feat_cols) or "NULL"
            sql = f"""
                SELECT DISTINCT ON (e.entity_id, e.event_timestamp)
                    e.entity_id, e.event_timestamp, {select_feats}
                FROM _entity_df e
                LEFT JOIN {table} f
                    ON f.entity_id = e.entity_id
                    AND f.event_timestamp <= e.event_timestamp
                ORDER BY e.entity_id, e.event_timestamp, f.event_timestamp DESC
            """
            joined = self.conn.execute(sql).fetchdf()

            new_cols = [f"{group_name}__{c}" for c in feat_cols]
            no_match = joined[new_cols].isna().all(axis=1) if new_cols else pd.Series([True] * len(joined))
            uncovered[group_name] = sorted(joined.loc[no_match, "entity_id"].unique().tolist())

            if fill_strategy == "zero" and new_cols:
                joined[new_cols] = joined[new_cols].fillna(0)

            result = result.merge(joined, on=["entity_id", "event_timestamp"], how="left")

        self.conn.unregister("_entity_df")
        return PITJoinResult(training_df=result, uncovered_entity_ids=uncovered)
