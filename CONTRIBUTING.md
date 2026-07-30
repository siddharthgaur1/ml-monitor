# Contributing

## Adding a new offline store backend

Subclass `featurestore.stores.offline.base.OfflineStore` and implement:

- `write(group_name: str, df: pd.DataFrame) -> int` — append rows. `df` is
  guaranteed to have `entity_id`, `event_timestamp`, plus feature columns.
- `read(group_name: str) -> pd.DataFrame` — return all historical rows.
- `delete_group(group_name: str) -> None`

`get_historical_features` has a default implementation on the base class
built from `read()` + `core.pit_join.point_in_time_join` (pandas
`merge_asof`). You only need to override it if your backend can express the
point-in-time join more efficiently in its own query language (see
`DuckDBOfflineStore` for an example using SQL `DISTINCT ON`).

Then wire it into `core/feature_store.py`'s `_build_offline_store()` and add
a `type:` value in `featurestore.yaml`.

## Adding a new online store backend

Subclass `featurestore.stores.online.base.OnlineStore` and implement:

- `write_batch(group_name, rows, ttl_hours=None) -> int` — `rows` is
  `[{"entity_id": ..., "features": {name: value}}, ...]`.
- `read_batch(group_name, entity_ids) -> dict[entity_id, dict[feature_name, value]]`
  — only return found, non-expired entries.

Wire it into `_build_online_store()` in `core/feature_store.py`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

If your change touches `core/pit_join.py` or an offline store's
`get_historical_features`, add or extend a leakage test in
`tests/test_pit_join.py` — assert no feature row with `event_timestamp`
later than a label's `event_timestamp` is ever selected.

## Style

- Keep backends dependency-optional where reasonable (see `redis_store.py`
  and `duckdb_store.py`: import failures raise a clear `ImportError` at
  *instantiation*, not at module import, so the core package works without
  the extra installed).
- No new abstractions beyond `OfflineStore`/`OnlineStore` unless a second
  concrete need demands it.
