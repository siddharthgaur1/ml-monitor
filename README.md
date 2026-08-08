# ml-platform

A feature store, a drift monitor, and a real-time fraud-scoring service that runs through both.

## The problem this solves

Three things have to agree for production ML to work: the features you train on,
the features you serve, and the distribution you compare live traffic against.
When those live in separate codebases they drift apart silently — a feature
renamed on one side, a reference distribution sliced by hand on the other, and
nothing fails until the model is quietly wrong.

These three subsystems were originally three repositories, deliberately designed
to interoperate but unable to depend on each other. The seams show what that
cost:

- `featurestore` shipped drift detection as a **documented stub**, because the
  maths lived in a "not-yet-built sibling project". It was listed as a known
  limitation.
- `ml-monitor` **duck-typed** its feature-store adapter — matching
  `offline_store.read()` and `registry.get_feature_group()` structurally,
  documented as "does NOT import lite-featurestore".
- `stream-fraud-detector` wrapped `import ml_monitor` in a `try/except
  ImportError` that returned `None`. A whole monitoring backend could silently
  do nothing.

Merging them closes all three. The stub is a real call, the adapter is a real
import, and the try/except is gone.

## What integrates, concretely

`src/ml_platform/pipeline.py` is one call that crosses every subsystem boundary:

```
fraud.common.features.FEATURE_NAMES   the feature schema, defined once
    -> features.FeatureStore          registered, validated, ingested
    -> features.offline_store         read back as a reference distribution
    -> monitoring.Monitor             live traffic compared against it
    -> monitoring.DriftReport         per-feature verdict
```

The schema is **imported** from the fraud subsystem, not restated. Two separate
repos had to copy that list, and a copied feature schema is where
training/serving skew starts.

`tests/test_pipeline_end_to_end.py` asserts the path holds together. It could
not have been written before the merge, because no repo could import the other
two.

## Results

| | |
|---|---|
| Tests | 113 passing (25 monitoring + 23 features + 60 fraud + 5 new cross-subsystem) |
| Subsystem boundaries crossed by one test | 3 |
| Duplicate code eliminated by the merge | **~0 lines — see Limitations** |
| Fraud service throughput | 29.7 req/s, median 1400 ms, p95 2300 ms, p99 2700 ms |
| Fraud service bottleneck | identified: `uvicorn` running without `--workers` |

Every number above is reproducible from this repo. The throughput figures come
from the committed locustfile in `loadtest/` and were measured on a laptop, not
extrapolated.

## Setup

```bash
pip install -e ".[dev]"          # feature store + monitoring
pip install -e ".[dev,fraud]"    # adds the fraud subsystem's stack
pytest tests/ tests_features/ tests_fraud/ -q
```

Python 3.11+. The fraud extra pulls Kafka, Redis, Postgres and XGBoost clients;
its tests stub every external service, so nothing needs to be running.

## Project structure

```
src/ml_platform/
    features/      offline+online stores, registry, point-in-time correct joins
    monitoring/    data / concept / prediction drift, alerting, SQLite history
    fraud/         the worked example: ingest -> score -> persist -> monitor
    pipeline.py    the path that crosses all three
tests/             monitoring + the end-to-end pipeline test
tests_features/    feature store
tests_fraud/       fraud service
docs/absorbed/     the original READMEs, RUNBOOK and SECURITY of the merged repos
loadtest/          locustfile behind the throughput numbers above
deploy/fraud/      the docker-compose stack the fraud subsystem actually runs on
deploy/features/   feature-store container assets
```

Paths inside `deploy/fraud/` (Dockerfile, compose, Makefile) still reference the
absorbed repo's own layout and have not been re-pointed at `src/ml_platform/`.
They are committed because the README's "local docker-compose stack" claim would
otherwise be unbacked, not because `docker compose up` works unchanged today.

## Limitations

Carried forward from each absorbed repository, because a merge is exactly where
caveats get quietly lost.

**The fraud subsystem's data is synthetic.** Fraud is injected as odd-hour +
high-amount + unseen-device, which is precisely what the features are built to
detect. Its accuracy says very little about real fraud — every metric it
produces measures the pipeline, not the modelling.

**The fraud service is not deployed. There is no public endpoint.** It is a
local docker-compose stack. It is deploy-ready in the ways that matter —
non-root multi-stage image, API-key auth, per-key rate limiting, `/health` with
real 503 semantics, `/metrics`, structured logs, images published to GHCR — but
provisioned nowhere. The honest blocker is cost: managed Kafka is out of budget,
and the substitute would be HTTP-only scoring without the broker, which is a
smaller system than what runs locally.

**The feature store is SQLite/parquet only.** No Postgres registry backend. PIT
joins are `merge_asof`/DuckDB, not Spark-scale. `fill_strategy="last_known"` is
an alias of `"none"` — `merge_asof` already returns the latest value known
at-or-before the label time.

**The merge eliminated almost no duplicate code, and that is worth stating
plainly.** These three repos shared a naming skeleton but not implementations:
measured file-similarity across them was 0.01–0.21. The case for merging was
never deduplication — it was that the three had real interfaces to each other
that repo boundaries forced into duck typing and optional imports. If you are
looking for "N thousand lines removed", it is not here.

**Two drift implementations run over the same fraud data** — Evidently and
`ml_platform.monitoring` — and they occasionally disagree. That is deliberate;
a disagreement is a signal worth seeing.

**Inherited lint debt.** CI ignores `E501,B905,B904,E741`: pre-existing findings
in absorbed code, mostly from the feature store, which had no ruff config and
was never linted. They are ignored rather than fixed so CI reports regressions
instead of sitting permanently red — not because they were dropped.

**The drift detector has no multiple-comparison correction.** Each feature is
tested independently at α=0.05, so with 11 features a clean run flags at least
one about 4 times in 10. The end-to-end test asserts a shifted distribution
drifts *more* than an unshifted one rather than asserting zero false positives,
because the latter would pass or fail on the random seed.
