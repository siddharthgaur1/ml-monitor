# ml-monitor

A standalone, model-agnostic ML monitoring system that detects **data drift**,
**concept drift**, and **prediction drift** for any sklearn/XGBoost/LightGBM
model in production. Drop it in front of a reference dataset and a live
prediction stream, get drift scores, alerts, and a dashboard.

Project #8 in a portfolio series.

## Install

```bash
pip install -e .                     # core (numpy/pandas/scipy/click/rich only)
pip install -e ".[sklearn]"          # + scikit-learn, for the examples
pip install -e ".[api]"              # + FastAPI/uvicorn for the REST API
pip install -e ".[dashboard]"        # + Streamlit/plotly for the dashboard
pip install -e ".[xgboost,lightgbm,torch]"  # optional model adapters
pip install -e ".[all,dev]"          # everything, plus pytest
```

Core detectors (`ml_monitor/detectors/*`, `ml_monitor/core/*`) depend only on
`numpy`/`pandas`/`scipy` — you can use them standalone without FastAPI,
Streamlit, or any ML framework installed.

## Quickstart

```python
import os
from ml_monitor import Monitor, DriftConfig, AlertConfig

monitor = Monitor(
    model=loaded_xgb_model,
    reference_data=X_train,
    reference_predictions=y_train_pred,
    config=DriftConfig(numerical_method="ks", categorical_method="chi2",
                        drift_threshold=0.05, window_size=500),
    alerts=AlertConfig(email=None, webhook_url=os.getenv("SLACK_WEBHOOK"),
                        severity_threshold="WARNING"),
)

# hot path: cheap, just appends to an in-memory window + SQLite row
monitor.log(features=X_live_row, prediction=y_pred, label=y_true_if_known)

# drift math runs here, lazily, not inside log()
report = monitor.drift_report()
report.to_html("reports/drift_report.html")

monitor.serve(port=8080)  # REST API (dashboard runs as a separate process, see below)
```

Run `examples/fraud_detection_example.py` for a full working demo (trains a
`RandomForestClassifier` on synthetic fraud-like data, then shows the drift
score jump from ~0.07 to ~0.92 after injecting drift into 1000 samples).
`examples/rag_monitor_example.py` shows the same flow with **zero labels**
(data + prediction drift only).

## Why `log()` is fast

`monitor.log()` never runs drift detectors inline — it only appends to a
fixed-size in-memory window (`WindowManager`, O(1) `deque`) and writes one row
to SQLite. Drift detectors (KS test, PSI, chi-squared, Spearman correlation,
ADWIN/DDM/Page-Hinkley) run lazily, the first time you call
`drift_report()` — which the REST API and dashboard call on every poll. This
is the simpler of the two options in the spec (vs. a background thread) and
avoids any locking between the hot logging path and drift computation.

## Drift types

- **Data drift** (`detectors/data_drift.py`) — is the distribution of an
  input feature shifting? KS test (numerical), chi-squared (categorical),
  PSI (either, quantile/category binned), Wasserstein distance (numerical).
- **Prediction drift** (`detectors/prediction_drift.py`) — is the model's
  output distribution shifting? PSI over prediction-score bins + a
  mean-shift-percentage alert. Works with **zero labels**.
- **Concept drift** (`detectors/concept_drift.py`) — is the relationship
  between inputs and the true outcome changing (model degrading)? Runs
  ADWIN, Page-Hinkley, or DDM over the stream of 0/1 error indicators.
  **Requires ground-truth labels** — silently skipped (returns `None`, never
  raises) when none have been logged.
- **Correlation drift** (`detectors/correlation_drift.py`) — has the
  Spearman correlation structure between features changed (e.g. two
  previously-correlated features have decoupled)?

Severity bands (shared across detectors, applied to a drift_score normalized
to roughly `[0, inf)`): `INFO` 0.1–0.2, `WARNING` 0.2–0.3, `CRITICAL` > 0.3.

## Framework integration

**sklearn**: any fitted estimator works directly — `Monitor(model=clf, ...)`
calls `clf.predict()`/`predict_proba()` as-is. `ml_monitor/adapters/sklearn.py`
extracts `feature_names_in_` when present.

**XGBoost**: pass either a `Booster` or the sklearn-API wrapper
(`XGBClassifier`/`XGBRegressor`) — `ml_monitor/adapters/xgboost.py` handles
`DMatrix` construction for raw boosters automatically. Requires
`pip install ml-monitor[xgboost]`.

Adapters for LightGBM, PyTorch (thin, `torch` imported lazily and only
required if you use it), and models served behind an HTTP API are also
provided (`ml_monitor/adapters/`); all are optional thin wrappers around
`predict()` + feature-name extraction, not required to use `Monitor` directly
with a raw model object.

## Store, alerts, dashboard, API

- **Store** (`store/sqlite_store.py`): every prediction logged goes to
  SQLite (timestamp, features JSON, prediction, nullable label, drift
  scores). `store/aggregator.py` buckets by hour/day. `purge_old()` enforces
  a configurable retention window (default 30 days).
- **Alerts** (`alerts/`): `logger.py` (structured JSON, always on),
  `slack.py`/`webhook.py` (POST formatted drift tables), `deduplicator.py`
  (won't re-fire the same feature+drift-type within 1hr by default).
- **REST API** (`ml_monitor/api.py`, FastAPI, `pip install ml-monitor[api]`):
  `POST /log`, `GET /drift/latest`, `GET /drift/history?hours=24`,
  `GET /drift/report`, `GET /predictions/stats`, `GET /health`,
  `POST /reference/update`, `GET /alerts/history`.
- **Dashboard** (`ml_monitor/dashboard.py`, Streamlit,
  `pip install ml-monitor[dashboard]`): run separately with
  `streamlit run ml_monitor/dashboard.py -- --db ml_monitor.db --reference reference.csv`.
  Shows status badge, feature drift table, top-5 drifted bar chart,
  prediction distribution, drift-over-time line chart, concept drift panel,
  reference-vs-current correlation heatmaps, and the alerts log. Auto-
  refreshes every 60s via `st.rerun()`.

## CLI

```bash
ml-monitor start --model model.pkl --reference reference.csv --port 8080
ml-monitor report --reference reference.csv --format html --output reports/drift.html
ml-monitor simulate --dataset data/reference.csv --drift-type gradual --feature amount --shift 2.0
ml-monitor status --db ml_monitor.db
```

## Drift simulator

`simulate/drift_simulator.py` injects `gradual`, `sudden`, or `seasonal`
drift into any numerical column of a CSV — used by the fraud example and
available standalone:

```bash
python simulate/drift_simulator.py --dataset data/reference.csv --drift-type gradual --feature amount --shift 2.0
```

## Docker

```bash
docker compose up --build
```

Minimal single-container setup: mount your model pickle at `/models/model.pkl`
and reference CSV at `/reference/reference.csv`, SQLite persists to a named
volume. Run the dashboard as a second process inside the same container:
`docker compose exec ml-monitor streamlit run ml_monitor/dashboard.py -- --db /data/ml_monitor.db`.

## Tests

```bash
pytest -q                                          # 25 tests
pytest -q --cov=ml_monitor --cov-report=term-missing
```

Detector tests assert drift **is** detected on synthetically shifted
distributions and **is not** falsely flagged on identical ones (KS, PSI,
chi-squared, Wasserstein, ADWIN, Page-Hinkley, DDM, Spearman correlation
break). `core/` + `detectors/` combined coverage is ~86%.

## Integration with stream-fraud-detector

`stream-fraud-detector` (a sibling portfolio project) is the intended
plug-in point for real-time scoring: its scoring loop would call
`monitor.log(features=transaction_features, prediction=fraud_score,
label=confirmed_fraud_if_known)` right after each prediction, giving
`ml-monitor` a live view of the feature and score distributions it's
scoring against. This repo does not import from or depend on
`stream-fraud-detector` — the integration point is just the `Monitor.log()`
call documented above, wired in from that project's side.

## Deliberate simplifications

- **Drift computed lazily, not on a background thread** — simpler, no
  locking between the hot `log()` path and drift math; the tradeoff is that
  `drift_report()` itself does the (still fast, single-window) computation
  synchronously when called.
- **SQLite behind a single lock** — fine for the write volumes a monitoring
  sidecar sees; swap for a connection pool/Postgres if this needs to serve
  many concurrent writers.
- **PyTorch/XGBoost/LightGBM adapters are thin wrappers** — `predict()`
  passthrough + feature-name extraction, no batching/device management
  beyond what's needed to run a forward pass.
- **HTML report is hand-built (no template engine)** — the report is one
  table plus two summary paragraphs; not worth a Jinja dependency.
