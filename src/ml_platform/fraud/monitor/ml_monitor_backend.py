"""Second monitoring backend, using `ml_platform.monitoring`, run alongside the
Evidently-based backend in `monitor/main.py`.

Two independent drift implementations over the same data is deliberate: they
disagree occasionally, and a disagreement is a signal worth seeing rather than
a bug to hide.

This used to guard `import ml_monitor` in a try/except and return None when it
was absent, because the two lived in separate repos and ml-monitor was never on
PyPI. In practice that meant a whole monitoring backend could silently do
nothing. Inside one package it is always available, so the guard is gone and a
failure here is a real failure.
"""
from __future__ import annotations

import pandas as pd
import structlog

from ml_platform.monitoring import DriftConfig, Monitor

log = structlog.get_logger()

# Kept as a module-level constant because monitor/main.py and the API's
# /monitoring endpoint branch on it. It is now always True; the name survives
# so those call sites need no change.
ML_MONITOR_AVAILABLE = True


def run_once(reference: pd.DataFrame, current: pd.DataFrame) -> dict | None:
    """Compare `current` (recent scored transactions) against `reference`
    (the training-time sample) using ml-monitor's KS/PSI detectors on
    `amount`/`is_fraud`, and prediction drift on `fraud_score`.

    Builds a fresh in-memory Monitor per call rather than keeping one alive
    across runs -- this backend is invoked on the same fixed interval as the
    Evidently one, each time against a fresh `current` snapshot, so there's
    no rolling state to preserve between calls.
    """
    if not ML_MONITOR_AVAILABLE:
        return None

    monitor = Monitor(
        reference_data=reference[["amount", "is_fraud"]],
        reference_predictions=reference["fraud_score"].to_numpy(),
        config=DriftConfig(window_size=max(len(current), 1)),
        db_path=":memory:",
    )
    for row in current.itertuples(index=False):
        monitor.log(features={"amount": row.amount, "is_fraud": row.is_fraud}, prediction=row.fraud_score)

    report = monitor.drift_report()
    log.info(
        "ml_monitor_report_written",
        n_samples=report.n_samples,
        drifted_features=report.drifted_features(),
    )
    return report.to_dict()
