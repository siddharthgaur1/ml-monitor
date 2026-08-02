"""Monitor: the drop-in monitoring layer wired around a trained model."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..alerts.deduplicator import Deduplicator
from ..alerts.logger import log_alert
from ..alerts.slack import send_slack_alert
from ..alerts.webhook import send_webhook
from ..detectors.concept_drift import detect_concept_drift
from ..detectors.correlation_drift import detect_correlation_drift
from ..detectors.data_drift import detect_data_drift
from ..detectors.prediction_drift import detect_prediction_drift
from ..store.sqlite_store import SQLiteStore
from .report import DriftReport
from .window_manager import LogRecord, WindowManager

_SEVERITY_RANK = {"INFO": 1, "WARNING": 2, "CRITICAL": 3}


@dataclass
class DriftConfig:
    numerical_method: str = "ks"          # "ks" | "wasserstein" | "psi"
    categorical_method: str = "chi2"      # "chi2" | "psi"
    drift_threshold: float = 0.05
    window_size: int = 500
    concept_drift_method: str = "ddm"     # "ddm" | "adwin" | "page_hinkley"
    correlation_change_threshold: float = 0.3


@dataclass
class AlertConfig:
    email: str | None = None
    webhook_url: str | None = None
    slack_webhook_url: str | None = None
    severity_threshold: str = "WARNING"
    dedup_cooldown_seconds: float = 3600.0


class Monitor:
    """Point at a reference dataset + live prediction stream, get drift
    scores, alerts, and degradation tracking over time.

    Design note: `log()` never runs drift math inline -- it only appends to
    the in-memory window and writes a row to SQLite (both O(1)-ish). Drift
    detectors run lazily, the first time `drift_report()` (or the API/
    dashboard, which call it on a poll) is invoked. This keeps the hot path
    fast without needing a background thread + lock dance.
    """

    def __init__(
        self,
        model: Any = None,
        reference_data: pd.DataFrame | None = None,
        reference_predictions: np.ndarray | None = None,
        config: DriftConfig | None = None,
        alerts: AlertConfig | None = None,
        db_path: str = "ml_monitor.db",
        retention_days: int = 30,
    ):
        self.model = model
        self.reference_data = pd.DataFrame(reference_data) if reference_data is not None else None
        self.reference_predictions = (
            np.asarray(reference_predictions) if reference_predictions is not None else None
        )
        self.config = config or DriftConfig()
        self.alert_config = alerts or AlertConfig()

        self.window = WindowManager(window_size=self.config.window_size)
        self.store = SQLiteStore(db_path=db_path, retention_days=retention_days)
        self.deduplicator = Deduplicator(cooldown_seconds=self.alert_config.dedup_cooldown_seconds)

        self._last_report: DriftReport | None = None

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def log(self, features, prediction: Any, label: Any | None = None) -> None:
        """Fast: no drift computation happens here."""
        if isinstance(features, pd.Series):
            features = features.to_dict()
        elif isinstance(features, dict):
            pass
        else:
            # array-like row -> pair with reference columns if we have them
            arr = np.asarray(features).ravel()
            cols = list(self.reference_data.columns) if self.reference_data is not None else [
                f"f{i}" for i in range(len(arr))
            ]
            features = {c: v for c, v in zip(cols, arr)}

        record = LogRecord(timestamp=time.time(), features=features, prediction=prediction, label=label)
        self.window.add(record)
        self.store.log_prediction(features=features, prediction=prediction, label=label)

    # ------------------------------------------------------------------
    # Lazy drift computation
    # ------------------------------------------------------------------
    def drift_report(self) -> DriftReport:
        records = self.window.records()
        report = DriftReport(n_samples=len(records))

        if not records:
            self._last_report = report
            return report

        current_df = self.window.features_frame()
        current_preds = self.window.predictions()

        if self.reference_data is not None and not current_df.empty:
            common_cols = [c for c in self.reference_data.columns if c in current_df.columns]
            if common_cols:
                report.data_drift = detect_data_drift(
                    self.reference_data[common_cols],
                    current_df[common_cols],
                    numerical_method=self.config.numerical_method,
                    categorical_method=self.config.categorical_method,
                    drift_threshold=self.config.drift_threshold,
                )
                report.correlation_drift = detect_correlation_drift(
                    self.reference_data[common_cols],
                    current_df[common_cols],
                    change_threshold=self.config.correlation_change_threshold,
                )

        if self.reference_predictions is not None and len(current_preds) > 0:
            try:
                report.prediction_drift = detect_prediction_drift(
                    self.reference_predictions, current_preds
                )
            except (TypeError, ValueError):
                report.prediction_drift = None

        labels, preds = self.window.labels_and_predictions()
        if labels:
            errors = [0 if float(l) == float(p) else 1 for l, p in zip(labels, preds)]
            report.concept_drift = detect_concept_drift(errors, method=self.config.concept_drift_method)

        self._last_report = report
        self._fire_alerts(report)
        return report

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    def _fire_alerts(self, report: DriftReport) -> None:
        threshold_rank = _SEVERITY_RANK.get(self.alert_config.severity_threshold, 2)
        due = []
        for feature, result in report.data_drift.items():
            severity = result.get("severity")
            if not severity or not result.get("is_drifted"):
                continue
            if _SEVERITY_RANK.get(severity, 0) < threshold_rank:
                continue
            if not self.deduplicator.should_alert("data_drift", feature):
                continue
            message = f"Data drift detected on '{feature}' (score={result['drift_score']:.4f}, severity={severity})"
            log_alert(severity, message, "data_drift", feature=feature, drift_score=result["drift_score"])
            self.store.log_alert(severity, message, "data_drift", feature=feature)
            due.append({"feature": feature, "severity": severity, "drift_score": result["drift_score"]})

        if report.prediction_drift and report.prediction_drift.get("is_drifted"):
            severity = report.prediction_drift.get("severity") or "WARNING"
            if _SEVERITY_RANK.get(severity, 0) >= threshold_rank and self.deduplicator.should_alert(
                "prediction_drift", None
            ):
                message = f"Prediction drift detected (score={report.prediction_drift['drift_score']:.4f})"
                log_alert(severity, message, "prediction_drift", drift_score=report.prediction_drift["drift_score"])
                self.store.log_alert(severity, message, "prediction_drift")
                due.append({"feature": "__prediction__", "severity": severity,
                             "drift_score": report.prediction_drift["drift_score"]})

        if (
            report.concept_drift
            and report.concept_drift.get("drift_detected")
            and self.deduplicator.should_alert("concept_drift", None)
        ):
            message = (
                f"Concept drift detected via {report.concept_drift['method']} "
                f"(baseline error={report.concept_drift['baseline_error_rate']:.3f}, "
                f"current error={report.concept_drift['current_error_rate']:.3f})"
            )
            log_alert("CRITICAL", message, "concept_drift")
            self.store.log_alert("CRITICAL", message, "concept_drift")
            due.append({"feature": "__concept__", "severity": "CRITICAL", "drift_score": 1.0})

        if due:
            if self.alert_config.webhook_url:
                send_webhook(self.alert_config.webhook_url, {"alerts": due})
            if self.alert_config.slack_webhook_url:
                send_slack_alert(self.alert_config.slack_webhook_url, due)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def serve(self, port: int = 8080, host: str = "0.0.0.0"):
        """Serve the REST API (dashboard is a separate `streamlit run dashboard.py` process)."""
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError("Install the 'api' extra: pip install ml-monitor[api]") from e
        from ..api import create_app

        app = create_app(self)
        uvicorn.run(app, host=host, port=port)

    def update_reference(self, reference_data: pd.DataFrame, reference_predictions: np.ndarray | None = None):
        self.reference_data = pd.DataFrame(reference_data)
        if reference_predictions is not None:
            self.reference_predictions = np.asarray(reference_predictions)
