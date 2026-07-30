"""FastAPI REST API around a Monitor instance. Optional extra: pip install ml-monitor[api]."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the 'api' extra: pip install ml-monitor[api]") from e


class LogRequest(BaseModel):
    features: Dict[str, Any]
    prediction: Any
    label: Optional[Any] = None


def create_app(monitor) -> "FastAPI":
    app = FastAPI(title="ml-monitor")

    @app.post("/log")
    def log(req: LogRequest):
        monitor.log(features=req.features, prediction=req.prediction, label=req.label)
        return {"status": "ok"}

    @app.get("/drift/latest")
    def drift_latest():
        report = monitor.drift_report()
        return report.to_dict()

    @app.get("/drift/history")
    def drift_history(hours: int = 24):
        from .store.aggregator import aggregate

        return aggregate(monitor.store, granularity="hourly", hours=hours)

    @app.get("/drift/report")
    def drift_report_json():
        return monitor.drift_report().to_dict()

    @app.get("/predictions/stats")
    def predictions_stats():
        return {
            "window_size": len(monitor.window),
            "total_logged": monitor.store.count(),
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "time": time.time()}

    @app.post("/reference/update")
    def reference_update(body: Dict[str, Any]):
        import pandas as pd

        monitor.update_reference(pd.DataFrame(body.get("reference_data", [])),
                                  body.get("reference_predictions"))
        return {"status": "ok"}

    @app.get("/alerts/history")
    def alerts_history(hours: int = 24):
        since = time.time() - hours * 3600
        return monitor.store.recent_alerts(since=since)

    return app
