"""ml-platform — a feature store, a drift monitor, and a worked example that uses both.

Three subsystems that were built as separate repos and deliberately designed to
interoperate, now in one package:

    ml_platform.features    offline/online feature store, point-in-time correct joins
    ml_platform.monitoring  data/concept/prediction drift detection and alerting
    ml_platform.fraud       real-time transaction scoring — the worked example

Before the merge these talked to each other through duck typing and optional
imports, because neither could depend on the other across repo boundaries:
`features.quality.drift_detector` documented itself as a stub delegating to "a
not-yet-built sibling project", and `fraud` wrapped `import ml_monitor` in a
try/except that silently disabled a whole monitoring backend when it was
missing. Both of those holes close here — see `ml_platform.pipeline`.
"""

from .monitoring import AlertConfig, DriftConfig, DriftReport, Monitor

__all__ = ["AlertConfig", "DriftConfig", "DriftReport", "Monitor"]

__version__ = "0.2.0"
