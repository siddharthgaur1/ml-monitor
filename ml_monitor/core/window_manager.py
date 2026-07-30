"""In-memory rolling window of the last N logged records."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class LogRecord:
    timestamp: float
    features: Dict[str, Any]
    prediction: Any
    label: Optional[Any] = None


class WindowManager:
    """Fixed-size FIFO window held in memory. Cheap O(1) append."""

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._records: Deque[LogRecord] = deque(maxlen=window_size)

    def add(self, record: LogRecord) -> None:
        self._records.append(record)

    def records(self) -> List[LogRecord]:
        return list(self._records)

    def features_frame(self):
        import pandas as pd

        return pd.DataFrame([r.features for r in self._records])

    def predictions(self):
        import numpy as np

        return np.array([r.prediction for r in self._records])

    def labels_and_predictions(self):
        """Returns (labels, predictions) for records that have a non-null label."""
        labels, preds = [], []
        for r in self._records:
            if r.label is not None:
                labels.append(r.label)
                preds.append(r.prediction)
        return labels, preds

    def __len__(self) -> int:
        return len(self._records)

    def is_full(self) -> bool:
        return len(self._records) >= self.window_size
