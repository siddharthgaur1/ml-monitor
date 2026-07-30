"""Smaller example: monitoring a label-free prediction stream, e.g. a RAG
pipeline's retrieval-similarity / confidence score. Shows Monitor working
with zero ground-truth labels (only data + prediction drift are available).

Run: python examples/rag_monitor_example.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_monitor import AlertConfig, DriftConfig, Monitor


def main():
    rng = np.random.default_rng(0)

    # Reference: query length + retrieved-doc similarity score, from a healthy period.
    reference = pd.DataFrame({
        "query_length_tokens": rng.normal(12, 3, 1000).clip(1),
        "top_k_similarity": rng.beta(6, 2, 1000),  # high similarity, healthy retrieval
    })
    reference_scores = rng.beta(6, 2, 1000)  # "confidence" of the generated answer

    monitor = Monitor(
        reference_data=reference,
        reference_predictions=reference_scores,
        config=DriftConfig(window_size=300),
        alerts=AlertConfig(severity_threshold="INFO"),
        db_path="examples/rag_demo.db",
    )

    print("Logging 300 healthy RAG queries (no labels available)...")
    for _ in range(300):
        row = {
            "query_length_tokens": float(rng.normal(12, 3)),
            "top_k_similarity": float(rng.beta(6, 2)),
        }
        confidence = float(rng.beta(6, 2))
        monitor.log(features=row, prediction=confidence)  # label=None: no ground truth in RAG

    report = monitor.drift_report()
    print(f"Drifted features: {report.drifted_features()}")
    print(f"Concept drift: {report.concept_drift}  (None expected -- no labels logged)")

    print("\nSimulating a corpus/index change: retrieval similarity degrades...")
    for _ in range(300):
        row = {
            "query_length_tokens": float(rng.normal(12, 3)),
            "top_k_similarity": float(rng.beta(2, 6)),  # much worse retrieval
        }
        confidence = float(rng.beta(2, 6))
        monitor.log(features=row, prediction=confidence)

    report2 = monitor.drift_report()
    print(f"Drifted features: {report2.drifted_features()}")
    print(f"Prediction drift detected: {report2.prediction_drift['is_drifted']} "
          f"(score={report2.prediction_drift['drift_score']:.4f})")


if __name__ == "__main__":
    main()
