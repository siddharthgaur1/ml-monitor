"""End-to-end demo: train a small sklearn model on synthetic fraud-like data,
wire it up to Monitor, then feed it ~1000 injected-drift samples (via
simulate/drift_simulator.py) and show the drift signal appearing.

Run: python examples/fraud_detection_example.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_monitor import AlertConfig, DriftConfig, Monitor
from simulate.drift_simulator import inject_drift


def make_synthetic_fraud_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    amount = rng.gamma(shape=2.0, scale=40.0, size=n)
    account_age_days = rng.exponential(scale=400, size=n)
    n_transactions_24h = rng.poisson(lam=3, size=n)
    merchant_risk_score = rng.beta(2, 8, size=n)

    fraud_logit = (
        -4.0
        + 0.02 * amount
        - 0.002 * account_age_days
        + 0.4 * n_transactions_24h
        + 5.0 * merchant_risk_score
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.uniform(size=n) < fraud_prob).astype(int)

    df = pd.DataFrame({
        "amount": amount,
        "account_age_days": account_age_days,
        "n_transactions_24h": n_transactions_24h,
        "merchant_risk_score": merchant_risk_score,
    })
    return df, is_fraud


def main():
    print("=== ml-monitor: fraud detection drift demo ===\n")

    X, y = make_synthetic_fraud_data(n=5000, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
    model.fit(X_train, y_train)
    print(f"Trained RandomForestClassifier on {len(X_train)} rows, "
          f"test accuracy={model.score(X_test, y_test):.3f}\n")

    y_train_pred = model.predict_proba(X_train)[:, 1]

    monitor = Monitor(
        model=model,
        reference_data=X_train,
        reference_predictions=y_train_pred,
        config=DriftConfig(numerical_method="ks", categorical_method="chi2",
                            drift_threshold=0.05, window_size=500),
        alerts=AlertConfig(severity_threshold="INFO"),
        db_path="examples/fraud_demo.db",
    )

    # --- Phase 1: ~500 samples from the SAME distribution (no drift expected) ---
    print("--- Phase 1: 500 in-distribution samples ---")
    phase1, _ = make_synthetic_fraud_data(n=500, seed=1)
    for _, row in phase1.iterrows():
        pred = model.predict_proba(row.to_frame().T)[0, 1]
        monitor.log(features=row, prediction=float(pred))
    report1 = monitor.drift_report()
    print(f"Drifted features: {report1.drifted_features()}")
    for f in report1.top_drifted(4):
        print(f"  {f['feature']:<22} score={f['drift_score']:.4f}  drifted={f['is_drifted']}")
    print(f"Prediction drift: is_drifted={report1.prediction_drift['is_drifted']}, "
          f"score={report1.prediction_drift['drift_score']:.4f}\n")

    # --- Phase 2: inject drift on 'amount' and 'merchant_risk_score', ~1000 samples ---
    print("--- Phase 2: 1000 samples with injected drift (amount shifted x3, "
          "merchant_risk_score shifted x2) ---")
    phase2_raw, _ = make_synthetic_fraud_data(n=1000, seed=2)
    phase2 = inject_drift(phase2_raw, feature="amount", drift_type="sudden", shift=3.0, start_frac=0.0)
    phase2 = inject_drift(phase2, feature="merchant_risk_score", drift_type="gradual", shift=2.0, start_frac=0.0)

    for _, row in phase2.iterrows():
        pred = model.predict_proba(row.to_frame().T)[0, 1]
        monitor.log(features=row, prediction=float(pred))

    report2 = monitor.drift_report()
    print(f"Drifted features: {report2.drifted_features()}")
    for f in report2.top_drifted(4):
        print(f"  {f['feature']:<22} score={f['drift_score']:.4f}  drifted={f['is_drifted']}  severity={f['severity']}")
    print(f"Prediction drift: is_drifted={report2.prediction_drift['is_drifted']}, "
          f"score={report2.prediction_drift['drift_score']:.4f}, "
          f"mean shift={report2.prediction_drift['mean_shift_pct']:.1%}\n")

    print("=== Before/after drift signal ===")
    before = report1.data_drift.get("amount", {}).get("drift_score", 0)
    after = report2.data_drift.get("amount", {}).get("drift_score", 0)
    print(f"'amount' drift_score: {before:.4f} (phase 1, no drift) -> {after:.4f} (phase 2, injected drift)")
    assert after > before, "expected drift score to increase after injecting drift"
    assert report2.data_drift["amount"]["is_drifted"], "expected 'amount' to be flagged as drifted"
    print("PASS: drift correctly detected after injected shift.\n")

    os.makedirs("examples/reports", exist_ok=True)
    report2.to_html("examples/reports/fraud_drift_report.html")
    print("HTML report written to examples/reports/fraud_drift_report.html")


if __name__ == "__main__":
    main()
