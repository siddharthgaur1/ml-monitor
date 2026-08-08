"""Streamlit dashboard, functional against the SQLite store directly.

Run with: streamlit run src/ml_platform/monitoring/dashboard.py -- --db ml_platform.db
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the 'dashboard' extra: pip install ml-monitor[dashboard]") from e

from ml_platform.monitoring.detectors.correlation_drift import detect_correlation_drift
from ml_platform.monitoring.detectors.data_drift import detect_data_drift
from ml_platform.monitoring.store.aggregator import aggregate
from ml_platform.monitoring.store.sqlite_store import SQLiteStore


def _get_db_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("ML_MONITOR_DB", "ml_platform.db"))
    parser.add_argument("--reference", default=os.environ.get("ML_MONITOR_REFERENCE", None))
    args, _ = parser.parse_known_args()
    return args.db, args.reference


def main():
    st.set_page_config(page_title="ml-monitor", layout="wide")
    st.title("ml-monitor dashboard")

    db_path, reference_path = _get_db_path()
    store = SQLiteStore(db_path=db_path)

    total = store.count()
    badge = "🟢 OK" if total > 0 else "⚪ No data yet"
    st.markdown(f"### Status: {badge}  &nbsp;|&nbsp; {total} predictions logged")

    rows = store.recent_predictions(limit=2000)
    if not rows:
        st.info("No predictions logged yet. Point Monitor.log() at this DB and refresh.")
        st.stop()

    current_df = pd.DataFrame([r["features"] for r in rows])
    pred_series = pd.Series([r["prediction"] for r in rows])
    pd.Series([r["timestamp"] for r in rows])

    reference_df = None
    if reference_path and os.path.exists(reference_path):
        reference_df = pd.read_csv(reference_path)

    if reference_df is not None:
        common = [c for c in reference_df.columns if c in current_df.columns]
        drift_results = detect_data_drift(reference_df[common], current_df[common])

        st.subheader("Feature drift table")
        drift_df = pd.DataFrame([{"feature": f, **r} for f, r in drift_results.items()]).sort_values(
            "drift_score", ascending=False
        )
        st.dataframe(drift_df[["feature", "method", "drift_score", "is_drifted", "severity"]])

        st.subheader("Top-5 drifted features")
        top5 = drift_df.head(5)
        st.plotly_chart(px.bar(top5, x="feature", y="drift_score", color="severity"), use_container_width=True)

        st.subheader("Prediction distribution: reference vs current")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pred_series, name="current", opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation heatmap: reference vs current")
        corr_result = detect_correlation_drift(reference_df[common], current_df[common])
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Reference")
            st.plotly_chart(px.imshow(corr_result["reference_matrix"]), use_container_width=True)
        with c2:
            st.caption("Current")
            st.plotly_chart(px.imshow(corr_result["current_matrix"]), use_container_width=True)
    else:
        st.warning("No --reference CSV configured; showing prediction stream only.")
        st.plotly_chart(px.histogram(pred_series, nbins=30, title="Prediction distribution"), use_container_width=True)

    st.subheader("Drift over time (hourly mean drift score)")
    agg = aggregate(store, granularity="hourly", hours=24 * 7)
    if agg:
        agg_df = pd.DataFrame(agg)
        agg_df["bucket_start"] = pd.to_datetime(agg_df["bucket_start"], unit="s")
        st.plotly_chart(px.line(agg_df, x="bucket_start", y="mean_drift_score"), use_container_width=True)
    else:
        st.caption("Not enough history yet.")

    st.subheader("Concept drift")
    labeled = [(r["label"], r["prediction"]) for r in rows if r["label"] is not None]
    if labeled:
        from ml_platform.monitoring.detectors.concept_drift import detect_concept_drift

        errors = [0 if float(l) == float(p) else 1 for l, p in labeled]
        cd = detect_concept_drift(errors)
        st.json(cd)
    else:
        st.caption("No ground-truth labels logged yet -- concept drift unavailable.")

    st.subheader("Alerts log")
    alerts = store.recent_alerts(limit=100)
    if alerts:
        st.dataframe(pd.DataFrame(alerts))
    else:
        st.caption("No alerts fired yet.")

    st.caption(f"Auto-refreshing every 60s. Last refresh: {time.strftime('%H:%M:%S')}")
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
