"""Real-time transaction scoring: the worked example that exercises the platform.

Kafka-compatible ingestion -> rolling per-user features in Redis -> XGBoost +
IsolationForest ensemble behind FastAPI -> scored rows to Postgres -> drift
reports.

**The transaction data is synthetic.** Fraud is injected as odd-hour +
high-amount + unseen-device, which is precisely what the features are built to
detect, so the model's accuracy on it says very little about real fraud. Every
metric this subsystem produces measures the pipeline, not the modelling. That
caveat came from this code's own README and is repeated here because a merge is
exactly where such a caveat gets quietly lost.
"""
