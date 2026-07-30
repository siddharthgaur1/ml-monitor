FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ml_monitor ./ml_monitor
COPY simulate ./simulate

RUN pip install --no-cache-dir ".[api,dashboard,sklearn]"

COPY . .

ENV ML_MONITOR_DB=/data/ml_monitor.db
VOLUME ["/data"]

EXPOSE 8080 8501

# Runs the REST API by default; run the dashboard with:
#   docker compose exec ml-monitor streamlit run ml_monitor/dashboard.py -- --db /data/ml_monitor.db
CMD ["python", "-m", "ml_monitor.cli", "status", "--db", "/data/ml_monitor.db"]
