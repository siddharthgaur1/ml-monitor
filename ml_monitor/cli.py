"""ml-monitor CLI (click)."""
from __future__ import annotations

import pickle

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from .core.monitor import AlertConfig, DriftConfig, Monitor
from .store.sqlite_store import SQLiteStore

console = Console()


def _load_model(path: str):
    # Pickle is the standard artifact format for sklearn/XGBoost models (joblib/pickle).
    # This loads a model file the operator points at deliberately via --model, the same
    # trust model as `joblib.load()` in any sklearn deployment -- not user-uploaded input.
    with open(path, "rb") as f:
        return pickle.load(f)


@click.group()
def cli():
    """ml-monitor: drift monitoring for any sklearn/XGBoost model."""


@cli.command()
@click.option("--model", "model_path", required=True, help="Pickled model file.")
@click.option("--reference", "reference_path", required=True, help="Reference dataset CSV.")
@click.option("--port", default=8080, help="Port for the REST API.")
@click.option("--db", default="ml_monitor.db", help="SQLite DB path.")
def start(model_path, reference_path, port, db):
    """Load a model + reference data and start the monitoring API server."""
    model = _load_model(model_path)
    reference_df = pd.read_csv(reference_path)
    monitor = Monitor(model=model, reference_data=reference_df, config=DriftConfig(), alerts=AlertConfig(),
                       db_path=db)
    console.print(f"[green]Starting ml-monitor API on port {port}[/green]")
    monitor.serve(port=port)


@cli.command()
@click.option("--db", default="ml_monitor.db", help="SQLite DB path.")
@click.option("--reference", "reference_path", required=True, help="Reference dataset CSV.")
@click.option("--format", "fmt", default="html", type=click.Choice(["html", "json"]))
@click.option("--output", default="reports/drift_report.html")
def report(db, reference_path, fmt, output):
    """Generate a drift report from the current SQLite store contents."""
    reference_df = pd.read_csv(reference_path)
    monitor = Monitor(reference_data=reference_df, db_path=db)
    # Rehydrate the in-memory window from persisted rows so the report reflects stored history.
    for row in monitor.store.recent_predictions(limit=monitor.config.window_size):
        monitor.log(features=row["features"], prediction=row["prediction"], label=row["label"])
    rep = monitor.drift_report()

    import os

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if fmt == "html":
        rep.to_html(output)
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(rep.to_json())
    console.print(f"[green]Report written to {output}[/green]")


@cli.command()
@click.option("--dataset", required=True)
@click.option("--drift-type", "drift_type", default="gradual", type=click.Choice(["gradual", "sudden", "seasonal"]))
@click.option("--feature", required=True)
@click.option("--shift", default=1.0, type=float)
@click.option("--output", default=None)
def simulate(dataset, drift_type, feature, shift, output):
    """Inject synthetic drift into a dataset (delegates to simulate/drift_simulator.py)."""
    from simulate.drift_simulator import inject_drift

    df = pd.read_csv(dataset)
    drifted = inject_drift(df, feature=feature, drift_type=drift_type, shift=shift)
    output = output or dataset.replace(".csv", f"_{drift_type}_drift.csv")
    drifted.to_csv(output, index=False)
    console.print(f"[green]Drifted dataset written to {output}[/green]")


@cli.command()
@click.option("--db", default="ml_monitor.db", help="SQLite DB path.")
def status(db):
    """Print current monitoring status."""
    store = SQLiteStore(db_path=db)
    table = Table(title="ml-monitor status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total predictions logged", str(store.count()))
    recent_alerts = store.recent_alerts(limit=10)
    table.add_row("Recent alerts", str(len(recent_alerts)))
    console.print(table)
    for a in recent_alerts:
        console.print(f"  [{a['severity']}] {a['message']}")


if __name__ == "__main__":
    cli()
