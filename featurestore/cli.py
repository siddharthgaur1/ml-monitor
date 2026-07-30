"""Command-line interface for lite-featurestore."""
from __future__ import annotations

import shutil
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from featurestore.core.feature_store import FeatureStore
from featurestore.materialization.materializer import materialize as materialize_fn
from featurestore.quality.validator import DataQualityError, check_quality
from featurestore.registry.validator import SchemaValidationError, validate_schema

console = Console()


def _load_store(config: str) -> FeatureStore:
    return FeatureStore(config=config)


def _read_data(file: str) -> pd.DataFrame:
    path = Path(file)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


@click.group()
@click.option("--config", default="featurestore.yaml", show_default=True, help="Path to featurestore.yaml")
@click.pass_context
def cli(ctx, config):
    """lite-featurestore: a small, self-hosted feature store."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.pass_context
def init(ctx):
    """Scaffold a featurestore.yaml and data/ directory in the current dir."""
    config_path = Path(ctx.obj["config"])
    if config_path.exists():
        console.print(f"[yellow]{config_path} already exists, skipping[/yellow]")
    else:
        example = Path(__file__).parent.parent / "featurestore.yaml.example"
        if example.exists():
            shutil.copy(example, config_path)
        else:
            config_path.write_text(
                "offline_store:\n  type: parquet\n  path: ./data/offline\n"
                "online_store:\n  type: sqlite\n  path: ./data/online.db\n  ttl_hours: 24\n"
                "registry:\n  type: sqlite\n  path: ./data/registry.db\n"
            )
        console.print(f"[green]Wrote {config_path}[/green]")
    Path("data").mkdir(exist_ok=True)
    console.print("[green]Initialized ./data[/green]")


@cli.command(name="list")
@click.pass_context
def list_groups(ctx):
    """List all registered feature groups."""
    fs = _load_store(ctx.obj["config"])
    groups = fs.list_feature_groups()
    table = Table(title="Feature Groups")
    for col in ("name", "entity_name", "online", "offline", "ttl_hours", "tags"):
        table.add_column(col)
    for g in groups:
        table.add_row(g["name"], g["entity_name"], str(g["online"]), str(g["offline"]),
                       str(g["ttl_hours"]), ",".join(g["tags"]))
    console.print(table)


@cli.command()
@click.argument("group")
@click.pass_context
def describe(ctx, group):
    """Show details for a feature group."""
    fs = _load_store(ctx.obj["config"])
    g = fs.registry.get_feature_group(group)
    if g is None:
        console.print(f"[red]No such feature group: {group}[/red]")
        raise SystemExit(1)
    console.print(g)
    table = Table(title=f"Features in {group}")
    table.add_column("name")
    table.add_column("dtype")
    table.add_column("description")
    for f in g["features"]:
        table.add_row(f["name"], f["dtype"], f["description"])
    console.print(table)


@cli.command()
@click.argument("keyword")
@click.pass_context
def search(ctx, keyword):
    """Search features by name, description, or tag."""
    fs = _load_store(ctx.obj["config"])
    results = fs.search_features(keyword)
    if not results:
        console.print("[yellow]No matches[/yellow]")
        return
    table = Table(title=f"Search: {keyword}")
    table.add_column("feature_group")
    table.add_column("name")
    table.add_column("dtype")
    table.add_column("description")
    for r in results:
        table.add_row(r["feature_group"], r["name"], r["dtype"], r["description"])
    console.print(table)


@cli.command()
@click.option("--group", required=True)
@click.option("--file", required=True, type=click.Path(exists=True))
@click.pass_context
def ingest(ctx, group, file):
    """Ingest a CSV/JSON/parquet file into a feature group."""
    fs = _load_store(ctx.obj["config"])
    df = _read_data(file)
    try:
        result = fs.ingest(feature_group=group, data=df)
    except (SchemaValidationError, DataQualityError) as e:
        console.print(f"[red]Ingest failed: {e}[/red]")
        raise SystemExit(1)
    console.print(f"[green]Ingested {result['rows_written']} rows into {group}[/green]")


@cli.command()
@click.option("--group", required=True)
@click.pass_context
def materialize(ctx, group):
    """Push latest offline feature values into the online store."""
    fs = _load_store(ctx.obj["config"])
    n = materialize_fn(fs, group)
    console.print(f"[green]Materialized {n} entities for {group}[/green]")


@cli.command()
@click.option("--group", required=True)
@click.pass_context
def stats(ctx, group):
    """Show the latest ingestion stats for a feature group."""
    fs = _load_store(ctx.obj["config"])
    s = fs.get_feature_stats(group)
    if s is None:
        console.print("[yellow]No stats yet (nothing ingested)[/yellow]")
        return
    table = Table(title=f"Stats: {group}")
    table.add_column("feature")
    table.add_column("null_rate")
    table.add_column("mean")
    table.add_column("std")
    table.add_column("min")
    table.add_column("max")
    for name, s_ in s.items():
        table.add_row(name, f"{s_['null_rate']:.2%}", str(s_["mean"]), str(s_["std"]), str(s_["min"]), str(s_["max"]))
    console.print(table)


@cli.command()
@click.option("--group", required=True)
@click.option("--file", required=True, type=click.Path(exists=True))
@click.pass_context
def validate(ctx, group, file):
    """Dry-run schema + quality validation against a registered group, without ingesting."""
    fs = _load_store(ctx.obj["config"])
    df = _read_data(file)
    reg_group = fs.registry.get_feature_group(group)
    if reg_group is None:
        console.print(f"[red]No such feature group: {group}[/red]")
        raise SystemExit(1)
    try:
        validate_schema(df, reg_group)
        check_quality(df, reg_group["features"])
    except (SchemaValidationError, DataQualityError) as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise SystemExit(1)
    console.print(f"[green]Valid: {len(df)} rows OK for {group}[/green]")


if __name__ == "__main__":
    cli()
