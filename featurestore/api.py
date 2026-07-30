"""REST API for lite-featurestore, built on FastAPI.

Run with: uvicorn featurestore.api:app
Config path is read from the FEATURESTORE_CONFIG env var (default
"featurestore.yaml").
"""
from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from featurestore.core.feature_store import Entity, Feature, FeatureGroup, FeatureStore
from featurestore.materialization.materializer import materialize as materialize_fn
from featurestore.quality.validator import DataQualityError
from featurestore.registry.validator import SchemaValidationError

app = FastAPI(title="lite-featurestore", version="0.1.0")

_fs: FeatureStore | None = None


def get_store() -> FeatureStore:
    global _fs
    if _fs is None:
        _fs = FeatureStore(config=os.environ.get("FEATURESTORE_CONFIG", "featurestore.yaml"))
    return _fs


class FeatureIn(BaseModel):
    name: str
    dtype: str = "float32"
    description: str = ""


class EntityIn(BaseModel):
    name: str
    dtype: str = "int64"
    description: str = ""


class FeatureGroupIn(BaseModel):
    name: str
    entity: EntityIn
    features: list[FeatureIn]
    ttl_hours: float | None = 24
    online: bool = True
    offline: bool = True
    tags: list[str] = []


class OnlineFeaturesRequest(BaseModel):
    feature_group: str
    entity_ids: list[Any]


class HistoricalFeaturesRequest(BaseModel):
    entity_rows: list[dict]  # rows with entity_id, event_timestamp, ...
    feature_groups: list[str]
    fill_strategy: str = "none"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/feature-groups")
def list_feature_groups():
    return get_store().list_feature_groups()


@app.get("/feature-groups/{name}")
def get_feature_group(name: str):
    group = get_store().registry.get_feature_group(name)
    if group is None:
        raise HTTPException(status_code=404, detail=f"feature group not found: {name}")
    return group


@app.post("/feature-groups", status_code=201)
def create_feature_group(body: FeatureGroupIn):
    fg = FeatureGroup(
        name=body.name,
        entity=Entity(**body.entity.model_dump()),
        features=[Feature(**f.model_dump()) for f in body.features],
        ttl_hours=body.ttl_hours,
        online=body.online,
        offline=body.offline,
        tags=body.tags,
    )
    get_store().register(fg)
    return get_store().registry.get_feature_group(body.name)


@app.post("/ingest/{name}")
async def ingest(
    name: str,
    file: UploadFile | None = File(None),
    rows: list[dict] | None = Body(None),
):
    """Accepts either a multipart parquet file upload, or a JSON body of rows."""
    if file is not None:
        content = await file.read()
        df = pd.read_parquet(io.BytesIO(content))
    elif rows:
        df = pd.DataFrame(rows)
    else:
        raise HTTPException(status_code=400, detail="provide a parquet `file` upload or a JSON `rows` body")

    try:
        result = get_store().ingest(feature_group=name, data=df)
    except (SchemaValidationError, DataQualityError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.post("/online-features")
def online_features(req: OnlineFeaturesRequest):
    try:
        return get_store().get_online_features(req.feature_group, req.entity_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/historical-features")
def historical_features(req: HistoricalFeaturesRequest):
    entity_df = pd.DataFrame(req.entity_rows)
    try:
        df = get_store().get_historical_features(
            entity_df, req.feature_groups, fill_strategy=req.fill_strategy
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return df.to_dict("records")


@app.post("/materialize/{name}")
def materialize(name: str):
    try:
        n = materialize_fn(get_store(), name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"feature_group": name, "entities_materialized": n}
