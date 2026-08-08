"""Error paths for the feature-store API.

These existed untested, which is how a bad automated edit during the merge put
`raise ... from exc` into handlers that bind `as e` -- a NameError on every
error response, invisible to a suite that only exercised success paths.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import ml_platform.features.api as api
    from ml_platform.features.core.config import FeatureStoreConfig
    from ml_platform.features.core.feature_store import FeatureStore

    cfg = FeatureStoreConfig.from_dict(
        {
            "offline_store": {"type": "parquet", "path": str(tmp_path / "offline")},
            "online_store": {"type": "sqlite", "path": str(tmp_path / "online.db")},
            "registry": {"path": str(tmp_path / "registry.db")},
        }
    )
    monkeypatch.setattr(api, "_fs", FeatureStore(config=cfg), raising=False)
    monkeypatch.setattr(api, "get_store", lambda: api._fs)
    return TestClient(api.app)


def test_unknown_feature_group_is_a_404_not_a_crash(client):
    r = client.get("/feature-groups/does-not-exist")
    assert r.status_code == 404
    assert "does-not-exist" in r.json()["detail"]


def test_ingest_without_a_body_is_a_400_not_a_crash(client):
    r = client.post("/feature-groups/anything/ingest")
    assert r.status_code in (400, 404, 422)
    assert isinstance(r.json().get("detail"), str)
