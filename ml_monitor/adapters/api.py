"""Adapter for a model served behind an HTTP API instead of loaded in-process."""
from __future__ import annotations

import json
import urllib.request


class APIModelAdapter:
    """Wraps a remote model endpoint that accepts {"features": [...]}-style
    JSON and returns {"prediction": ...}.
    """

    def __init__(self, endpoint_url: str, feature_names=None, timeout: float = 10.0):
        self.endpoint_url = endpoint_url
        self.feature_names = feature_names
        self.timeout = timeout

    def predict(self, X):
        rows = X.tolist() if hasattr(X, "tolist") else X
        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps({"features": rows}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        return body["prediction"]
