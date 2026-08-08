"""Thin adapter around a LightGBM model. Optional dep."""
from __future__ import annotations


class LightGBMAdapter:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or self._extract_feature_names(model)

    @staticmethod
    def _extract_feature_names(model):
        names = getattr(model, "feature_name_", None) or getattr(model, "feature_names_in_", None)
        return list(names) if names is not None else None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying model has no predict_proba")
