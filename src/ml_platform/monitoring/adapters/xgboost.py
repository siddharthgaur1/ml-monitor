"""Thin adapter around an XGBoost model (Booster or sklearn-API). Optional dep."""
from __future__ import annotations


class XGBoostAdapter:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or self._extract_feature_names(model)

    @staticmethod
    def _extract_feature_names(model):
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return list(names)
        names = getattr(model, "feature_names", None)  # xgb.Booster
        return list(names) if names else None

    def predict(self, X):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required: pip install ml-monitor[xgboost]") from e
        if isinstance(self.model, xgb.Booster):
            return self.model.predict(xgb.DMatrix(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying model has no predict_proba")
