"""ml_platform.features: a small, self-hosted feature store.

Public SDK surface: FeatureStore, FeatureGroup, Feature, Entity
"""
from ml_platform.features.core.feature_store import Entity, Feature, FeatureGroup, FeatureStore

__all__ = ["Entity", "Feature", "FeatureGroup", "FeatureStore"]
__version__ = "0.1.0"
