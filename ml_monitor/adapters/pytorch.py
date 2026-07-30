"""Thin adapter around a PyTorch nn.Module. torch is an optional dep, imported lazily."""
from __future__ import annotations


class PyTorchAdapter:
    def __init__(self, model, feature_names=None, device: str = "cpu"):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("torch is required: pip install ml-monitor[torch]") from e
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()

    def predict(self, X):
        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            out = self.model(tensor)
            return out.cpu().numpy()
