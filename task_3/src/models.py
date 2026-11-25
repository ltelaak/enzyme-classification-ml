"""Portable adaptation of the original hierarchical Task3Model.

The caller provides the trained classifiers and their fitted PCA transformers.
This avoids machine-specific paths and implicit loading of serialized objects.
"""
import torch
from torch import nn


class Task3Model(nn.Module):
    """Predict 0 for non-enzymes, otherwise the first-level EC class (1–7)."""

    def __init__(self, model_one, model_two, pca_one, pca_two):
        super().__init__()
        self.model_one = model_one.eval()
        self.model_two = model_two.eval()
        self.pca_one = pca_one
        self.pca_two = pca_two

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("Expected a batch of protein embeddings (N, D).")
        self.model_one.eval()
        self.model_two.eval()
        raw = x.detach().cpu().numpy()
        param_one = next(self.model_one.parameters())
        param_two = next(self.model_two.parameters())
        x_one = torch.as_tensor(self.pca_one.transform(raw),
                               dtype=param_one.dtype, device=param_one.device)
        x_two = torch.as_tensor(self.pca_two.transform(raw),
                               dtype=param_two.dtype, device=param_two.device)
        enzyme = torch.round(self.model_one(x_one)).reshape(-1).bool().cpu()
        ec = self.model_two(x_two).argmax(dim=1).cpu() + 1
        return torch.where(enzyme, ec, torch.zeros_like(ec))
