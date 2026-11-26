"""Load fitted PCA parameters from portable NPZ files."""

import numpy as np


class NpzPCA:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.mean_ = np.asarray(
                data["mean"], dtype=np.float64
            )
            self.components_ = np.asarray(
                data["components"], dtype=np.float64
            )

        if self.mean_.shape != (1024,):
            raise ValueError("Invalid PCA mean shape.")

        if self.components_.shape != (128, 1024):
            raise ValueError("Invalid PCA component shape.")

        if not np.isfinite(self.mean_).all():
            raise ValueError("Non-finite PCA mean.")

        if not np.isfinite(self.components_).all():
            raise ValueError("Non-finite PCA components.")

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != 1024:
            raise ValueError("Expected embeddings of shape (N, 1024).")

        result = np.einsum(
            "ni,ki->nk",
            X - self.mean_,
            self.components_,
            optimize=False
        )

        if not np.isfinite(result).all():
            raise ValueError("Non-finite PCA output.")

        return result
