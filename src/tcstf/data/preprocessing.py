from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Training-only standardizer with explicit fitted state."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = np.maximum(x.std(axis=0), self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit on training data first")
        return ((np.asarray(x, dtype=float) - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit on training data first")
        return np.asarray(x) * self.scale_ + self.mean_
