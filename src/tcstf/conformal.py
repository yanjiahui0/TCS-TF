from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SplitConformalPathSet:
    """Independent post-selection pathwise recalibration from Eqs. (57)–(58)."""

    alpha: float = 0.10
    min_scale: float = 1e-6
    q_alpha: float | None = None

    def nonconformity(
        self,
        y: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        scale = np.maximum(np.asarray(scale, dtype=float), self.min_scale)
        z = np.abs(np.asarray(y, dtype=float) - np.asarray(center, dtype=float)) / scale
        axes = tuple(range(1, z.ndim))
        return z.max(axis=axes)

    def calibrate_scores(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        n = len(scores)
        if n < 1:
            raise ValueError("Need at least one recalibration score")
        ordered = np.sort(scores)
        k = int(math.ceil((n + 1) * (1 - self.alpha)))
        # Manuscript convention defines s_(n+1)=+infinity.
        self.q_alpha = float("inf") if k == n + 1 else float(ordered[k - 1])
        return self.q_alpha

    def calibrate(
        self,
        y: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> float:
        return self.calibrate_scores(self.nonconformity(y, center, scale))

    def contains(self, y: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if self.q_alpha is None:
            raise RuntimeError("Call calibrate first")
        return self.nonconformity(y, center, scale) <= self.q_alpha

    def bounds(self, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_alpha is None:
            raise RuntimeError("Call calibrate first")
        radius = self.q_alpha * np.maximum(np.asarray(scale, dtype=float), self.min_scale)
        center = np.asarray(center, dtype=float)
        return center - radius, center + radius
