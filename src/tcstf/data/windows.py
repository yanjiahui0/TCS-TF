from __future__ import annotations

import numpy as np


def rolling_windows(
    values: np.ndarray,
    history: int,
    horizon: int,
    known_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create simple rolling history/future arrays from a 1D or 2D time series."""

    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values) - history - horizon + 1
    if n <= 0:
        raise ValueError("Series is too short for requested history/horizon")
    x, y = [], []
    for i in range(n):
        hist = values[i : i + history].reshape(-1)
        if known_future is not None:
            kf = np.asarray(known_future)[i + history : i + history + horizon].reshape(-1)
            hist = np.concatenate([hist, kf])
        x.append(hist)
        y.append(values[i + history : i + history + horizon])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
