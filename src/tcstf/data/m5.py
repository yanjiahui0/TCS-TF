from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PanelWindows:
    x: np.ndarray
    y: np.ndarray
    series_id: np.ndarray
    origin_time: np.ndarray


def prepare_m5_long_dataframe(
    df: pd.DataFrame,
    *,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    demand_col: str = "demand",
    history: int = 56,
    horizon: int = 28,
    min_nonzero_fraction: float = 0.05,
) -> PanelWindows:
    """Build rolling-origin M5-style windows from a long demand table.

    This function deliberately does not invent initial inventory or pipeline
    orders. Those decision-state warm starts belong in the experiment manifest.
    """

    required = {series_col, time_col, demand_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([series_col, time_col])

    xs, ys, ids, times = [], [], [], []
    for sid, g in data.groupby(series_col, sort=False):
        demand = g[demand_col].to_numpy(dtype=np.float32)
        timestamps = g[time_col].to_numpy()
        if len(demand) < history + horizon:
            continue
        if np.mean(demand > 0) < min_nonzero_fraction:
            continue
        # Minimal forecast-time context: raw history + robust history summaries.
        for i in range(history, len(demand) - horizon + 1):
            hist = demand[i - history : i]
            future = demand[i : i + horizon]
            summaries = np.asarray(
                [hist.mean(), hist.std(), np.median(hist), hist[-1], np.quantile(hist, 0.9)],
                dtype=np.float32,
            )
            xs.append(np.concatenate([hist, summaries]))
            ys.append(future[:, None])
            ids.append(str(sid))
            times.append(timestamps[i])
    if not xs:
        raise ValueError("No admissible M5 windows were created")
    return PanelWindows(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        series_id=np.asarray(ids),
        origin_time=np.asarray(times),
    )
