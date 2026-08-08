from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BatteryWindows:
    x: np.ndarray
    y: np.ndarray
    origin_time: np.ndarray


def prepare_battery_dataframe(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    load_col: str = "load",
    solar_col: str = "solar",
    buy_col: str = "buy_price",
    sell_col: str = "sell_price",
    history: int = 168,
    horizon: int = 24,
) -> BatteryWindows:
    """Create battery forecasting windows with Y=(net demand,buy,sell).

    Calendar features are forecast-time known and may enter X. Quantities placed in
    Y are not duplicated as future-known features, respecting the manuscript's
    leakage rule.
    """

    required = {time_col, load_col, solar_col, buy_col, sell_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).drop_duplicates(time_col)
    net = (data[load_col] - data[solar_col]).to_numpy(dtype=np.float32)
    buy = data[buy_col].to_numpy(dtype=np.float32)
    sell = data[sell_col].to_numpy(dtype=np.float32)
    ts = data[time_col]

    xs, ys, origins = [], [], []
    for i in range(history, len(data) - horizon + 1):
        hist_net = net[i - history : i]
        hist_buy = buy[i - history : i]
        hour = np.arange(i, i + horizon)
        future_ts = ts.iloc[i : i + horizon]
        cal = np.column_stack(
            [
                np.sin(2 * np.pi * future_ts.dt.hour.to_numpy() / 24),
                np.cos(2 * np.pi * future_ts.dt.hour.to_numpy() / 24),
                future_ts.dt.dayofweek.to_numpy() / 6.0,
            ]
        ).astype(np.float32)
        x = np.concatenate(
            [
                hist_net,
                hist_buy,
                np.asarray([hist_net.mean(), hist_net.std(), hist_buy.mean()], dtype=np.float32),
                cal.reshape(-1),
            ]
        )
        y = np.column_stack([net[i : i + horizon], buy[i : i + horizon], sell[i : i + horizon]])
        xs.append(x)
        ys.append(y)
        origins.append(ts.iloc[i].to_datetime64())
    if not xs:
        raise ValueError("No admissible battery windows were created")
    return BatteryWindows(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(origins))
