from __future__ import annotations

import numpy as np
import pandas as pd


def past_only_filled(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, list[float]]:
    """Treat zero as missing per the METR-LA preprocessing convention; fill only from the past."""
    observed = raw.astype(float, copy=True)
    observed[observed <= 0] = np.nan
    train_medians = np.nanmedian(observed[:train_end], axis=0)
    if np.isnan(train_medians).any():
        raise ValueError("A sensor has no valid training readings for imputation.")
    frame = pd.DataFrame(observed).ffill()
    filled = frame.fillna(pd.Series(train_medians, index=frame.columns)).to_numpy(float)
    return filled, train_medians.tolist()


def feature_names(use_neighbors: bool, history_steps: int = 12) -> list[str]:
    names = [f"speed_lag_{lag}" for lag in range(1, history_steps + 1)]
    names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    if use_neighbors:
        names += [f"neighbor_mean_lag_{lag}" for lag in range(1, history_steps + 1)]
    return names


def build_features(
    timestamps: pd.DatetimeIndex,
    filled: np.ndarray,
    selected: list[str],
    column_index: dict[str, int],
    neighbors: dict[str, list[str]],
    use_neighbors: bool,
    history_steps: int = 12,
) -> dict[str, np.ndarray]:
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    dow = timestamps.dayofweek.to_numpy()
    calendar = np.column_stack([
        np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])
    result: dict[str, np.ndarray] = {}
    for sensor in selected:
        idx = column_index[sensor]
        columns = [filled[:, idx][history_steps - lag : -(lag) if lag else None] for lag in range(1, history_steps + 1)]
        columns += [calendar[history_steps:]]
        if use_neighbors:
            neighbor_matrix = np.mean([filled[:, column_index[other]] for other in neighbors[sensor]], axis=0)
            columns += [neighbor_matrix[history_steps - lag : -(lag) if lag else None, None] for lag in range(1, history_steps + 1)]
        result[sensor] = np.column_stack(columns)
    return result


def target_arrays(raw: np.ndarray, selected: list[str], column_index: dict[str, int], horizon: int, history_steps: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for sensor in selected:
        values = raw[:, column_index[sensor]]
        result[sensor] = values[history_steps + horizon :]
    return result
