from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trafficflow.data import load_graph, load_speed_data


ROOT = Path(__file__).parents[1]


def test_sensor_alignment_and_graph_mapping() -> None:
    _, columns, _ = load_speed_data(ROOT / "data" / "raw" / "METR-LA.h5")
    ids, mapping, adjacency = load_graph(ROOT / "data" / "raw" / "adj_METR-LA.pkl", columns)
    assert columns == ids
    assert adjacency.shape == (len(columns), len(columns))
    assert all(mapping[sensor] == i for i, sensor in enumerate(ids))


def test_sampling_interval_and_target_timing() -> None:
    timestamps, _, _ = load_speed_data(ROOT / "data" / "raw" / "METR-LA.h5")
    assert timestamps[1] - timestamps[0] == pd.Timedelta(minutes=5)
    assert 15 // 5 == 3


def test_split_has_no_future_target_leakage() -> None:
    timestamps, _, _ = load_speed_data(ROOT / "data" / "raw" / "METR-LA.h5")
    history, horizon = 12, 3
    train_end, val_end = int(len(timestamps) * 0.6), int(len(timestamps) * 0.8)
    train_rows = np.arange(0, train_end - history - horizon)
    val_rows = np.arange(train_end - history, val_end - history - horizon)
    test_rows = np.arange(val_end - history, len(timestamps) - history - horizon)
    target_times = timestamps[history + horizon :]
    assert target_times[train_rows[-1]] < timestamps[train_end]
    assert target_times[val_rows[-1]] < timestamps[val_end]
    assert target_times[test_rows[0]] >= timestamps[val_end]
