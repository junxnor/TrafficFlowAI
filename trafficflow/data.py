from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def load_speed_data(path: str | Path) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """Read the pandas HDF5 layout directly, avoiding a PyTables dependency at runtime."""
    with h5py.File(path, "r") as handle:
        group = handle["df"]
        columns = [value.decode("ascii") for value in group["axis0"][:]]
        epoch_ns = group["axis1"][:]
        values = np.asarray(group["block0_values"][:], dtype=float)
    timestamps = pd.to_datetime(epoch_ns)
    if len(timestamps) < 2:
        raise ValueError("The HDF5 file has too few timestamps.")
    interval = timestamps[1] - timestamps[0]
    if interval != pd.Timedelta(minutes=5):
        raise ValueError(f"Expected 5-minute sampling, found {interval}.")
    if values.shape != (len(timestamps), len(columns)):
        raise ValueError("HDF5 values do not align with its timestamp and column axes.")
    return timestamps, columns, values


def load_graph(path: str | Path, sensor_columns: list[str]) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Load and validate the trusted local METR-LA graph pickle."""
    with Path(path).open("rb") as handle:
        payload: Any = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ValueError("Expected [sensor_ids, id_to_index, adjacency_matrix] in graph pickle.")
    sensor_ids, id_to_index, adjacency = payload
    if not isinstance(sensor_ids, list) or not isinstance(id_to_index, dict):
        raise ValueError("Graph pickle has unexpected sensor ID or mapping types.")
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError("Adjacency matrix shape does not match graph sensor IDs.")
    if set(sensor_columns) != set(sensor_ids) or set(sensor_columns) != set(id_to_index):
        raise ValueError("HDF5 sensor columns and graph sensor IDs do not match exactly.")
    expected_positions = {sensor_id: position for position, sensor_id in enumerate(sensor_ids)}
    if any(id_to_index[sensor_id] != expected_positions[sensor_id] for sensor_id in sensor_ids):
        raise ValueError("The graph ID mapping is not the adjacency matrix order.")
    return sensor_ids, {str(key): int(value) for key, value in id_to_index.items()}, adjacency


def choose_connected_sensors(sensor_columns: list[str], mapping: dict[str, int], adjacency: np.ndarray, count: int = 15) -> list[str]:
    """Choose a deterministic connected subgraph using breadth-first traversal."""
    start = sensor_columns[0]
    selected: list[str] = []
    queue = [start]
    seen: set[str] = set()
    while queue and len(selected) < count:
        sensor = queue.pop(0)
        if sensor in seen:
            continue
        seen.add(sensor)
        selected.append(sensor)
        row = adjacency[mapping[sensor]]
        neighbors = [sensor_columns[i] for i in np.flatnonzero(row > 0) if sensor_columns[i] not in seen]
        queue.extend(sorted(neighbors, key=lambda item: mapping[item]))
    if len(selected) < count:
        raise ValueError(f"Could only find {len(selected)} connected sensors.")
    return selected


def graph_neighbors(selected: list[str], mapping: dict[str, int], adjacency: np.ndarray) -> dict[str, list[str]]:
    selected_set = set(selected)
    result: dict[str, list[str]] = {}
    for sensor in selected:
        row = adjacency[mapping[sensor]]
        column = adjacency[:, mapping[sensor]]
        # METR-LA's adjacency can be directed; for local context, either
        # direction means the two selected sensors are graph neighbors.
        result[sensor] = [other for other in selected if other != sensor and (row[mapping[other]] > 0 or column[mapping[other]] > 0)]
    if any(not neighbors for neighbors in result.values()):
        raise ValueError("Selected connected subgraph contains a sensor without a local graph neighbor.")
    return result


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
