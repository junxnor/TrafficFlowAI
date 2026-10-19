from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trafficflow.data import choose_connected_sensors, graph_neighbors, load_graph, load_speed_data, save_json
from trafficflow.features import build_features, feature_names, past_only_filled


HORIZONS = {15: 3, 30: 6, 45: 9}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate TrafficFlow AI.")
    parser.add_argument("--data", default="data/raw/METR-LA.h5")
    parser.add_argument("--graph", default="data/raw/adj_METR-LA.pkl")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    timestamps, columns, raw = load_speed_data(args.data)
    graph_ids, mapping, adjacency = load_graph(args.graph, columns)
    selected = choose_connected_sensors(columns, mapping, adjacency, count=15)
    neighbors = graph_neighbors(selected, mapping, adjacency)
    local_index = {sensor: i for i, sensor in enumerate(selected)}
    column_index = {sensor: columns.index(sensor) for sensor in selected}
    history_steps = 12
    max_horizon = max(HORIZONS.values())
    n = len(timestamps)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    filled, train_medians = past_only_filled(raw[:, [column_index[s] for s in selected]], train_end)
    train_valid_speeds = raw[:train_end, [column_index[s] for s in selected]]
    train_valid_speeds = train_valid_speeds[train_valid_speeds > 0]
    congestion_thresholds = {
        "low_speed_cutoff": float(np.quantile(train_valid_speeds, 0.33)),
        "high_speed_cutoff": float(np.quantile(train_valid_speeds, 0.66)),
    }
    all_features = {
        kind: build_features(timestamps, filled, selected, local_index, neighbors, kind == "graph", history_steps)
        for kind in ("base", "graph")
    }

    # Every horizon uses the same chronological boundaries. For each horizon,
    # training targets end before validation and validation targets end before test.
    common_length = n - history_steps - max_horizon
    forecast_times = timestamps[history_steps : history_steps + common_length]
    models: dict[str, dict[int, dict[str, object]]] = {
        "autoregressive": {}, "base": {}, "graph": {}
    }
    records: list[dict[str, object]] = []
    overall_by_horizon: dict[str, dict[str, dict[str, float]]] = {}
    test_counts_by_horizon: dict[str, int] = {}
    cached_predictions: dict[tuple[int, str, str], np.ndarray] = {}
    cached_actuals: dict[tuple[int, str], np.ndarray] = {}

    for minutes, horizon in HORIZONS.items():
        target_indices = np.arange(history_steps + horizon, history_steps + horizon + common_length)
        train_rows = np.arange(0, max(0, train_end - history_steps - horizon))
        val_rows = np.arange(max(0, train_end - history_steps), max(0, val_end - history_steps - horizon))
        test_rows = np.arange(max(0, val_end - history_steps), common_length)
        horizon_models = {kind: {} for kind in models}
        aggregate_actual: dict[str, list[np.ndarray]] = {name: [] for name in ("persistence", "autoregressive_linear", "random_forest_base", "random_forest_graph")}
        aggregate_predicted: dict[str, list[np.ndarray]] = {name: [] for name in aggregate_actual}

        for sensor in selected:
            y = raw[target_indices, column_index[sensor]].astype(float, copy=True)
            valid_train = train_rows[y[train_rows] > 0]
            valid_val = val_rows[y[val_rows] > 0]
            valid_test = test_rows[y[test_rows] > 0]
            if len(valid_train) < 100 or len(valid_val) < 20 or len(valid_test) < 20:
                raise ValueError(f"Not enough valid examples for {sensor} at {minutes} minutes.")
            x_base = all_features["base"][sensor]
            x_graph = all_features["graph"][sensor]
            fitted = {
                "autoregressive": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                "base": RandomForestRegressor(n_estimators=12, max_depth=16, max_samples=0.4, random_state=42, n_jobs=1, min_samples_leaf=3, max_features="sqrt"),
                "graph": RandomForestRegressor(n_estimators=12, max_depth=16, max_samples=0.4, random_state=42, n_jobs=1, min_samples_leaf=3, max_features="sqrt"),
            }
            fitted["autoregressive"].fit(x_base[valid_train], y[valid_train])
            fitted["base"].fit(x_base[valid_train], y[valid_train])
            fitted["graph"].fit(x_graph[valid_train], y[valid_train])
            for kind, estimator in fitted.items():
                horizon_models[kind][sensor] = estimator

            predictions = {
                "persistence": x_base[valid_test, 0],
                "autoregressive_linear": fitted["autoregressive"].predict(x_base[valid_test]),
                "random_forest_base": fitted["base"].predict(x_base[valid_test]),
                "random_forest_graph": fitted["graph"].predict(x_graph[valid_test]),
            }
            actual = y[valid_test]
            cached_actuals[(minutes, sensor)] = actual
            for name, prediction in predictions.items():
                cached_predictions[(minutes, sensor, name)] = prediction
                aggregate_actual[name].append(actual)
                aggregate_predicted[name].append(prediction)
                records.append({"horizon_minutes": minutes, "model": name, "sensor_id": sensor, "n_test": int(len(valid_test)), **metrics(actual, prediction)})

        models["autoregressive"][horizon] = horizon_models["autoregressive"]
        models["base"][horizon] = horizon_models["base"]
        models["graph"][horizon] = horizon_models["graph"]
        overall_by_horizon[str(minutes)] = {
            name: metrics(np.concatenate(aggregate_actual[name]), np.concatenate(aggregate_predicted[name]))
            for name in aggregate_actual
        }
        test_counts_by_horizon[str(minutes)] = int(sum(len(values) for values in aggregate_actual["persistence"]))

    evaluation = pd.DataFrame(records)
    replay_chunks: list[pd.DataFrame] = []
    # Replay rows have valid actual observations at all three horizons so the
    # dashboard can compare the full multi-step forecast at one historical time.
    replay_start = max(0, val_end - history_steps)
    for sensor in selected:
        rows = np.arange(replay_start, common_length)
        actual_by_horizon = {
            minutes: raw[history_steps + horizon + rows, column_index[sensor]]
            for minutes, horizon in HORIZONS.items()
        }
        valid = np.logical_and.reduce([
            np.isfinite(values) & (values > 0) for values in actual_by_horizon.values()
        ])
        rows = rows[valid]
        if not len(rows):
            continue
        replay_data: dict[str, object] = {
            "timestamp": forecast_times[rows].astype(str),
            "sensor_id": np.repeat(sensor, len(rows)),
            "current_speed": all_features["base"][sensor][rows, 0],
        }
        for minutes, horizon in HORIZONS.items():
            replay_data[f"actual_future_speed_{minutes}"] = actual_by_horizon[minutes][valid]
            replay_data[f"predicted_autoregressive_speed_{minutes}"] = models["autoregressive"][horizon][sensor].predict(all_features["base"][sensor][rows])
            replay_data[f"predicted_base_speed_{minutes}"] = models["base"][horizon][sensor].predict(all_features["base"][sensor][rows])
            replay_data[f"predicted_graph_speed_{minutes}"] = models["graph"][horizon][sensor].predict(all_features["graph"][sensor][rows])
        replay_chunks.append(pd.DataFrame(replay_data))

    per_sensor_path = artifact_dir / "evaluation_per_sensor.csv"
    evaluation.to_csv(per_sensor_path, index=False)
    save_json(artifact_dir / "evaluation_overall.json", overall_by_horizon["15"])
    save_json(artifact_dir / "evaluation_by_horizon.json", overall_by_horizon)
    joblib.dump(models, artifact_dir / "models.joblib")
    save_json(artifact_dir / "sensor_mapping.json", {"selected_sensors": selected, "hdf5_columns": columns, "graph_ids": graph_ids, "id_to_index": mapping, "neighbors": neighbors})
    save_json(artifact_dir / "feature_config.json", {
        "horizon_steps": HORIZONS[15], "horizon_minutes": 15,
        "horizons_minutes": list(HORIZONS), "horizons_steps": HORIZONS,
        "history_steps": history_steps, "sampling_minutes": 5,
        "base_features": feature_names(False, history_steps),
        "graph_features": feature_names(True, history_steps),
        "train_medians": train_medians, "train_end": train_end,
        "validation_end": val_end, "selected_sensors": selected,
        "congestion_thresholds": congestion_thresholds,
    })
    replay_frame = pd.concat(replay_chunks, ignore_index=True)
    replay_frame.to_csv(artifact_dir / "test_replay.csv", index=False)
    print(json.dumps({
        "rows": len(raw), "sensors": len(columns), "selected_sensors": selected,
        "zero_readings": int(np.sum(raw <= 0)), "horizons_minutes": list(HORIZONS),
        "train_end": train_end, "validation_end": val_end,
        "test_examples_per_horizon": test_counts_by_horizon,
        "overall_by_horizon": overall_by_horizon,
        "replay_rows": len(replay_frame),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
