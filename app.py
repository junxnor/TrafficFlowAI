from __future__ import annotations

import json
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trafficflow.data import load_graph, load_speed_data
from trafficflow.features import build_features, past_only_filled


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def inject_design_css() -> None:
    """Apply the downloaded design tokens to Streamlit's rendered components."""
    built_tailwind = ROOT / "theme" / "tailwind.css"
    css_files = [built_tailwind if built_tailwind.exists() else ROOT / "theme" / "variables.css", ROOT / "theme" / "dashboard.css"]
    css = "\n".join(path.read_text(encoding="utf-8") for path in css_files if path.exists())
    wallpaper = ROOT / "assets" / "images" / "IMG_1244.jpg"
    if wallpaper.exists():
        image_data = base64.b64encode(wallpaper.read_bytes()).decode("ascii")
        css += f"""
/* Local traffic photograph used as a subdued, non-live dashboard wallpaper. */
.stApp {{
  background-image: linear-gradient(rgba(254, 255, 252, .86), rgba(254, 255, 252, .86)), url('data:image/jpeg;base64,{image_data}');
  background-attachment: fixed;
  background-position: center;
  background-size: cover;
}}
"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data
def load_artifacts():
    timestamps, columns, raw = load_speed_data(ROOT / "data" / "raw" / "METR-LA.h5")
    mapping_data = json.loads((ARTIFACTS / "sensor_mapping.json").read_text(encoding="utf-8"))
    config = json.loads((ARTIFACTS / "feature_config.json").read_text(encoding="utf-8"))
    overall = json.loads((ARTIFACTS / "evaluation_overall.json").read_text(encoding="utf-8"))
    overall_by_horizon = json.loads((ARTIFACTS / "evaluation_by_horizon.json").read_text(encoding="utf-8"))
    per_sensor = pd.read_csv(ARTIFACTS / "evaluation_per_sensor.csv")
    replay = pd.read_csv(ARTIFACTS / "test_replay.csv", parse_dates=["timestamp"])
    per_sensor["sensor_id"] = per_sensor["sensor_id"].astype(str)
    replay["sensor_id"] = replay["sensor_id"].astype(str)
    _, graph_mapping, adjacency = load_graph(ROOT / "data" / "raw" / "adj_METR-LA.pkl", columns)
    selected = mapping_data["selected_sensors"]
    local_raw = raw[:, [columns.index(sensor) for sensor in selected]]
    filled, _ = past_only_filled(local_raw, config["train_end"])
    local_index = {sensor: i for i, sensor in enumerate(selected)}
    features = {kind: build_features(pd.DatetimeIndex(timestamps), filled, selected, local_index, mapping_data["neighbors"], kind == "graph", config["history_steps"]) for kind in ("base", "graph")}
    models = joblib.load(ARTIFACTS / "models.joblib")
    return timestamps, selected, mapping_data, config, overall, overall_by_horizon, per_sensor, replay, features, models, graph_mapping, adjacency, filled


def congestion_label(speed: float, thresholds: dict[str, float]) -> tuple[str, str]:
    if speed < thresholds["low_speed_cutoff"]:
        return "High congestion", "high"
    if speed < thresholds["high_speed_cutoff"]:
        return "Moderate congestion", "moderate"
    return "Low congestion", "low"


def light_chart_layout(**kwargs):
    """Keep Plotly on the same light token system as the Streamlit shell."""
    layout = dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="af, Inter, sans-serif", color="#171717", size=14),
        hoverlabel=dict(bgcolor="#1f1f29", font_color="#ffffff", font_size=14),
        margin=dict(l=48, r=24, t=24, b=48),
    )
    layout.update(kwargs)
    return layout


def render_landing() -> None:
    st.markdown('<div class="tfai-landing-nav"><div class="tfai-brand"><span class="tfai-brand-mark">🚦</span><span>TrafficFlow AI</span></div><div class="tfai-nav-note">TEAM MEOWMEOW · URBAN MOBILITY</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tfai-hero"><div class="tfai-hero-copy"><div class="tfai-landing-pill">AI-POWERED TRAFFIC FORECASTING</div><h1>Know the road<br><span>before you go.</span></h1><p>TrafficFlow AI forecasts traffic speed 15, 30, and 45 minutes ahead, so you can explore how congestion may change.</p></div></div>', unsafe_allow_html=True)
    action_col, note_col = st.columns([1, 2])
    with action_col:
        if st.button("Open forecast dashboard →", type="primary", key="open_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    with note_col:
        st.markdown('<div class="tfai-hero-note">Historical replay · 15 connected sensors · Random Forest + graph context</div>', unsafe_allow_html=True)
    st.markdown('<div class="tfai-landing-section"><div class="tfai-section-kicker">A clearer view of the commute</div><h2>From raw sensor readings<br>to a confident next step.</h2><div class="tfai-feature-grid"><div class="tfai-feature"><div class="tfai-feature-icon green">↗</div><h3>Forecast ahead</h3><p>Explore speed forecasts 15, 30, and 45 minutes into the future.</p></div><div class="tfai-feature"><div class="tfai-feature-icon blue">◌</div><h3>Understand the signal</h3><p>Compare autoregressive and Random Forest forecasts with held-out observations.</p></div><div class="tfai-feature"><div class="tfai-feature-icon ink">✦</div><h3>Explore the network</h3><p>Follow congestion across a small connected sensor network in one readable view.</p></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tfai-landing-band"><div><div class="tfai-section-kicker">BUILT FOR A TWO-MINUTE DEMO</div><h2>Simple enough to explain.<br>Useful enough to remember.</h2></div><div class="tfai-band-copy">Choose a sensor, replay a historical moment, and see the model make its call. No live APIs, no route planning, no black box infrastructure.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tfai-landing-footer"><span>TrafficFlow AI</span><span>Track 1 · Challenge 06 · Team meowmeow</span></div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="TrafficFlow AI", page_icon="🚦", layout="wide", initial_sidebar_state="collapsed")
    inject_design_css()
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if st.session_state.page == "landing":
        render_landing()
        return
    try:
        timestamps, selected, mapping_data, config, overall, overall_by_horizon, per_sensor, replay, features, models, graph_mapping, adjacency, filled = load_artifacts()
    except FileNotFoundError:
        st.error("Artifacts are missing. Run `python train.py` first.")
        st.stop()
    location_names = {sensor_id: f"Location {index + 1}" for index, sensor_id in enumerate(selected)}
    with st.container(key="top_nav"):
        brand_col, nav_col = st.columns([5, 1], vertical_alignment="center")
        with brand_col:
            st.markdown('<div class="tfai-topnav-name"><span>🚦</span> TrafficFlow AI</div>', unsafe_allow_html=True)
        with nav_col:
            if st.button("Home", key="back_home", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()
    with st.expander("Forecast options", expanded=False):
        location_col, time_col, option_horizon = st.columns([1.1, 2, 1])
        with location_col:
            sensor = st.selectbox(
                "Location",
                selected,
                format_func=lambda sensor_id: location_names[sensor_id],
                help="Choose which road location to forecast. Sensor IDs are anonymized in this dataset, so the choices use friendly labels.",
            )
        sensor_replay = replay[replay.sensor_id == sensor].sort_values("timestamp")
        with time_col:
            choice = st.selectbox(
                "Historical starting time",
                sensor_replay.timestamp.dt.strftime("%Y-%m-%d %H:%M").tolist(),
                index=max(0, len(sensor_replay) - 1),
                help="Choose a historical moment. The app forecasts speed 15, 30, and 45 minutes after this time.",
            )
        with option_horizon:
            horizon_minutes = st.selectbox("Headline forecast", config["horizons_minutes"], format_func=lambda value: f"{value} min", help="Choose which horizon to feature in the speed cards. The chart always shows all three horizons.")
        option_gap, option_rain, option_incident = st.columns(3)
        with option_gap:
            missing_minutes = st.selectbox("Missing-reading simulation", [0, 15, 30, 60], format_func=lambda value: "Off" if value == 0 else f"Last {value} min", help="Hide recent readings for the selected location. The model fills the gap using only the last earlier available reading.")
        with option_rain:
            rain_case = st.selectbox("Rain what-if", ["None", "Light rain", "Heavy rain"], help="METR-LA has no weather data. These are manual scenarios that apply fixed illustrative speed reductions.")
        with option_incident:
            incident_case = st.selectbox("Incident what-if", ["None", "Minor blockage", "Major incident"], help="METR-LA has no incident data. This applies a manual illustrative reduction to the model forecast.")
        st.caption("Scenario assumptions: light/heavy rain reduces speed by 3%/8%; a minor/major incident by 8%/20%. These are not learned effects or live reports.")
    st.caption("Historical replay · sensor IDs are anonymized, so road locations use friendly labels.")
    st.markdown('<div class="tfai-kicker">URBAN MOBILITY INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("🚦 TrafficFlow AI")
    st.caption("Historical replay · Team meowmeow · Track 1 / Challenge 06")
    st.info("This is a Historical replay of METR-LA observations, not live traffic.")
    selected_time = pd.Timestamp(choice)
    row = sensor_replay.iloc[(sensor_replay.timestamp - selected_time).abs().argmin()]
    future = float(row[f"actual_future_speed_{horizon_minutes}"])
    predicted = float(row[f"predicted_graph_speed_{horizon_minutes}"])
    horizon_steps = int(config["horizons_steps"][str(horizon_minutes)])
    timestamp_index = pd.DatetimeIndex(timestamps)
    forecast_idx = int(timestamp_index.get_indexer([selected_time], method="nearest")[0])
    feature_row = forecast_idx - int(config["history_steps"])
    graph_input = features["graph"][sensor][feature_row].copy()
    masked_steps = min(missing_minutes // config["sampling_minutes"], config["history_steps"])
    current_input_speed = float(row.current_speed)
    if masked_steps:
        replacement = float(filled[forecast_idx - masked_steps - 1, selected.index(sensor)])
        graph_input[:masked_steps] = replacement
        current_input_speed = float(replacement)
        model_for_horizon = models["graph"][horizon_steps][sensor]
        predicted = float(model_for_horizon.predict(graph_input.reshape(1, -1))[0])
    rain_reduction = {"None": 0.0, "Light rain": 0.03, "Heavy rain": 0.08}[rain_case]
    incident_reduction = {"None": 0.0, "Minor blockage": 0.08, "Major incident": 0.20}[incident_case]
    scenario_speed = predicted * (1 - rain_reduction) * (1 - incident_reduction)
    scenario_active = rain_case != "None" or incident_case != "None"
    label, status_class = congestion_label(predicted, config["congestion_thresholds"])
    c1, c2, c3 = st.columns(3)
    current_label = "Last known speed before gap" if masked_steps else "Current observed speed"
    c1.markdown(f'<div class="tfai-metric"><div class="tfai-metric-label">{current_label}</div><div class="tfai-metric-value">{current_input_speed:.1f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="tfai-metric"><div class="tfai-metric-label">Predicted speed in {horizon_minutes} min</div><div class="tfai-metric-value-row"><span class="tfai-metric-value">{predicted:.1f}</span><span class="tfai-metric-delta">{predicted - current_input_speed:+.1f}</span></div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="tfai-metric"><div class="tfai-metric-label">Actual future speed · {horizon_minutes} min</div><div class="tfai-metric-value">{future:.1f}</div></div>', unsafe_allow_html=True)
    if scenario_active:
        st.metric("Illustrative what-if speed", f"{scenario_speed:.1f}", delta=f"{scenario_speed - predicted:+.1f}")
    if masked_steps:
        st.info(f"Missing-data simulation: the latest {missing_minutes} minutes for {location_names[sensor]} are hidden from the model input and filled from the most recent earlier available value.")
    st.markdown(f'<div class="tfai-status"><span class="tfai-status-dot {status_class}">●</span><span>{label}</span><span class="tfai-status-tag">{horizon_minutes} MIN FORECAST</span></div>', unsafe_allow_html=True)
    st.caption(f"Prototype thresholds from training data only: high congestion < {config['congestion_thresholds']['low_speed_cutoff']:.1f}; moderate {config['congestion_thresholds']['low_speed_cutoff']:.1f}–{config['congestion_thresholds']['high_speed_cutoff']:.1f}; low ≥ {config['congestion_thresholds']['high_speed_cutoff']:.1f} speed units.")
    sensor_idx = selected.index(sensor)
    recent_start = max(config["history_steps"], forecast_idx - config["history_steps"] + 1)
    recent = pd.DataFrame({"timestamp": timestamp_index[recent_start : forecast_idx + 1], "Observed speed": features["base"][sensor][recent_start - config["history_steps"] : forecast_idx - config["history_steps"] + 1, 0]})
    if masked_steps:
        recent.loc[recent.index[-masked_steps:], "Observed speed"] = np.nan
    chart = go.Figure(go.Scatter(x=recent.timestamp, y=recent["Observed speed"], mode="lines+markers", name="Observed", line=dict(color="#282834", width=2), marker=dict(color="#282834", size=6)))
    forecast_points, actual_points, scenario_points = [], [], []
    for minutes in config["horizons_minutes"]:
        steps = int(config["horizons_steps"][str(minutes)])
        point_prediction = float(models["graph"][steps][sensor].predict(graph_input.reshape(1, -1))[0]) if masked_steps else float(row[f"predicted_graph_speed_{minutes}"])
        point_time = selected_time + pd.Timedelta(minutes=minutes)
        forecast_points.append((point_time, point_prediction))
        actual_points.append((point_time, float(row[f"actual_future_speed_{minutes}"])))
        if scenario_active:
            scenario_points.append((point_time, point_prediction * (1 - rain_reduction) * (1 - incident_reduction)))
    chart.add_trace(go.Scatter(x=[point[0] for point in forecast_points], y=[point[1] for point in forecast_points], mode="lines+markers", marker=dict(size=10, color="#41a1cf", line=dict(color="#ffffff", width=2)), line=dict(color="#41a1cf", dash="dash"), name="Forecast · 15/30/45 min"))
    chart.add_trace(go.Scatter(x=[point[0] for point in actual_points], y=[point[1] for point in actual_points], mode="lines+markers", marker=dict(size=9, color="#0081c0", line=dict(color="#ffffff", width=2)), line=dict(color="#0081c0", dash="dot"), name="Actual future"))
    if scenario_active:
        chart.add_trace(go.Scatter(x=[point[0] for point in scenario_points], y=[point[1] for point in scenario_points], mode="lines+markers", marker=dict(size=8, color="#d97706"), line=dict(color="#d97706", dash="longdash"), name="What-if scenario"))
    chart.update_layout(**light_chart_layout(
        yaxis_title="Speed",
        xaxis_title="Time",
        height=360,
        legend=dict(orientation="h", y=-0.18, font=dict(color="#171717", size=13)),
        xaxis=dict(showgrid=False, color="#171717", title_font=dict(color="#171717", size=14), tickfont=dict(color="#2c2c2c", size=13)),
        yaxis=dict(gridcolor="#c7cccf", color="#171717", title_font=dict(color="#171717", size=14), tickfont=dict(color="#2c2c2c", size=13)),
    ))
    st.plotly_chart(chart, use_container_width=True)
    st.subheader("Schematic sensor network")
    st.caption("No geographic coordinates are included in this dataset; node positions below are schematic only.")
    positions = {sid: (np.cos(2 * np.pi * i / len(selected)), np.sin(2 * np.pi * i / len(selected))) for i, sid in enumerate(selected)}
    graph_fig = go.Figure()
    for sid in selected:
        for other in mapping_data["neighbors"][sid]:
            if selected.index(sid) < selected.index(other):
                graph_fig.add_trace(go.Scatter(x=[positions[sid][0], positions[other][0]], y=[positions[sid][1], positions[other][1]], mode="lines", line=dict(color="#8a9298", width=1.4), hoverinfo="skip", showlegend=False))
    speeds = []
    for sid in selected:
        item = replay[(replay.sensor_id == sid) & (replay.timestamp == selected_time)]
        speeds.append(float(item[f"predicted_graph_speed_{horizon_minutes}"].iloc[0]) if not item.empty else np.nan)
    speeds[selected.index(sensor)] = scenario_speed if scenario_active else predicted
    colors = [congestion_label(speed, config["congestion_thresholds"])[0] if np.isfinite(speed) else "No test value" for speed in speeds]
    graph_fig.add_trace(go.Scatter(x=[positions[sid][0] for sid in selected], y=[positions[sid][1] for sid in selected], mode="markers+text", text=[location_names[sid] for sid in selected], textposition="top center", textfont=dict(color="#171717", size=13), marker=dict(size=18, color=[{"High congestion": "#1f1f29", "Moderate congestion": "#41a1cf", "Low congestion": "#f9faf7", "No test value": "#646464"}[c] for c in colors], line=dict(color="#2c2c2c", width=1.5)), hovertext=[f"{location_names[sid]} ({sid})<br>Predicted speed: {speed:.1f}<br>{color}" for sid, speed, color in zip(selected, speeds, colors)], hovertemplate="%{hovertext}<extra></extra>", name="Sensors"))
    graph_fig.update_layout(**light_chart_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="af, Inter, sans-serif", color="#171717", size=14), xaxis=dict(visible=False), yaxis=dict(visible=False), height=480, margin=dict(l=20, r=20, t=20, b=20), showlegend=False))
    st.plotly_chart(graph_fig, use_container_width=True)
    st.subheader("Overall model performance")
    selected_overall = overall_by_horizon[str(horizon_minutes)]
    st.caption(f"Test results across all selected road locations at the {horizon_minutes}-minute horizon, using identical held-out examples for each model.")
    overall_table = pd.DataFrame(selected_overall).T.rename_axis("model").reset_index()
    overall_table["model"] = overall_table["model"].replace({
        "persistence": "Persistence baseline",
        "autoregressive_linear": "Autoregressive linear model",
        "random_forest_base": "Random Forest · single location",
        "random_forest_graph": "Random Forest · connected locations",
    })
    st.dataframe(overall_table, hide_index=True, use_container_width=True)

    st.subheader("Model evaluation")
    persistence_mae = float(selected_overall["persistence"]["MAE"])
    graph_mae = float(selected_overall["random_forest_graph"]["MAE"])
    persistence_rmse = float(selected_overall["persistence"]["RMSE"])
    graph_rmse = float(selected_overall["random_forest_graph"]["RMSE"])
    sensor_scores = per_sensor[per_sensor.horizon_minutes == horizon_minutes].pivot(index="sensor_id", columns="model", values="MAE")
    graph_wins = int((sensor_scores["random_forest_graph"] < sensor_scores["persistence"]).sum())
    total_locations = len(sensor_scores)
    e1, e2, e3, e4 = st.columns(4)
    best_model_key = min(selected_overall, key=lambda name: selected_overall[name]["MAE"])
    best_model_name = {"persistence": "Persistence", "autoregressive_linear": "Auto-regressive", "random_forest_base": "Base RF", "random_forest_graph": "Graph RF"}[best_model_key]
    e1.metric("Best model", best_model_name)
    e2.metric("MAE improvement", f"{(persistence_mae - graph_mae) / persistence_mae:.1%}")
    e3.metric("RMSE improvement", f"{(persistence_rmse - graph_rmse) / persistence_rmse:.1%}")
    e4.metric("Locations improved", f"{graph_wins}/{total_locations}")
    st.caption("Lower MAE and RMSE scores indicate better predictions. Improvements are measured against the persistence baseline.")


if __name__ == "__main__":
    main()
