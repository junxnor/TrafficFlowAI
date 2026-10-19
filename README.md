# TrafficFlow AI

Hackathon prototype for Track 1: AI/ML Methodology, Challenge 06: Predicting Urban Traffic Congestion. Team **meowmeow**; team leader **KOK JIN YUAN**.

TrafficFlow AI predicts speed 15, 30, and 45 minutes ahead for 15 connected METR-LA sensors. It compares a persistence baseline, an autoregressive linear model, a Random Forest using the selected sensor's history, and a graph Random Forest that also uses neighboring sensor history. The Streamlit screen is explicitly a **Historical replay**, not a live traffic feed.

## Project layout

```text
TrafficFlowAI/
├── app.py                 # Streamlit entry point
├── train.py               # Training and evaluation command
├── trafficflow/            # Data loading and feature logic
├── tests/                  # Focused project checks
├── data/raw/               # Local METR-LA source files (not committed)
├── artifacts/              # Trained model and evaluation outputs (not committed)
├── assets/images/          # Dashboard wallpaper
├── theme/                  # Design tokens and Streamlit styles
└── tools/                  # Local Tailwind CLI
```

## Windows setup

From this folder, use Python 3.10–3.13 (Python 3.12 was used for the verified run):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the raw files in `data/raw/`: `METR-LA.h5` and `adj_METR-LA.pkl`. They are ignored by Git by default. The dashboard wallpaper is in `assets/images/`.

## Train and run

```powershell
python train.py
python -m pytest tests
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Training is offline and runs once; the app loads the saved artifacts from `artifacts/`.

## Design system and Tailwind

The app uses a landing page plus dashboard flow. Its styling is adapted from the supplied General Intelligence Company / literary-journal theme: parchment canvas, editorial serif display type, quiet sans-serif UI, hairline green-gray borders, Signal Blue outlined accents, and a restrained Dusk action band. The source copies live in `theme/`; Streamlit injects the compiled Tailwind CSS plus a small Streamlit-specific adapter. Tailwind CSS v4.3.3 is available as a standalone Windows CLI in `tools/tailwindcss.exe` and is ignored by Git because it is a platform binary.

To rebuild the CSS after changing the design tokens:

```powershell
.\tools\tailwindcss.exe -i .\theme\input.css -o .\theme\tailwind.css --minify
```

The app still has a readable fallback through `theme/variables.css` and `theme/dashboard.css` if the generated CSS is removed. The landing page's “Open forecast dashboard” button switches to the forecast view, and “Back to home” returns to the landing page.

## Data and preprocessing

The input is the METR-LA traffic speed benchmark, commonly distributed with the DCRNN project (https://github.com/liyaguang/DCRNN). This workspace file has 34,272 timestamps, 207 sensor columns, and 5-minute sampling, so 15 minutes is 3 steps. The HDF5 contains no prose documentation or units attribute; values are numeric speeds in the benchmark's speed units. The local pickle is treated as a trusted project input and contains `[sensor_ids, id_to_index, adjacency_matrix]`. The implementation validates that all three sensor orders agree before selecting a deterministic 15-node connected BFS subgraph.

The file contains 575,302 zero readings. Following the standard METR-LA preprocessing convention, non-positive values are treated as missing; this is a documented project decision because the supplied HDF5 itself does not label them. Inputs are forward-filled from past observations only. Leading gaps use per-sensor medians fitted on the training period. Targets remain raw and missing targets are excluded from evaluation. No future row is used to fill an input.

The timeline is split chronologically 60% train, 20% validation, 20% test. Target windows are kept within their split. At each horizon, persistence, autoregressive linear, base Random Forest, and graph Random Forest use identical held-out examples. Metrics are saved in `artifacts/evaluation_by_horizon.json`, `artifacts/evaluation_overall.json` (15-minute compatibility view), and `artifacts/evaluation_per_sensor.csv`.

The dashboard lets you choose 15-, 30-, or 45-minute forecasts and plots all three forecast points against later observations. “Simulate missing readings” hides the most recent 15, 30, or 60 minutes of the selected sensor's input and applies past-only filling. METR-LA does not contain weather or incident fields, so the rain and incident controls are manual what-if scenarios. Their speed reductions are fixed prototype assumptions, shown separately from the learned forecast; they are not live conditions or trained effects.

## Model terms

- **Random Forest:** many decision trees whose predictions are averaged, giving a compact nonlinear model for a laptop-sized prototype.
- **Autoregressive linear model:** a Ridge regression using the selected sensor's recent speed lags and calendar features as a conventional time-series comparison.
- **MAE:** average absolute speed error; lower is better.
- **RMSE:** square-root average squared error; lower is better and large misses count more.
- **Persistence baseline:** predicts that the future speed equals the current speed. The prototype reports honestly if a Random Forest does not beat it.

Congestion labels are prototype categories based only on the 33rd and 66th percentile speeds in the training subset. They are not road-authority thresholds.

## Two-minute demo

1. Run the training command and then launch Streamlit.
2. Keep the “Historical replay” banner visible; select one connected sensor and a historical forecast timestamp.
3. Show current observed speed, the 15-minute prediction, actual future speed, and the time-series forecast point.
4. Open the schematic sensor network to show predicted congestion colors; explain that no geographic coordinates were provided.
5. Show saved overall/per-sensor metrics and mention the persistence comparison and missing-data policy.

Limitations: this is a small offline sensor subset, not a live service; geographic coordinates are unavailable; the categories are illustrative; and Random Forests do not model long-range temporal dynamics or incidents.
