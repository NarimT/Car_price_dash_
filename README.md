
# Car_price_dash

Dash + Plotly web app for car-price **bin** prediction (interactive UI).  
This repository demonstrates an end-to-end workflow: EDA → model training & logging with MLflow → model registry → simple Dash app for inference and inspection.

---

## Features
- Interactive Dash UI to:
  - Enter MLflow Tracking URI, username, password
  - Specify registered model name and stage/version
  - Load a model from MLflow on-demand and display clear load/predict debug output
  - Enter car features (`year`, `max_power`, `mileage`, `brand`, `fuel`) and predict a price bin (0..3)
- Robust loading & inference:
  - Loads `models:/{name}/{version}` ; falls back to latest READY version
  - Attempts to load `assets.json` and `preprocessor` artifacts from the model run
  - If no preprocessor, deterministic fallback encoding for categorical inputs (brand→index, fuel→map)
  - Converts inputs to `numpy.float64` and adjusts shapes for an intercept column if required by the model signature
  - If PyFunc `.predict()` fails, tries to extract underlying estimator weights (e.g., `coef_` or custom `W`) and compute `X @ W` manually

---

## Quick start

> Recommended Python: **3.10** or **3.11** (better wheel support in CI)

1. Clone the repo:
```bash
git clone https://github.com/NarimT/Car_price_dash_.git
cd Car_price_dash_

2. Install dependencies:
python -m pip install --upgrade pip setuptools wheel
# IMPORTANT: ensure requirements.txt does NOT contain stdlib names like `time`
python -m pip install -r requirements.txt

3 Run the app 
export MLFLOW_TRACKING_URI="https://admin:password@mlflow.ml.brain.cs.ait.ac.th"
export MODEL_NAME="A3_st125983"
export MODEL_VERSION="1"
python app.py

