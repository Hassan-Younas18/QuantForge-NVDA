"""
Reads the artefacts the pipeline already persists under outputs/ — no
separate database, no duplicate bookkeeping. `run_summary.json` (written by
`main.run()`) is the single source of truth for "what was the last trained
state"; scalers + checkpoints are reloaded from outputs/models/ when a fresh
forecast is needed without retraining.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_DIR, RESULTS_DIR
from src.models.deep_models import build_model
from src.utils.io import load_json


def has_trained_artifacts() -> bool:
    return (RESULTS_DIR / "run_summary.json").exists()


def load_run_summary() -> dict | None:
    path = RESULTS_DIR / "run_summary.json"
    return load_json(path) if path.exists() else None


def load_model_comparison() -> pd.DataFrame | None:
    path = RESULTS_DIR / "model_comparison.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def load_feature_importance() -> dict | None:
    path = RESULTS_DIR / "feature_importance.json"
    return load_json(path) if path.exists() else None


def load_forecast_csv() -> pd.DataFrame | None:
    path = RESULTS_DIR / "forecast.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None


def load_eda() -> dict | None:
    path = RESULTS_DIR / "eda.json"
    return load_json(path) if path.exists() else None


def load_scalers():
    scaler_dir = MODEL_DIR / "scalers"
    f_path, t_path = scaler_dir / "feature_scaler.joblib", scaler_dir / "target_scaler.joblib"
    if not (f_path.exists() and t_path.exists()):
        return None, None
    return joblib.load(f_path), joblib.load(t_path)


def load_best_model(n_features: int):
    """Rebuild the winning architecture and load its trained weights."""
    summary = load_run_summary()
    if summary is None:
        return None, None
    name = summary["best_model"]
    ckpt = MODEL_DIR / f"{name}.pt"
    if not ckpt.exists():
        return None, None
    import torch

    model = build_model(name, n_features)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, name
