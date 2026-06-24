"""
Walk-forward (rolling-window) retraining backtest.

Instead of a single static train/test split, we repeatedly:
    train on [0 : t]  ->  predict the next ``step`` days  ->  advance t
This is the gold-standard evaluation for trading models because it mimics how
the model would actually be used in production and reveals whether edge decays
over time. It is expensive (many re-trains), so ``step`` is configurable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..data.preprocessing import build_dataset
from ..evaluation.metrics import evaluate
from ..models.deep_models import build_model
from ..training.trainer import train_model, predict

logger = logging.getLogger("nvda")


def walk_forward_backtest(feat: pd.DataFrame, feature_cols: list[str],
                          model_name: str, cfg: Config,
                          initial_train_days: int = 1000,
                          step: int = 60, n_folds: int = 6) -> pd.DataFrame:
    """
    Roll a re-trained model forward across the series. Returns a per-fold
    metrics DataFrame so you can see if performance is stable or degrading.
    """
    rows = []
    start = initial_train_days
    for fold in range(n_folds):
        end_train = start + fold * step
        end_test = end_train + step
        if end_test > len(feat):
            break
        sub = feat.iloc[:end_test].copy()

        # Re-derive a split where THIS fold's test window is the last `step`.
        train_frac = (end_train) / len(sub)
        local = Config()
        local.split.train_frac = train_frac * 0.85
        local.split.val_frac = train_frac * 0.15
        ds = build_dataset(sub, feature_cols, cfg.data.target,
                           local.split, cfg.window)

        model = build_model(model_name, ds.X_train.shape[-1])
        train_model(model, ds, cfg.train, tag=f"wf_{model_name}_{fold}")

        pred_t = ds.target_scaler.inverse_transform(
            predict(model, ds.X_test).reshape(-1, 1)).ravel()
        prev = ds.close_test
        if cfg.data.target == "log_return":
            y_true = prev * np.exp(
                ds.target_scaler.inverse_transform(ds.y_test).ravel())
            y_pred = prev * np.exp(pred_t)
        else:
            y_true = ds.target_scaler.inverse_transform(ds.y_test).ravel()
            y_pred = pred_t

        m = evaluate(y_true, y_pred, prev)
        m["fold"] = fold
        m["test_start"] = str(sub.index[end_train].date())
        rows.append(m)
        logger.info("[walk-forward] fold %d rmse=%.4f dir_acc=%.1f",
                    fold, m["rmse"], m.get("dir_acc", float("nan")))

    return pd.DataFrame(rows).set_index("fold")
