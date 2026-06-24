"""
Evaluation metrics (RMSE, MAE, MAPE, R²) plus directional accuracy, and the
automatic model-selection logic that picks the winner from the bake-off.

Metrics are always computed in **price space** so different target choices
(return vs price) are compared on the same, interpretable footing.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("nvda")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    denom = np.where(np.abs(y) < 1e-8, np.nan, y)
    return float(np.nanmean(np.abs((y - yhat) / denom)) * 100)


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def directional_accuracy(prev_close, y_true, y_pred):
    """% of days the model gets the up/down direction right vs prev close."""
    true_dir = np.sign(y_true - prev_close)
    pred_dir = np.sign(y_pred - prev_close)
    mask = true_dir != 0
    return float(np.mean(true_dir[mask] == pred_dir[mask]) * 100)


def evaluate(y_true_price, y_pred_price, prev_close=None) -> Dict[str, float]:
    out = {
        "rmse": rmse(y_true_price, y_pred_price),
        "mae": mae(y_true_price, y_pred_price),
        "mape": mape(y_true_price, y_pred_price),
        "r2": r2(y_true_price, y_pred_price),
    }
    if prev_close is not None:
        out["dir_acc"] = directional_accuracy(prev_close, y_true_price,
                                              y_pred_price)
    return out


def comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    cols = [c for c in ["rmse", "mae", "mape", "r2", "dir_acc"] if c in df]
    return df[cols].sort_values("rmse")


def select_best(results: Dict[str, Dict[str, float]],
                metric: str = "rmse") -> str:
    """
    Pick the winner. For error metrics lower is better; for r2/dir_acc higher.
    Baseline rows (prefixed 'naive'/'arima') are excluded from selection but
    kept in the table for honest comparison.
    """
    candidates = {k: v for k, v in results.items()
                  if not k.startswith(("naive", "arima"))}
    higher_better = metric in ("r2", "dir_acc")
    best = max(candidates, key=lambda k: candidates[k][metric]) if higher_better \
        else min(candidates, key=lambda k: candidates[k][metric])
    logger.info("Selected best model: %s (%s=%.5f)",
                best, metric, candidates[best][metric])
    return best
