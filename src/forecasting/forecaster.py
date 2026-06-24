"""
Forecasting: next-day, next-7 and next-30 trading-day predictions.

Two pieces of machinery:
  1. Recursive roll-forward — predict one step, synthesise the implied OHLCV
     bar, recompute *causal* indicators, and feed the new window back in.
  2. MC-Dropout uncertainty — keep dropout active at inference and sample the
     network many times to obtain a predictive distribution, then widen the
     band with horizon to reflect compounding error.

Caveat (documented loudly in the README): recursive forecasts drift, and the
intervals capture *model* uncertainty, NOT true market risk. Treat the 7/30-day
paths as scenario illustrations, not tradeable signals.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..config import ForecastConfig
from ..utils.io import get_device

logger = logging.getLogger("nvda")


def _mc_dropout_predict(model, x: np.ndarray, n_samples: int,
                        device: str) -> tuple[float, float]:
    """Return (mean, std) of the target over ``n_samples`` dropout passes."""
    model.to(device).train()                     # dropout ON
    xb = torch.from_numpy(x.astype(np.float32)).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(xb).cpu().numpy().ravel()[0])
    model.eval()
    arr = np.asarray(preds)
    return float(arr.mean()), float(arr.std())


def _target_to_next_close(target_kind: str, value: float,
                          last_close: float) -> float:
    if target_kind == "log_return":
        return last_close * float(np.exp(value))
    return float(value)                          # already a price


def recursive_forecast(
    model,
    history: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    feature_cols: list[str],
    feature_scaler,
    target_scaler,
    target_kind: str,
    lookback: int,
    fcfg: ForecastConfig,
    max_horizon: int,
) -> pd.DataFrame:
    """
    Roll the model forward ``max_horizon`` trading days.

    Returns a DataFrame indexed by future business days with columns:
    ['pred_close', 'lower', 'upper'].
    """
    device = get_device()
    work = history.copy()
    rows = []
    cum_var = 0.0                                # accumulate predictive variance

    last_date = work.index[-1]
    future_dates = pd.bdate_range(last_date, periods=max_horizon + 1)[1:]

    for step, fdate in enumerate(future_dates, start=1):
        feat = feature_builder(work)
        feat = feat.dropna(subset=feature_cols)
        if len(feat) < lookback:
            logger.warning("Not enough history for forecast window; stopping.")
            break
        window = feat[feature_cols].to_numpy(np.float32)[-lookback:]
        window_scaled = feature_scaler.transform(window)[None, ...]

        mean_s, std_s = _mc_dropout_predict(
            model, window_scaled, fcfg.mc_dropout_samples, device)

        # Invert target scaling.
        mean = target_scaler.inverse_transform([[mean_s]])[0, 0]
        std = std_s * float(target_scaler.scale_[0])

        last_close = float(work["Close"].iloc[-1])
        pred_close = _target_to_next_close(target_kind, mean, last_close)

        # Propagate uncertainty into price space and let it compound.
        if target_kind == "log_return":
            price_sigma = pred_close * std
        else:
            price_sigma = std
        cum_var += price_sigma ** 2
        band = fcfg.ci_z * float(np.sqrt(cum_var))

        rows.append({
            "date": fdate,
            "pred_close": pred_close,
            "lower": pred_close - band,
            "upper": pred_close + band,
        })

        # Synthesise a placeholder bar so indicators can roll forward.
        new_bar = {
            "Open": pred_close, "High": pred_close, "Low": pred_close,
            "Close": pred_close, "Adj Close": pred_close,
            "Volume": float(work["Volume"].iloc[-21:].mean()),
        }
        work.loc[fdate] = {c: new_bar.get(c, np.nan) for c in work.columns}

    out = pd.DataFrame(rows).set_index("date")
    return out


def summarise_forecasts(forecast_df: pd.DataFrame,
                        horizons=(1, 7, 30)) -> dict:
    """Pull out the headline numbers for the requested horizons."""
    summary = {}
    for h in horizons:
        if len(forecast_df) >= h:
            row = forecast_df.iloc[h - 1]
            summary[f"day_{h}"] = {
                "date": str(forecast_df.index[h - 1].date()),
                "pred_close": float(row["pred_close"]),
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
            }
    return summary
