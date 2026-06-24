"""
Re-forecast against the latest data using the already-trained best model,
without retraining. Reuses the exact same building blocks `main.run()` uses
for its forecast stage (`recursive_forecast`, the `_feature_frame` builder),
just reloading the model + scalers from disk instead of from a live training
run in memory.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial

import pandas as pd

import main as pipeline_main
from src.config import Config, ForecastConfig
from src.data.loader import download_data, get_earnings_dates
from src.data.preprocessing import clean_prices
from src.forecasting.forecaster import recursive_forecast, summarise_forecasts

from . import registry


class ForecastUnavailable(RuntimeError):
    pass


def refresh_forecast(years: int = 5, horizons: tuple[int, ...] = (1, 7, 30)) -> tuple[pd.DataFrame, dict, str, float]:
    summary = registry.load_run_summary()
    if summary is None:
        raise ForecastUnavailable("No trained model yet — run /api/train first.")

    feature_names = summary["feature_names"]
    f_scaler, t_scaler = registry.load_scalers()
    if f_scaler is None or t_scaler is None:
        raise ForecastUnavailable("Scalers not found — retrain to regenerate them.")

    model, model_name = registry.load_best_model(len(feature_names))
    if model is None:
        raise ForecastUnavailable("Best model checkpoint not found — retrain.")

    ticker = summary["ticker"]
    target = summary["target"]
    lookback = summary["config"]["window"]["lookback"]

    data_cfg = replace(Config().data, ticker=ticker, period_years=years, target=target)
    raw = clean_prices(download_data(data_cfg))
    earnings = get_earnings_dates(ticker)
    builder = partial(pipeline_main._feature_frame, earnings=earnings)

    fcfg = ForecastConfig(horizons=horizons)
    fc = recursive_forecast(model, raw, builder, feature_names, f_scaler, t_scaler,
                            target, lookback, fcfg, max(horizons))
    last_close = float(raw["Close"].iloc[-1])
    return fc, summarise_forecasts(fc, horizons), model_name, last_close
