"""
End-to-end pipeline orchestration.

Run:
    python main.py                      # full run with defaults
    python main.py --years 10 --tune    # 10y history + hyper-parameter search
    python main.py --target price        # predict price levels instead of returns

Stages: download -> clean -> features -> EDA -> dataset -> model bake-off
-> selection -> test evaluation -> explainability -> multi-horizon forecast.
Every artefact (models, plots, JSON/CSV results) lands under outputs/.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd

import joblib

from src.config import Config, RESULTS_DIR, MODEL_DIR
from src.data.loader import download_data, get_earnings_dates
from src.data.preprocessing import clean_prices, winsorise_returns, build_dataset
from src.features.indicators import (add_technical_indicators,
                                     add_earnings_features, DEFAULT_FEATURES)
from src.analysis.eda import run_eda
from src.models.deep_models import build_model, count_params
from src.models.baseline import naive_last_value
from src.training.trainer import train_model, predict
from src.training.tuning import tune_model
from src.evaluation.metrics import evaluate, comparison_table, select_best
from src.explainability.explain import permutation_importance
from src.forecasting.forecaster import recursive_forecast, summarise_forecasts
from src.utils.io import setup_logging, set_seed, save_json, get_device
from src.utils.plotting import (plot_model_comparison, plot_actual_vs_pred,
                                plot_feature_importance)

logger = logging.getLogger("nvda")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _feature_frame(raw: pd.DataFrame, earnings) -> pd.DataFrame:
    """Build the full feature frame from raw OHLCV (used live + in forecasting)."""
    df = add_technical_indicators(raw)
    df = add_earnings_features(df, earnings)
    return df


def _to_price_space(pred_target, prev_close, target_kind):
    """Convert model output (return or price) to a comparable price."""
    if target_kind == "log_return":
        return prev_close * np.exp(pred_target)
    return pred_target


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> dict:
    set_seed(cfg.train.seed)
    logger.info("Device: %s | target: %s | candidates: %s",
                get_device(cfg.train.device), cfg.data.target,
                list(cfg.candidate_models))

    # 1. Data -------------------------------------------------------------- #
    raw = download_data(cfg.data)
    raw = clean_prices(raw)
    earnings = get_earnings_dates(cfg.data.ticker)

    # 2. Features + EDA ---------------------------------------------------- #
    feat = _feature_frame(raw, earnings)
    feat = winsorise_returns(feat, "log_return")
    run_eda(raw, cfg.data.ticker)

    feature_cols = [c for c in DEFAULT_FEATURES if c in feat.columns]
    if "days_to_earnings" in feat.columns:
        feature_cols += ["days_to_earnings", "is_earnings_window"]

    # 3. Dataset (leakage-free) ------------------------------------------- #
    ds = build_dataset(feat, feature_cols, cfg.data.target,
                       cfg.split, cfg.window)
    n_features = ds.X_train.shape[-1]

    # Persist scalers so a server process can re-forecast later without
    # retraining (the CLI itself never needs these files back).
    scaler_dir = MODEL_DIR / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ds.feature_scaler, scaler_dir / "feature_scaler.joblib")
    joblib.dump(ds.target_scaler, scaler_dir / "target_scaler.joblib")

    # Helper: convert a test-set target prediction to price for fair metrics.
    prev_close_test = ds.close_test
    if cfg.data.target == "log_return":
        y_true_price = prev_close_test * np.exp(
            ds.target_scaler.inverse_transform(ds.y_test).ravel())
    else:
        y_true_price = ds.target_scaler.inverse_transform(ds.y_test).ravel()

    # 4. Model bake-off ---------------------------------------------------- #
    results, trained = {}, {}
    for name in cfg.candidate_models:
        logger.info("=== Training %s ===", name)
        kwargs = {}
        if cfg.tuning.enabled:
            best = tune_model(name, ds, cfg.train, cfg.tuning)
            kwargs = {k: v for k, v in best.items() if k != "lr"}
            train_cfg = replace(cfg.train, lr=best.get("lr", cfg.train.lr))
        else:
            train_cfg = cfg.train
        model = build_model(name, n_features, **kwargs)
        logger.info("%s params: %d", name, count_params(model))
        train_model(model, ds, train_cfg, tag=name)

        pred_t = ds.target_scaler.inverse_transform(
            predict(model, ds.X_test).reshape(-1, 1)).ravel()
        pred_price = _to_price_space(pred_t, prev_close_test, cfg.data.target)
        results[name] = evaluate(y_true_price, pred_price, prev_close_test)
        trained[name] = model

    # Baseline (random walk) for honest comparison.
    naive_price = naive_last_value(prev_close_test)
    results["naive_rw"] = evaluate(y_true_price, naive_price, prev_close_test)

    # 5. Compare + select -------------------------------------------------- #
    table = comparison_table(results)
    table.to_csv(RESULTS_DIR / "model_comparison.csv")
    plot_model_comparison(table, cfg.selection_metric)
    logger.info("\n%s", table.round(4).to_string())

    best_name = select_best(results, cfg.selection_metric)
    best_model = trained[best_name]

    # 6. Test plot of the winner ------------------------------------------ #
    best_pred_t = ds.target_scaler.inverse_transform(
        predict(best_model, ds.X_test).reshape(-1, 1)).ravel()
    best_pred_price = _to_price_space(best_pred_t, prev_close_test,
                                      cfg.data.target)
    plot_actual_vs_pred(ds.test_dates, y_true_price, best_pred_price,
                        f"{cfg.data.ticker} — {best_name} (test set)",
                        f"{cfg.data.ticker}_best_test.png")

    # 7. Explainability ---------------------------------------------------- #
    importance = permutation_importance(best_model, ds.X_test, ds.y_test,
                                        ds.feature_names)
    plot_feature_importance(importance, f"{cfg.data.ticker}_feature_importance.png")
    save_json(importance, RESULTS_DIR / "feature_importance.json")

    # 8. Forecast next 1/7/30 trading days -------------------------------- #
    builder = partial(_feature_frame, earnings=earnings)
    fc = recursive_forecast(
        best_model, raw, builder, ds.feature_names,
        ds.feature_scaler, ds.target_scaler, cfg.data.target,
        cfg.window.lookback, cfg.forecast, max(cfg.forecast.horizons))
    fc.to_csv(RESULTS_DIR / "forecast.csv")
    fc_summary = summarise_forecasts(fc, cfg.forecast.horizons)

    recent = raw["Close"].iloc[-120:]
    plot_actual_vs_pred(
        list(recent.index) + list(fc.index),
        list(recent.values) + [np.nan] * len(fc),
        [np.nan] * len(recent) + list(fc["pred_close"]),
        f"{cfg.data.ticker} — {max(cfg.forecast.horizons)}-day forecast ({best_name})",
        f"{cfg.data.ticker}_forecast.png",
        lower=[np.nan] * len(recent) + list(fc["lower"]),
        upper=[np.nan] * len(recent) + list(fc["upper"]),
    )

    # 9. Persist run summary ---------------------------------------------- #
    run_summary = {
        "ticker": cfg.data.ticker,
        "target": cfg.data.target,
        "best_model": best_name,
        "metrics": results,
        "selection_metric": cfg.selection_metric,
        "forecast": fc_summary,
        "top_features": list(importance.items())[:10],
        "feature_names": ds.feature_names,
        "config": cfg.to_dict(),
    }
    save_json(run_summary, RESULTS_DIR / "run_summary.json")
    logger.info("DONE. Best=%s. Artefacts in outputs/.", best_name)
    return run_summary


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="NVDA deep-learning forecaster")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--target", choices=["log_return", "price"],
                   default="log_return")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of candidate models to run.")
    a = p.parse_args()

    cfg = Config()
    cfg.data = replace(cfg.data, ticker=a.ticker, period_years=a.years,
                       target=a.target)
    cfg.train = replace(cfg.train, epochs=a.epochs)
    cfg.window = replace(cfg.window, lookback=a.lookback)
    cfg.tuning = replace(cfg.tuning, enabled=a.tune)
    if a.models:
        cfg.candidate_models = tuple(a.models)
    return cfg


if __name__ == "__main__":
    setup_logging()
    run(parse_args())
