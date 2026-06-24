"""
Smoke test: runs the full spine on a synthetic OHLCV series so we verify the
code executes end-to-end WITHOUT touching Yahoo Finance. Not a substitute for
a real run — just a correctness/wiring check for CI.
"""
import logging
import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocessing import clean_prices, winsorise_returns, build_dataset
from src.features.indicators import (add_technical_indicators,
                                     add_earnings_features, DEFAULT_FEATURES)
from src.analysis.eda import run_eda
from src.models.deep_models import build_model, count_params, REGISTRY
from src.models.baseline import naive_last_value
from src.training.trainer import train_model, predict
from src.evaluation.metrics import evaluate, comparison_table, select_best
from src.explainability.explain import permutation_importance
from src.forecasting.forecaster import recursive_forecast, summarise_forecasts
from src.utils.io import setup_logging, set_seed, get_device
from dataclasses import replace
from functools import partial

setup_logging(logging.INFO)
set_seed(0)


def synthetic_ohlcv(n=1400, seed=0):
    """Geometric random walk with drift + vol clustering -> looks stock-ish."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    vol = 0.012 + 0.006 * np.abs(np.sin(np.arange(n) / 90))
    rets = rng.normal(0.0006, 1, n) * vol
    close = 20 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.integers(5e6, 5e7, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Adj Close": close, "Volume": volume}, index=dates).rename_axis("Date")


def main():
    cfg = Config()
    # Shrink everything so the smoke test finishes in seconds.
    cfg.train = replace(cfg.train, epochs=4, patience=3, batch_size=32)
    cfg.window = replace(cfg.window, lookback=30)
    cfg.forecast = replace(cfg.forecast, horizons=(1, 7, 30),
                           mc_dropout_samples=20)

    raw = clean_prices(synthetic_ohlcv())
    earnings = pd.DatetimeIndex([])
    feat = add_earnings_features(add_technical_indicators(raw), earnings)
    feat = winsorise_returns(feat, "log_return")
    run_eda(raw, "SYNTH")

    feature_cols = [c for c in DEFAULT_FEATURES if c in feat.columns]
    ds = build_dataset(feat, feature_cols, cfg.data.target, cfg.split, cfg.window)
    n_features = ds.X_train.shape[-1]
    print(f"\nfeatures={n_features} | train/val/test ="
          f" {len(ds.X_train)}/{len(ds.X_val)}/{len(ds.X_test)}")

    prev_close = ds.close_test
    y_true_price = prev_close * np.exp(
        ds.target_scaler.inverse_transform(ds.y_test).ravel())

    results, trained = {}, {}
    for name in REGISTRY:                     # test ALL architectures
        model = build_model(name, n_features)
        print(f"  {name:11s} params={count_params(model):>7d}", end=" ")
        train_model(model, ds, cfg.train, tag=name)
        pred_t = ds.target_scaler.inverse_transform(
            predict(model, ds.X_test).reshape(-1, 1)).ravel()
        pred_price = prev_close * np.exp(pred_t)
        results[name] = evaluate(y_true_price, pred_price, prev_close)
        trained[name] = model
    results["naive_rw"] = evaluate(y_true_price, naive_last_value(prev_close),
                                   prev_close)

    table = comparison_table(results)
    print("\n=== Comparison ===\n", table.round(4).to_string())
    best = select_best(results, "rmse")

    imp = permutation_importance(trained[best], ds.X_test, ds.y_test,
                                 ds.feature_names, n_repeats=2)
    print("Top-3 features:", list(imp.items())[:3])

    builder = partial(lambda r, earnings: add_earnings_features(
        add_technical_indicators(r), earnings), earnings=earnings)
    fc = recursive_forecast(trained[best], raw, builder, ds.feature_names,
                            ds.feature_scaler, ds.target_scaler,
                            cfg.data.target, cfg.window.lookback,
                            cfg.forecast, 30)
    print("\n=== Forecast head ===\n", fc.head().round(3).to_string())
    print("\nSummary:", summarise_forecasts(fc, (1, 7, 30)))
    print("\nSMOKE TEST PASSED (device:", get_device(), ")")


if __name__ == "__main__":
    main()
