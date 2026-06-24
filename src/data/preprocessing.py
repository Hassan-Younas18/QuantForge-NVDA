"""
Cleaning, outlier handling, leakage-free splitting, scaling and windowing.

Design principles:
  * Time order is sacred — splits are chronological, never shuffled.
  * Scalers are fit on the TRAIN split only, then applied to val/test.
  * Outliers are winsorised (not dropped) to preserve the daily cadence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import SplitConfig, WindowConfig

logger = logging.getLogger("nvda")


# --------------------------------------------------------------------------- #
#  Cleaning
# --------------------------------------------------------------------------- #
def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data errors on raw OHLCV."""
    df = df.copy()
    # Drop fully-empty rows, then forward-fill short gaps in prices only.
    df = df[~df[["Open", "High", "Low", "Close"]].isna().all(axis=1)]
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"]
                  if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "Volume" in df:
        df["Volume"] = df["Volume"].fillna(0)
    # Non-positive prices are data errors.
    df = df[(df[price_cols] > 0).all(axis=1)]
    n_missing = df.isna().sum().sum()
    if n_missing:
        logger.info("Dropping %d residual NaNs after fill", int(n_missing))
        df = df.dropna()
    return df


def winsorise_returns(df: pd.DataFrame, col: str = "log_return",
                      z: float = 6.0) -> pd.DataFrame:
    """
    Clip extreme daily returns at +/- z robust-sigmas. Using a high z (6)
    keeps genuine earnings-day moves while neutralising data-glitch spikes.
    """
    if col not in df:
        return df
    s = df[col]
    med = s.median()
    mad = (s - med).abs().median() * 1.4826  # robust sigma
    if mad == 0:
        return df
    lo, hi = med - z * mad, med + z * mad
    n_clip = int(((s < lo) | (s > hi)).sum())
    if n_clip:
        logger.info("Winsorising %d extreme '%s' values", n_clip, col)
    df = df.copy()
    df[col] = s.clip(lo, hi)
    return df


# --------------------------------------------------------------------------- #
#  Splitting
# --------------------------------------------------------------------------- #
@dataclass
class SplitIndices:
    train: slice
    val: slice
    test: slice


def chronological_split(n: int, cfg: SplitConfig) -> SplitIndices:
    i_train = int(n * cfg.train_frac)
    i_val = int(n * (cfg.train_frac + cfg.val_frac))
    return SplitIndices(slice(0, i_train), slice(i_train, i_val), slice(i_val, n))


# --------------------------------------------------------------------------- #
#  Target construction
# --------------------------------------------------------------------------- #
def make_target(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Next-step supervised target.
      * 'log_return': y_t = log(C_{t+1}/C_t)  -> stationary, honest.
      * 'price'     : y_t = C_{t+1}           -> level; flatters metrics.
    """
    close = df["Close"]
    if target == "price":
        return close.shift(-1)
    if target == "log_return":
        return np.log(close.shift(-1) / close)
    raise ValueError(f"Unknown target '{target}'")


# --------------------------------------------------------------------------- #
#  Scaling + windowing
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    test_dates: pd.DatetimeIndex
    close_test: np.ndarray          # aligned Close_t for return->price recovery


def _windowize(X: np.ndarray, y: np.ndarray, lookback: int):
    """Turn (T, F) into overlapping (N, lookback, F) windows -> next-step y."""
    xs, ys, idx = [], [], []
    for t in range(lookback, len(X)):
        xs.append(X[t - lookback:t])
        ys.append(y[t])
        idx.append(t)
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32).reshape(-1, 1),
            np.asarray(idx))


def build_dataset(df: pd.DataFrame, feature_cols: list[str], target: str,
                  split_cfg: SplitConfig, win_cfg: WindowConfig) -> Dataset:
    """
    Full pipeline: target -> chronological split -> fit scalers on train only
    -> window each split independently (so a test window never peeks into
    training rows and vice-versa).
    """
    work = df.copy()
    work["__y__"] = make_target(work, target)
    work = work.dropna(subset=feature_cols + ["__y__"])

    X = work[feature_cols].to_numpy(np.float32)
    y = work["__y__"].to_numpy(np.float32)
    close = work["Close"].to_numpy(np.float32)
    dates = work.index

    sp = chronological_split(len(work), split_cfg)

    f_scaler = StandardScaler().fit(X[sp.train])
    t_scaler = StandardScaler().fit(y[sp.train].reshape(-1, 1))
    Xs = f_scaler.transform(X)
    ys = t_scaler.transform(y.reshape(-1, 1)).ravel()

    def cut(s: slice):
        return _windowize(Xs[s], ys[s], win_cfg.lookback)

    Xtr, ytr, _ = cut(sp.train)
    Xva, yva, _ = cut(sp.val)
    Xte, yte, idx_te = cut(sp.test)

    # Map test-window targets back to absolute dates / close prices.
    test_abs = np.arange(len(work))[sp.test][idx_te]
    logger.info("Windows -> train %d | val %d | test %d",
                len(Xtr), len(Xva), len(Xte))
    return Dataset(
        Xtr, ytr, Xva, yva, Xte, yte,
        feature_names=feature_cols,
        feature_scaler=f_scaler, target_scaler=t_scaler,
        test_dates=dates[test_abs], close_test=close[test_abs],
    )
