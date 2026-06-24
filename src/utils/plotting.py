"""Centralised plotting helpers. All figures are saved to outputs/plots."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend so this runs on a server / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import PLOT_DIR

logger = logging.getLogger("nvda")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})


def _save(fig: plt.Figure, name: str) -> Path:
    path = PLOT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved plot -> %s", path)
    return path


def plot_price_history(df: pd.DataFrame, ticker: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df.index, df["Close"], lw=1, label="Close")
    if "SMA_50" in df:
        axes[0].plot(df.index, df["SMA_50"], lw=1, label="SMA 50", alpha=.8)
    if "SMA_200" in df:
        axes[0].plot(df.index, df["SMA_200"], lw=1, label="SMA 200", alpha=.8)
    axes[0].set_title(f"{ticker} closing price")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left")
    axes[1].bar(df.index, df["Volume"], width=1.0, color="grey", alpha=.6)
    axes[1].set_ylabel("Volume")
    return _save(fig, f"{ticker}_price_history.png")


def plot_returns_distribution(returns: pd.Series, ticker: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(returns.dropna(), bins=120, color="steelblue")
    axes[0].set_title("Daily log-return distribution")
    axes[1].plot(returns.index, returns.rolling(21).std() * np.sqrt(252))
    axes[1].set_title("Annualised rolling volatility (21d)")
    return _save(fig, f"{ticker}_returns_volatility.png")


def plot_model_comparison(metrics: pd.DataFrame, metric: str = "rmse") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = metrics.sort_values(metric)
    ax.barh(order.index, order[metric], color="teal")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model comparison ({metric.upper()}, lower is better)")
    ax.invert_yaxis()
    return _save(fig, f"model_comparison_{metric}.png")


def plot_actual_vs_pred(dates, actual, pred, title: str, fname: str,
                        lower=None, upper=None) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual", lw=1.4)
    ax.plot(dates, pred, label="Predicted", lw=1.4, alpha=.85)
    if lower is not None and upper is not None:
        ax.fill_between(dates, lower, upper, alpha=.2, color="orange",
                        label="95% interval")
    ax.set_title(title)
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    return _save(fig, fname)


def plot_feature_importance(importance: Mapping[str, float], fname: str,
                            top_n: int = 20) -> Path:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.3)))
    ax.barh(names, vals, color="indianred")
    ax.set_xlabel("Permutation importance (Δ RMSE)")
    ax.set_title("Feature importance")
    return _save(fig, fname)
