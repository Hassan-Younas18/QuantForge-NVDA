"""Exploratory data analysis: summary stats, trend/volatility/volume plots."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR
from ..utils.io import save_json
from ..utils.plotting import plot_price_history, plot_returns_distribution

logger = logging.getLogger("nvda")


def run_eda(df: pd.DataFrame, ticker: str) -> dict:
    """
    Produce a compact EDA summary + the key diagnostic plots, and persist them.

    Returns a dict of headline statistics (also written to results/eda.json).
    """
    close = df["Close"]
    rets = np.log(close / close.shift(1)).dropna()

    summary = {
        "rows": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "price_min": float(close.min()),
        "price_max": float(close.max()),
        "price_last": float(close.iloc[-1]),
        "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "cagr_pct": float(((close.iloc[-1] / close.iloc[0]) **
                           (252 / len(close)) - 1) * 100),
        "daily_vol_pct": float(rets.std() * 100),
        "annual_vol_pct": float(rets.std() * np.sqrt(252) * 100),
        "ann_sharpe_naive": float((rets.mean() / rets.std()) * np.sqrt(252)),
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurtosis()),
        "max_daily_gain_pct": float(rets.max() * 100),
        "max_daily_loss_pct": float(rets.min() * 100),
        "missing_values": int(df.isna().sum().sum()),
    }

    # Augment for the price plot if SMA columns are present.
    plot_df = df.copy()
    if "SMA_50" not in plot_df:
        plot_df["SMA_50"] = close.rolling(50).mean()
        plot_df["SMA_200"] = close.rolling(200).mean()

    plot_price_history(plot_df, ticker)
    plot_returns_distribution(rets, ticker)
    save_json(summary, RESULTS_DIR / "eda.json")

    logger.info("EDA: %d rows %s->%s | CAGR %.1f%% | ann.vol %.1f%%",
                summary["rows"], summary["start"], summary["end"],
                summary["cagr_pct"], summary["annual_vol_pct"])
    return summary
