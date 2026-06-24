"""
Thin wrappers around the existing data/feature pipeline (src/data, src/features).
No re-implementation — every function here just calls into the pipeline and
shapes the result for JSON.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import download_data, get_company_info as _get_company_info
from src.data.preprocessing import clean_prices
from src.features.indicators import add_technical_indicators


def _history_df(ticker: str, years: int) -> pd.DataFrame:
    cfg = replace(Config().data, ticker=ticker, period_years=years)
    raw = download_data(cfg)
    return clean_prices(raw)


def get_history(ticker: str, years: int) -> list[dict]:
    df = _history_df(ticker, years)
    out = df.reset_index().rename(columns={"index": "Date"})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.replace({np.nan: None}).to_dict(orient="records")


def get_indicators(ticker: str, years: int) -> list[dict]:
    df = _history_df(ticker, years)
    feat = add_technical_indicators(df)
    cols = [
        "Close", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_10", "EMA_20", "EMA_50",
        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_mid", "BB_lower",
    ]
    cols = [c for c in cols if c in feat.columns]
    out = feat[cols].reset_index().rename(columns={"index": "Date"})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.replace({np.nan: None}).to_dict(orient="records")


def get_stock_info(ticker: str) -> dict:
    info = _get_company_info(ticker)
    df = _history_df(ticker, 1)
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    change = last_close - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    info.update({
        "last_close": last_close,
        "change": change,
        "change_pct": change_pct,
        "last_date": str(df.index[-1].date()),
        "volume": float(df["Volume"].iloc[-1]) if "Volume" in df else None,
    })
    return info
