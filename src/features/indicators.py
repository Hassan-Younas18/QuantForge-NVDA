"""
Technical indicators — implemented in pure pandas/numpy so the project has
**no native TA-Lib dependency** (TA-Lib is painful to install in CI/cloud).

Every indicator below is causal: value at time t uses only data up to t,
which is essential to avoid look-ahead leakage in a forecasting model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(s: pd.Series, window: int = 20, k: float = 2.0):
    mid = sma(s, window)
    std = s.rolling(window).std()
    upper, lower = mid + k * std, mid - k * std
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, pct_b, bandwidth


def momentum(s: pd.Series, window: int = 10) -> pd.Series:
    return s.diff(window)


def roc(s: pd.Series, window: int = 10) -> pd.Series:
    return s.pct_change(window) * 100


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — volatility measure using H/L/C."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the full indicator set requested in the brief plus a few extras
    (log return, ATR, OBV, calendar) that materially help the models.
    Returns a new DataFrame; the caller decides which columns become features.
    """
    out = df.copy()
    close = out["Close"]

    # Returns (the model's natural input scale)
    out["log_return"] = np.log(close / close.shift(1))
    out["return_5"] = close.pct_change(5)
    out["return_21"] = close.pct_change(21)

    # Trend — SMA / EMA at several horizons
    for w in (10, 20, 50, 200):
        out[f"SMA_{w}"] = sma(close, w)
        out[f"EMA_{w}"] = ema(close, w)
    out["sma_50_200_ratio"] = out["SMA_50"] / out["SMA_200"]

    # Momentum
    out["RSI_14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = hist
    out["MOM_10"] = momentum(close, 10)
    out["ROC_10"] = roc(close, 10)

    # Volatility — Bollinger Bands + ATR + realised vol
    up, mid, low, pct_b, bw = bollinger_bands(close, 20, 2.0)
    out["BB_upper"], out["BB_mid"], out["BB_lower"] = up, mid, low
    out["BB_pct_b"], out["BB_bandwidth"] = pct_b, bw
    out["ATR_14"] = atr(out, 14)
    out["volatility_21"] = out["log_return"].rolling(21).std()

    # Volume
    out["OBV"] = obv(out)
    out["volume_sma_20"] = sma(out["Volume"], 20)
    out["volume_ratio"] = out["Volume"] / out["volume_sma_20"]

    # Calendar / seasonality
    idx = out.index
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    return out


def add_earnings_features(df: pd.DataFrame,
                          earnings_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Optional: distance-to-earnings and an earnings-window flag."""
    out = df.copy()
    if earnings_dates is None or len(earnings_dates) == 0:
        out["days_to_earnings"] = 0.0
        out["is_earnings_window"] = 0
        return out
    ed = np.sort(earnings_dates.values.astype("datetime64[D]"))
    idx = out.index.values.astype("datetime64[D]")
    pos = np.searchsorted(ed, idx)
    nxt = np.clip(pos, 0, len(ed) - 1)
    days = (ed[nxt] - idx).astype("timedelta64[D]").astype(float)
    out["days_to_earnings"] = days
    out["is_earnings_window"] = (np.abs(days) <= 2).astype(int)
    return out


# Default feature set used by the pipeline (numeric, causal, leakage-free).
DEFAULT_FEATURES = [
    "log_return", "return_5", "return_21",
    "SMA_10", "SMA_20", "SMA_50", "SMA_200",
    "EMA_10", "EMA_20", "EMA_50", "sma_50_200_ratio",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "MOM_10", "ROC_10",
    "BB_pct_b", "BB_bandwidth", "ATR_14", "volatility_21",
    "OBV", "volume_ratio",
    "day_of_year_sin", "day_of_year_cos",
]
