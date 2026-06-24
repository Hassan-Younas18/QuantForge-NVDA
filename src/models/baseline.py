"""
Baseline forecasters. The single most important sanity check in financial
forecasting is: *does the fancy model beat a naive random walk?* For price
levels it usually does NOT, because tomorrow's price is overwhelmingly
explained by today's. We always report these so the deep models are judged
against an honest bar.
"""
from __future__ import annotations

import numpy as np


def naive_last_value(close_t: np.ndarray) -> np.ndarray:
    """Random-walk: predict next price == current price."""
    return close_t.copy()


def naive_drift(close_t: np.ndarray, drift: float) -> np.ndarray:
    """Random-walk with drift in price space."""
    return close_t * (1.0 + drift)


def zero_return_forecast(n: int) -> np.ndarray:
    """For a return target, the naive baseline predicts zero return."""
    return np.zeros(n, dtype=np.float32)


def arima_forecast(train_close: np.ndarray, steps: int,
                   order=(5, 1, 0)) -> np.ndarray:
    """
    Optional classical baseline. Requires statsmodels; returns NaNs if absent
    so the pipeline degrades gracefully.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train_close, order=order).fit()
        return np.asarray(model.forecast(steps), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return np.full(steps, np.nan, dtype=np.float32)
