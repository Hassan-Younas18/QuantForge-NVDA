from __future__ import annotations

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: str
    pred_close: float
    lower: float
    upper: float


class ForecastHorizonSummary(BaseModel):
    date: str
    pred_close: float
    lower: float
    upper: float
    pct_change: float


class ForecastResponse(BaseModel):
    model_name: str
    last_close: float
    path: list[ForecastPoint]
    summary: dict[str, ForecastHorizonSummary]
