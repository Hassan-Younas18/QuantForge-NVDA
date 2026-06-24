from __future__ import annotations

from pydantic import BaseModel


class TrainRequest(BaseModel):
    ticker: str = "NVDA"
    years: int = 10
    target: str = "log_return"  # {"log_return", "price"}
    models: list[str] | None = None
    epochs: int = 80
    lookback: int = 60
    tune: bool = False


class TrainJobResponse(BaseModel):
    job_id: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    error: str | None = None


class ModelMetric(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    dir_acc: float | None = None


class ModelComparisonResponse(BaseModel):
    best_model: str
    metrics: dict[str, ModelMetric]
