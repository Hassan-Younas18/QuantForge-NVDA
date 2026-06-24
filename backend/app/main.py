"""
FastAPI app exposing the existing NVDA forecasting pipeline (src/, main.py)
over REST. Interactive API docs are auto-generated at /docs and /redoc.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.io import setup_logging

from .api import routes_forecast, routes_insights, routes_models, routes_stock, routes_train
from .core.settings import settings

setup_logging()
logger = logging.getLogger("nvda.api")

app = FastAPI(
    title="NVDA Forecasting API",
    description="REST API over the NVDA deep-learning forecasting pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_stock.router)
app.include_router(routes_train.router)
app.include_router(routes_models.router)
app.include_router(routes_forecast.router)
app.include_router(routes_insights.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
