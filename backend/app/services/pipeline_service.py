"""
Training orchestration — calls main.run(cfg) directly. This is the same
function the CLI (`python main.py`) calls; the API just builds the same kind
of Config object `main.parse_args()` builds from argv, then runs it in a
background thread so the HTTP request returns immediately with a job id.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import main as pipeline_main
from src.config import Config

from ..core.jobs import job_store

logger = logging.getLogger("nvda.api")


def build_config(ticker: str, years: int, target: str, models: list[str] | None,
                  epochs: int, lookback: int, tune: bool) -> Config:
    cfg = Config()
    cfg.data = replace(cfg.data, ticker=ticker, period_years=years, target=target)
    cfg.train = replace(cfg.train, epochs=epochs)
    cfg.window = replace(cfg.window, lookback=lookback)
    cfg.tuning = replace(cfg.tuning, enabled=tune)
    if models:
        cfg.candidate_models = tuple(models)
    return cfg


def run_training_job(job_id: str, cfg: Config) -> None:
    job_store.update(job_id, status="running", message="Training in progress...")
    try:
        summary = pipeline_main.run(cfg)
        job_store.update(job_id, status="completed",
                         message=f"Done. Best model: {summary['best_model']}",
                         result=summary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job %s failed", job_id)
        job_store.update(job_id, status="failed", message="Training failed",
                         error=str(exc))
