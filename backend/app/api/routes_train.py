from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..core.jobs import job_store
from ..schemas.models import TrainJobResponse, TrainRequest, TrainStatusResponse
from ..services.pipeline_service import build_config, run_training_job

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("", response_model=TrainJobResponse)
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    cfg = build_config(req.ticker, req.years, req.target, req.models,
                       req.epochs, req.lookback, req.tune)
    job = job_store.create()
    background_tasks.add_task(run_training_job, job.id, cfg)
    return TrainJobResponse(job_id=job.id)


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
def training_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return TrainStatusResponse(job_id=job.id, status=job.status,
                               message=job.message, error=job.error)


@router.get("/result/{job_id}")
def training_result(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not completed")
    return job.result
