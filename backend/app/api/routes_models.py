from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services import registry

router = APIRouter(prefix="/api/models", tags=["models"])


def _require_summary() -> dict:
    summary = registry.load_run_summary()
    if summary is None:
        raise HTTPException(status_code=404,
                            detail="No trained model yet — POST /api/train first.")
    return summary


@router.get("/comparison")
def model_comparison():
    summary = _require_summary()
    return {"best_model": summary["best_model"], "metrics": summary["metrics"]}


@router.get("/metrics")
def model_metrics():
    summary = _require_summary()
    return summary["metrics"]


@router.get("/importance")
def model_importance():
    _require_summary()
    importance = registry.load_feature_importance()
    if importance is None:
        raise HTTPException(status_code=404, detail="No feature-importance artefact yet.")
    return importance
