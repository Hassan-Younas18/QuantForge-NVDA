from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services import insights_service, registry

router = APIRouter(prefix="/api", tags=["insights"])


@router.get("/insights")
def insights():
    return insights_service.build_insights()


@router.get("/eda")
def eda():
    data = registry.load_eda()
    if data is None:
        raise HTTPException(status_code=404, detail="No EDA available yet.")
    return data
