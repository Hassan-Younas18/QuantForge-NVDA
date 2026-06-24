from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import data_service

router = APIRouter(prefix="/api/stock", tags=["stock"])


@router.get("/info")
def stock_info(ticker: str = Query("NVDA")):
    try:
        return data_service.get_stock_info(ticker)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/history")
def stock_history(ticker: str = Query("NVDA"), years: int = Query(10, ge=1, le=20)):
    try:
        return data_service.get_history(ticker, years)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/indicators")
def stock_indicators(ticker: str = Query("NVDA"), years: int = Query(10, ge=1, le=20)):
    try:
        return data_service.get_indicators(ticker, years)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
