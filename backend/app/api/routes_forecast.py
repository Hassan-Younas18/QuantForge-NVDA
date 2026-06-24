from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import data_service, registry
from ..services.forecast_service import ForecastUnavailable, refresh_forecast

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


def _with_pct_change(summary: dict, last_close: float) -> dict:
    out = {}
    for horizon, point in summary.items():
        pct = (point["pred_close"] - last_close) / last_close * 100 if last_close else 0.0
        out[horizon] = {**point, "pct_change": pct}
    return out


@router.get("")
def forecast(refresh: bool = Query(False)):
    if refresh:
        try:
            fc, fc_summary, model_name, last_close = refresh_forecast()
        except ForecastUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        path = [
            {"date": str(idx.date()), "pred_close": float(row.pred_close),
             "lower": float(row.lower), "upper": float(row.upper)}
            for idx, row in fc.iterrows()
        ]
        return {
            "model_name": model_name,
            "last_close": last_close,
            "path": path,
            "summary": _with_pct_change(fc_summary, last_close),
        }

    summary = registry.load_run_summary()
    fc_df = registry.load_forecast_csv()
    if summary is None or fc_df is None:
        raise HTTPException(status_code=404,
                            detail="No forecast available yet — POST /api/train first.")
    last_close = data_service.get_stock_info(summary["ticker"])["last_close"]
    path = [
        {"date": str(idx.date()), "pred_close": float(row.pred_close),
         "lower": float(row.lower), "upper": float(row.upper)}
        for idx, row in fc_df.iterrows()
    ]
    return {
        "model_name": summary["best_model"],
        "last_close": last_close,
        "path": path,
        "summary": _with_pct_change(summary["forecast"], last_close),
    }
