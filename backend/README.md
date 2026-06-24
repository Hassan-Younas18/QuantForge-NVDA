# NVDA Forecasting API

FastAPI service exposing the existing `src/` pipeline (and `main.py`) over REST.
Nothing in `src/` or `main.py`'s CLI behavior changes — this is a thin,
additive layer on top.

## Run locally

```bash
# from the project root, with the root venv active
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs` (Swagger) or `/redoc`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/stock/info?ticker=NVDA` | Company info, last price, day change, key stats |
| GET | `/api/stock/history?ticker=&years=` | OHLCV bars |
| GET | `/api/stock/indicators?ticker=&years=` | SMA/EMA/RSI/MACD/Bollinger series |
| POST | `/api/train` | Start a model bake-off (background job) → `{job_id}` |
| GET | `/api/train/status/{job_id}` | Poll job status/message |
| GET | `/api/train/result/{job_id}` | Full run summary once `status == completed` |
| GET | `/api/models/comparison` | Latest metrics table + winning model |
| GET | `/api/models/metrics` | Per-model RMSE/MAE/MAPE/R²/directional accuracy |
| GET | `/api/models/importance` | Permutation feature importance |
| GET | `/api/forecast?refresh=false` | 1–30 day forecast path + 1/7/30-day summary. `refresh=true` reloads the trained checkpoint + scalers and recomputes against the latest data without retraining |
| GET | `/api/insights` | Narrative: why the model was selected, recent trend, confidence |
| GET | `/api/eda` | Summary statistics (CAGR, volatility, skew, etc.) |

## Architecture notes

- `app/services/pipeline_service.py` calls `main.run(cfg)` directly — the same
  function the CLI uses — so training logic lives in exactly one place.
- `app/services/forecast_service.py` reloads the winning model checkpoint
  (`outputs/models/<name>.pt`) and the scalers persisted at training time
  (`outputs/models/scalers/*.joblib`) to produce a fresh forecast without
  retraining.
- `app/services/registry.py` reads whatever `main.run()` already persists
  under `outputs/results/` and `outputs/models/` — there's no separate
  database; the filesystem artefacts are the single source of truth.
- `app/core/jobs.py` is an in-memory job tracker for the async training
  endpoint. It's intentionally simple (single process, no persistence across
  restarts) — swap in Celery/RQ + Redis if you need multi-worker scaling.

## Configuration

Environment variables (prefix `NVDA_API_`):

| Variable | Default | Purpose |
|---|---|---|
| `NVDA_API_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins |
| `NVDA_API_DEFAULT_TICKER` | `NVDA` | Default ticker when not specified |
