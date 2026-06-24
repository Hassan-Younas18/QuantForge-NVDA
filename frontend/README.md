# NVDA Forecasting Dashboard

React + TypeScript + Tailwind CSS v4 + Recharts dashboard for the
[NVDA forecasting pipeline](../README.md). Talks to the FastAPI backend in
[`../backend`](../backend) over `/api/*`.

## Run locally

```bash
npm install
npm run dev
```

Opens on `http://localhost:5173`. The dev server proxies `/api/*` requests to
`http://localhost:8000` (override with `VITE_API_PROXY_TARGET`), so start the
backend first — see [`../backend/README.md`](../backend/README.md).

## Build

```bash
npm run build      # type-checks (tsc -b) then builds to dist/
npm run preview     # serve the production build locally
```

## Structure

- `src/api/client.ts` — typed fetch wrapper over the backend REST API.
- `src/types/api.ts` — TypeScript types mirroring the backend's Pydantic schemas.
- `src/hooks/` — data-fetching (`useAsync`, `useStockData`) and the training-job poller (`useTrainJob`).
- `src/components/` — presentational pieces grouped by feature area (layout, charts, forecast, models, insights, common).
- `src/views/` — one component per nav section (Dashboard, History, Predictions, Models, Insights), composed in `App.tsx`.

See the project root [README.md, §8](../README.md#8-web-dashboard-fastapi--react) for the full local/Docker run instructions.
