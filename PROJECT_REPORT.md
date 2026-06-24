# Project Report — NVDA Deep-Learning Forecasting System

## 1. What this project is

This project is an end-to-end quantitative research and engineering system
that forecasts NVIDIA (NVDA) stock prices using deep learning, and exposes
that system through a production-style web application. It has two layers:

1. **A research pipeline** (`src/`, `main.py`) that downloads market data,
   engineers technical-analysis features, trains and compares six
   deep-learning architectures, selects a winner under a strict
   anti-leakage protocol, evaluates it against classical baselines, and
   produces multi-day forecasts with uncertainty bands.
2. **A web application** (`backend/`, `frontend/`) that wraps that pipeline
   in a REST API and a React dashboard, so the same research can be
   triggered, inspected, and consumed interactively rather than only from
   the command line.

The project is explicitly framed as a **research and engineering portfolio
piece, not a trading system**. A core design goal — reflected throughout the
codebase — is to make it *hard to fool yourself* about whether the models
have real predictive skill, because the most common failure mode in
financial ML is mistaking autocorrelation for prediction.

---

## 2. Problem framing and why it's harder than it looks

Naively, forecasting "tomorrow's NVDA close" looks like an ordinary
regression problem: build features, fit a sequence model, minimize error.
In practice, three properties of financial time series make that naive
approach actively misleading:

- **Extreme autocorrelation at the price level.** Today's close is an
  excellent predictor of tomorrow's close. A model that learns to copy the
  last observed price will score a high R² and low MAPE while having *zero*
  trading edge — it has simply learned the "naive random walk."
- **Near-efficiency.** If a simple, exploitable pattern in daily OHLCV data
  existed, it would be arbitraged away quickly. Genuine edge, if any, is
  small and unstable across regimes.
- **Non-stationarity.** NVDA's behavior during 2016–2019, the 2020 COVID
  shock, and the 2023–2025 AI-driven re-rating are different statistical
  regimes. A model fit across all of them is fit to a moving target.

The project's engineering choices are largely a response to these three
facts: predict returns instead of price levels, always benchmark against a
naive baseline, judge models on directional accuracy rather than level
error, and report uncertainty rather than a single point forecast.

---

## 3. Pipeline architecture and techniques used

### 3.1 Data acquisition — `src/data/loader.py`

- **Technique:** `yfinance` download with `auto_adjust=False`, requesting
  both `Close` and `Adj Close`, normalized against two real-world quirks:
  single-ticker downloads sometimes return `MultiIndex` columns, and
  `auto_adjust` can silently collapse the adjusted/unadjusted columns into
  one. Results are cached to **Parquet** keyed by ticker/period/interval.
- **Why:** Yahoo Finance's API is not contractually stable; defending
  against its known failure modes up front avoids silent data corruption
  rather than crashing loudly or — worse — training on wrong data. Parquet
  caching makes repeated runs instant and avoids hammering the API.
- **Engineering note:** the cache write was made **atomic** (write to a temp
  file, then `os.replace`) after a real bug surfaced during development: the
  web dashboard fires concurrent requests (history + indicators) for the
  same ticker, and two simultaneous cache-misses both writing to the same
  path produced a truncated, corrupt Parquet file. This is the kind of bug
  that only appears once you add concurrent consumers — a good example of
  why the web layer earns its keep even as a verification harness.

### 3.2 Cleaning & outlier handling — `src/data/preprocessing.py`

- **Technique:** forward-fill short gaps in price columns (not feature
  columns), drop non-positive/garbage rows, and **winsorize** daily
  log-returns at ±6 robust sigmas (median absolute deviation, not standard
  deviation, since MAD is robust to the very outliers being measured).
- **Why clip instead of drop:** dropping a row breaks the daily cadence
  that sequence models rely on (windows assume contiguous trading days). A
  high threshold (6σ) preserves genuine large moves (e.g. earnings
  reactions) while neutralizing data glitches.

### 3.3 Feature engineering — `src/features/indicators.py`

25–27 features, computed in **pure pandas/NumPy** (deliberately no TA-Lib,
which has notoriously painful native-build requirements in CI/cloud
environments):

| Family | Features |
|---|---|
| Returns | `log_return`, `return_5`, `return_21` |
| Trend | SMA/EMA at 10/20/50/200, `sma_50_200_ratio` |
| Momentum | RSI(14, Wilder's smoothing), MACD (line/signal/histogram), momentum(10), ROC(10) |
| Volatility | Bollinger %b and bandwidth, ATR(14), realized volatility(21) |
| Volume | OBV, volume ratio vs 20-day average |
| Calendar | day-of-week, month, sin/cos day-of-year encoding |
| Optional | days-to-earnings, earnings-window flag |

- **Why these specific indicators:** they're the standard technical-analysis
  toolkit (trend, momentum, volatility, volume) — a reasonable, well-understood
  basis for a feature set rather than an ad hoc one.
- **The one non-negotiable property: every feature is causal.** A feature's
  value at time *t* uses only data ≤ *t*. This sounds obvious but is the
  single most common source of look-ahead leakage in time-series ML (e.g. a
  centered rolling window, or a scaler fit on the full series). Getting this
  wrong silently inflates backtest performance in a way that doesn't survive
  contact with live data.
- **Cyclical calendar encoding** (`sin`/`cos` of day-of-year) is used instead
  of a raw integer so the model sees December 31 and January 1 as adjacent,
  not 364 days apart.

### 3.4 Leakage-free dataset construction — `build_dataset()`

Three guards, in order of how often they're violated in practice:

1. **Chronological split (70/15/15), never shuffled.** Random shuffling
   before splitting is the single most common time-series ML bug — it lets
   the model "see the future" during training.
2. **Scalers (`StandardScaler`) fit on the training split only**, then
   applied to validation/test. Fitting a scaler on the full dataset leaks
   future distributional information (mean/variance) into the training
   process.
3. **Per-split windowing.** Sliding windows are built independently *within*
   each split, so a test window can never reach back into training rows
   (and vice versa) — windowing before splitting would let test-set windows
   straddle the train/test boundary.

### 3.5 Target choice — return vs. price

The default target is **next-day log-return**, `y_t = log(C_{t+1}/C_t)`,
not the raw price. This is the project's central honesty mechanism:

- Log-returns are approximately stationary; price levels are not (a model
  trained on $20–$50 NVDA generalizes poorly to $200 NVDA without return-space
  training).
- Critically, a model minimizing MSE on price levels converges toward
  "predict tomorrow = today," because that's the loss-minimizing strategy
  under near-random-walk dynamics — and it will *look* excellent on
  RMSE/MAPE/R² while having no real skill. Training on returns doesn't
  eliminate this pathology but makes it visible (see §3.7).
- `--target price` is still supported (the brief many such projects are
  built against explicitly asks for price forecasts), with the README's
  *Honest Limitations* section explaining why its headline metrics are
  misleading on their own.

### 3.6 Model architectures — `src/models/deep_models.py`

All six candidates share one contract — `(batch, lookback=60, n_features) →
(batch, 1)` — so the trainer, evaluator, and forecaster are architecture-agnostic:

| Model | Mechanism | Why included |
|---|---|---|
| **LSTM** | Gated recurrence, 2 layers, hidden=64 | The standard long-memory sequence baseline |
| **GRU** | Lighter gating than LSTM, fewer parameters | Tests whether LSTM's extra gate complexity is paying for itself |
| **BiLSTM** | Forward + backward LSTM within each lookback window | Extra context *within* a fixed window (not leakage — the window itself only contains past data) |
| **CNN-LSTM** | 1-D convolution extracts local shape patterns, LSTM models their evolution | Tests whether local pattern extraction (à la candlestick shapes) helps before recurrence |
| **Transformer** | Self-attention + sinusoidal positional encoding over the lookback | Tests whether attention-based long-range dependency modeling beats recurrence |
| **TFT** *(optional, off by default)* | Variable selection + interpretable attention | Available in `src/models/tft.py` behind an optional heavy dependency (`pytorch-forecasting`) |

Every model exposes **MC-Dropout** (dropout layers stay active at inference
when the model is toggled to `.train()` mode) — used later for uncertainty
quantification (§3.9) without needing a separate probabilistic architecture.

### 3.7 Training — `src/training/trainer.py`

- **Technique:** Adam optimizer, MSE loss, gradient-norm clipping,
  `ReduceLROnPlateau` scheduling, **early stopping** on validation loss with
  best-checkpoint save/restore, full deterministic seeding (Python/NumPy/
  PyTorch RNGs + deterministic cuDNN), automatic device selection
  (CUDA/MPS/CPU).
- **Why:** this is a standard, well-tested recipe for stable training on a
  comparatively small, noisy financial dataset. Determinism is prioritized
  over raw training speed because this is a portfolio/research artifact
  where reproducibility matters more than shaving milliseconds.
- **Hyperparameter search** (`src/training/tuning.py`, opt-in via `--tune`):
  Optuna (TPE Bayesian search) over a small per-architecture grid (hidden
  size, layers, dropout, learning rate), with a deterministic grid-search
  fallback if Optuna isn't installed — the pipeline degrades gracefully
  rather than hard-failing on an optional dependency.

### 3.8 Evaluation — `src/evaluation/metrics.py`

- **Metrics:** RMSE, MAE, MAPE, R², and **directional accuracy** (% of days
  the model gets the up/down move right relative to the previous close).
- **All metrics are computed in price space**, even for the return-target
  model — return predictions are converted back to price (`price_t *
  exp(predicted_return)`) before scoring, so return-target and price-target
  runs are judged on the same, interpretable footing.
- **Why directional accuracy is the headline metric, not RMSE:** this is
  the project's strongest opinion. A model can post RMSE that's
  statistically indistinguishable from the naive baseline while telling you
  almost nothing about whether the price will go up or down — and
  direction is what a trading decision actually requires. Reporting both,
  side by side, is what prevents the "naive-predictor trap" from going
  unnoticed.

### 3.9 Baselines — `src/models/baseline.py`

- **Naive random walk** (`predict next price == current price`) is always
  computed and shown in every comparison table, never excluded as "too
  trivial." **Selection logic explicitly excludes baselines from winning**
  the bake-off (so the system always nominates a *learned* model) **but
  keeps them in every report** — the discipline is to never let the
  comparison table go out without its honesty check.
- **ARIMA** (`statsmodels`, optional) is available as a classical baseline,
  degrading to `NaN` if `statsmodels` is absent rather than crashing.

### 3.10 Forecasting — `src/forecasting/forecaster.py`

- **Recursive (roll-forward) multi-step forecasting:** predict one step →
  synthesize the implied OHLCV bar from that prediction → recompute the
  *same causal indicators* on the extended series → repeat for up to 30
  trading days.
- **MC-Dropout uncertainty:** ~200 stochastic forward passes per step (with
  dropout active) give a predictive mean and standard deviation; the
  resulting variance is **compounded across steps** (`cum_var += σ_t²`) so
  the confidence band visibly widens with horizon — a 30-day forecast has a
  much wider band than a 1-day one, reflecting genuinely compounding
  uncertainty rather than a constant-width band that would overstate
  near-term confidence and understate long-term uncertainty.
- **Why MC-Dropout over a dedicated probabilistic model:** it requires zero
  architecture changes and works identically across all six candidate
  models, at the cost of representing only *model* uncertainty, not full
  market risk — a limitation the project documents rather than hides.

### 3.11 Explainability — `src/explainability/explain.py`

- **Technique:** model-agnostic permutation feature importance — shuffle
  one feature channel across all test windows, measure the resulting RMSE
  increase, repeat 5× per feature and average.
- **Why this over gradient-based attribution (e.g. SHAP/Integrated
  Gradients):** it works identically for every architecture in the bake-off
  (recurrent, convolutional, attention-based) with no framework-specific
  hooks, at the cost of being more expensive to compute and only measuring
  marginal, not interactive, importance — an acceptable trade for a
  six-model bake-off where consistency of method matters more than maximal
  fidelity for any one model.

### 3.12 Reproducibility

Global seeding, deterministic cuDNN, a pinned `requirements.txt`,
Parquet-cached raw data, and every run's full configuration + metrics
serialized to `outputs/results/run_summary.json`. A `tests/smoke_test.py`
exercises the entire pipeline against synthetic data — every code path runs
with zero network dependency, which is what was used throughout development
to verify changes without waiting on live downloads.

---

## 4. The web application

### 4.1 Why a web layer, and how it's integrated

The web app is an **additive, non-breaking layer** on top of the CLI
pipeline — a deliberate constraint, not an afterthought. `backend/` calls
`main.run(cfg)` and the `src/` modules directly rather than re-implementing
any modeling logic; the CLI (`python main.py`) and the synthetic smoke test
were re-verified to behave identically after every change. The only
modifications to the original pipeline were small and additive:

- Persisting the fitted `StandardScaler`s (`outputs/models/scalers/*.joblib`)
  and the full feature-name list, so a server process can reload a trained
  model and re-forecast later **without retraining** — the CLI never needed
  this because it forecasts immediately after training in the same process.
- Persisting the *full* permutation-importance dictionary (previously only
  the top 10 were kept in `run_summary.json`), so the dashboard's
  feature-importance chart isn't artificially truncated.
- One new read-only helper (`get_company_info()`) for ticker metadata.

### 4.2 Backend — FastAPI (`backend/`)

- **Service layer** (`backend/app/services/`) is a thin orchestration layer:
  `pipeline_service` calls `main.run()` for training,
  `forecast_service` reloads a checkpoint + scalers to recompute a forecast
  on demand, `data_service` wraps the existing data/indicator functions,
  `registry` reads whatever `main.run()` already persists under `outputs/`
  (no separate database — the filesystem artefacts the pipeline already
  produces *are* the source of truth).
- **Async training jobs:** a model bake-off takes minutes on CPU, so
  `POST /api/train` returns a `job_id` immediately and runs training via
  FastAPI `BackgroundTasks`, tracked in a simple in-memory job store
  (`backend/app/core/jobs.py`). This is intentionally lightweight — no
  Celery/Redis — appropriate for a single-process deployment; the README
  notes the upgrade path if multi-worker scaling is ever needed.
- **REST surface:** stock info/history/indicators, train + status/result
  polling, model comparison/metrics/importance, forecast (cached or
  `?refresh=true` for a live recompute), insights, and EDA. Interactive
  documentation is generated automatically by FastAPI at `/docs`.

### 4.3 Frontend — React + TypeScript (`frontend/`)

- **Stack:** Vite, React 19, TypeScript, **Tailwind CSS v4** (CSS-first
  config via `@theme`, class-based dark mode), **Recharts** for
  visualization.
- **Why Recharts over Plotly:** smaller bundle size and more idiomatic
  composition with React/Tailwind; its `Brush` component directly satisfies
  the zoom-and-pan requirement on the historical price chart without a
  custom interaction layer.
- **Structure:** a typed API client (`src/api/client.ts`) and matching
  TypeScript types mirror the backend's Pydantic schemas; data-fetching
  hooks (`useAsync`, `useStockData`, `useTrainJob`) separate network state
  from presentation; components are grouped by feature area (layout,
  charts, forecast, models, insights) and composed into one view per nav
  section.
- **UI:** dark-mode-by-default, NVIDIA-green accent on a near-black
  Bloomberg/TradingView-style palette, loading states and error banners on
  every async view, responsive layout (grid stats collapse to two columns
  on narrow viewports, nav scrolls horizontally on mobile).

### 4.4 Deployment

Docker Compose runs two containers: the backend (installs the root +
backend `requirements.txt`, runs `uvicorn`) and the frontend (multi-stage
build — Node compiles the static bundle, then `nginx` serves it and proxies
`/api/*` to the backend container, so the browser only ever talks to one
origin in production). `outputs/` is volume-mounted so trained models and
cached data survive container restarts.

---

## 5. Results and what they actually show

From a real run against 10 years of NVDA daily data:

| Model | RMSE | MAE | MAPE | R² | Directional Accuracy |
|---|---|---|---|---|---|
| naive_rw (baseline) | 4.054 | 3.081 | 1.70% | 0.9701 | 0.0%* |
| **Transformer (selected)** | 4.056 | 3.076 | 1.70% | 0.9701 | 54.2% |
| CNN-LSTM | 4.083 | 3.096 | 1.71% | 0.9697 | 54.2% |
| BiLSTM | 4.075 | 3.102 | 1.71% | 0.9698 | 46.2% |
| GRU | 4.093 | 3.124 | 1.72% | 0.9695 | 49.3% |
| LSTM | 4.106 | 3.137 | 1.73% | 0.9693 | 46.2% |

*the naive baseline's directional accuracy is reported as 0% by
construction — it always predicts "no change," so it has no defined
direction to score.

**The result that matters more than the table:** every model — including
the winner — is statistically indistinguishable from the naive baseline on
RMSE/R², and directional accuracy hovers within a few points of a 50%
coin-flip. This is presented in the README and the dashboard's Insights
panel as the headline finding, not buried as a caveat — it's the honest,
reproducible conclusion the entire pipeline is built to surface.

---

## 6. Engineering practices used and why

| Practice | Where | Why |
|---|---|---|
| Config-as-dataclasses, single source of truth | `src/config.py` | Every run's parameters are reproducible and diffable |
| Synthetic-data smoke test, zero network dependency | `tests/smoke_test.py` | Verify the full pipeline without waiting on live downloads or risking flaky API calls |
| Additive-only integration (web layer never edits pipeline behavior) | `backend/`, `main.py` diff | The CLI remains a trustworthy, independently runnable artifact |
| Atomic file writes for shared caches | `src/data/loader.py` | Prevent corruption under concurrent readers/writers (caught via real usage, not theorized) |
| Graceful optional-dependency degradation | Optuna/statsmodels/lxml fallbacks | The pipeline never hard-fails because of an optional extra |
| Model-agnostic interfaces (shared `forward` contract, shared MC-Dropout) | `src/models/deep_models.py` | One trainer, one evaluator, one forecaster serve all six architectures |
| Typed API contracts | Pydantic (backend) + TypeScript (frontend) | Frontend/backend schema drift fails at compile/validation time, not in production |

---

## 7. Honest limitations (carried over from the README, restated for completeness)

- **The naive-predictor trap.** Minimizing MSE on price levels rewards
  "predict no change" — addressed by defaulting to return targets and
  always reporting directional accuracy next to RMSE/R².
- **Markets are near-efficient and non-stationary.** A model fit on one
  regime (e.g. pre-2023 NVDA) degrades in another (the 2023–25 AI re-rating).
  The codebase includes a walk-forward backtest (`src/evaluation/backtest.py`)
  as the more realistic test, separate from the single chronological split.
- **Recursive forecasts drift.** Errors compound step over step; the 30-day
  path is a scenario, not a point forecast — this is why the confidence
  band widens with horizon rather than staying constant.
- **MC-Dropout intervals measure model uncertainty only.** They do not
  capture macro shocks, fundamentals, or news — true tail risk is wider
  than the bands shown.
- **Not financial advice.** The system is a research and engineering
  artifact; past performance does not predict future results.

---

## 8. Technology summary

| Layer | Technology |
|---|---|
| Core ML | Python, PyTorch, scikit-learn, NumPy, pandas, Optuna, statsmodels |
| Data | yfinance, Parquet (pyarrow) |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS v4, Recharts |
| Deployment | Docker, Docker Compose, nginx |
| Testing | Synthetic-data smoke test (no network) |
