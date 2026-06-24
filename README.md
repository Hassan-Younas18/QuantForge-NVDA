# NVDA Deep-Learning Forecasting System

A production-structured quantitative research project that downloads NVIDIA
(NVDA) history from Yahoo Finance, engineers technical features, runs an
**automatic bake-off across six deep-learning architectures**, selects the
best on a leakage-free validation split, evaluates on a held-out test set, and
produces 1 / 7 / 30-trading-day forecasts with confidence intervals.

> **This is a research and engineering portfolio project, not investment
> advice.** Read the *Honest Limitations* section before drawing any
> conclusion about tradeability. The single most important finding it
> reproduces is that, for daily price levels, a **naive random walk is a
> brutally strong baseline** that deep nets rarely beat in any economically
> meaningful way.

---

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full run (downloads ~10y NVDA, trains all 6 candidates, forecasts):
python main.py

# Common variants:
python main.py --years 10 --tune                  # + Optuna hyper-search
python main.py --target price                      # predict price levels
python main.py --models lstm gru transformer       # subset of models

# Verify the code without any network (synthetic data):
PYTHONPATH=. python tests/smoke_test.py
```

All artefacts are written to `outputs/`: trained weights (`models/`),
figures (`plots/`), and metrics/forecasts (`results/`).

---

## 2. Project structure

```
nvda_forecasting/
├── main.py                     # end-to-end orchestration (CLI)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py               # single source of truth for every setting
│   ├── data/
│   │   ├── loader.py           # yfinance download + parquet cache + earnings
│   │   └── preprocessing.py    # cleaning, outliers, splits, scaling, windowing
│   ├── features/
│   │   ├── indicators.py       # SMA/EMA/RSI/MACD/Bollinger/Momentum/ATR/OBV
│   │   └── sentiment.py        # OPTIONAL FinBERT news sentiment scaffold
│   ├── analysis/
│   │   └── eda.py              # summary stats + trend/vol/volume plots
│   ├── models/
│   │   ├── deep_models.py      # LSTM, GRU, BiLSTM, CNN-LSTM, Transformer
│   │   ├── tft.py              # OPTIONAL Temporal Fusion Transformer
│   │   └── baseline.py         # random-walk & ARIMA baselines
│   ├── training/
│   │   ├── trainer.py          # loop, early stopping, checkpointing, sched
│   │   └── tuning.py           # Optuna search (grid fallback)
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE/MAE/MAPE/R²/dir-acc + model selection
│   │   └── backtest.py         # walk-forward rolling retraining
│   ├── forecasting/
│   │   └── forecaster.py       # recursive multi-step + MC-dropout intervals
│   ├── explainability/
│   │   └── explain.py          # permutation feature importance
│   └── utils/                  # io (seed/device/json) + plotting
├── tests/
│   └── smoke_test.py           # full-spine check on synthetic data (no network)
└── outputs/                    # models · plots · results · data_cache
```

---

## 3. What each stage does (and the reasoning)

### 3.1 Data collection — `src/data/loader.py`
Downloads ≥10 years of daily OHLCV + Adjusted Close via `yfinance`
(`auto_adjust=False` so both `Close` and `Adj Close` survive). Hardened against
the two modern `yfinance` quirks — MultiIndex columns and the auto-adjust
column collapse — and caches to parquet so re-runs are instant and API-friendly.
Optionally pulls historical earnings dates.

### 3.2 Cleaning & outliers — `src/data/preprocessing.py`
Forward-fills short price gaps, drops non-positive/garbage rows, and
**winsorises** daily log-returns at ±6 robust sigmas. We clip rather than drop
because dropping a row breaks the daily cadence the sequence models depend on;
a high threshold preserves real earnings-day moves while killing data glitches.

### 3.3 Feature engineering — `src/features/indicators.py`
All indicators are **pure pandas/numpy** (no TA-Lib build pain) and **causal**
(value at *t* uses only data ≤ *t*). The set covers every item in the brief —
SMA, EMA, RSI (Wilder), MACD (line/signal/histogram), Bollinger Bands
(%b + bandwidth), momentum/ROC — plus ATR, OBV, realised volatility, return
horizons, and cyclical calendar encodings. 25 features by default.

### 3.4 EDA — `src/analysis/eda.py`
Summary stats (CAGR, annualised vol, skew, kurtosis, Sharpe, max draw days)
plus price/SMA/volume and return-distribution/rolling-volatility figures.

### 3.5 Leakage-free dataset — `build_dataset`
The three guards against the most common time-series ML mistakes:
1. **Chronological split** (70/15/15), never shuffled.
2. **Scalers fit on TRAIN only**, then applied to val/test.
3. **Per-split windowing** so a test window can't reach back into training rows.

The default learning target is the **next-day log-return** (stationary, honest),
with `--target price` available because the brief asks for price forecasts —
see §6 for why level metrics are misleading.

### 3.6 Automatic model selection — `src/models/` + `main.py`
Trains each candidate, evaluates **in price space** on the validation set, and
picks the winner by `selection_metric` (default RMSE). Candidates:

| Model | Idea |
|-------|------|
| LSTM | Gated recurrence; long-memory baseline |
| GRU | Lighter gating, fewer params |
| BiLSTM | Forward+backward context (within each window) |
| CNN-LSTM | Conv1d extracts local patterns → LSTM models their evolution |
| Transformer | Self-attention + positional encoding over the lookback |
| TFT *(optional)* | Variable selection + interpretable attention (`tft.py`) |

### 3.7 Training — `src/training/trainer.py`
Adam + MSE, gradient clipping, `ReduceLROnPlateau`, **early stopping** on
validation loss, **best-checkpoint** save/restore, deterministic seeding,
auto device (CUDA/MPS/CPU). Hyper-parameter search via **Optuna** (`--tune`),
with a deterministic grid fallback if Optuna isn't installed.

### 3.8 Evaluation — `src/evaluation/metrics.py`
RMSE, MAE, MAPE, R² — **plus directional accuracy** (the metric that actually
matters for trading) — all computed in price space so `return` and `price`
targets are judged on the same footing. Comparison table → CSV + bar chart.

### 3.9 Forecasting — `src/forecasting/forecaster.py`
Recursive roll-forward for 1/7/30 days: predict one step → synthesise the
implied bar → recompute causal indicators → repeat. **MC-Dropout** (dropout
left on at inference, sampled ~200×) yields a predictive distribution; the band
widens with horizon to reflect compounding error.

### 3.10 Explainability — `src/explainability/explain.py`
Model-agnostic **permutation importance**: shuffle each feature channel, measure
the RMSE increase. Works for every architecture (unlike framework-specific
gradient methods). SHAP DeepExplainer is noted as a deeper, optional follow-up.

### 3.11 Advanced (optional)
- **Walk-forward retraining** — `src/evaluation/backtest.py` (the realistic test).
- **News sentiment** — `src/features/sentiment.py` (FinBERT scaffold; mind look-ahead).
- **Earnings features** — days-to-earnings + earnings-window flag (built in).
- **Baselines** — random walk (always) + ARIMA (`baseline.py`).

---

## 4. Performance comparison (illustrative, from the synthetic smoke test)

Run on a synthetic geometric random walk so the numbers are reproducible
without market data. **Real NVDA numbers differ, but the *ranking pattern* is
the lesson.**

| Model        | RMSE   | MAE    | MAPE  | R²     | Dir-Acc |
|--------------|--------|--------|-------|--------|---------|
| **naive_rw** | **0.187** | **0.152** | **1.43%** | **0.980** | 50.0% |
| transformer  | 0.188 | 0.152 | 1.43% | 0.980 | 50.0% |
| cnn_lstm     | 0.188 | 0.153 | 1.44% | 0.979 | 44.7% |
| lstm         | 0.189 | 0.153 | 1.44% | 0.979 | 46.0% |
| gru          | 0.189 | 0.153 | 1.44% | 0.979 | 46.0% |
| bilstm       | 0.190 | 0.154 | 1.45% | 0.979 | 46.0% |

Read this carefully: every deep model posts a gorgeous **R² ≈ 0.98** — and is
still **beaten by predicting "tomorrow = today."** Directional accuracy hovers
at a coin-flip ~50%. High R²/low MAPE on price levels is an artefact of
autocorrelation, not predictive skill.

---

## 5. Final recommendation

For NVDA daily forecasting, the defensible recommendation is:

1. **Headline model:** the **Transformer** (or GRU) is the best *learned*
   model — competitive RMSE for the fewest pathologies and the cleanest
   attention-based interpretability story. The pipeline selects it
   automatically when it wins on validation RMSE.
2. **But always ship it next to the random-walk baseline** and judge it on
   **directional accuracy and walk-forward Sharpe**, not level RMSE/R².
3. **Predict returns, not prices** (`--target log_return`, the default), and
   treat the 7/30-day paths as scenario illustrations with widening intervals,
   not point bets.

A model is only worth deploying if it **beats the naive baseline on a
walk-forward backtest using a trading-relevant metric.** This repo is built to
make that test easy and honest.

---

## 6. Honest limitations & overfitting risks

- **The naive-predictor trap.** A network minimising MSE on price levels
  converges to "repeat the last price," which looks excellent on RMSE/MAPE/R²
  while having ~zero edge. We default to returns and report directional
  accuracy specifically to avoid fooling ourselves.
- **Markets are near-efficient & non-stationary.** Regimes shift (COVID, the
  2023–24 AI surge); a model fit on one regime degrades in the next. Use the
  walk-forward backtest to see decay.
- **Recursive forecasts drift.** Errors compound; 30-day paths are scenarios.
- **MC-Dropout intervals are *model* uncertainty, not market risk.** They
  understate true tail risk and ignore fundamentals, macro and news shocks.
- **Overfitting controls used:** chronological splits, train-only scaling,
  early stopping, dropout, weight decay, gradient clipping, and small search
  spaces. Even so, with enough features and tuning a model *will* fit noise —
  the walk-forward test is the real guard.
- **Not financial advice.** Past performance does not predict future results.

---

## 7. Reproducibility

Global seeding (`set_seed`), deterministic cuDNN, pinned `requirements.txt`,
parquet-cached raw data, and every run's full config + metrics serialised to
`outputs/results/run_summary.json`.

---

## 8. Web dashboard (FastAPI + React)

The CLI pipeline above is also exposed as a REST API with a React dashboard on
top. Nothing about the CLI changes — `backend/` and `frontend/` are additive
layers that call into the same `src/` pipeline and `main.run()`.

### Run locally (two terminals)

```bash
# Terminal 1 — backend (from the project root, same venv as the CLI)
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
# Docs at http://localhost:8000/docs

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
# App at http://localhost:5173 (proxies /api -> :8000 automatically in dev)
```

### Run with Docker Compose

```bash
cp .env.example .env   # adjust ports/origins if needed
docker compose up --build
# Dashboard at http://localhost:3000 (nginx proxies /api -> the backend container)
# API directly at http://localhost:8000
```

The first training run (triggered from the **Models** tab, or `POST /api/train`)
populates `outputs/`, which is volume-mounted into the backend container so
artefacts survive restarts. See [`backend/README.md`](backend/README.md) for
the full endpoint reference.
