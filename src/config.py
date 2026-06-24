"""
Central configuration for the NVDA forecasting system.

Everything that controls a run lives here so experiments are reproducible
and a single source of truth governs paths, hyper-parameters and the model
search space. Values can be overridden at runtime via ``Config.from_overrides``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "data_cache"

for _d in (OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    ticker: str = "NVDA"
    period_years: int = 10            # how many years of history to request
    interval: str = "1d"             # daily bars
    # Target the model learns. "log_return" is the *honest* default; "price"
    # is provided because the brief asks for price forecasts, but see the
    # README/limitations on why level-prediction metrics flatter the model.
    target: str = "log_return"        # {"log_return", "price"}
    use_cache: bool = True
    cache_max_age_days: int = 1


@dataclass
class SplitConfig:
    # Chronological splits — NEVER shuffle time series.
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15            # remainder; kept explicit for clarity


@dataclass
class WindowConfig:
    lookback: int = 60                # input sequence length (trading days)
    horizon: int = 1                  # direct-prediction horizon (single step)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12                # early-stopping patience
    min_delta: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"              # {"auto", "cpu", "cuda", "mps"}
    num_workers: int = 0


@dataclass
class TuningConfig:
    enabled: bool = False             # turn on Optuna search
    n_trials: int = 20
    timeout_sec: int | None = 1800


@dataclass
class ForecastConfig:
    horizons: Sequence[int] = (1, 7, 30)
    mc_dropout_samples: int = 200     # for predictive uncertainty
    ci_z: float = 1.96                # 95% interval


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    # Which architectures to put in the automatic selection bake-off.
    candidate_models: Sequence[str] = (
        "lstm",
        "gru",
        "bilstm",
        "cnn_lstm",
        "transformer",
        # "tft",  # Temporal Fusion Transformer — enable if pytorch-forecasting
        #          is installed (see src/models/tft.py). Heavy; off by default.
    )
    # Metric used to pick the winner on the validation set (lower is better
    # for all error metrics; "r2" handled specially).
    selection_metric: str = "rmse"

    def to_dict(self) -> dict:
        return asdict(self)
