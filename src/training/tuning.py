"""
Hyper-parameter tuning. Uses Optuna when available (Bayesian/TPE search),
otherwise falls back to a small deterministic grid so the pipeline still runs.
The objective is validation loss from a short training budget.
"""
from __future__ import annotations

import logging
from dataclasses import replace

from ..config import TrainConfig, TuningConfig
from ..models.deep_models import build_model
from .trainer import train_model

logger = logging.getLogger("nvda")


# Per-architecture search spaces kept deliberately small for tractability.
SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
}


def _suggest_optuna(trial):
    return {
        "hidden": trial.suggest_categorical("hidden", SEARCH_SPACE["hidden"]),
        "layers": trial.suggest_categorical("layers", SEARCH_SPACE["layers"]),
        "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
        "lr": trial.suggest_categorical("lr", SEARCH_SPACE["lr"]),
    }


def tune_model(model_name: str, ds, base_train: TrainConfig,
               tcfg: TuningConfig) -> dict:
    """Return the best hyper-parameter dict for ``model_name``."""
    n_features = ds.X_train.shape[-1]
    # Short budget per trial keeps search affordable.
    trial_train = replace(base_train, epochs=min(25, base_train.epochs),
                          patience=6)

    def fit_eval(params):
        model_kwargs = {k: v for k, v in params.items() if k != "lr"}
        model = build_model(model_name, n_features, **model_kwargs)
        cfg = replace(trial_train, lr=params["lr"])
        res = train_model(model, ds, cfg, tag=f"tune_{model_name}")
        return res.best_val_loss

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            return fit_eval(_suggest_optuna(trial))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tcfg.n_trials,
                       timeout=tcfg.timeout_sec, show_progress_bar=False)
        logger.info("[tune:%s] best val %.6f params=%s",
                    model_name, study.best_value, study.best_params)
        return study.best_params
    except ImportError:
        logger.warning("Optuna missing -> small grid fallback for %s", model_name)
        best, best_params = float("inf"), {}
        grid = [
            {"hidden": h, "layers": l, "dropout": 0.2, "lr": lr}
            for h in (64, 128) for l in (1, 2) for lr in (1e-3, 3e-3)
        ]
        for params in grid:
            val = fit_eval(params)
            if val < best:
                best, best_params = val, params
        logger.info("[tune:%s] grid best val %.6f params=%s",
                    model_name, best, best_params)
        return best_params
