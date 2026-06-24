"""
Explainability: model-agnostic permutation feature importance.

For each feature we shuffle its column across the test windows and measure the
increase in RMSE. A large increase => the model relied on that feature. This
works for every architecture here (and any black box), unlike gradient-based
methods that need framework-specific hooks. SHAP DeepExplainer is noted as a
deeper alternative but kept optional to avoid a heavy dependency.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from ..evaluation.metrics import rmse
from ..training.trainer import predict

logger = logging.getLogger("nvda")


def permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: list[str], n_repeats: int = 5,
                           seed: int = 42) -> Dict[str, float]:
    """
    X_test: (N, lookback, F). We permute one feature channel (across the time
    and sample axes) at a time and record the mean RMSE degradation.
    """
    rng = np.random.default_rng(seed)
    base = rmse(y_test.ravel(), predict(model, X_test))
    importance: Dict[str, float] = {}

    for f_idx, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            flat = Xp[:, :, f_idx].ravel()
            rng.shuffle(flat)
            Xp[:, :, f_idx] = flat.reshape(Xp[:, :, f_idx].shape)
            deltas.append(rmse(y_test.ravel(), predict(model, Xp)) - base)
        importance[name] = float(np.mean(deltas))

    ranked = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    top = list(ranked.items())[:5]
    logger.info("Top features: %s", ", ".join(f"{k}={v:.4f}" for k, v in top))
    return ranked
