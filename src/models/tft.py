"""
Optional Temporal Fusion Transformer (TFT).

The TFT is a heavyweight, attention-based architecture with built-in variable
selection and interpretable attention. It is gated behind an optional
dependency (``pytorch-forecasting`` + ``pytorch-lightning``) because those
packages pin specific torch/lightning versions and materially inflate install
size. Enable by adding "tft" to ``Config.candidate_models`` once installed.

This module deliberately keeps a thin, self-contained interface and is skipped
automatically (with a logged message) if the dependency is missing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nvda")

TFT_AVAILABLE = False
try:  # pragma: no cover - optional heavy dependency
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    import lightning.pytorch as pl

    TFT_AVAILABLE = True
except Exception:  # noqa: BLE001
    logger.info("pytorch-forecasting not installed -> TFT disabled.")


def train_tft(df: pd.DataFrame, feature_cols: list[str], target: str,
              max_encoder_length: int = 60, max_epochs: int = 30):
    """
    Train a TFT on a long-format frame. Returns (model, predictions) or raises
    a clear error if the optional dependency is unavailable.

    NOTE: kept intentionally minimal — see pytorch-forecasting docs for the
    full quantile-loss / interpretation API.
    """
    if not TFT_AVAILABLE:
        raise ImportError(
            "TFT requires `pip install pytorch-forecasting lightning`."
        )

    data = df.copy().reset_index(drop=False)
    data["time_idx"] = np.arange(len(data))
    data["group"] = 0
    data[target] = data[target].astype(float)

    cutoff = int(len(data) * 0.85)
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,
        time_varying_unknown_reals=[target] + feature_cols,
        target_normalizer=GroupNormalizer(groups=["group"]),
        allow_missing_timesteps=True,
    )
    val = TimeSeriesDataSet.from_dataset(training, data, predict=True,
                                         stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=64)
    val_loader = val.to_dataloader(train=False, batch_size=256)

    tft = TemporalFusionTransformer.from_dataset(
        training, hidden_size=32, attention_head_size=2,
        dropout=0.2, learning_rate=1e-3,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=False,
                         enable_checkpointing=False, logger=False)
    trainer.fit(tft, train_loader, val_loader)
    preds = tft.predict(val_loader)
    return tft, np.asarray(preds).ravel()
