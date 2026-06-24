"""
Generic training engine shared by every architecture: mini-batch SGD with
Adam, gradient clipping, ReduceLROnPlateau, early stopping on validation loss
and best-checkpoint restoration.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import MODEL_DIR, TrainConfig
from ..utils.io import get_device

logger = logging.getLogger("nvda")


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_run: int
    history: dict = field(default_factory=dict)
    checkpoint_path: str = ""


def _loaders(ds, batch_size, num_workers):
    def mk(X, y, shuffle):
        tds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=False)
    # Shuffling *windows* (not time steps) is fine and aids optimisation;
    # leakage is already prevented by the chronological split upstream.
    return mk(ds.X_train, ds.y_train, True), mk(ds.X_val, ds.y_val, False)


def train_model(model, ds, cfg: TrainConfig, tag: str = "model") -> TrainResult:
    device = get_device(cfg.device)
    model.to(device)
    train_loader, val_loader = _loaders(ds, cfg.batch_size, cfg.num_workers)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 3))
    loss_fn = torch.nn.MSELoss()

    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    hist = {"train_loss": [], "val_loss": []}
    ckpt = Path(MODEL_DIR) / f"{tag}.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb), yb).item() * len(xb)
        va_loss /= len(val_loader.dataset)
        sched.step(va_loss)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)

        if va_loss < best_val - cfg.min_delta:
            best_val, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt)          # checkpoint best only
        else:
            bad += 1
            if bad >= cfg.patience:
                logger.info("[%s] early stop @ epoch %d (best val %.6f)",
                            tag, epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)          # restore best weights
    logger.info("[%s] done: best val %.6f @ epoch %d", tag, best_val, best_epoch)
    return TrainResult(best_val, best_epoch, hist, str(ckpt))


@torch.no_grad()
def predict(model, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or get_device()
    model.to(device).eval()
    xb = torch.from_numpy(X.astype(np.float32)).to(device)
    return model(xb).cpu().numpy().ravel()
