"""
Deep-learning architectures for sequence-to-one forecasting.

All models share one contract:
    input : (batch, lookback, n_features)
    output: (batch, 1)              # next-step target (return or price)

MC-Dropout is available on every model (dropout layers stay active at
inference when ``model.train()`` is toggled), which the forecaster uses to
build predictive confidence intervals.
"""
from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn


class _Base(nn.Module):
    """Common base so the trainer can treat every model identically."""

    name: str = "base"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Recurrent family
# --------------------------------------------------------------------------- #
class LSTMModel(_Base):
    name = "lstm"

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class GRUModel(_Base):
    name = "gru"

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class BiLSTMModel(_Base):
    name = "bilstm"

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


# --------------------------------------------------------------------------- #
#  Convolutional-recurrent hybrid
# --------------------------------------------------------------------------- #
class CNNLSTMModel(_Base):
    """1-D conv stack extracts local patterns, LSTM models their evolution."""

    name = "cnn_lstm"

    def __init__(self, n_features, channels=64, hidden=64, layers=1,
                 dropout=0.2, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.ReLU(),
        )
        self.rnn = nn.LSTM(channels, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (B, L, F)
        z = self.conv(x.transpose(1, 2))  # -> (B, C, L)
        z = z.transpose(1, 2)             # -> (B, L, C)
        out, _ = self.rnn(z)
        return self.head(self.drop(out[:, -1]))


# --------------------------------------------------------------------------- #
#  Transformer encoder for time series
# --------------------------------------------------------------------------- #
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(_Base):
    name = "transformer"

    def __init__(self, n_features, d_model=64, nhead=4, layers=2,
                 dim_ff=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(self.drop(z[:, -1]))


# --------------------------------------------------------------------------- #
#  Registry + factory
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, Type[_Base]] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "bilstm": BiLSTMModel,
    "cnn_lstm": CNNLSTMModel,
    "transformer": TransformerModel,
}


def build_model(name: str, n_features: int, **kwargs) -> _Base:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choices: {list(REGISTRY)}")
    return REGISTRY[name](n_features=n_features, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
