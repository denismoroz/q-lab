"""A small causal convolutional network over the last hours of wave features
(docs/research/waves-1-sequence-protocol-2026-11-05.md). Its output at hour t reads only hours up to
t (tested)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

WINDOW = 168


class CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, kernel: int = 3) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    def __init__(self, n_features: int, channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(CausalConv(n_features, channels, 1), nn.ReLU(),
                                 CausalConv(channels, channels, 4), nn.ReLU(),
                                 CausalConv(channels, channels, 16), nn.ReLU())
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: batch x features x time -> the probability logit at the last time step."""
        return self.head(self.net(x)[:, :, -1]).squeeze(-1)


def windows(x: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """batch x features x WINDOW: the WINDOW hours ending at each of `ends` (NaN -> 0)."""
    idx = ends[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
    w = x[np.clip(idx, 0, None)]
    w[idx < 0] = 0.0
    return np.nan_to_num(w).transpose(0, 2, 1).astype(np.float32)


def fit(x: np.ndarray, ends: np.ndarray, y: np.ndarray, seed: int = 7, epochs: int = 5,
        batch: int = 256, lr: float = 1e-3) -> TCN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TCN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        order = rng.permutation(len(ends))
        for k in range(0, len(order), batch):
            sel = order[k:k + batch]
            xb = torch.from_numpy(windows(x, ends[sel]))
            yb = torch.from_numpy(y[sel].astype(np.float32))
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def predict(model: TCN, x: np.ndarray, ends: np.ndarray, batch: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for k in range(0, len(ends), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(windows(x, ends[k:k + batch]))))
                       .numpy())
    return np.concatenate(out) if out else np.array([])


__all__ = ["TCN", "WINDOW", "fit", "predict", "windows"]
