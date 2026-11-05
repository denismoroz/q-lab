"""One-coin forecast pilot (docs/research/forecast-pilot-*-protocol-2026-11-05.md): demeaned,
volatility-scaled future log returns as targets, fall/rise classes, and the measures of a forecast.
Every target statistic reads only outcomes that have already ended (tested)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qlab import ml_money as mm


def forward_return(logc: np.ndarray, h: int) -> np.ndarray:
    """Per decision t on a regular grid: logc[t + h] - logc[t] (NaN past the end)."""
    out = np.full(len(logc), np.nan)
    out[: len(logc) - h] = logc[h:] - logc[: len(logc) - h]
    return out


def z_target(logc: np.ndarray, h: int, window: int = 8760, min_obs: int = 2160
             ) -> tuple[np.ndarray, np.ndarray]:
    """(z of the next h-step return, z of the past h-step return), both scaled by the mean and
    standard deviation of the h-step returns that had ENDED by t, over the last `window` steps."""
    import pandas as pd

    fwd = forward_return(logc, h)
    ended = pd.Series(fwd).shift(h)  # the return that started h steps ago and ends at t
    mu = ended.rolling(window, min_periods=min_obs).mean().to_numpy()
    sd = ended.rolling(window, min_periods=min_obs).std().to_numpy()
    past = ended.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (fwd - mu) / sd, (past - mu) / sd


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a positive gets a higher score than a negative (ties count half)."""
    from scipy.stats import rankdata

    ok = ~(np.isnan(score) | np.isnan(label))
    s, y = score[ok], label[ok].astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def hit_rate(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y)) & (x != 0) & (y != 0)
    return float((np.sign(x[ok]) == np.sign(y[ok])).mean()) if ok.any() else float("nan")


def r2_vs_zero(pred: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(pred) | np.isnan(y))
    return float(1.0 - ((y[ok] - pred[ok]) ** 2).sum() / (y[ok] ** 2).sum()) if ok.any() else \
        float("nan")


class ForecastTCN(nn.Module):
    """The small causal encoder of `ml_money.TCNPolicy` with `n_out` linear outputs."""

    def __init__(self, n_static: int, n_out: int, ch: int = 8, hidden: int = 16) -> None:
        super().__init__()
        layers, c_in = [], mm.CHANNELS
        for d in (1, 4, 16):
            layers += [nn.ConstantPad1d((4 * d, 0), 0.0), nn.Conv1d(c_in, ch, 5, dilation=d),
                       nn.ReLU()]
            c_in = ch
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(ch + n_static, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, win: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.conv(win)[:, :, -1], static], dim=-1))


__all__ = ["ForecastTCN", "auc", "forward_return", "hit_rate", "pearson", "r2_vs_zero",
           "spearman", "z_target"]
