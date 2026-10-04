"""Recurrent and convolutional networks for the regime detector (docs/TASKS.md
T39; owner, 2026-10-04: «pytorch ставь, делай LSTM и CNN»).

Same protocol as detector_nets.py: input is the last 90 one-bar returns in
units of the series' volatility (oldest first), trained on hourly BTC with
hourly labels (every 4th hour), applied to daily data, UP and DOWN
separately, refit at the start of every month on data known before it.

Architectures, small and standard, nothing tuned:
- LSTM: one layer, 32 hidden units, the last state into a linear output;
- CNN: two 1-D convolutions (16 channels, kernel 5, ReLU), global average
  pooling, a linear output.
Adam (PyTorch's default learning rate 1e-3), batch 256, at most 30 epochs,
early stopping after 3 epochs without improvement on the chronologically
last tenth of the training rows; seed 0.

Development only (2020-04 .. 2023-12); `--holdout NAME` scores one variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from detector_binary import binary_scores, combine  # noqa: E402
from detector_nets import SEQ, STRIDE, sequence  # noqa: E402
from detector_timeframes import DEV, HOLDOUT, WINDOW, bar_labels, hourly_btc  # noqa: E402

from qlab import regime_detect as rd  # noqa: E402
from qlab.regimes import load, market_closes  # noqa: E402

EPOCHS, PATIENCE, BATCH = 30, 3, 256


class LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):  # x: (batch, SEQ)
        _, (h, _) = self.rnn(x.unsqueeze(-1))
        return self.out(h[-1]).squeeze(-1)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.ReLU(), nn.Conv1d(16, 16, 5), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)


ARCH = {"lstm": LSTM, "cnn": CNN}


def fit_predict(arch: str, x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    torch.manual_seed(0)
    model = ARCH[arch]()
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    split = int(len(x) * 0.9)
    xt, yt = torch.tensor(x[:split], dtype=torch.float32), torch.tensor(y[:split], dtype=torch.float32)
    xv, yv = torch.tensor(x[split:], dtype=torch.float32), torch.tensor(y[split:], dtype=torch.float32)
    best, best_state, waited = float("inf"), None, 0
    gen = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(BATCH):
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item() if len(xv) else 0.0
        if val < best - 1e-4:
            best, waited = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x_new, dtype=torch.float32))).numpy()


def walk(arch: str, train_x: pd.DataFrame, train_closes: pd.Series, daily_x: pd.DataFrame,
         target: str, window) -> pd.Series:
    cols = [f"r_{k}" for k in reversed(range(SEQ))]  # oldest first
    train_x, daily_x = train_x[cols].dropna(), daily_x[cols].dropna()
    bar = pd.Timedelta(hours=1)
    out = []
    for month in pd.date_range(window[0], window[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        labels = bar_labels(train_closes, month, bar)
        gap = max(pd.Timedelta(days=rd.EMBARGO_DAYS), WINDOW * bar)
        labels = labels[labels.index <= month - gap]
        rows = train_x.index.intersection(labels.index)[::STRIDE]
        y = (labels.loc[rows] == target).astype(int)
        part = daily_x[(daily_x.index >= month) & (daily_x.index < min(nxt, window[1]))]
        if part.empty or y.nunique() < 2:
            continue
        p = fit_predict(arch, train_x.loc[rows].to_numpy(), y.to_numpy(), part.to_numpy())
        out.append(pd.Series(p, index=part.index))
        print(f"  {arch} {target} {month:%Y-%m} trained on {len(rows)} rows", file=sys.stderr,
              flush=True)
    return pd.concat(out)


def main() -> None:
    daily, hourly = market_closes(), hourly_btc()
    labels = load().labels
    train_x, daily_x = sequence(hourly), sequence(daily)

    def run(arch: str, window) -> None:
        up = walk(arch, train_x, hourly, daily_x, "bull", window)
        down = walk(arch, train_x, hourly, daily_x, "bear", window)
        su, sd = binary_scores(up, labels, "bull"), binary_scores(down, labels, "bear")
        acc = rd.accuracy(combine(up, down), labels)["accuracy"]
        print(f"{arch:<6} | UP auc {su['auc']:.3f} prec {su['precision']:.2f} rec "
              f"{su['recall']:.2f} | DOWN auc {sd['auc']:.3f} prec {sd['precision']:.2f} rec "
              f"{sd['recall']:.2f} | both acc {acc:.3f}", flush=True)

    if "--holdout" in sys.argv:
        run(sys.argv[sys.argv.index("--holdout") + 1], HOLDOUT)
        return
    for arch in ARCH:
        run(arch, DEV)


if __name__ == "__main__":
    main()
