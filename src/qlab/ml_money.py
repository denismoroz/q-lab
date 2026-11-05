"""A hedge-share policy learned on money from minute data (docs/research/ml-15m-*-2026-11-05.md;
owner 2026-11-05: «обучать на минутке а чтобы работала на 15»; «или часе»).

At each decision (every `step` minutes) the policy reads the last `WINDOW` minutes of five
normalised channels, the coin's hourly wave features, the coin, and its current hedge share, and
outputs the share for the next step. It is trained by unrolling over consecutive decisions on the
money of the simplified book (spot plus hedge per unit of position), net of the cost of changing
the share and with funding on the short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

WINDOW = 240
CHANNELS = 5
COST = (7.5 + 0.9) / 1e4  # the stand's Binance cost for Bv2, per unit of share changed


def minute_channels(m: pd.DataFrame) -> np.ndarray:
    """minutes x 5: return and range in units of the past day's minute volatility, log volume and
    log trades as deviations from their past day, taker-buy imbalance. Rolling statistics end at
    the minute before (past only)."""
    c = m["close"]
    r = np.log(c).diff()
    sig = r.rolling(1440, min_periods=240).std().shift(1)
    rng = (m["high"] - m["low"]) / c
    lv = np.log1p(m["volume"])
    lt = np.log1p(m["trades"])

    def z(x):
        mu = x.rolling(1440, min_periods=240).mean().shift(1)
        sd = x.rolling(1440, min_periods=240).std().shift(1)
        return (x - mu) / sd

    imb = (m["taker_buy_base"] / m["volume"]).where(m["volume"] > 0) - 0.5
    out = np.column_stack([(r / sig).clip(-10, 10), (rng / sig).clip(0, 20), z(lv).clip(-5, 5),
                           imb.clip(-0.5, 0.5), z(lt).clip(-5, 5)])
    return np.nan_to_num(out).astype(np.float32)


def decision_grid(open_time: pd.DatetimeIndex, close: np.ndarray, step: int) -> pd.DataFrame:
    """Decision times every `step` minutes (the end of each step), the position of the last minute
    before each (the window's end), and the perp's return over the next step (NaN if a minute is
    missing at either end)."""
    open_time = pd.DatetimeIndex(open_time).as_unit("ns")
    first = open_time[0].ceil(f"{step}min")
    taus = pd.date_range(first + pd.Timedelta(minutes=step), open_time[-1],
                         freq=f"{step}min").as_unit("ns")
    pos = np.searchsorted(open_time.asi8, taus.asi8, side="left") - 1  # last minute before tau
    nxt = np.searchsorted(open_time.asi8, (taus + pd.Timedelta(minutes=step)).asi8,
                          side="left") - 1
    ok_now = (pos >= 0) & (open_time[np.clip(pos, 0, None)] == taus - pd.Timedelta(minutes=1))
    ok_next = (nxt < len(open_time)) & (open_time[np.clip(nxt, 0, len(open_time) - 1)]
                                        == taus + pd.Timedelta(minutes=step - 1))
    ret = np.full(len(taus), np.nan)
    good = ok_now & ok_next
    ret[good] = close[nxt[good]] / close[pos[good]] - 1.0
    return pd.DataFrame({"tau": taus, "pos": pos, "ret": ret, "ok": ok_now})


def window_summary(ch: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Mean of each channel over the WINDOW minutes ending at `pos` (inclusive)."""
    cs = np.vstack([np.zeros((1, ch.shape[1]), dtype=np.float64), np.cumsum(ch, axis=0)])
    hi = pos + 1
    lo = np.clip(hi - WINDOW, 0, None)
    return ((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)


def windows(ch: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """batch x channels x WINDOW, the minutes ending at `pos`."""
    idx = pos[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
    w = ch[np.clip(idx, 0, None)]
    w[idx < 0] = 0.0
    return w.transpose(0, 2, 1)


# ── policies ───────────────────────────────────────────────────────────────


class LinearPolicy(nn.Module):
    def __init__(self, n_static: int) -> None:
        super().__init__()
        self.lin = nn.Linear(n_static + CHANNELS + 1, 1)
        self.uses_window = False

    def encode(self, win, summ, static):
        return torch.cat([static, summ], dim=-1)

    def step(self, enc, prev):
        return torch.sigmoid(self.lin(torch.cat([enc, prev[:, None]], dim=-1))).squeeze(-1)


class TCNPolicy(nn.Module):
    def __init__(self, n_static: int, ch: int = 8, hidden: int = 16) -> None:
        super().__init__()
        layers, c_in = [], CHANNELS
        for d in (1, 4, 16):
            layers += [nn.ConstantPad1d((4 * d, 0), 0.0), nn.Conv1d(c_in, ch, 5, dilation=d),
                       nn.ReLU()]
            c_in = ch
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(ch + n_static + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.uses_window = True

    def encode(self, win, summ, static):
        shape = win.shape  # ... x channels x WINDOW
        flat = win.reshape(-1, shape[-2], shape[-1])
        e = self.conv(flat)[:, :, -1].reshape(*shape[:-2], -1)
        return torch.cat([e, static], dim=-1)

    def step(self, enc, prev):
        return torch.sigmoid(self.mlp(torch.cat([enc, prev[:, None]], dim=-1))).squeeze(-1)


def make_policy(kind: str, n_static: int) -> nn.Module:
    return LinearPolicy(n_static) if kind == "linear" else TCNPolicy(n_static)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def unroll(model, enc: torch.Tensor, ret: torch.Tensor, fund: torch.Tensor, scale: torch.Tensor,
           prev0: torch.Tensor, cost_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Shares and money per step over a batch of sequences (batch x steps)."""
    prev = prev0
    shares, money = [], []
    for t in range(enc.shape[1]):
        s = model.step(enc[:, t], prev)
        m = scale * ((1.0 - s) * ret[:, t] + s * fund[:, t] - cost_mult * COST * (s - prev).abs())
        shares.append(s)
        money.append(m)
        prev = s
    return torch.stack(shares, 1), torch.stack(money, 1)


def loss_of(money: torch.Tensor, penalty: str) -> torch.Tensor:
    if penalty == "sortino":
        down = torch.clamp(money, max=0.0)
        return -money.mean() / (torch.sqrt((down ** 2).mean()) + 1e-8)
    return -money.mean()


__all__ = ["COST", "LinearPolicy", "TCNPolicy", "decision_grid", "loss_of", "make_policy",
           "minute_channels", "n_params", "unroll", "window_summary", "windows"]
