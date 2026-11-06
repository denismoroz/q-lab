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


EMA_SPANS = (12, 24, 72, 168, 720)
EMA_CROSSES = ((12, 72), (24, 168), (72, 720))


def ema_features(close: np.ndarray) -> np.ndarray:
    """13 price-free EMA features per hourly bar (docs/research/forecast-pilot-ema-protocol):
    distance of the close to each EMA, three crossovers and each EMA's slope over a quarter of its
    span, all in units of the past 720 hours' hourly volatility scaled to the horizon. Reads only
    bars up to and including each bar."""
    import pandas as pd

    c = pd.Series(close).ffill()
    sig = np.log(c).diff().rolling(720, min_periods=360).std()
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in EMA_SPANS}
    cols = [(c / ema[n] - 1.0) / (sig * np.sqrt(n)) for n in EMA_SPANS]
    cols += [(ema[f] - ema[s]) / (c * sig * np.sqrt(s)) for f, s in EMA_CROSSES]
    cols += [(ema[n] - ema[n].shift(n // 4)) / (c * sig * np.sqrt(n / 4)) for n in EMA_SPANS]
    return np.column_stack([x.to_numpy() for x in cols]).clip(-20, 20)


STANDARD_NAMES = (
    [f"ema_dist_{n}" for n in EMA_SPANS] + [f"ema_cross_{f}_{s}" for f, s in EMA_CROSSES]
    + [f"ema_slope_{n}" for n in EMA_SPANS]
    + ["macd", "macd_signal", "macd_hist", "rsi_14h", "rsi_14d", "boll_b", "boll_width",
       "atr_rel", "atr_vs_median", "adx", "di_diff", "di_plus", "donchian_24", "donchian_168",
       "donchian_720", "obv_slope_24", "obv_slope_168", "vwap_24", "vwap_168"])


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def rsi_daily(c, n):
    """RSI over closes spaced 24 hours apart and ending at each hour (a daily RSI that is current
    at every hour; the audit found the calendar-day version one day stale at 00:00 decisions).
    `c`: hourly closes on a regular hourly index."""
    import pandas as pd

    out = pd.Series(np.nan, index=c.index)
    for k in range(24):
        sub = c.iloc[k::24]
        out.iloc[k::24] = _rsi(sub, n).to_numpy()
    return out


def winsorised(z: np.ndarray, limit: float = 3.0) -> np.ndarray:
    """z clipped to [-limit, limit] (NaN kept): the fat-tailed target's declared variant."""
    return np.clip(z, -limit, limit)


def standard_features(bars, day_index=None) -> np.ndarray:
    """The 32 standard indicators (docs/research/forecast-pilot-standard-protocol) on hourly bars
    (`bars`: DataFrame with close, high, low, volume, indexed by the bar's open time, UTC), each
    known at the bar's close and read only from bars up to it. RSI over 14 days is computed on
    closes 24 hours apart ending at the hour (`rsi_daily`)."""
    import pandas as pd

    c = bars["close"].ffill()
    hi, lo, v = bars["high"].ffill(), bars["low"].ffill(), bars["volume"].fillna(0.0)
    sig = np.log(c).diff().rolling(720, min_periods=360).std()
    ema = pd.DataFrame(ema_features(c.to_numpy()), index=bars.index)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    signal = macd.ewm(span=9, adjust=False).mean()
    scale = c * sig * np.sqrt(26)
    cols = [macd / scale, signal / scale, (macd - signal) / scale]
    cols.append((_rsi(c, 14) - 50) / 50)
    cols.append((rsi_daily(c, 14) - 50) / 50)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    cols += [(c - (ma - 2 * sd)) / (4 * sd) - 0.5, (4 * sd / ma) / (sig * np.sqrt(20))]
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    cols += [(atr / c) / sig, atr / atr.rolling(90 * 24, min_periods=30 * 24).median()]
    upm, dnm = hi.diff(), -lo.diff()
    plus = upm.where((upm > dnm) & (upm > 0), 0.0)
    minus = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    cols += [adx / 100, (pdi - mdi) / 100, pdi / 100]
    for n in (24, 168, 720):
        h_n, l_n = hi.rolling(n).max(), lo.rolling(n).min()
        cols.append((c - l_n) / (h_n - l_n) - 0.5)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    for n in (24, 168):
        cols.append((obv - obv.shift(n)) / v.rolling(n).sum())
    for n in (24, 168):
        vwap = (c * v).rolling(n).sum() / v.rolling(n).sum()
        cols.append((c / vwap - 1) / (sig * np.sqrt(n)))
    std = np.column_stack([ema.to_numpy()] + [x.to_numpy() for x in cols])
    return np.nan_to_num(std, nan=np.nan, posinf=np.nan, neginf=np.nan).clip(-20, 20)


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
