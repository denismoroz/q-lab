"""Waves on the lower timeframe, trend on the higher one (owner 2026-11-05: «нужен механизм который
бы уверенно работал с волнами на более мелком тайфрейме и определял тренд на более высоком»;
docs/research/waves-*-2026-11-05.md). Pure functions; the runner is scripts/research/waves.py.

Every mechanism reads only the past:
- `directional_change`: event-time waves (a wave ends when the price turns by theta from its
  extreme); state = the direction of the last confirmed change; also the overshoots.
- `structure_state`: higher highs / lower lows of the confirmed zig-zag pivots.
- `causal_atrous`: one-sided Haar a-trous layers (no look-ahead at the edge, tested).
- `online_jump`: a two-state trend with a switching penalty, forward pass only.
- `confirmed_falls`: big-fall labels as known on a given day (for training the learned model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MINUTE_COLUMNS = ["open_time", "close", "volume", "taker_buy_base"]


def load_minutes(root: Path, coin: str, market: str = "futures") -> pd.DataFrame:
    """All months of one coin's minute candles, indexed by the minute's open time (UTC)."""
    folder = Path(root) / market / coin
    parts = [pd.read_parquet(p, columns=MINUTE_COLUMNS) for p in sorted(folder.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").set_index("open_time")
    return df.sort_index()


def daily_sigma(close: pd.Series, days: int = 30) -> pd.Series:
    """Per minute: the standard deviation of daily log returns over the `days` days before the
    minute's day (yesterday's reading)."""
    day_close = close.resample("D").last()
    sig = np.log(day_close).diff().rolling(days, min_periods=days // 2).std().shift(1)
    return sig.reindex(close.index.floor("D")).set_axis(close.index)


def directional_change(px: np.ndarray, theta: np.ndarray) -> dict:
    """Event-time waves. `theta[i]` is the reversal threshold (a fraction) in force at bar i.

    Returns `state` (+1 after an upturn, -1 after a downturn, 0 before the first), the event bars,
    their directions, the extreme each wave reached (`ext_bar`: where the wave that just ended
    peaked or bottomed), and the overshoot of each finished wave (how far the price went past
    the event that started it, in units of that event's theta)."""
    import math

    n = len(px)
    pl, tl = np.asarray(px, dtype=float).tolist(), np.asarray(theta, dtype=float).tolist()
    state_l = [0] * n
    events, dirs, ext_bars, overshoots = [], [], [], []
    mode = 0
    hi = lo = -1
    start_px = start_theta = math.nan
    for i in range(n):
        p, th = pl[i], tl[i]
        if p != p or th != th:  # NaN: nothing is decided on this bar
            state_l[i] = mode
            continue
        if hi < 0 or p > pl[hi]:
            hi = i
        if lo < 0 or p < pl[lo]:
            lo = i
        if mode >= 0 and p <= pl[hi] * (1.0 - th):
            if mode == 1:
                overshoots.append(math.log(pl[hi] / start_px) / start_theta)
            events.append(i)
            dirs.append(-1)
            ext_bars.append(hi)
            mode, lo, start_px, start_theta = -1, i, p, th
        elif mode <= 0 and p >= pl[lo] * (1.0 + th):
            if mode == -1:
                overshoots.append(math.log(start_px / pl[lo]) / start_theta)
            events.append(i)
            dirs.append(1)
            ext_bars.append(lo)
            mode, hi, start_px, start_theta = 1, i, p, th
        state_l[i] = mode
    state = np.array(state_l, dtype=np.int8)
    return {"state": state, "events": np.array(events, dtype=int),
            "dirs": np.array(dirs, dtype=int), "ext_bars": np.array(ext_bars, dtype=int),
            "overshoot": np.array(overshoots)}


def structure_state(px: np.ndarray, dc: dict, n: int) -> np.ndarray:
    """-1 when the last confirmed high is below the previous high and the last confirmed low is
    below the previous low, +1 for the opposite, otherwise unchanged; changes only at the bar a
    pivot is confirmed (a directional-change event)."""
    out = np.zeros(n, dtype=np.int8)
    highs: list[float] = []
    lows: list[float] = []
    state = 0
    last = 0
    for e, d, x in zip(dc["events"], dc["dirs"], dc["ext_bars"], strict=True):
        out[last:e] = state
        if d == -1:
            highs.append(px[x])  # a downturn confirms the high it turned from
        else:
            lows.append(px[x])
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                state = -1
            elif highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                state = 1
        last = e
    out[last:] = state
    return out


def causal_atrous(x: np.ndarray, levels: int) -> list[np.ndarray]:
    """One-sided Haar a-trous smoothing layers c_1 .. c_levels of `x`: c_j(t) = (c_{j-1}(t) +
    c_{j-1}(t - 2^(j-1))) / 2; a value at t reads only x up to t."""
    layers = []
    c = np.asarray(x, dtype=float)
    for j in range(1, levels + 1):
        lag = 2 ** (j - 1)
        prev = np.full(len(c), np.nan)
        prev[lag:] = c[:-lag]
        c = (c + prev) / 2.0
        layers.append(c)
    return layers


def online_jump(features: np.ndarray, lam: float) -> np.ndarray:
    """Two trend states (+1 up, -1 down) with a switching penalty `lam`, filtered forward only.

    `features` (bars x k) are signed: positive agrees with "up". The cost of a state at a bar is
    the sum of the features that disagree with it; V_s(t) = cost_s(t) + min(V_s(t-1),
    V_other(t-1) + lam); the state at t is the cheaper V at t (never revised later). Rows with a
    NaN keep the previous state."""
    n = len(features)
    out = np.zeros(n, dtype=np.int8)
    v_up = v_down = 0.0
    state = 0
    for t in range(n):
        f = features[t]
        if np.isnan(f).any():
            out[t] = state
            continue
        c_up = float(np.clip(-f, 0, None).sum())
        c_down = float(np.clip(f, 0, None).sum())
        new_up = c_up + min(v_up, v_down + lam)
        new_down = c_down + min(v_down, v_up + lam)
        v_up, v_down = new_up, new_down
        m = min(v_up, v_down)
        v_up, v_down = v_up - m, v_down - m  # keep the numbers small; the choice is unchanged
        state = 1 if v_up < v_down else (-1 if v_down < v_up else state)
        out[t] = state
    return out


def confirmed_falls(close: pd.Series, x: float, min_days: int,
                    upto: pd.Timestamp) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]],
                                                 pd.Timestamp | None]:
    """The big falls (peak, trough) known on `upto` from the daily closes up to it, and the day of
    the last confirmed pivot (hours after it have no label yet)."""
    from qlab.bv3_parts import zigzag_falls

    c = close[close.index <= upto].dropna()
    falls = zigzag_falls(c, x, min_days)
    # the last confirmed pivot: the trough of the last fall, or the peak a later downturn confirmed
    v = c.to_numpy()
    last_pivot = None
    hi = lo = 0
    trend = None
    for i in range(1, len(v)):
        if v[i] > v[hi]:
            hi = i
        if v[i] < v[lo]:
            lo = i
        if trend in (None, "up") and v[i] <= v[hi] * (1 - x):
            last_pivot, trend, lo = c.index[hi], "down", i
        elif trend in (None, "down") and v[i] >= v[lo] * (1 + x):
            last_pivot, trend, hi = c.index[lo], "up", i
        if trend == "down" and v[i] < v[lo]:
            lo = i
        if trend == "up" and v[i] > v[hi]:
            hi = i
    return list(zip(falls["peak"], falls["trough"], strict=True)), last_pivot


__all__ = ["causal_atrous", "confirmed_falls", "daily_sigma", "directional_change",
           "load_minutes", "online_jump", "structure_state"]
