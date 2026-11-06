"""The reversal detector (docs/research/reversal-detector-protocol-2026-11-06.md): bars of any
size from the minutes, the wave machine that emits «possible top / bottom» events, the
first-passage label (reads the future by design: it is the yardstick), and scale-free features in
bar units read only from bars up to the event bar (tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.forecast import _rsi

SIGMA_BARS = 96
MAX_BARS = 200
RET_K = (1, 2, 4, 8, 16, 32)
FEATURE_NAMES = (
    [f"ret_{k}" for k in RET_K] + [f"rsi_{n}" for n in (4, 8, 14)]
    + [f"ema_dist_{n}" for n in (4, 8, 16, 32)] + ["boll_b_10", "boll_b_20", "donch_8",
                                                   "donch_32"]
    + [f"imb_{k}" for k in (1, 4, 16)] + ["vol_z16", "trades_z16", "rv_ratio"]
    + ["bars_since_pivot", "last_swing", "leg_ratio", "higher_high", "higher_low", "speed",
       "step_sigma"])


def make_bars(m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Bars of `minutes` from the minute rows (index: the bar's open time, UTC); missing bars
    are absent. `end` is the bar's close time."""
    rule = "D" if minutes == 1440 else f"{minutes}min"
    g = m.groupby(m.index.floor(rule))
    b = g.agg(high=("high", "max"), low=("low", "min"), close=("close", "last"),
              volume=("volume", "sum"), trades=("trades", "sum"),
              buy=("taker_buy_base", "sum"))
    b.index = pd.DatetimeIndex(b.index).as_unit("ns")
    b["end"] = b.index.asi8 + minutes * 60_000_000_000
    return b


def bar_sigma(close: np.ndarray) -> np.ndarray:
    """The past volatility per bar: the deviation of log returns over the previous 96 bars."""
    r = pd.Series(np.log(close)).diff()
    return r.rolling(SIGMA_BARS, min_periods=SIGMA_BARS // 2).std().shift(1).to_numpy()


def wave_events(close: np.ndarray, sigma: np.ndarray, x: float, big: float) -> dict:
    """Walk the bars once (past only): emit a «possible top» (side +1) when a close falls x·σ
    below the running high of an up leg, at most once per high; a top is confirmed when the
    close falls `big`·σ (σ of that event) below the high, the leg turns down; bottoms mirror.
    Returns per event its bar, side, the extreme and its bar, σ, and the wave state then."""
    lc = np.log(close)
    n = len(lc)
    leg, ext, ext_i, armed, sig_e = 1, lc[0], 0, True, np.nan
    pivots: list[tuple[int, float, int]] = []  # (bar, log price, +1 top / -1 bottom)
    rows = []
    for t in range(1, n):
        s = sigma[t]
        if leg == 1:
            if lc[t] > ext:
                ext, ext_i, armed, sig_e = lc[t], t, True, np.nan
                continue
        else:
            if lc[t] < ext:
                ext, ext_i, armed, sig_e = lc[t], t, True, np.nan
                continue
        if np.isnan(s):
            continue
        move = (ext - lc[t]) * leg  # how far the close has come back from the extreme
        if armed and move >= x * s:
            tops = [p for p in pivots if p[2] == 1]
            bots = [p for p in pivots if p[2] == -1]
            last = pivots[-1] if pivots else None
            prev = pivots[-2] if len(pivots) > 1 else None
            swing = abs(last[1] - prev[1]) / s if (last and prev) else np.nan
            leg_size = abs(ext - last[1]) if last else np.nan
            ratio = leg_size / abs(last[1] - prev[1]) if (last and prev and last[1] != prev[1]) \
                else np.nan
            if leg == 1:
                hh = float(ext > tops[-1][1]) if tops else np.nan
                hl = float(bots[-1][1] > bots[-2][1]) if len(bots) > 1 else np.nan
            else:
                hh = float(tops[-1][1] > tops[-2][1]) if len(tops) > 1 else np.nan
                hl = float(ext > bots[-1][1]) if bots else np.nan
            rows.append((t, leg, ext, ext_i, s, t - last[0] if last else np.nan, swing, ratio,
                         hh, hl, t - ext_i, move / s))
            armed, sig_e = False, s
        if not np.isnan(sig_e) and move >= big * sig_e:
            pivots.append((ext_i, ext, leg))
            leg, ext, ext_i, armed, sig_e = -leg, lc[t], t, True, np.nan
    cols = ["bar", "side", "ext", "ext_bar", "sigma", "bars_since_pivot", "last_swing",
            "leg_ratio", "higher_high", "higher_low", "speed", "step_sigma"]
    a = np.array(rows, dtype=float).reshape(-1, len(cols))
    return {c: a[:, i] for i, c in enumerate(cols)}


def first_passage(close: np.ndarray, ev: dict, big: float, max_bars: int = MAX_BARS
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per event: 1 if the close reaches the extreme minus `big`·σ (toward the reversal) before
    it passes the extreme again, within `max_bars` bars; the bar it resolved on; time-out flag.
    Reads the future by design."""
    lc = np.log(close)
    n = len(lc)
    lab = np.zeros(len(ev["bar"]))
    res = np.zeros(len(ev["bar"]), dtype=np.int64)
    timeout = np.zeros(len(ev["bar"]), dtype=bool)
    for k, (t, side, ext, s) in enumerate(zip(ev["bar"].astype(int), ev["side"], ev["ext"],
                                             ev["sigma"], strict=True)):
        target = ext - side * big * s
        done = False
        for u in range(t, min(n, t + max_bars + 1)):
            if side * (lc[u] - target) <= 0:
                lab[k], res[k], done = 1.0, u, True
                break
            if u > t and side * (lc[u] - ext) > 0:
                res[k], done = u, True
                break
        if not done:
            res[k], timeout[k] = min(n - 1, t + max_bars), True
    return lab, res, timeout


def bar_features(b: pd.DataFrame, sigma: np.ndarray) -> np.ndarray:
    """bars x the bar-unit features (before the wave columns), each read from bars up to it."""
    c = b["close"]
    lc = np.log(c)
    s = pd.Series(sigma, index=b.index)
    cols = []
    for k in RET_K:
        cols.append((lc - lc.shift(k)) / (s * np.sqrt(k)))
    for n in (4, 8, 14):
        cols.append((_rsi(c, n) - 50) / 50)
    for n in (4, 8, 16, 32):
        cols.append((lc - np.log(c.ewm(span=n, adjust=False).mean())) / s)
    for n in (10, 20):
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        cols.append((c - ma) / (4 * sd))
    for n in (8, 32):
        hn, ln = b["high"].rolling(n).max(), b["low"].rolling(n).min()
        cols.append((c - ln) / (hn - ln) - 0.5)
    for k in (1, 4, 16):
        cols.append(b["buy"].rolling(k).sum() / b["volume"].rolling(k).sum() - 0.5)
    for col in ("volume", "trades"):
        lv = np.log1p(b[col])
        cols.append((lv - lv.rolling(16).mean().shift(1)) / lv.rolling(16).std().shift(1))
    r = lc.diff()
    cols.append(r.rolling(8).std() / r.rolling(64).std())
    x = np.column_stack([np.asarray(v, dtype=float) for v in cols])
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)


def event_features(bf: np.ndarray, ev: dict) -> np.ndarray:
    t = ev["bar"].astype(int)
    wave = np.column_stack([ev[k] for k in ("bars_since_pivot", "last_swing", "leg_ratio",
                                            "higher_high", "higher_low", "speed",
                                            "step_sigma")])
    return np.column_stack([bf[t], wave])


__all__ = ["FEATURE_NAMES", "bar_features", "bar_sigma", "event_features", "first_passage",
           "make_bars", "wave_events"]
