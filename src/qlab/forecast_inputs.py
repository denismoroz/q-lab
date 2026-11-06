"""Forecast pilot 2 (docs/research/forecast-pilot-2-*-2026-11-05.md): the shuffled-target nulls,
the size baseline, and the "fast" inputs other than raw minutes: minute summaries, time bars and
dollar bars. Everything an input reads at a decision time `tau` ended by `tau` (tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3_600_000_000_000
DAY = 24 * HOUR
MINUTE = 60_000_000_000
BAR_CHANNELS = 5
Z_BARS = 96  # bars in the past window that scales a bar's volume / duration / trades


# ── nulls: where each training row takes its shuffled label from ─────────────


def _days(tau: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    day = tau // DAY
    uniq, start = np.unique(day, return_index=True)
    ends = np.r_[start[1:], len(tau)]
    return uniq, [np.arange(s, e) for s, e in zip(start, ends, strict=True)]


def day_block_source(tau: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The pilot-1 null: whole days permuted, the k-th row of a day takes the k-th row of
    another day (the hour of the day is kept). `tau` sorted; returns row positions."""
    _, blocks = _days(tau)
    order = rng.permutation(len(blocks))
    src = np.concatenate([blocks[k] for k in order])
    if len(src) < len(tau):
        src = np.r_[src, np.arange(len(src), len(tau))]
    return src[: len(tau)]


def within_day_source(tau: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each day's labels shifted circularly by a random 1..n-1 rows inside that day."""
    _, blocks = _days(tau)
    out = np.arange(len(tau))
    for b in blocks:
        if len(b) > 1:
            out[b] = np.roll(b, -int(rng.integers(1, len(b))))
    return out


def other_day_source(tau: np.ndarray, rng: np.random.Generator, min_gap_days: int = 7
                     ) -> np.ndarray:
    """The primary null: each day takes the labels of another random day at least
    `min_gap_days` away (days permuted), circularly shifted by a random 1..23 hours."""
    uniq, blocks = _days(tau)
    n = len(uniq)
    perm = rng.permutation(n)
    for _ in range(50):  # repair the pairs that landed too close
        bad = np.flatnonzero(np.abs(uniq[perm] - uniq) < min_gap_days)
        if not len(bad):
            break
        for i in bad:
            j = int(rng.integers(n))
            if (abs(uniq[perm[j]] - uniq[i]) >= min_gap_days
                    and abs(uniq[perm[i]] - uniq[j]) >= min_gap_days):
                perm[i], perm[j] = perm[j], perm[i]
    out = np.empty(len(tau), dtype=np.int64)
    for i, b in enumerate(blocks):
        src = blocks[perm[i]]
        shift = int(rng.integers(1, 24))
        out[b] = src[(np.arange(len(b)) + shift) % len(src)]
    return out


NULLS = {"other day": other_day_source, "within day": within_day_source,
         "day block": day_block_source}


def shuffled_labels(y: np.ndarray, mask: np.ndarray, tau: np.ndarray, how: str,
                    rng: np.random.Generator) -> np.ndarray:
    """`y` with its rows under `mask` replaced by the labels the null `how` assigns them."""
    idx = np.flatnonzero(mask)
    src = NULLS[how](tau[idx], rng)
    out = y.copy()
    out[idx] = y[idx[src]]
    return out


# ── the size baseline ────────────────────────────────────────────────────────


def size_baseline(logc: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per hourly decision: log of the realised volatility of hourly returns over 24, 168, 720
    hours relative to the last year's, and one-hot hour of the day and day of the week. Returns
    end at the decision (past only)."""
    r = pd.Series(logc).diff()
    year = r.rolling(8760, min_periods=2160).std()
    cols, names = [], []
    for n in (24, 168, 720):
        cols.append(np.log(r.rolling(n, min_periods=n // 2).std() / year).to_numpy())
        names.append(f"rv{n}")
    t = pd.DatetimeIndex(pd.to_datetime(tau, utc=True))
    hour, dow = t.hour.to_numpy(), t.dayofweek.to_numpy()
    cols += [(hour == k).astype(float) for k in range(24)]
    names += [f"hour{k}" for k in range(24)]
    cols += [(dow == k).astype(float) for k in range(7)]
    names += [f"dow{k}" for k in range(7)]
    x = np.column_stack(cols)
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan), names


def calendar_and_size(logc: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """The five compact columns of the size baseline for feature selection: rv24, rv168,
    rv720, hour of the day, day of the week."""
    x, names = size_baseline(logc, tau)
    t = pd.DatetimeIndex(pd.to_datetime(tau, utc=True))
    return (np.column_stack([x[:, :3], t.hour.to_numpy(), t.dayofweek.to_numpy()]),
            ["rv24", "rv168", "rv720", "hour", "dow"])


# ── minutes, cleaned ─────────────────────────────────────────────────────────


def clean_minutes(m: pd.DataFrame, bad: np.ndarray) -> pd.DataFrame:
    """Minutes with the flagged spike minutes' prices replaced by the previous close (so a spike
    and its reversal both vanish); volume and trades kept."""
    m = m.copy()
    c = m["close"].where(~bad).ffill()
    for col in ("close", "high", "low"):
        m[col] = np.where(bad, c, m[col])
    return m


def minute_sigma(close: pd.Series) -> pd.Series:
    """The past day's minute volatility, ending at the minute before (as `ml_money`)."""
    r = np.log(close).diff()
    return r.rolling("1440min", min_periods=240).std().shift(1)


# ── (b) minute summaries ─────────────────────────────────────────────────────

SUMMARY_NAMES = ("rv1h", "rv1d", "imb1h", "imb1d", "trades1h", "spikes1d")


def minute_summaries(m: pd.DataFrame) -> np.ndarray:
    """minutes x 6, each over windows ending at (and including) that minute: log realised minute
    volatility of the last hour and day relative to the last 30 days, taker-buy imbalance of the
    last hour and day, log trades of the last hour relative to the 30-day hourly mean, number of
    minutes in the last day whose move exceeded 4 past-day minute sigmas."""
    r = np.log(m["close"]).diff()
    rv30 = r.rolling("30D", min_periods=1440 * 7).std()
    vol, buy, tr = m["volume"], m["taker_buy_base"], m["trades"]
    cols = [np.log(r.rolling("60min", min_periods=30).std() / rv30),
            np.log(r.rolling("1440min", min_periods=240).std() / rv30)]
    for w in ("60min", "1440min"):
        cols.append(buy.rolling(w).sum() / vol.rolling(w).sum() - 0.5)
    cols.append(np.log1p(tr.rolling("60min").sum())
                - np.log1p(tr.rolling("30D", min_periods=1440 * 7).sum() / 720))
    sig = minute_sigma(m["close"])
    cols.append((r.abs() > 4 * sig).astype(float).rolling("1440min").sum())
    x = np.column_stack([c.to_numpy() for c in cols])
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)


# ── (c, d, e) bars ───────────────────────────────────────────────────────────


def _z_past(x: pd.Series, n: int = Z_BARS) -> pd.Series:
    mu = x.rolling(n, min_periods=n // 4).mean().shift(1)
    sd = x.rolling(n, min_periods=n // 4).std().shift(1)
    return (x - mu) / sd


def bars_from_ids(m: pd.DataFrame, ids: np.ndarray, sigma: pd.Series, dollar: bool
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate minutes into bars by a non-decreasing bar id per minute. Returns (bars x 5
    channels, the end time of each bar in ns = its last minute's open + 1 minute). Channels:
    return and range in units of the minute sigma at the bar's end times sqrt(bar minutes), log
    volume (log duration for dollar bars) and log trades as deviations from the past bars,
    taker-buy imbalance."""
    t = pd.DatetimeIndex(m.index).as_unit("ns").asi8
    r = np.log(m["close"]).diff().fillna(0.0).to_numpy()
    df = pd.DataFrame({"id": ids, "r": r, "hi": m["high"].to_numpy(), "lo": m["low"].to_numpy(),
                       "c": m["close"].to_numpy(), "v": m["volume"].to_numpy(),
                       "b": m["taker_buy_base"].to_numpy(), "n": m["trades"].to_numpy(),
                       "t": t, "s": sigma.to_numpy()})
    g = df.groupby("id", sort=True)
    a = g.agg(r=("r", "sum"), hi=("hi", "max"), lo=("lo", "min"), c=("c", "last"),
              v=("v", "sum"), b=("b", "sum"), n=("n", "sum"), t0=("t", "first"),
              t1=("t", "last"), s=("s", "last"))
    end = a["t1"].to_numpy() + MINUTE
    dur = (end - a["t0"].to_numpy()) / MINUTE
    scale = a["s"].to_numpy() * np.sqrt(dur)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = a["r"].to_numpy() / scale
        rng_ = ((a["hi"] - a["lo"]) / a["c"]).to_numpy() / scale
        imb = (a["b"] / a["v"]).where(a["v"] > 0).to_numpy() - 0.5
    third = np.log(pd.Series(dur)) if dollar else np.log1p(a["v"].reset_index(drop=True))
    trades = np.log1p(a["n"].reset_index(drop=True))
    out = np.column_stack([np.clip(ret, -10, 10), np.clip(rng_, 0, 20),
                           _z_past(third).clip(-5, 5).to_numpy(), np.clip(imb, -0.5, 0.5),
                           _z_past(trades).clip(-5, 5).to_numpy()])
    return np.nan_to_num(out).astype(np.float32), end


def time_bar_ids(m: pd.DataFrame, minutes: int) -> np.ndarray:
    t = pd.DatetimeIndex(m.index).as_unit("ns").asi8
    return t // (minutes * MINUTE)


def dollar_bar_ids(m: pd.DataFrame, amount: float) -> np.ndarray:
    """A minute belongs to the bar counted by the traded value before it: the minute that takes
    the running total over a multiple of `amount` closes its bar (known at that minute's end)."""
    qv = m["quote_volume"].to_numpy().astype(np.float64)
    before = np.cumsum(qv) - qv
    return np.floor(before / amount).astype(np.int64)


def bar_windows(chan: np.ndarray, end: np.ndarray, tau: np.ndarray, length: int) -> np.ndarray:
    """batch x channels x length: the last `length` bars that had ended by each `tau`."""
    last = np.searchsorted(end, tau, side="right") - 1
    idx = last[:, None] - np.arange(length - 1, -1, -1)[None, :]
    w = chan[np.clip(idx, 0, None)]
    w[idx < 0] = 0.0
    return w.transpose(0, 2, 1)


def bar_window_means(chan: np.ndarray, end: np.ndarray, tau: np.ndarray, length: int
                     ) -> np.ndarray:
    """Mean of each channel over the last `length` ended bars (for the tabular models)."""
    cs = np.vstack([np.zeros((1, chan.shape[1])), np.cumsum(chan, axis=0, dtype=np.float64)])
    hi = np.searchsorted(end, tau, side="right")
    lo = np.clip(hi - length, 0, None)
    return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)[:, None]


__all__ = ["NULLS", "bar_window_means", "bar_windows", "bars_from_ids", "calendar_and_size",
           "clean_minutes", "day_block_source", "dollar_bar_ids", "minute_sigma",
           "minute_summaries", "other_day_source", "shuffled_labels", "size_baseline",
           "time_bar_ids", "within_day_source"]


# ── simpler targets (pilot 3 addendum) ───────────────────────────────────────


def past_h_sigma(logc: np.ndarray, h: int, window: int = 8760, min_obs: int = 2160
                 ) -> np.ndarray:
    """Standard deviation of the h-step log returns that had ENDED by each step, over the last
    `window` steps (the scale of `forecast.z_target`)."""
    fwd = np.full(len(logc), np.nan)
    fwd[: len(logc) - h] = logc[h:] - logc[: len(logc) - h]
    ended = pd.Series(fwd).shift(h)
    return ended.rolling(window, min_periods=min_obs).std().to_numpy()


def triple_barrier(close: np.ndarray, open_ns: np.ndarray, pos: np.ndarray, tau: np.ndarray,
                   h: int, width: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Per decision: +1 if the minute closes after `tau` reach log C(tau) + width before
    log C(tau) - width within h hours, -1 for the lower barrier first, 0 if neither (time out),
    NaN if the window has a missing minute or no width. C(tau) is the close of minute `pos`."""
    n_min = 60 * h
    logm = np.log(close.astype(np.float64))
    out = np.full(len(tau), np.nan)
    end = pos + n_min
    ok = (pos >= 0) & (end < len(close)) & ~np.isnan(width)
    idx_ok = np.flatnonzero(ok)
    good = open_ns[end[idx_ok]] == tau[idx_ok] + (n_min - 1) * MINUTE
    idx_ok = idx_ok[good]
    steps = np.arange(1, n_min + 1)
    for b in range(0, len(idx_ok), chunk):
        ii = idx_ok[b:b + chunk]
        rel = logm[pos[ii][:, None] + steps[None, :]] - logm[pos[ii]][:, None]
        w = width[ii][:, None]
        up, dn = rel >= w, rel <= -w
        first_up = np.where(up.any(1), up.argmax(1), n_min)
        first_dn = np.where(dn.any(1), dn.argmax(1), n_min)
        out[ii] = np.where(first_up < first_dn, 1.0, np.where(first_dn < first_up, -1.0, 0.0))
    return out
