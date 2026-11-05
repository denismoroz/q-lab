"""Bv3 split into parts (owner 2026-11-05: «нужно распилить её на части и пробовать улучшить
каждую под задачу»; docs/research/bv3-parts-*-2026-11-05.md). Pure functions; the runner is
scripts/research/bv3_parts.py.

- `zigzag_falls`: big falls of a daily price (the yardstick; it reads the future).
- `rolling_inertia`, `rolling_vr`: the coin's CURRENT inertia, past-only.
- `machine`: the hedge decision assembled from parts -- phase, entry, exit, re-entry, minimum time
  and cooldown. With the production parts it is the live book's own wish (tested).
- part metrics: `phase_metric` (A), `entry_metric` (B), `exit_metric` (C), `flip_metric` (D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── labels ──────────────────────────────────────────────────────────────


def zigzag_falls(close: pd.Series, x: float, min_days: int) -> pd.DataFrame:
    """Down legs of the zig-zag with threshold `x` (a leg ends once the price turned by `x` from
    its extreme) at least `x` deep and `min_days` long: peak, trough, the next peak (end of the
    rebound; NaT when the data ends first), depth, rebound."""
    c = close.dropna()
    v, idx = c.to_numpy(), c.index
    if len(v) < 3:
        return pd.DataFrame(columns=["peak", "trough", "next_peak", "depth", "rebound"])
    pivots: list[tuple[int, str]] = []
    trend = None
    hi = lo = 0
    for i in range(1, len(v)):
        if v[i] > v[hi]:
            hi = i
        if v[i] < v[lo]:
            lo = i
        if trend in (None, "up") and v[i] <= v[hi] * (1 - x):
            pivots.append((hi, "peak"))
            trend, lo = "down", i
        elif trend in (None, "down") and v[i] >= v[lo] * (1 + x):
            pivots.append((lo, "trough"))
            trend, hi = "up", i
        if trend == "down" and v[i] < v[lo]:
            lo = i
        if trend == "up" and v[i] > v[hi]:
            hi = i
    rows = []
    for k, (p, kind) in enumerate(pivots):
        if kind != "peak" or k + 1 >= len(pivots):
            continue
        t = pivots[k + 1][0]
        nxt = pivots[k + 2][0] if k + 2 < len(pivots) else None
        depth = v[t] / v[p] - 1.0
        if -depth < x or (idx[t] - idx[p]).days < min_days:
            continue
        rows.append({"peak": idx[p], "trough": idx[t],
                     "next_peak": idx[nxt] if nxt is not None else pd.NaT,
                     "depth": depth,
                     "rebound": (v[nxt] / v[t] - 1.0) if nxt is not None else np.nan})
    return pd.DataFrame(rows, columns=["peak", "trough", "next_peak", "depth", "rebound"])


# ── inertia, past only ──────────────────────────────────────────────────


def rolling_inertia(close: pd.Series, window_days: int = 730, past: int = 30,
                    nxt: int = 14) -> pd.Series:
    """Per day: the correlation of the past `past`-day log return with the next `nxt`-day one,
    over the pairs of the last `window_days` days whose next window has already ended."""
    lc = np.log(close)
    x = (lc - lc.shift(past)).shift(nxt)  # the pair that started `nxt` days ago
    y = (lc.shift(-nxt) - lc).shift(nxt)  # ... and its outcome, known today
    return x.rolling(window_days - nxt, min_periods=365).corr(y)


def rolling_vr(close: pd.Series, h: int = 30, window_days: int = 730) -> pd.Series:
    lc = np.log(close)
    r1 = lc.diff()
    rh = lc - lc.shift(h)
    return rh.rolling(window_days, min_periods=365).var() / (
        h * r1.rolling(window_days, min_periods=365).var())


def daily_to_hours(daily: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """A daily value known at the day's close, applied to the hours of the next day."""
    key = index.tz_convert("UTC").tz_localize(None).floor("D") - pd.Timedelta(days=1)
    return daily.reindex(key).to_numpy(dtype=float)


# ── the assembled decision ──────────────────────────────────────────────


@dataclass(frozen=True)
class Parts:
    sticky: int = 12
    min_hold: int = 0
    cooldown: int = 0
    rebound_x: float | None = None
    reentry_y: float | None = None
    reentry_bars: int = 14 * 24


def machine(px: np.ndarray, phase: np.ndarray, entry: np.ndarray | None,
            exit_extra: np.ndarray | None, parts: Parts) -> np.ndarray:
    """The hedge wish for the bar after each bar (1.0/0.0; NaN where unpriced).

    Off -> on: inside the phase when `entry` holds and the cooldown has passed; or, with
    `reentry_y`, when within `reentry_bars` after a release the price is `reentry_y` below the
    release price. On -> off: after `sticky` bars in a row that are out of the phase, or meet
    `exit_extra`, or (with `rebound_x`) stand `rebound_x` above the lowest price since the on,
    and not before `min_hold` bars under hedge."""
    wish = np.full(len(px), np.nan)
    on = False
    off_run = held = 0
    since_off = since_rel = 10**9
    trough = rel_px = np.inf
    p = parts
    for i in np.flatnonzero(~np.isnan(px)):
        price = px[i]
        if not on:
            since_off += 1
            since_rel += 1
            reent = (p.reentry_y is not None and since_rel <= p.reentry_bars
                     and price <= rel_px * (1.0 - p.reentry_y))
            if (phase[i] and (entry is None or entry[i]) and since_off > p.cooldown) or reent:
                on, off_run, held, trough = True, 0, 0, price
        else:
            held += 1
            trough = min(trough, price)
            calm = (not phase[i]) or (exit_extra is not None and bool(exit_extra[i])) or (
                p.rebound_x is not None and price >= trough * (1.0 + p.rebound_x))
            off_run = off_run + 1 if calm else 0
            if off_run >= p.sticky and held >= p.min_hold:
                on, since_off, since_rel, rel_px = False, 0, 0, price
        wish[i] = 1.0 if on else 0.0
    return wish


def run_length(flag: np.ndarray) -> np.ndarray:
    """Bars in a row the flag has held, up to and including each bar."""
    out = np.zeros(len(flag), dtype=int)
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        out[i] = run
    return out


# ── part metrics (on one coin's stepped bars) ───────────────────────────


def fall_positions(falls: pd.DataFrame, at: pd.DatetimeIndex) -> list[tuple[int, int, int]]:
    """Each fall's (peak, trough, next peak) as positions among the bars `at` (the last bar of the
    day; -1 for a next peak that is unknown or outside the bars)."""
    days = at.tz_convert("UTC").tz_localize(None).floor("D")
    last = pd.Series(np.arange(len(at)), index=days).groupby(level=0).last()
    out = []
    for f in falls.itertuples():
        if f.peak not in last.index or f.trough not in last.index:
            continue
        nxt = int(last[f.next_peak]) if pd.notna(f.next_peak) and f.next_peak in last.index else -1
        out.append((int(last[f.peak]), int(last[f.trough]), nxt))
    return out


def in_falls(n: int, falls: list[tuple[int, int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for p0, p1, _ in falls:
        m[p0:p1 + 1] = True
    return m


def phase_metric(phase: np.ndarray, fall_mask: np.ndarray, sel: np.ndarray) -> tuple[float, ...]:
    """(fall hours in phase, fall hours, other hours in phase, other hours) over `sel`."""
    a = phase & fall_mask & sel
    b = fall_mask & sel
    c = phase & ~fall_mask & sel
    d = ~fall_mask & sel
    return float(a.sum()), float(b.sum()), float(c.sum()), float(d.sum())


def entry_metric(h: np.ndarray, px: np.ndarray, falls, sel: np.ndarray) -> tuple[list, int]:
    """Per fall (peak in `sel`): the share of its depth lost before the hedge was on; and the
    switch-ons outside every fall, counted on `sel`."""
    losses = []
    fm = in_falls(len(h), falls)
    for p0, p1, _ in falls:
        if not sel[p0]:
            continue
        depth = 1.0 - px[p1] / px[p0]
        if depth <= 0:
            continue
        if h[p0]:
            losses.append(0.0)
            continue
        on = np.flatnonzero(h[p0:p1 + 1])
        losses.append(min(1.0, (1.0 - px[p0 + on[0]] / px[p0]) / depth) if len(on) else 1.0)
    starts = np.r_[False, h[1:] & ~h[:-1]]
    return losses, int((starts & ~fm & sel).sum())


def exit_metric(h: np.ndarray, px: np.ndarray, falls, sel: np.ndarray, drop: float = 0.10,
                horizon: int = 30 * 24) -> tuple[list, int]:
    """Per fall with a known rebound (peak in `sel`): the share of the rebound missed under the
    hedge; and the premature releases on `sel` (the price fell `drop` more before the next on,
    within `horizon` bars)."""
    missed = []
    for p0, p1, p2 in falls:
        if not sel[p0] or p2 < 0 or px[p2] <= px[p1]:
            continue
        if not h[p1]:
            missed.append(0.0)
            continue
        offs = np.flatnonzero(~h[p1:p2 + 1])
        if not len(offs):
            missed.append(1.0)
            continue
        r = p1 + offs[0]
        missed.append(float(np.clip((px[r] - px[p1]) / (px[p2] - px[p1]), 0.0, 1.0)))
    releases = np.flatnonzero(np.r_[False, ~h[1:] & h[:-1]] & sel)
    premature = 0
    for r in releases:
        nxt_on = np.flatnonzero(h[r:r + horizon])
        end = r + (nxt_on[0] if len(nxt_on) else min(horizon, len(h) - r))
        if end > r and np.nanmin(px[r:end]) <= px[r] * (1.0 - drop):
            premature += 1
    return missed, premature


def flip_metric(h: np.ndarray, px: np.ndarray, sel: np.ndarray, fee: float,
                within: int = 7 * 24) -> list[float]:
    """Cost of each re-on within `within` bars of a release on `sel`: (release price - re-on
    price) / release price plus two fees."""
    costs = []
    releases = np.flatnonzero(np.r_[False, ~h[1:] & h[:-1]] & sel)
    for r in releases:
        nxt = np.flatnonzero(h[r:r + within])
        if len(nxt):
            o = r + nxt[0]
            costs.append((px[r] - px[o]) / px[r] + 2 * fee)
    return costs


__all__ = ["Parts", "daily_to_hours", "entry_metric", "exit_metric", "fall_positions",
           "flip_metric", "in_falls", "machine", "phase_metric", "rolling_inertia", "rolling_vr",
           "run_length", "zigzag_falls"]
