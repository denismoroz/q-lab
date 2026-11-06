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
    key = m.index.floor(rule)
    g = m.groupby(key)
    agg = {"high": ("high", "max"), "low": ("low", "min"), "close": ("close", "last"),
           "volume": ("volume", "sum"), "trades": ("trades", "sum"),
           "buy": ("taker_buy_base", "sum")}
    if "open" in m:
        agg["open"] = ("open", "first")
    b = g.agg(**agg)
    if "open" not in b:
        b["open"] = g["close"].first()
    # the range made in the bar's last quarter of minutes
    offset = (m.index - key).total_seconds().to_numpy() / 60.0
    q = m[offset >= 0.75 * minutes]
    gq = q.groupby(q.index.floor(rule))
    b["lq_high"], b["lq_low"] = gq["high"].max(), gq["low"].min()
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
    state_leg = np.zeros(n)
    state_move = np.full(n, np.nan)
    state_leg[0] = leg
    for t in range(1, n):
        s = sigma[t]
        state_leg[t] = leg
        if leg == 1:
            if lc[t] > ext:
                ext, ext_i, armed, sig_e = lc[t], t, True, np.nan
                state_move[t] = 0.0
                continue
        else:
            if lc[t] < ext:
                ext, ext_i, armed, sig_e = lc[t], t, True, np.nan
                state_move[t] = 0.0
                continue
        if np.isnan(s):
            continue
        move = (ext - lc[t]) * leg  # how far the close has come back from the extreme
        state_move[t] = move / s
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
            p1 = pivots[-1] if pivots else (np.nan, np.nan, 0)
            p2 = pivots[-2] if len(pivots) > 1 else (np.nan, np.nan, 0)
            p3 = pivots[-3] if len(pivots) > 2 else (np.nan, np.nan, 0)
            rows.append((t, leg, ext, ext_i, s, t - last[0] if last else np.nan, swing, ratio,
                         hh, hl, t - ext_i, move / s, p1[0], p1[1], p2[0], p2[1], p3[0]))
            armed, sig_e = False, s
        if not np.isnan(sig_e) and move >= big * sig_e:
            pivots.append((ext_i, ext, leg))
            leg, ext, ext_i, armed, sig_e = -leg, lc[t], t, True, np.nan
    cols = ["bar", "side", "ext", "ext_bar", "sigma", "bars_since_pivot", "last_swing",
            "leg_ratio", "higher_high", "higher_low", "speed", "step_sigma", "piv1_bar",
            "piv1_price", "piv2_bar", "piv2_price", "piv3_bar"]
    a = np.array(rows, dtype=float).reshape(-1, len(cols))
    out = {c: a[:, i] for i, c in enumerate(cols)}
    out["state_leg"], out["state_move"] = state_leg, state_move
    return out


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


PACK_GROUPS = {
    "rsi_div": ["rsi_div_4", "rsi_div_8", "rsi_div_14", "new_extreme"],
    "imb_div": ["imb_div"],
    "volume": ["ext_vol_z", "ext_trades_z", "leg_volume_ratio"],
    "wicks": ["ext_upper", "ext_lower", "ext_close_pos", "ev_upper", "ev_lower", "ev_close_pos",
              "ev_last_quarter"],
    "funding": ["funding", "funding_pct", "funding_change"],
    "basis": ["basis_z", "basis_change"],
    "mtf": ["next_same_dir", "next_move", "next_rsi8", "next_ema16"],
}
PACK_NAMES = [n for g in PACK_GROUPS.values() for n in g]


def bar_series(values: np.ndarray, value_end: np.ndarray, bar_end: np.ndarray) -> np.ndarray:
    """The last value known at each bar's close (`values` stamped at `value_end`)."""
    pos = np.searchsorted(value_end, bar_end, side="right") - 1
    out = np.full(len(bar_end), np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def reversal_pack(b: pd.DataFrame, ev: dict, sigma: np.ndarray, bf: np.ndarray,
                  funding: tuple[np.ndarray, np.ndarray] | None, basis: np.ndarray | None,
                  mtf: np.ndarray) -> np.ndarray:
    """events x PACK_NAMES: how the price arrived at the extreme (module docstring of the
    protocol): divergences with RSI and taker imbalance, volume climax, wicks, funding, basis
    and the larger bar size's state (`mtf`, events x 4, computed by the caller). Past only:
    every bar used is at or before the event bar."""
    c = b["close"]
    side = ev["side"]
    e = ev["ext_bar"].astype(int)
    t = ev["bar"].astype(int)
    n_ev = len(t)

    def at(arr, idx):
        out = np.full(n_ev, np.nan)
        ok = ~np.isnan(idx)
        out[ok] = np.asarray(arr)[idx[ok].astype(int)]
        return out

    cols = []
    for n in (4, 8, 14):
        r = ((_rsi(c, n) - 50) / 50).to_numpy()
        cols.append(side * (at(r, ev["piv2_bar"]) - r[e]))
    cols.append(np.where(np.isnan(ev["piv2_price"]), np.nan,
                         (side * (ev["ext"] - ev["piv2_price"]) > 0).astype(float)))
    cb = np.r_[0.0, np.cumsum(b["buy"].to_numpy(dtype=float))]
    cv = np.r_[0.0, np.cumsum(b["volume"].to_numpy(dtype=float))]

    def span(cum, a, z):
        ok = ~(np.isnan(a) | np.isnan(z))
        out = np.full(n_ev, np.nan)
        aa, zz = a[ok].astype(int), z[ok].astype(int)
        out[ok] = cum[zz + 1] - cum[aa]
        return out

    with np.errstate(invalid="ignore", divide="ignore"):
        cur_b, cur_v = span(cb, ev["piv1_bar"], e.astype(float)), span(cv, ev["piv1_bar"],
                                                                       e.astype(float))
        prv_b, prv_v = span(cb, ev["piv3_bar"], ev["piv2_bar"]), span(cv, ev["piv3_bar"],
                                                                      ev["piv2_bar"])
        cols.append(side * ((prv_b / prv_v) - (cur_b / cur_v)))
        vz = bf[:, FEATURE_NAMES.index("vol_z16")]
        tz = bf[:, FEATURE_NAMES.index("trades_z16")]
        cols += [vz[e], tz[e]]
        cur_len = e - ev["piv1_bar"] + 1
        prv_len = ev["piv2_bar"] - ev["piv3_bar"] + 1
        cols.append(np.log((cur_v / cur_len) / (prv_v / prv_len)))
        o, h, lo, cl = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        rng_ = h - lo
        for idx in (e, t):
            cols += [(h[idx] - np.maximum(o[idx], cl[idx])) / rng_[idx],
                     (np.minimum(o[idx], cl[idx]) - lo[idx]) / rng_[idx],
                     (cl[idx] - lo[idx]) / rng_[idx]]
        cols.append((b["lq_high"].to_numpy(dtype=float)[t] - b["lq_low"].to_numpy(dtype=float)[t])
                    / rng_[t])
    if funding is None:
        cols += [np.full(n_ev, np.nan)] * 3
    else:
        rate, pct = funding  # per bar: the last settled rate and its past-90-day percentile
        cols += [rate[t], pct[t], rate[t] - at(rate, ev["piv1_bar"])]
    if basis is None:
        cols += [np.full(n_ev, np.nan)] * 2
    else:
        bs = pd.Series(basis)
        z = ((bs - bs.rolling(96, min_periods=48).mean().shift(1))
             / bs.rolling(96, min_periods=48).std().shift(1)).to_numpy()
        cols += [z[t], basis[t] - at(basis, ev["piv1_bar"])]
    x = np.column_stack(cols + [mtf])
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)


__all__ = ["FEATURE_NAMES", "PACK_GROUPS", "PACK_NAMES", "bar_features", "bar_series",
           "bar_sigma", "event_features", "first_passage", "make_bars", "reversal_pack",
           "wave_events"]
