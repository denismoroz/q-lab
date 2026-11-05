"""Follow-up of the hedge-trigger study (docs/research/hedge-trigger-*-2026-11-05.md; owner
2026-11-05: «запускай все по порядку»). Pure functions; the runner is
scripts/research/hedge_followup.py.

- Task 1, momentum: `momentum_pairs`, `variance_ratio`, `block_bootstrap`.
- Task 3, band and pause: `band_wish` (the production 14/30-day wish with an entry band, an exit
  band, a minimum time under hedge and a cooldown; with all four at zero it is the book's own
  wish), `proxy_curve` (spot plus hedge per unit of position, the drawdown it implies).
- Task 6, simple books: `downside_vol_exposure`.
- Regimes: `regime_of_hours` (the lab's labelling, `qlab.regimes.label_days`, on BTC's own closes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from qlab.regimes import label_days

HORIZONS = (1, 3, 7, 14, 30)


# ── regimes ───────────────────────────────────────────────────────────────


def daily_closes(hourly: pd.Series) -> pd.Series:
    """The last price of each UTC day (a day without one is NaN), indexed by the day."""
    s = hourly.copy()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.resample("D").last()


def regime_of_days(btc_hourly: pd.Series) -> pd.Series:
    """bull / flat / bear per day: BTC's 30-day return around the day, terciles of the whole
    history (the lab's descriptive labelling; it looks ahead, for measurement only)."""
    labels, _, _ = label_days(daily_closes(btc_hourly).dropna())
    return labels


def regime_of_hours(btc_hourly: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    days = regime_of_days(btc_hourly)
    key = index.tz_convert("UTC").tz_localize(None).floor("D")
    return days.reindex(key).to_numpy(dtype=object)


# ── task 1: momentum ─────────────────────────────────────────────────────


def momentum_pairs(close: pd.Series, h_past: int, h_next: int) -> pd.DataFrame:
    """Per day t: log return over the h_past days ending at t and over the h_next days after t."""
    lc = np.log(close)
    return pd.DataFrame({"past": lc - lc.shift(h_past), "next": lc.shift(-h_next) - lc},
                        index=close.index)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def sign_excess(x: np.ndarray, y: np.ndarray) -> float:
    """Share of equal signs minus the share expected if the two were independent."""
    ok = ~(np.isnan(x) | np.isnan(y)) & (x != 0) & (y != 0)
    if ok.sum() < 10:
        return float("nan")
    a, b = x[ok] > 0, y[ok] > 0
    agree = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    return agree - float(pa * pb + (1 - pa) * (1 - pb))


def variance_ratio(close: pd.Series, h: int) -> float:
    """Var(h-day log return) / (h * Var(1-day log return)), overlapping h-day returns."""
    lc = np.log(close)
    r1 = (lc - lc.shift(1)).dropna()
    rh = (lc - lc.shift(h)).dropna()
    if len(r1) < 3 * h or r1.var() == 0:
        return float("nan")
    return float(rh.var() / (h * r1.var()))


def block_bootstrap(dates: np.ndarray, stat: Callable[[np.ndarray], float], block_days: int = 60,
                    draws: int = 1000, seed: int = 3) -> tuple[float, float, float]:
    """(point, low, high): `stat(rows)` on all rows and its 95% interval when blocks of
    `block_days` consecutive days are drawn with replacement. `dates` is one day per row
    (rows of several coins on one day move together)."""
    point = stat(np.arange(len(dates)))
    day = pd.DatetimeIndex(dates).normalize().as_unit("ns")
    uniq = np.unique(day.asi8)
    if len(uniq) < 2 * block_days:
        return point, float("nan"), float("nan")
    # calendar days from the first to the last, so a block is consecutive days
    first = uniq[0]
    span = int((uniq[-1] - first) // 86_400_000_000_000) + 1
    pos = ((day.asi8 - first) // 86_400_000_000_000).astype(int)
    order = np.argsort(pos, kind="stable")
    counts = np.bincount(pos, minlength=span)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    n_blocks = int(np.ceil(span / block_days))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        starts = rng.integers(0, max(1, span - block_days + 1), size=n_blocks)
        picked = (starts[:, None] + np.arange(block_days)[None, :]).ravel()
        lens = counts[picked]
        total = int(lens.sum())
        if not total:
            continue
        begin = np.repeat(offsets[picked] - np.r_[0, np.cumsum(lens)[:-1]], lens)
        out.append(stat(order[begin + np.arange(total)]))
    out = np.asarray(out, dtype=float)
    out = out[np.isfinite(out)]
    if not len(out):
        return point, float("nan"), float("nan")
    return point, float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ── task 3: band and pause ───────────────────────────────────────────────


def lagged(values: np.ndarray, priced: np.ndarray, bars: int) -> np.ndarray:
    """value / value `bars` priced bars earlier - 1 (the book counts priced closes)."""
    out = np.full(len(values), np.nan)
    at = np.flatnonzero(priced)
    s = values[at]
    if len(s) > bars:
        out[at[bars:]] = s[bars:] / s[:-bars] - 1.0
    return out


def band_wish(perp: np.ndarray, enter_below: float = 0.0, exit_above: float = 0.0,
              min_hold: int = 0, cooldown: int = 0, sticky: int = 12,
              short_days: int = 14, long_days: int = 30) -> np.ndarray:
    """The hedge wish for the bar after each bar (1.0/0.0, NaN where unpriced).

    The production rule (`frab.strategy.b2.book._signals` with threshold 0, sticky exit 12): the
    signal is m = min(14-day return, 30-day return); hedge on when m <= 0, off after `sticky`
    bars in a row with m > 0. Here: on when m <= -enter_below; off after `sticky` bars in a row
    with m > exit_above, and not before `min_hold` bars under hedge; after an off, no new on for
    `cooldown` bars. Unknown m (too little history) wants no hedge, as the book. With every knob
    at zero the result is the book's own wish (tested against the live book)."""
    priced = ~np.isnan(perp)
    short = lagged(perp, priced, short_days * 24)
    long_ = lagged(perp, priced, long_days * 24)
    wish = np.full(len(perp), np.nan)
    on, off_run, held, since_off = False, 0, 0, 10**9
    for i in np.flatnonzero(priced):
        known = not np.isnan(long_[i])
        s = short[i] if not np.isnan(short[i]) else -np.inf
        m = min(s, long_[i]) if known else np.nan
        if not on:
            since_off += 1
            if known and m <= -enter_below and since_off > cooldown:
                on, off_run, held = True, 0, 0
        else:
            held += 1
            # unknown history reads as "no hedge wanted", as the book's rule
            calm = (not known) or m > exit_above
            off_run = off_run + 1 if calm else 0
            if off_run >= sticky and held >= min_hold:
                on, since_off = False, 0
        wish[i] = 1.0 if on else 0.0
    return wish


def proxy_curve(spot_ret: np.ndarray, hedge_money: np.ndarray) -> np.ndarray:
    """Per bar: one unit of spot plus its hedge (hedge money already per unit of position)."""
    return np.nan_to_num(spot_ret) + np.nan_to_num(hedge_money)


def max_drawdown(per_bar: np.ndarray) -> float:
    """Deepest fall of the summed curve from its running peak (the curve starts at 0)."""
    curve = np.cumsum(np.nan_to_num(per_bar))
    peak = np.maximum.accumulate(np.r_[0.0, curve])[1:]
    return float((curve - np.maximum(peak, 0.0)).min()) if len(curve) else float("nan")


def downside_vol_exposure(perp: pd.Series, scale: float = 1.0, window: int = 30 * 24,
                          min_days: int = 90) -> np.ndarray:
    """Task 6: spot exposure min(1, scale * past median / current) of the downside volatility
    (root mean square of the negative hourly returns over `window` bars), the median over the
    coin's own previous days; 1 until `min_days` of history. Read once a day, held for the day."""
    r = np.log(perp.ffill()).diff()
    down = r.clip(upper=0.0)
    dvol = np.sqrt((down ** 2).rolling(window, min_periods=window // 2).mean())
    day = dvol.groupby(dvol.index.floor("D")).last()
    median = day.expanding(min_periods=min_days).median().shift(1)
    expo_day = (scale * median / day).clip(upper=1.0)
    expo = expo_day.shift(1).reindex(dvol.index.floor("D")).to_numpy()  # yesterday's reading
    return np.where(np.isnan(expo), 1.0, expo)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ok = x.notna() & y.notna()
    return float(x[ok].corr(y[ok], method="spearman")) if ok.sum() > 2 else float("nan")


__all__ = [
    "HORIZONS", "band_wish", "block_bootstrap", "corr", "daily_closes", "downside_vol_exposure",
    "lagged", "max_drawdown", "momentum_pairs", "proxy_curve", "regime_of_days",
    "regime_of_hours", "sign_excess", "spearman", "variance_ratio",
]
