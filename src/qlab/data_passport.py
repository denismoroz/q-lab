"""A passport of minute candles before any model reads them (docs/research/ml-15m-2-discipline-
protocol-2026-11-05.md; owner 2026-11-05: «чтобы не было garbage in - garbage out»).

`passport(frame)` measures one instrument's minutes: gaps, duplicates, order, impossible prices,
spikes, empty stretches, taker volume above volume, the first days of trading. `cross_check`
compares minutes resampled to hours (or days) with the lab's cached candles. `bad_tick_mask` is
the declared treatment of spikes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPIKE_SIGMAS = 20.0
VOLUME_SPIKE = 50.0


def _runs(flag: np.ndarray) -> tuple[int, int]:
    """(number of runs of True, longest run)."""
    if not flag.any():
        return 0, 0
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(len(starts)), int((ends - starts).max())


def bad_tick_mask(close: pd.Series, sigmas: float = SPIKE_SIGMAS) -> np.ndarray:
    """Minutes whose log return exceeds `sigmas` times the past day's minute volatility."""
    r = np.log(close).diff()
    sig = r.rolling(1440, min_periods=240).std().shift(1)
    return ((r.abs() > sigmas * sig) & sig.notna()).to_numpy()


def passport(raw: pd.DataFrame) -> dict:
    """`raw`: minute rows as read (open_time, open, high, low, close, volume, trades,
    taker_buy_base), before any de-duplication."""
    t = pd.DatetimeIndex(raw["open_time"])
    out: dict = {"rows": int(len(raw)), "first": str(t.min()), "last": str(t.max())}
    out["duplicates"] = int(t.duplicated().sum())
    out["non_monotonic"] = int((np.diff(t.as_unit("ns").asi8) < 0).sum())
    df = raw.drop_duplicates("open_time").set_index("open_time").sort_index()
    idx = pd.DatetimeIndex(df.index)
    expected = int((idx[-1] - idx[0]) / pd.Timedelta(minutes=1)) + 1
    out["expected_minutes"] = expected
    out["missing_minutes"] = expected - len(df)
    out["missing_share"] = out["missing_minutes"] / expected
    step = np.round(np.diff(idx.as_unit("ns").asi8) / 60e9).astype(np.int64)
    gaps = np.flatnonzero(step > 1)
    out["gaps"] = int(len(gaps))
    order = gaps[np.argsort(-step[gaps])][:10]
    out["longest_gaps"] = [{"after": str(idx[g]), "minutes_missing": int(step[g] - 1)}
                           for g in order]
    px = df[["open", "high", "low", "close"]]
    out["nonpositive_prices"] = int((px <= 0).any(axis=1).sum())
    out["high_below_low"] = int((df["high"] < df["low"]).sum())
    out["close_outside_range"] = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    out["open_outside_range"] = int(((df["open"] > df["high"]) | (df["open"] < df["low"])).sum())
    bad = bad_tick_mask(df["close"])
    r = np.log(df["close"]).diff().to_numpy()
    reverted = bad[:-1] & (np.sign(r[1:]) == -np.sign(r[:-1])) & (np.abs(r[1:]) > 0.5 * np.abs(
        r[:-1]))
    out["spikes"] = int(bad.sum())
    out["spikes_reverted_next_minute"] = int(reverted.sum())
    zero = (df["volume"] <= 0).to_numpy()
    out["zero_volume_minutes"] = int(zero.sum())
    out["zero_volume_longest"] = _runs(zero)[1]
    for col in ("volume", "trades"):
        med = df[col].rolling(1440, min_periods=240).median().shift(1)
        out[f"{col}_spikes"] = int((df[col] > VOLUME_SPIKE * med).sum())
    out["taker_above_volume"] = int((df["taker_buy_base"] > df["volume"] * (1 + 1e-9)).sum())
    dv = (df["close"] * df["volume"]).resample("D").sum()
    if len(dv) > 97:
        out["first_week_turnover_ratio"] = float(dv.iloc[:7].mean() / dv.iloc[7:97].median())
    return out


def align(perp: pd.Series, spot: pd.Series) -> dict:
    """Spot against perpetual minute closes on the perpetual's minutes."""
    both = pd.concat([perp.rename("p"), spot.rename("s")], axis=1, join="inner")
    gap = np.abs(np.log(both["s"] / both["p"]))
    return {"perp_minutes_with_spot": float(len(both) / len(perp)),
            "median_abs_gap": float(gap.median()), "p999_abs_gap": float(gap.quantile(0.999)),
            "minutes_gap_over_2pct": int((gap > 0.02).sum())}


def cross_check(minutes: pd.DataFrame, cached: pd.DataFrame, freq: str, price_tol: float = 1e-6,
                volume_tol: float = 1e-3) -> dict:
    """Minutes resampled to `freq` (close = last, volume = sum, only complete periods) against the
    cached candles (`price`, `volume`), on the periods both have."""
    per = {"h": 60, "D": 1440}[freq]
    g = minutes.resample(freq)
    agg = pd.DataFrame({"close": g["close"].last(), "volume": g["volume"].sum(),
                        "n": g["close"].count()})
    agg = agg[agg["n"] == per]
    cached = cached.copy()
    cached.index = pd.DatetimeIndex(cached.index).tz_convert("UTC") if cached.index.tz else \
        pd.DatetimeIndex(cached.index).tz_localize("UTC")
    both = agg.join(cached[["price", "volume"]].rename(columns={"volume": "cvol"}), how="inner")
    dp = np.abs(both["close"] / both["price"] - 1)
    dv = np.abs(both["volume"] / both["cvol"].replace(0, np.nan) - 1)
    bad_p = dp > price_tol
    bad_v = dv > volume_tol
    worst = both.assign(dp=dp, dv=dv).sort_values("dp", ascending=False).head(5)
    return {"periods": int(len(both)), "price_mismatch": int(bad_p.sum()),
            "volume_mismatch": int(bad_v.sum()),
            "mismatch_share": float((bad_p | bad_v).mean()) if len(both) else None,
            "worst": [{"time": str(i), "rel_price": float(r.dp), "rel_volume": float(r.dv)}
                      for i, r in worst.iterrows()]}


__all__ = ["align", "bad_tick_mask", "cross_check", "passport"]
