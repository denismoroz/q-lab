"""The minute passport finds what it is meant to find."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import data_passport as dp


def _raw(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    vol = rng.uniform(1, 2, n)
    return pd.DataFrame({"open_time": t, "open": close, "high": close * 1.0005,
                         "low": close * 0.9995, "close": close, "volume": vol,
                         "trades": rng.integers(5, 10, n), "taker_buy_base": vol * 0.5})


def test_a_clean_file_has_a_clean_passport() -> None:
    p = dp.passport(_raw())
    assert p["missing_minutes"] == 0 and p["duplicates"] == 0 and p["non_monotonic"] == 0
    assert p["spikes"] == 0 and p["taker_above_volume"] == 0 and p["high_below_low"] == 0


def test_the_passport_finds_gaps_duplicates_spikes_and_bad_rows() -> None:
    raw = _raw()
    raw = raw.drop(index=range(1000, 1030))  # a 30-minute outage
    raw = pd.concat([raw, raw.iloc[[5]]])  # a duplicate stamp
    raw.loc[2000, "close"] *= 1.5  # a one-minute spike that reverts
    raw.loc[2500, "high"] = raw.loc[2500, "low"] * 0.9
    raw.loc[2600, "taker_buy_base"] = raw.loc[2600, "volume"] * 2
    p = dp.passport(raw)
    assert p["missing_minutes"] == 30 and p["longest_gaps"][0]["minutes_missing"] == 30
    assert p["duplicates"] == 1 and p["non_monotonic"] >= 1
    assert p["spikes"] >= 1 and p["spikes_reverted_next_minute"] >= 1
    assert p["high_below_low"] == 1 and p["taker_above_volume"] == 1


def test_cross_check_counts_hours_that_disagree() -> None:
    raw = _raw(600).set_index("open_time")
    g = raw.resample("h")
    cached = pd.DataFrame({"price": g["close"].last(), "volume": g["volume"].sum()})
    assert dp.cross_check(raw, cached, "h")["price_mismatch"] == 0
    cached.iloc[2, 0] *= 1.01
    out = dp.cross_check(raw, cached, "h")
    assert out["price_mismatch"] == 1 and out["periods"] == 10
