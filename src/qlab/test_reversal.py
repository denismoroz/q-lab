"""The reversal detector: events and features read only the past; the label means what it says."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import reversal as rv


def _walk(n: int, seed: int) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))


def test_events_up_to_a_bar_do_not_depend_on_later_bars() -> None:
    c = _walk(3000, 1)
    s = rv.bar_sigma(c)
    ev = rv.wave_events(c, s, 0.5, 1.5)
    c2 = c.copy()
    c2[2000:] *= 1.5
    ev2 = rv.wave_events(c2, rv.bar_sigma(c2), 0.5, 1.5)
    k = int(np.sum(ev["bar"] < 2000))
    assert k > 50
    for key in ev:
        assert np.allclose(ev[key][:k], ev2[key][:k], equal_nan=True), key


def test_the_first_passage_label() -> None:
    # up to 110, a dip to 108 (the event), then 104 (below 110·e^-0.05 = 104.6) before any new
    # high: a top, resolved on that bar
    c = np.r_[np.linspace(100, 110, 200), [108.0, 104.0, 95.0, 96.0], np.full(10, 96.0)]
    s = np.full(len(c), 0.01)
    ev = rv.wave_events(c, s, 1.0, 5.0)
    assert ev["side"][0] == 1 and ev["bar"][0] == 200
    lab, res, timeout = rv.first_passage(c, ev, 5.0)
    assert lab[0] == 1 and res[0] == 201 and not timeout[0]
    # the same dip followed by a new high first: not a big reversal
    c2 = np.r_[np.linspace(100, 110, 200), [108.0, 111.0, 95.0], np.full(10, 95.0)]
    ev2 = rv.wave_events(c2, s[: len(c2)], 1.0, 5.0)
    lab2, res2, _ = rv.first_passage(c2, ev2, 5.0)
    assert lab2[0] == 0 and res2[0] == 201


def test_bar_features_read_only_bars_up_to_them() -> None:
    n = 1500
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(4)
    c = _walk(n, 2)
    b = pd.DataFrame({"high": c * 1.002, "low": c * 0.998, "close": c,
                      "volume": rng.uniform(1, 10, n),
                      "trades": rng.integers(10, 100, n).astype(float),
                      "buy": rng.uniform(0, 5, n)}, index=idx)
    f = rv.bar_features(b, rv.bar_sigma(c))
    b2 = b.copy()
    b2.iloc[1000:] *= 1.7
    f2 = rv.bar_features(b2, rv.bar_sigma(b2["close"].to_numpy()))
    assert f.shape[1] == len(rv.FEATURE_NAMES) - 7
    assert np.allclose(f[:1000], f2[:1000], equal_nan=True)


def _minute_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="min", tz="UTC").as_unit("ns")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0003,
                         "low": np.minimum(o, c) * 0.9997, "close": c, "volume": v,
                         "trades": rng.integers(5, 50, n).astype(float),
                         "taker_buy_base": v * rng.uniform(0.3, 0.7, n)}, index=idx)


def test_the_reversal_pack_reads_only_bars_up_to_the_event() -> None:
    m = _minute_frame(30 * 1440, 6)
    cut = 25 * 1440  # minutes from here on change
    m2 = m.copy()
    for col in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        m2.iloc[cut:, m2.columns.get_loc(col)] *= 1.3

    def pack(frame):
        b = rv.make_bars(frame, 15)
        c = b["close"].to_numpy()
        s = rv.bar_sigma(c)
        ev = rv.wave_events(c, s, 0.5, 1.5)
        bf = rv.bar_features(b, s)
        f = (np.zeros(len(b)), np.zeros(len(b)))
        x = rv.reversal_pack(b, ev, s, bf, f, np.zeros(len(b)), np.zeros((len(ev["bar"]), 4)))
        return x, b["end"].to_numpy()[ev["bar"].astype(int)]

    x, end = pack(m)
    x2, _ = pack(m2)
    k = int(np.sum(end <= m.index.asi8[cut]))
    assert x.shape[1] == len(rv.PACK_NAMES) and k > 20
    assert np.allclose(x[:k], x2[:k], equal_nan=True)
