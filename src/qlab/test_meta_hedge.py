"""Bv2's level 2 learns whether a hedge pays from the past only (docs/TASKS.md T39)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import meta_hedge as mh

HOURS = pd.date_range("2021-01-01", periods=24 * 200, freq="1h", tz="UTC")


def _coin(seed: int):
    rng = np.random.default_rng(seed)
    drift = np.sin(np.arange(len(HOURS)) / 300.0) * 0.0006
    perp = pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, len(HOURS)))),
                     index=HOURS)
    funding = pd.Series(0.00001, index=HOURS)
    return perp, funding


def _level1() -> pd.DataFrame:
    days = pd.date_range("2020-12-01", periods=260, freq="1D", tz="UTC")
    return pd.DataFrame({"p_bull": 0.3, "p_flat": 0.4, "p_bear": 0.3}, index=days)


def test_a_hedge_pays_when_the_coin_falls_more_than_the_round_trip() -> None:
    idx = pd.date_range("2025-01-01", periods=mh.HORIZON_HOURS + 2, freq="1h", tz="UTC")
    falling = pd.Series(np.linspace(100, 90, len(idx)), index=idx)
    rising = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    zero = pd.Series(0.0, index=idx)
    assert mh.hedge_pays(falling, zero, 0.0017).iloc[0] == 1.0
    assert mh.hedge_pays(rising, zero, 0.0017).iloc[0] == 0.0
    assert np.isnan(mh.hedge_pays(falling, zero, 0.0017).iloc[-1])  # not known yet


def test_a_label_is_known_only_after_its_horizon() -> None:
    perp, funding = _coin(1)
    labels = mh.hedge_pays(perp, funding, 0.0017)
    as_of = HOURS[3000]
    known = mh.known_at(labels, as_of)
    assert known.index.max() + pd.Timedelta(hours=mh.HORIZON_HOURS + 1) <= as_of


def test_features_do_not_see_the_coins_future() -> None:
    perp, funding = _coin(2)
    cut = HOURS[3000]
    changed = perp.copy()
    changed[changed.index > cut] *= 2.0
    a = mh.coin_features(perp, funding, _level1())
    b = mh.coin_features(changed, funding, _level1())
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut])


def test_decisions_do_not_move_with_the_future() -> None:
    coins = {"A": _coin(3), "B": _coin(4)}
    lvl = _level1()

    def build(cut=None):
        feats, labels = {}, {}
        for name, (perp, funding) in coins.items():
            p = perp.copy()
            if cut is not None:
                p[p.index > cut] *= np.linspace(0.5, 2.0, (p.index > cut).sum())
            feats[name] = mh.coin_features(p, funding, lvl)
            labels[name] = mh.hedge_pays(p, funding, 0.0017)
        return mh.walk_forward(feats, labels, HOURS[24 * 90])

    base = build()
    cut = HOURS[24 * 160]
    moved = build(cut)
    pd.testing.assert_frame_equal(base[base.index <= cut], moved[moved.index <= cut])


def test_the_sticky_exit_holds_the_hedge_for_its_hours() -> None:
    from qlab.strategies.live.bv2 import _sticky

    raw = np.array([np.nan, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    assert _sticky(raw, 3).tolist()[1:] == [1, 1, 1, 0, 1, 1, 1, 0, 0]
