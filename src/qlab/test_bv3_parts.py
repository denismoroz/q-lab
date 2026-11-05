"""Bv3 by parts: the labels find the falls, the assembled machine with the production parts is the
live book's wish, the part metrics read what they say."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import bv3_parts as bp
from qlab import hedge_followup as hf
from qlab import hedge_study as hs
from qlab.strategies.live import _bv2_testkit as kit
from qlab.test_hedge_study import COINS, _market


def _prod_phase(px: np.ndarray) -> np.ndarray:
    priced = ~np.isnan(px)
    s, lg = hf.lagged(px, priced, 14 * 24), hf.lagged(px, priced, 30 * 24)
    m = np.fmin(np.where(np.isnan(s), -np.inf, s), lg)
    return ~np.isnan(lg) & (m <= 0)


def test_the_machine_with_production_parts_is_the_live_book_s_wish() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    rec = hs.run_book(panel, params)
    for c in COINS:
        px = panel.prices[c].to_numpy(dtype=float)
        wish = bp.machine(px, _prod_phase(px), None, None, bp.Parts())
        assert np.array_equal(np.nan_to_num(wish), np.nan_to_num(hf.band_wish(px)))
        st = rec.stepped[c].to_numpy()
        assert ((wish[st] > 0.5) == rec.own[c].to_numpy()[st]).all()


def test_min_hold_cooldown_rebound_and_reentry() -> None:
    px = np.array([100, 99, 98, 97, 103, 104, 105, 95, 90, 89.0])
    phase = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    base = bp.machine(px, phase, None, None, bp.Parts(sticky=1))
    assert base.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    held = bp.machine(px, phase, None, None, bp.Parts(sticky=1, min_hold=5))
    assert held[:6].tolist() == [1, 1, 1, 1, 1, 0]
    rebound = bp.machine(px, np.ones(10, dtype=bool), None, None,
                         bp.Parts(sticky=1, rebound_x=0.05, cooldown=100))
    assert rebound[4] == 0.0  # 103 is 6% above the low of 97
    reent = bp.machine(px, phase, None, None, bp.Parts(sticky=1, reentry_y=0.05))
    assert reent[7] == 1.0  # 95 is more than 5% below the release price of 103


def test_zigzag_finds_a_big_fall_and_its_rebound() -> None:
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    v = np.r_[np.linspace(100, 150, 20), np.linspace(150, 90, 20), np.linspace(90, 140, 20)]
    f = bp.zigzag_falls(pd.Series(v, index=days), 0.25, 7)
    assert len(f) == 1
    assert f["peak"].iloc[0] == days[19] and f["trough"].iloc[0] == days[39]
    assert f["depth"].iloc[0] == pytest.approx(-0.4)
    assert pd.isna(f["next_peak"].iloc[0])  # the rally has not turned 25% yet
    assert bp.zigzag_falls(pd.Series(v, index=days), 0.45, 7).empty


def test_rolling_inertia_reads_only_finished_pairs() -> None:
    days = pd.date_range("2020-01-01", periods=900, freq="D")
    c = pd.Series(np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.03, 900))), index=days)
    full = bp.rolling_inertia(c)
    cut = 700
    head = bp.rolling_inertia(c.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], head)


def test_entry_and_exit_metrics() -> None:
    px = np.array([100, 95, 90, 80, 70, 75, 85, 95, 100.0])
    falls = [(0, 4, 8)]
    sel = np.ones(9, dtype=bool)
    h = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    losses, false_starts = bp.entry_metric(h, px, falls, sel)
    assert losses == [pytest.approx((1 - 0.9) / 0.3)] and false_starts == 0
    missed, premature = bp.exit_metric(h, px, falls, sel)
    assert missed == [pytest.approx((85 - 70) / 30)] and premature == 0
    costs = bp.flip_metric(np.array([1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool), px, sel, 0.0)
    assert costs == [pytest.approx((95 - 80) / 95)]
