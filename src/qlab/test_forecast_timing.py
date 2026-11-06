"""The timing hint: the percentile reads only the past; the delays move moments as declared."""

from __future__ import annotations

import numpy as np

from qlab import forecast_timing as ft


def test_the_percentile_reads_only_past_forecasts() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=3000)
    a = ft.past_percentile(f, window=500, min_obs=100)
    g = f.copy()
    g[2000:] += 10.0
    b = ft.past_percentile(g, window=500, min_obs=100)
    assert np.allclose(a[:2000], b[:2000], equal_nan=True)
    assert np.isnan(a[:100]).all() and not np.isnan(a[100])
    # its value: the share of the previous window strictly below
    i = 1500
    assert a[i] == (f[i - 500:i] < f[i]).mean()


def test_no_hint_gives_the_books_own_wish() -> None:
    own = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    gate = ft.DelayGate(np.full(len(own), np.nan), entry=True, exit=True, q=0.25, wait=3)
    assert np.array_equal(ft.delay_wish(own, gate), own)


def test_entry_waits_for_a_low_forecast_or_the_maximum_wait() -> None:
    own = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    pct = np.array([0.5, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
    out = ft.delay_wish(own, ft.DelayGate(pct, entry=True, exit=False, q=0.25, wait=10))
    assert out.tolist() == [False, False, False, True, True, True, True, False]
    out = ft.delay_wish(own, ft.DelayGate(np.full(8, 0.9), entry=True, exit=False, q=0.25,
                                          wait=2))
    assert out.tolist() == [False, False, False, True, True, True, True, False]
    # the book changes its mind before the hint allows: no entry at all
    own2 = np.array([0, 1, 1, 0, 0], dtype=bool)
    out = ft.delay_wish(own2, ft.DelayGate(np.full(5, 0.9), entry=True, exit=False, q=0.25,
                                           wait=10))
    assert not out.any()


def test_exit_waits_for_a_high_forecast_or_the_maximum_wait() -> None:
    own = np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)
    pct = np.array([0.5, 0.5, 0.1, 0.1, 0.9, 0.1, 0.1])
    out = ft.delay_wish(own, ft.DelayGate(pct, entry=False, exit=True, q=0.25, wait=10),
                        start_in_pos=True)
    assert out.tolist() == [True, True, True, True, False, False, False]
    out = ft.delay_wish(own, ft.DelayGate(np.full(7, 0.1), entry=False, exit=True, q=0.25,
                                          wait=1), start_in_pos=True)
    assert out.tolist() == [True, True, True, False, False, False, False]


def test_the_size_gate_without_a_forecast_is_the_books_own_wish_at_full_size() -> None:
    own = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    gate = ft.SizeGate(np.full(6, np.nan), None)
    wish, share = ft.size_wish(own, gate)
    assert np.array_equal(wish, own) and (share == 1.0).all()


def test_the_size_moves_only_by_the_band_and_an_opening_takes_the_target() -> None:
    own = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    inside = np.array([0.5, 0.6, 0.7, 0.9, 0.55, 0.5])
    wish, share = ft.size_wish(own, ft.SizeGate(inside, None, band=0.25))
    assert wish.tolist() == own.tolist()
    # opens at 0.6; 0.7 is inside the band; 0.9 is 0.3 away; 0.55 is 0.35 back; 0.5 stays
    assert share[1:].tolist() == [0.6, 0.6, 0.9, 0.55, 0.55]


def test_the_extra_hedge_opens_on_the_condition_and_is_held_for_the_hold() -> None:
    own = np.zeros(8, dtype=bool)
    extra = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    wish, share = ft.size_wish(own, ft.SizeGate(np.full(8, np.nan), extra, beta=0.5, hold=3))
    assert wish.tolist() == [False, True, True, True, False, False, False, False]
    assert (share[1:4] == 0.5).all()
