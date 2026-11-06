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
