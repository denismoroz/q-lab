"""Waves: event-time changes, structure, causal layers and the online two-state trend read only
the past."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import waves as wv


def test_directional_change_marks_reversals_of_theta() -> None:
    px = np.array([100, 105, 110, 104, 98, 97, 103, 108, 101.0])
    dc = wv.directional_change(px, np.full(len(px), 0.10))
    # 110 -> 98 is 10.9% down: downturn at bar 4; 97 -> 108 is 11.3% up: upturn at bar 7
    assert dc["events"].tolist() == [4, 7] and dc["dirs"].tolist() == [-1, 1]
    assert dc["state"].tolist() == [0, 0, 0, 0, -1, -1, -1, 1, 1]
    assert dc["ext_bars"].tolist() == [2, 5]
    assert dc["overshoot"][0] == pytest.approx(np.log(98 / 97) / 0.10)


def test_directional_change_does_not_read_ahead() -> None:
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    full = wv.directional_change(px, np.full(3000, 0.03))["state"]
    head = wv.directional_change(px[:2000], np.full(2000, 0.03))["state"]
    assert np.array_equal(full[:2000], head)


def test_structure_turns_down_on_lower_highs_and_lower_lows() -> None:
    px = np.array([100, 120, 100, 125, 105, 115, 95, 110, 90, 92.0])
    dc = wv.directional_change(px, np.full(len(px), 0.08))
    s = wv.structure_state(px, dc, len(px))
    assert s[-1] == -1
    up = np.array([100, 110, 101, 115, 105, 120, 110, 125.0])
    dcu = wv.directional_change(up, np.full(len(up), 0.06))
    assert wv.structure_state(up, dcu, len(up))[-1] == 1


def test_causal_atrous_has_no_look_ahead_at_the_edge() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 2000))
    full = wv.causal_atrous(x, 8)
    head = wv.causal_atrous(x[:1500], 8)
    for a, b in zip(full, head, strict=True):
        assert np.allclose(a[:1500], b, equal_nan=True)
    # a layer of a constant is the constant
    assert np.allclose(wv.causal_atrous(np.full(600, 3.0), 5)[-1][100:], 3.0)


def test_online_jump_switches_only_when_disagreement_beats_the_penalty() -> None:
    f = np.r_[np.full(50, 1.0), np.full(3, -1.0), np.full(50, 1.0), np.full(50, -1.0)][:, None]
    s = wv.online_jump(f, lam=5.0)
    assert (s[:50] == 1).all()
    assert (s[50:53] == 1).all()  # a three-bar wobble does not pay the penalty
    assert (s[-40:] == -1).all()
    head = wv.online_jump(f[:120], lam=5.0)
    assert np.array_equal(s[:120], head)


def test_confirmed_falls_know_only_the_past() -> None:
    days = pd.date_range("2024-01-01", periods=80, freq="D")
    v = np.r_[np.linspace(100, 150, 20), np.linspace(150, 90, 20), np.linspace(90, 130, 40)]
    c = pd.Series(v, index=days)
    falls, pivot = wv.confirmed_falls(c, 0.25, 7, days[45])
    assert falls == [] and pivot == days[19]  # the trough is not confirmed yet
    falls, pivot = wv.confirmed_falls(c, 0.25, 7, days[79])
    assert falls == [(days[19], days[39])] and pivot == days[39]


def test_the_tcn_output_at_an_hour_reads_no_later_hour() -> None:
    from qlab import waves_tcn as wt

    rng = np.random.default_rng(3)
    x = rng.normal(size=(600, 4))
    ends = np.arange(200, 400)
    model = wt.fit(x, ends, (x[ends, 0] > 0), epochs=1)
    base = wt.predict(model, x, ends)
    later = x.copy()
    later[401:] = 100.0  # change every hour after the last end
    assert np.allclose(wt.predict(model, later, ends), base)
    shifted = x.copy()
    shifted[300] += 5.0  # an hour inside some windows does matter
    assert not np.allclose(wt.predict(model, shifted, ends), base)


def test_fall_weights_price_what_an_unhedged_hour_costs() -> None:
    px = np.array([100, 100, 90, 80, 60, 70, 80.0])
    falls = [(1, 4, 6)]
    w = wv.fall_weights(px, falls, "to_trough")
    assert w[0] == 0 and w[5] == 0
    assert w[1] == pytest.approx(0.4) and w[3] == pytest.approx(0.25) and w[4] == 0
    h = wv.fall_weights(px, falls, "hour")
    assert h[1] == pytest.approx(0.1) and h[3] == pytest.approx(0.25) and h[4] == 0
    assert wv.fall_weights(px, falls, "to_trough", min_depth=0.5).sum() == 0
