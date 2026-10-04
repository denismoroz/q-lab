"""The regime detector learns only from the past (docs/TASKS.md T39)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import regime_detect as rd


def _btc(days: int = 900, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Alternating drifts so every regime appears.
    drift = np.sin(np.arange(days) / 40.0) * 0.004
    closes = 10_000 * np.exp(np.cumsum(drift + rng.normal(0, 0.02, days)))
    index = pd.date_range("2019-09-09", periods=days, freq="1D", tz="UTC")
    return pd.Series(closes, index=index)


def test_features_do_not_see_the_future() -> None:
    btc = _btc()
    cut = btc.index[500]
    changed = btc.copy()
    changed[changed.index > cut] *= np.linspace(0.3, 3.0, (changed.index > cut).sum())
    a, b = rd.features(btc), rd.features(changed)
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut])


def test_labels_known_at_a_date_need_fifteen_days_after_them() -> None:
    btc = _btc()
    as_of = btc.index[400]
    labels = rd.known_labels(btc, as_of)
    assert labels.index.max() == as_of - pd.Timedelta(days=rd.HALF)
    # Thresholds from what was known then: the future does not move them.
    changed = btc.copy()
    changed[changed.index > as_of] *= 5.0
    pd.testing.assert_series_equal(labels, rd.known_labels(changed, as_of))


def test_walk_forward_predictions_do_not_move_with_the_future() -> None:
    btc = _btc()
    feats = rd.features(btc)
    start = btc.index[300]
    base = rd.walk_forward(btc, feats, start)
    cut = btc.index[600]
    changed = btc.copy()
    changed[changed.index > cut] *= np.linspace(0.3, 3.0, (changed.index > cut).sum())
    moved = rd.walk_forward(changed, rd.features(changed), start)
    keep = base.index <= cut
    pd.testing.assert_frame_equal(base[keep], moved[moved.index <= cut])
    # Every model trained on labels known a full embargo before its month.
    month_start = base.index.to_series().dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    first_day = month_start.where(month_start >= start, start)
    gap = (first_day - pd.to_datetime(base["trained_until"])).dt.days
    assert (gap >= rd.EMBARGO_DAYS).all()


def test_bv2_rule_hedges_unless_up_over_both_windows() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC")
    rising = pd.Series(np.linspace(100, 140, 40), index=index)
    assert not rd.bv2_rule(rising).dropna().any()
    assert rd.bv2_rule(rising[::-1].set_axis(index)).dropna().all()


def test_hedge_score_measures_lag_into_a_fall() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    labels = pd.Series(["bull"] * 3 + ["bear"] * 5 + ["flat"] * 2, index=index)
    hedge = pd.Series([False] * 5 + [True] * 5, index=index)  # two days late
    s = rd.score_hedge(hedge, labels)
    assert s.median_lag_days == 2.0
    assert s.bear_hedged == pytest.approx(3 / 5)
    assert s.bull_hedged == 0.0


def test_a_shorter_window_is_known_sooner_and_never_overlaps_its_gap() -> None:
    btc = _btc()
    as_of = btc.index[400]
    labels = rd.known_labels(btc, as_of, window=14)
    assert labels.index.max() == as_of - pd.Timedelta(days=7)
    base = rd.walk_forward(btc, rd.features(btc), btc.index[300], window=14)
    month = base.index.to_series().dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    first = month.where(month >= btc.index[300], btc.index[300])
    assert ((first - pd.to_datetime(base["trained_until"])).dt.days >= 14).all()
    assert rd.return_spans(14) == [3, 7, 14, 28, 42]


def test_a_switching_penalty_makes_the_regime_hold_and_reads_only_the_past() -> None:
    days = pd.date_range("2024-01-01", periods=8, freq="1D", tz="UTC")
    # Most probable state flickers bull/bear every other day.
    p_bull = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
    proba = pd.DataFrame({"p_bull": p_bull, "p_flat": 0.1,
                          "p_bear": [0.9 - b for b in p_bull]}, index=days)
    raw = rd.persistent(proba, 0.0)
    assert list(raw) == ["bull", "bear"] * 4  # no penalty: the most probable state
    held = rd.persistent(proba, 2.0)
    assert set(held) == {"bull"}  # the flicker is not worth a switch
    changed = proba.copy()
    changed.iloc[5:] = [0.01, 0.01, 0.98]
    pd.testing.assert_series_equal(rd.persistent(changed, 2.0).iloc[:5], held.iloc[:5])
