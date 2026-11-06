"""Forecast pilot: the targets' scaling reads only ended outcomes; the measures are right."""

from __future__ import annotations

import numpy as np
import pytest

from qlab import forecast as fc


def test_the_target_scaling_reads_only_returns_that_have_ended() -> None:
    rng = np.random.default_rng(0)
    logc = np.cumsum(rng.normal(0, 0.01, 6000))
    z, past = fc.z_target(logc, 24, window=1000, min_obs=200)
    later = logc.copy()
    later[4000:] += 5.0  # change everything from bar 4000 on
    z2, past2 = fc.z_target(later, 24, window=1000, min_obs=200)
    t = 4000 - 24  # its own target ends at 4000; its scaling may use only returns ended by t
    assert np.allclose(past[:4000], past2[:4000], equal_nan=True)
    # the scaling statistics at t are unchanged: z differs only through the target itself
    assert np.allclose(z[:t], z2[:t], equal_nan=True)


def test_auc_and_correlations() -> None:
    y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    assert fc.auc(np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]), y) == pytest.approx(1.0)
    assert fc.auc(np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3]), y) == pytest.approx(0.0)
    x = np.arange(20.0)
    assert fc.pearson(x, 2 * x) == pytest.approx(1.0)
    assert fc.spearman(x, x ** 3) == pytest.approx(1.0)
    assert fc.hit_rate(np.array([1, -1, 1.0]), np.array([2, -3, -1.0])) == pytest.approx(2 / 3)
    assert fc.r2_vs_zero(np.zeros(5), np.ones(5)) == pytest.approx(0.0)


def test_ema_features_do_not_read_later_bars() -> None:
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    full = fc.ema_features(close)
    head = fc.ema_features(close[:2000])
    assert full.shape == (3000, 13)
    assert np.allclose(full[:2000], head, equal_nan=True)


def test_standard_indicators_do_not_read_later_bars() -> None:
    import pandas as pd

    rng = np.random.default_rng(2)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    bars = pd.DataFrame({"close": close, "high": close * 1.004, "low": close * 0.996,
                         "volume": rng.uniform(1, 2, n)}, index=idx)
    full = fc.standard_features(bars)
    head = fc.standard_features(bars.iloc[:2000])
    assert full.shape == (n, len(fc.STANDARD_NAMES)) == (n, 32)
    assert np.allclose(full[:2000], head, equal_nan=True)


def test_the_daily_rsi_is_current_at_every_hour_and_reads_only_the_past() -> None:
    import pandas as pd

    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=24 * 60, freq="h", tz="UTC")
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    r = fc.rsi_daily(c, 14)
    t = 24 * 40  # a 00:00 hour
    c2 = c.copy()
    c2.iloc[t] *= 1.05  # the close of this very hour moves the RSI at this hour
    assert not np.isclose(fc.rsi_daily(c2, 14).iloc[t], r.iloc[t])
    c3 = c.copy()
    c3.iloc[t + 1:] *= 1.3
    assert np.allclose(fc.rsi_daily(c3, 14).iloc[: t + 1], r.iloc[: t + 1], equal_nan=True)


def test_winsorised_z_clips_and_keeps_missing() -> None:
    z = np.array([-5.0, -1.0, np.nan, 2.0, 7.0])
    w = fc.winsorised(z, 3.0)
    assert w[0] == -3.0 and w[4] == 3.0 and w[1] == -1.0 and np.isnan(w[2])
