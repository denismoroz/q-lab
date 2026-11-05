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
