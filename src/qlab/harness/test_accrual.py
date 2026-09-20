"""Tests for accrual.compute_accrual and the NO_ACCRUAL sentinel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.accrual import NO_ACCRUAL, AccrualError, compute_accrual


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")


def test_omitting_accrual_raises_type_error() -> None:
    held = pd.DataFrame(1.0, index=_index(2), columns=["X"])
    with pytest.raises(TypeError, match="required"):
        compute_accrual(held, None)


def test_no_accrual_sentinel_yields_zero_series() -> None:
    held = pd.DataFrame(1.0, index=_index(3), columns=["X"])
    accrual = compute_accrual(held, NO_ACCRUAL)
    assert (accrual == 0.0).all()
    assert list(accrual.index) == list(held.index)


def test_nan_funding_while_held_raises() -> None:
    idx = _index(3)
    held = pd.DataFrame({"X": [1.0, 1.0, 0.0]}, index=idx)  # held rows 0,1; flat row 2
    funding = pd.DataFrame({"X": [0.001, np.nan, np.nan]}, index=idx)
    with pytest.raises(AccrualError, match="NaN"):
        compute_accrual(held, funding)


def test_nan_funding_while_flat_does_not_raise() -> None:
    idx = _index(3)
    held = pd.DataFrame({"X": [1.0, 0.0, 0.0]}, index=idx)  # only row 0 held
    funding = pd.DataFrame({"X": [0.001, np.nan, np.nan]}, index=idx)  # NaN only while flat
    result = compute_accrual(held, funding)
    assert result.iloc[0] == pytest.approx(0.001)
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 0.0


def test_known_accrual_matches_hand_computation() -> None:
    idx = _index(4)
    # Long 2 units of X, short 3 units of Y, constant funding rates.
    held = pd.DataFrame({"X": [2.0] * 4, "Y": [-3.0] * 4}, index=idx)
    funding = pd.DataFrame({"X": [0.0003] * 4, "Y": [-0.0002] * 4}, index=idx)
    result = compute_accrual(held, funding)
    # X: 2 * 0.0003 = 0.0006 each period.
    # Y: -3 * -0.0002 = 0.0006 each period (short with negative rate earns).
    expected = 0.0006 + 0.0006
    assert np.allclose(result.to_numpy(), expected)
