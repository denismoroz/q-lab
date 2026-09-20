"""Tests for CostModel."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.costs import CostModel


def test_cost_model_requires_explicit_values() -> None:
    # No defaults exist to fall back on -- omitting either argument is a
    # TypeError, not a free-trading simulation.
    with pytest.raises(TypeError):
        CostModel()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CostModel(taker_fee_bps=3.5)  # type: ignore[call-arg]


def test_cost_model_rejects_negative_fees() -> None:
    with pytest.raises(ValueError, match="taker_fee_bps"):
        CostModel(taker_fee_bps=-1.0, slippage_bps=0.5)
    with pytest.raises(ValueError, match="slippage_bps"):
        CostModel(taker_fee_bps=1.0, slippage_bps=-0.5)


def test_total_bps_sums_fee_and_slippage() -> None:
    costs = CostModel(taker_fee_bps=3.5, slippage_bps=0.9)
    assert costs.total_bps == pytest.approx(4.4)


def test_cost_of_turnover_hand_computed() -> None:
    costs = CostModel(taker_fee_bps=3.0, slippage_bps=2.0)  # total 5 bps
    turnover = pd.Series([2.0, 0.0, 4.0], index=pd.RangeIndex(3))
    cost = costs.cost_of_turnover(turnover)
    # 2.0 * 5/1e4 = 0.001 ; 0.0 ; 4.0 * 5/1e4 = 0.002
    assert cost.iloc[0] == pytest.approx(0.001)
    assert cost.iloc[1] == pytest.approx(0.0)
    assert cost.iloc[2] == pytest.approx(0.002)
