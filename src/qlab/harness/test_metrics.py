"""Tests for qlab.harness.metrics: compute_metrics, periods_per_year, and
min_capital_usd.

Expected values are computed independently in each test (plain numpy/python
arithmetic against the documented formulas), not by calling the functions
under test with different inputs -- this is a hand-computation cross-check,
not a tautology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.metrics import compute_metrics, min_capital_usd, periods_per_year
from qlab.harness.run import RunResult


def _run_result(
    net_return: list[float], turnover: list[float], index: pd.DatetimeIndex
) -> RunResult:
    r = pd.Series(net_return, index=index)
    t = pd.Series(turnover, index=index)
    zeros = pd.Series(0.0, index=index)
    return RunResult(
        snapshot_id="snap-metrics",
        net_return=r,
        gross_return=r,
        turnover=t,
        cost=zeros,
        accrual=zeros,
    )


# --- periods_per_year --------------------------------------------------


def test_periods_per_year_daily_index() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    ppy = periods_per_year(idx)
    assert ppy == pytest.approx(365.25)


def test_periods_per_year_hourly_index() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    ppy = periods_per_year(idx)
    assert ppy == pytest.approx(365.25 * 24)


def test_periods_per_year_needs_at_least_two_timestamps() -> None:
    idx = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="fewer than 2"):
        periods_per_year(idx)


# --- compute_metrics: hand-computed values ------------------------------


def test_compute_metrics_matches_hand_computation() -> None:
    # 4 consecutive days (same ISO week: Mon-Thu) so worst_week is the
    # compounded total over all 4 periods.
    idx = pd.date_range("2026-03-02", periods=4, freq="D", tz="UTC")  # Mon..Thu
    net = [0.10, -0.20, 0.05, 0.10]
    turnover = [2.0, 0.0, 1.0, 0.0]
    result = _run_result(net, turnover, idx)

    metrics = compute_metrics(result)

    n = 4
    ppy = 365.25  # daily spacing
    r = np.array(net)

    # ann_return_net: geometric compounding.
    total_growth = np.prod(1.0 + r)
    expected_ann_return = total_growth ** (ppy / n) - 1.0
    assert metrics["ann_return_net"] == pytest.approx(expected_ann_return)

    # sharpe_net: sample mean / sample std (ddof=1) * sqrt(ppy).
    expected_sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(ppy)
    assert metrics["sharpe_net"] == pytest.approx(expected_sharpe)

    # max_dd: equity curve 1.10, 0.88, 0.924, 1.0164; running max caps at
    # 1.10 throughout (never exceeded again) -> worst drawdown at step 1:
    # 0.88 / 1.10 - 1 = -0.2 exactly.
    assert metrics["max_dd"] == pytest.approx(-0.2)

    # turnover: annualised mean of the per-period turnover series.
    expected_turnover = np.mean(turnover) * ppy
    assert metrics["turnover"] == pytest.approx(expected_turnover)

    # skew: Fisher-Pearson adjusted (bias-corrected) sample skew.
    mean = r.mean()
    std_pop = r.std(ddof=0)
    m3 = np.mean((r - mean) ** 3)
    g1 = m3 / std_pop**3
    expected_skew = np.sqrt(n * (n - 1)) / (n - 2) * g1
    assert metrics["skew"] == pytest.approx(expected_skew)

    # worst_week: only one calendar week present -> compounded total growth.
    assert metrics["worst_week"] == pytest.approx(total_growth - 1.0)

    assert metrics["n_periods"] == 4


def test_compute_metrics_rejects_empty_result() -> None:
    idx = pd.DatetimeIndex([], tz="UTC")
    result = _run_result([], [], idx)
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(result)


def test_compute_metrics_single_period_is_undefined_but_does_not_crash() -> None:
    idx = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    result = _run_result([0.05], [1.0], idx)
    metrics = compute_metrics(result)
    assert np.isnan(metrics["ann_return_net"])
    assert np.isnan(metrics["sharpe_net"])
    assert np.isnan(metrics["turnover"])
    assert metrics["n_periods"] == 1


def test_compute_metrics_total_wipeout_floors_ann_return_at_minus_one() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    result = _run_result([-1.0, 0.0], [0.0, 0.0], idx)  # -100% on day 1
    metrics = compute_metrics(result)
    assert metrics["ann_return_net"] == -1.0


# --- min_capital_usd -----------------------------------------------------


def test_min_capital_usd_hand_computed() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame(
        {"A": [0.5, 0.02], "B": [-0.5, 0.0]},  # smallest non-zero |weight| is 0.02
        index=idx,
    )
    result = min_capital_usd(weights, min_leg_notional=12.0)
    assert result == pytest.approx(12.0 / 0.02)  # == 600.0


def test_min_capital_usd_uses_smallest_weight_over_whole_period_not_typical() -> None:
    # 99 periods of a large, comfortable weight, and ONE period with a tiny
    # leg -- the rare small leg must still set the floor.
    idx = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    values = [0.5] * 99 + [0.001]
    weights = pd.DataFrame({"A": values}, index=idx)
    result = min_capital_usd(weights, min_leg_notional=10.0)
    assert result == pytest.approx(10.0 / 0.001)


def test_min_capital_usd_requires_min_leg_notional_explicitly() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"A": [0.5, 0.5]}, index=idx)
    with pytest.raises(TypeError):
        min_capital_usd(weights)  # type: ignore[call-arg]


def test_min_capital_usd_rejects_non_positive_min_leg_notional() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"A": [0.5, 0.5]}, index=idx)
    with pytest.raises(ValueError, match="min_leg_notional"):
        min_capital_usd(weights, min_leg_notional=0.0)


def test_min_capital_usd_rejects_all_flat_book() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"A": [0.0, 0.0]}, index=idx)
    with pytest.raises(ValueError, match="never take"):
        min_capital_usd(weights, min_leg_notional=12.0)
