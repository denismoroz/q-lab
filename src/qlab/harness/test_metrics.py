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

from qlab.harness.metrics import (
    ann_return_net_ex_best_1pct,
    compute_metrics,
    effective_profitable_days,
    effective_profitable_days_share,
    fragility_days,
    fragility_share,
    min_capital_usd,
    periods_per_year,
)
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

    # fragility_days: sorted desc [0.10, 0.10, 0.05, -0.20], total 0.05.
    # Removing the single best day (0.10) leaves 0.05 - 0.10 = -0.05 <= 0.
    assert metrics["fragility_days"] == 1
    assert metrics["fragility_share"] == pytest.approx(1 / 4)

    # ann_return_net_ex_best_1pct: round(4 * 0.01) == 0 best days removed at
    # this n, so it must equal ann_return_net exactly.
    assert metrics["ann_return_net_ex_best_1pct"] == pytest.approx(metrics["ann_return_net"])


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


# --- fragility_days / fragility_share (T27, gap 1) -----------------------


def test_fragility_days_is_zero_when_total_is_negative() -> None:
    # sum = -0.1 - 0.05 + 0.02 = -0.13 <= 0: nothing to remove, the run is
    # unprofitable outright, not "propped up" by any days.
    r = pd.Series([-0.10, -0.05, 0.02])
    assert fragility_days(r) == 0


def test_fragility_days_is_zero_when_total_is_exactly_zero() -> None:
    r = pd.Series([0.10, -0.10])
    assert fragility_days(r) == 0


def test_fragility_days_one_day_flips_the_sign() -> None:
    # total = 0.10 - 0.03 - 0.02 - 0.01 = 0.04 > 0. Removing the single best
    # day (0.10) leaves -0.03 - 0.02 - 0.01 = -0.06 <= 0.
    r = pd.Series([0.10, -0.03, -0.02, -0.01])
    assert fragility_days(r) == 1


def test_fragility_days_needs_several_removals() -> None:
    # Hand-computed: total = 0.15 + 0.10 + 0.08 - 0.03 = 0.30.
    #   remove top 1 (0.15) -> remaining 0.10 + 0.08 - 0.03 = 0.15 > 0
    #   remove top 2 (0.15, 0.10) -> remaining 0.08 - 0.03 = 0.05 > 0
    #   remove top 3 (0.15, 0.10, 0.08) -> remaining -0.03 <= 0
    r = pd.Series([0.15, 0.10, 0.08, -0.03])
    assert fragility_days(r) == 3


def test_fragility_days_never_needs_the_sentinel_guard() -> None:
    # Removing every day always drives the remaining sum to
    # total - total == 0 <= 0, so a large, irregular, strictly positive
    # series must still return a count no bigger than its length (the
    # RuntimeError branch stays unreachable, per the docstring's proof).
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(loc=0.01, scale=0.05, size=500))
    if float(r.sum()) <= 0:
        r = r - r.mean() + 0.01  # force a strictly positive total
    n = fragility_days(r)
    assert 0 < n <= len(r)


def test_fragility_share_is_fragility_days_over_n_periods() -> None:
    r = pd.Series([0.10, -0.03, -0.02, -0.01])  # fragility_days == 1, n == 4
    assert fragility_share(r) == pytest.approx(0.25)


def test_fragility_share_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="empty"):
        fragility_share(pd.Series([], dtype=float))


# --- ann_return_net_ex_best_1pct ------------------------------------------


def test_ann_return_net_ex_best_1pct_hand_computed() -> None:
    # 200 daily periods: two big +50% days (the "best 1%", round(200*0.01)
    # == 2) and 198 days of a small constant loss.
    idx = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
    values = [0.5, 0.5] + [-0.001] * 198
    r = pd.Series(values, index=idx)

    result = ann_return_net_ex_best_1pct(r)

    ppy = 365.25  # daily spacing
    n_remaining = 198
    total_growth_remaining = (1.0 - 0.001) ** 198
    expected = total_growth_remaining ** (ppy / n_remaining) - 1.0
    assert result == pytest.approx(expected)
    # Sanity: excluding the two spikes should leave a clearly negative
    # annualised return, unlike the full series (which is dominated by them).
    assert result < 0


def test_ann_return_net_ex_best_1pct_matches_ann_return_net_when_n_best_is_zero() -> None:
    # n=4 -> round(4 * 0.01) == 0 best days removed.
    idx = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    r = pd.Series([0.10, -0.20, 0.05, 0.10], index=idx)

    total_growth = float((1.0 + r).prod())
    expected = total_growth ** (365.25 / 4) - 1.0
    assert ann_return_net_ex_best_1pct(r) == pytest.approx(expected)


def test_ann_return_net_ex_best_1pct_is_nan_below_two_periods() -> None:
    idx = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    r = pd.Series([0.05], index=idx)
    assert np.isnan(ann_return_net_ex_best_1pct(r))


def test_ann_return_net_ex_best_1pct_can_stay_positive_while_fragility_days_is_high() -> None:
    """Verified claim in the docstring: removing the best 1% of days is NOT
    enough, on its own, to catch a result carried by a few best days --
    this is exactly why `fragility_days` is the primary metric. Built with
    n=627 (matching the real xsmom-honest-20legs run's period count): ten
    "big" days carry the whole positive total (so `fragility_days` needs
    all ten removed to flip the sign, since any nine leave a positive
    remainder), while the best-1% cut (round(627 * 0.01) == 6 days) removes
    only six of those ten and still leaves a positive compounded result.
    """
    idx = pd.date_range("2025-01-01", periods=627, freq="D", tz="UTC")
    n_tail = 627 - 10
    big_days = [0.10] * 9 + [0.05]  # 9 equal big days + a 10th, smaller one
    small_days = [-0.03 / n_tail] * n_tail  # tail sums to exactly -0.03
    r = pd.Series(big_days + small_days, index=idx)
    # total = 0.90 + 0.05 - 0.03 = 0.92; removing the top 9 leaves
    # 0.05 - 0.03 = 0.02 > 0, removing all 10 leaves -0.03 <= 0.
    assert fragility_days(r) == 10  # every one of the ten big days is needed

    n_best = round(627 * 0.01)
    assert n_best == 6
    remaining_after_best_1pct = r.drop(r.nlargest(n_best).index)
    assert float((1.0 + remaining_after_best_1pct).prod()) > 1.0  # still net positive

    result = ann_return_net_ex_best_1pct(r)
    assert result > 0.0


# --- effective_profitable_days: the scale-free concentration measure -------


def test_effective_profitable_days_counts_evenly_spread_profit_exactly() -> None:
    """Profit spread evenly over m days scores exactly m -- the property that
    makes the number readable as "days' worth of profit"."""
    idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    r = pd.Series([0.01] * 4 + [0.0] * 6, index=idx)
    assert effective_profitable_days(r) == pytest.approx(4.0)


def test_effective_profitable_days_is_one_for_a_single_profitable_day() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    r = pd.Series([0.0, 0.0, 0.5, -0.01, -0.02], index=idx)
    assert effective_profitable_days(r) == pytest.approx(1.0)


def test_effective_profitable_days_is_invariant_to_scaling_the_book() -> None:
    """The defect this metric exists to avoid: `fragility_days` rises purely
    because there is more profit to cancel, so it cannot be compared across
    runs of different size. This one does not move at all."""
    idx = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    r = pd.Series([0.03, -0.01, 0.005, 0.02, -0.004, 0.001, 0.0, 0.012], index=idx)
    assert effective_profitable_days(10.0 * r) == pytest.approx(effective_profitable_days(r))
    assert effective_profitable_days(0.1 * r) == pytest.approx(effective_profitable_days(r))


def test_fragility_days_is_not_scale_invariant() -> None:
    """Pins the confound itself, so nobody later builds a rule on
    `fragility_days` believing it measures shape. Scaling every return by 10
    leaves the SHAPE identical while the day count needed to cancel the total
    stays tied to the run's own size."""
    idx = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    r = pd.Series([0.05, 0.04, -0.02, -0.02, -0.02, -0.02], index=idx)
    shifted = r + 0.01  # same shape, more profit -- more days needed to cancel
    assert fragility_days(shifted) > fragility_days(r)
    assert effective_profitable_days(shifted) != pytest.approx(effective_profitable_days(r))


def test_effective_profitable_days_is_zero_without_a_profitable_day() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    r = pd.Series([-0.01, -0.02, 0.0, -0.005], index=idx)
    assert effective_profitable_days(r) == 0.0


def test_effective_profitable_days_share_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError, match="empty"):
        effective_profitable_days_share(pd.Series(dtype=float))
