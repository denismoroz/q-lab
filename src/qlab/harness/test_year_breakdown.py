"""Tests for qlab.harness.metrics.year_breakdown (docs/YEARS.md).

Expected values are worked out by hand from the documented formulas, not by calling
the function under test again.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qlab.harness.metrics import year_breakdown


def _daily(start: str, end: str, value: float | list[float]) -> pd.Series:
    index = pd.date_range(start, end, freq="1D", tz="UTC")
    return pd.Series(value, index=index, dtype=float)


def _by_year(parts: list[pd.Series]) -> pd.Series:
    return pd.concat(parts)


def test_partial_years_are_listed_with_their_day_counts() -> None:
    # 2020: 184 days at +0.1%, 2021: 365 at -0.05%, 2022: 90 at +0.2%.
    r = _by_year([_daily("2020-07-01", "2020-12-31", 0.001),
                  _daily("2021-01-01", "2021-12-31", -0.0005),
                  _daily("2022-01-01", "2022-03-31", 0.002)])
    out = year_breakdown(r)

    assert (out["year_2020_days"], out["year_2021_days"], out["year_2022_days"]) == (184, 365, 90)
    # The year's own total, not annualised.
    assert out["year_2020_return"] == pytest.approx(1.001**184 - 1)
    assert out["year_2021_return"] == pytest.approx(0.9995**365 - 1)
    assert out["year_2022_return"] == pytest.approx(1.002**90 - 1)
    # Within a falling year the drawdown is the whole fall (equity starts at 1.0).
    assert out["year_2021_max_dd"] == pytest.approx(0.9995**365 - 1)
    # A year that only rises never dips.
    assert out["year_2020_max_dd"] == 0.0


def test_counted_years_need_120_days_and_sharpe_needs_60() -> None:
    def alternating(start: str, days: int) -> pd.Series:
        return pd.Series([0.001, 0.003] * days, index=pd.date_range(
            start, periods=days * 2, freq="1D", tz="UTC")).iloc[:days]

    # 2020 has 61 days: listed with a Sharpe, but not counted; 2021 has 365: counted.
    out = year_breakdown(pd.concat([alternating("2020-11-01", 61),
                                    _daily("2021-01-01", "2021-12-31", 0.001)]))
    assert out["year_2020_days"] == 61 and "year_2020_sharpe" in out
    assert out["years_counted"] == 1
    # 31 and 28 days: no Sharpe for either.
    short = year_breakdown(pd.concat([_daily("2020-12-01", "2020-12-31", 0.001),
                                      alternating("2021-01-01", 28)]))
    assert short["year_2020_days"] == 31 and "year_2020_sharpe" not in short
    assert "year_2021_sharpe" not in short
    # 59 days: just under; 60: just enough.
    assert "year_2021_sharpe" not in year_breakdown(alternating("2021-01-01", 59).pipe(
        lambda s: pd.concat([_daily("2020-12-30", "2020-12-31", 0.001), s])))
    assert "year_2021_sharpe" in year_breakdown(alternating("2021-01-01", 60).pipe(
        lambda s: pd.concat([_daily("2020-12-30", "2020-12-31", 0.001), s])))


def test_year_sharpe_is_mean_over_spread_with_the_series_annualisation() -> None:
    values = [0.01, -0.005, 0.02, 0.0] * 30  # 120 days of 2021
    r = pd.concat([_daily("2020-12-30", "2020-12-31", 0.001),
                   pd.Series(values, index=pd.date_range("2021-01-01", periods=120, tz="UTC"))])
    expected = np.mean(values) / np.std(values, ddof=1) * math.sqrt(365.25)
    assert year_breakdown(r)["year_2021_sharpe"] == pytest.approx(expected)


def test_summaries_over_counted_years() -> None:
    r = _by_year([_daily("2020-01-01", "2020-12-31", 0.002),   # 366 days, best
                  _daily("2021-01-01", "2021-12-31", -0.0004),
                  _daily("2022-01-01", "2022-12-31", 0.0005),
                  _daily("2023-01-01", "2023-02-09", 0.01)])    # 40 days: listed, not counted
    out = year_breakdown(r)

    assert out["years_counted"] == 3 and out["years_positive"] == 2
    assert out["years_positive_share"] == pytest.approx(2 / 3)

    # Best counted year is 2020 (the 40-day 2023 is huge but not counted).
    rest = np.concatenate([np.full(365, 0.9996), np.full(365, 1.0005), np.full(40, 1.01)])
    n, ppy = len(rest), 365.25
    assert out["ann_return_net_ex_best_year"] == pytest.approx(rest.prod() ** (ppy / n) - 1)

    total = 0.002 * 366 - 0.0004 * 365 + 0.0005 * 365 + 0.01 * 40
    assert out["best_year_share_of_profit"] == pytest.approx(0.002 * 366 / total)


def test_best_year_removal_leaves_the_other_years_whole() -> None:
    # Two equal years but one lucky: dropping the lucky one leaves the dull one's pace.
    r = _by_year([_daily("2021-01-01", "2021-12-31", 0.003),
                  _daily("2022-01-01", "2022-12-31", 0.0001)])
    out = year_breakdown(r)
    assert out["ann_return_net_ex_best_year"] == pytest.approx(1.0001**365.25 - 1, rel=1e-9)
    assert out["best_year_share_of_profit"] == pytest.approx(
        0.003 * 365 / (0.003 * 365 + 0.0001 * 365))


def test_fewer_than_two_counted_years_has_no_ex_best_and_no_share() -> None:
    out = year_breakdown(_daily("2021-03-01", "2021-12-31", 0.001))  # 306 days, one year
    assert out["years_counted"] == 1 and out["years_positive_share"] == 1.0
    assert "ann_return_net_ex_best_year" not in out
    assert "best_year_share_of_profit" not in out

    # Two years in the series but only one reaches 120 days: still one counted year.
    two = year_breakdown(pd.concat([_daily("2021-12-01", "2021-12-31", 0.01),
                                    _daily("2022-01-01", "2022-12-31", 0.001)]))
    assert two["years_counted"] == 1 and "ann_return_net_ex_best_year" not in two


def test_no_counted_year_has_no_summaries() -> None:
    out = year_breakdown(_daily("2021-01-01", "2021-03-31", 0.001))
    assert out["year_2021_days"] == 90
    assert not any(k.startswith("years_") for k in out)


def test_all_negative_years() -> None:
    r = _by_year([_daily("2021-01-01", "2021-12-31", -0.001),
                  _daily("2022-01-01", "2022-12-31", -0.0005)])
    out = year_breakdown(r)
    assert out["years_positive"] == 0 and out["years_positive_share"] == 0.0
    # Dropping the better of two losing years leaves the worse one's pace.
    assert out["ann_return_net_ex_best_year"] == pytest.approx(0.999**365.25 - 1)
    # No profit to be a share of.
    assert "best_year_share_of_profit" not in out


def test_removing_the_best_year_can_wipe_the_book_out() -> None:
    r = _by_year([_daily("2021-01-01", "2021-12-31", 0.01),
                  _daily("2022-01-01", "2022-12-31", 0.0)])
    r.iloc[-1] = -1.0  # the last day of 2022 loses everything
    assert year_breakdown(r)["ann_return_net_ex_best_year"] == -1.0


def test_hourly_bars_count_calendar_days_not_bars() -> None:
    index = pd.date_range("2020-12-31 12:00", "2021-01-02 11:00", freq="1h", tz="UTC")
    out = year_breakdown(pd.Series(0.0001, index=index))
    assert out["year_2020_days"] == 1 and out["year_2021_days"] == 2
    assert out["year_2020_return"] == pytest.approx(1.0001**12 - 1)


def test_too_short_series_has_no_years() -> None:
    assert year_breakdown(pd.Series([0.01], index=pd.date_range("2021-01-01", periods=1))) == {}
