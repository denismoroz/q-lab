"""Tests for `qlab.strategies._periods.periods_for`."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.strategies._periods import periods_for


def test_periods_for_daily_index_matches_calendar_days() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    assert periods_for(index, pd.Timedelta(days=14)) == 14
    assert periods_for(index, pd.Timedelta(days=1)) == 1


def test_periods_for_hourly_index_matches_calendar_hours() -> None:
    index = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    assert periods_for(index, pd.Timedelta(hours=12)) == 12
    assert periods_for(index, pd.Timedelta(days=1)) == 24


def test_periods_for_floors_at_one_bar() -> None:
    # A daily index: a 12-hour duration is less than one bar, but there is
    # no such thing as "zero bars of patience".
    index = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    assert periods_for(index, pd.Timedelta(hours=12)) == 1


def test_periods_for_rejects_too_short_index() -> None:
    index = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="fewer than 2"):
        periods_for(index, pd.Timedelta(days=1))
