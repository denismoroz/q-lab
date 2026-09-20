"""Tests for align_funding_to_index: the core "NaN means unknown, never
silently zero" logic shared by every source fetcher."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.data.sources.base import align_funding_to_index


def test_exact_frequency_match_passes_through():
    """Native interval == panel bar: every bar with a real print keeps it,
    a bar with no print is NaN (not zero)."""
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    raw = pd.Series(
        {
            pd.Timestamp("2024-01-01 00:00", tz="UTC"): 0.0001,
            pd.Timestamp("2024-01-01 01:00", tz="UTC"): 0.0002,
            # 02:00 missing entirely
            pd.Timestamp("2024-01-01 03:00", tz="UTC"): 0.0004,
        }
    )
    out = align_funding_to_index(raw, index, pd.Timedelta(hours=1))
    assert out.loc["2024-01-01 00:00"] == 0.0001
    assert out.loc["2024-01-01 01:00"] == 0.0002
    assert np.isnan(out.loc["2024-01-01 02:00"])
    assert out.loc["2024-01-01 03:00"] == 0.0004


def test_coarser_bar_sums_complete_bucket():
    """Panel bar coarser than native settlement (daily bar, hourly funding):
    a full day's worth of settlements sums; a short bucket is NaN."""
    index = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    hours = pd.date_range("2023-12-31 01:00", "2024-01-01 00:00", freq="1h", tz="UTC")
    raw = pd.Series(0.0001, index=hours)  # exactly 24 settlements ending at day 1 label
    out = align_funding_to_index(raw, index, pd.Timedelta(hours=1))
    assert np.isclose(out.iloc[0], 24 * 0.0001)
    # second day has no data at all -> unknown, not zero
    assert np.isnan(out.iloc[1])


def test_finer_bar_places_single_settlement_rest_nan():
    """Panel bar finer than native settlement (hourly bar, 8h funding):
    the settlement lands on exactly one bar; every other bar is NaN, never 0."""
    index = pd.date_range("2024-01-01 00:00", periods=9, freq="1h", tz="UTC")
    raw = pd.Series({pd.Timestamp("2024-01-01 08:00", tz="UTC"): 0.0003})
    out = align_funding_to_index(raw, index, pd.Timedelta(hours=8))
    assert out.loc["2024-01-01 08:00"] == 0.0003
    non_settlement = out.drop(pd.Timestamp("2024-01-01 08:00", tz="UTC"))
    assert non_settlement.isna().all()


def test_settlement_jitter_past_the_mark_stays_in_its_own_bucket():
    """Real Hyperliquid fundingHistory timestamps land a few ms *after* the
    hour (e.g. HH:00:00.025), not exactly on it. A naive (prev, t] window
    would push that settlement into the NEXT hour's bucket; flooring to the
    native interval must keep it in the hour it actually belongs to."""
    index = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    raw = pd.Series(
        {
            pd.Timestamp("2024-01-01 00:00:00.025", tz="UTC"): 0.0001,
            pd.Timestamp("2024-01-01 01:00:00.037", tz="UTC"): 0.0002,
        }
    )
    out = align_funding_to_index(raw, index, pd.Timedelta(hours=1))
    assert out.loc["2024-01-01 00:00"] == 0.0001
    assert out.loc["2024-01-01 01:00"] == 0.0002
    assert np.isnan(out.loc["2024-01-01 02:00"])


def test_empty_index_returns_empty_series():
    index = pd.DatetimeIndex([], tz="UTC")
    out = align_funding_to_index(pd.Series(dtype=float), index, pd.Timedelta(hours=1))
    assert len(out) == 0


def test_empty_raw_series_is_all_nan():
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    out = align_funding_to_index(pd.Series(dtype=float), index, pd.Timedelta(hours=1))
    assert out.isna().all()
