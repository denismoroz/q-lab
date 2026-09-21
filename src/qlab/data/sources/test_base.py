"""Tests for align_funding_to_index: the core "NaN means unknown, never
silently zero" logic shared by every source fetcher. Also covers the
per-instrument cache's `has_funding` persistence (docs/TASKS.md, T17)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qlab.data.sources.base import (
    InstrumentHistory,
    align_funding_to_index,
    load_cached_history,
    store_cached_history,
)


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


# --------------------------------------------------------------------------
# Per-instrument cache: has_funding round trip (docs/TASKS.md, T17)
# --------------------------------------------------------------------------


def _history(instrument: str, index: pd.DatetimeIndex, *, has_funding: bool) -> InstrumentHistory:
    funding = pd.Series(0.0001, index=index) if has_funding else pd.Series(dtype=float)
    return InstrumentHistory(
        instrument=instrument,
        prices=pd.Series([1.0, 2.0], index=index),
        funding=funding,
        first_seen=index[0],
        last_seen=index[-1],
        is_delisted=False,
        has_funding=has_funding,
    )


def test_cache_round_trip_persists_has_funding_false(tmp_path):
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    hist = _history("BTC-SPOT", index, has_funding=False)

    store_cached_history(tmp_path, "hyperliquid", "1h", index[0], index[-1], hist)
    loaded = load_cached_history(tmp_path, "hyperliquid", "BTC-SPOT", "1h", index[0], index[-1])

    assert loaded is not None
    assert loaded.has_funding is False


def test_cache_round_trip_persists_has_funding_true(tmp_path):
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    hist = _history("BTC", index, has_funding=True)

    store_cached_history(tmp_path, "hyperliquid", "1h", index[0], index[-1], hist)
    loaded = load_cached_history(tmp_path, "hyperliquid", "BTC", "1h", index[0], index[-1])

    assert loaded is not None
    assert loaded.has_funding is True


def test_cache_backfills_has_funding_true_for_legacy_entries_without_the_key(tmp_path):
    """A cache entry written before spot support existed has no
    "has_funding" key at all -- it is necessarily a perp fetch, so True is
    the correct backfill, not a guess."""
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    hist = _history("BTC", index, has_funding=True)
    store_cached_history(tmp_path, "hyperliquid", "1h", index[0], index[-1], hist)

    meta_path = tmp_path / "hyperliquid" / "1h" / f"BTC__{index[0].date()}__{index[-1].date()}.json"
    meta = json.loads(meta_path.read_text())
    del meta["has_funding"]
    meta_path.write_text(json.dumps(meta))

    loaded = load_cached_history(tmp_path, "hyperliquid", "BTC", "1h", index[0], index[-1])
    assert loaded is not None
    assert loaded.has_funding is True
