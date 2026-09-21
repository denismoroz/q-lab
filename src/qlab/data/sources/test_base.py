"""Tests for align_funding_to_index: the core "NaN means unknown, never
silently zero" logic shared by every source fetcher. Also covers the
per-instrument cache's `has_funding` persistence (docs/TASKS.md, T17)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qlab.data.sources.base import (
    MAX_PLAUSIBLE_BAR_MOVE,
    InstrumentHistory,
    align_funding_to_index,
    detect_bad_price_bars,
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


# --------------------------------------------------------------------------
# detect_bad_price_bars: quote sanity at collection time (docs/TASKS.md, T17)
#
# Reproduces the live defect found in Hyperliquid's free spot API: a newly
# created spot pair with no real trading yet returns a repeating, non-zero
# placeholder "close" (Hyperliquid's UBTC/USDC pair, @142, printed a
# constant 6969696 then 7979573 for 11 daily bars in Feb 2025 -- about 82x
# BTC's real price -- before jumping to a real ~$97.6k print the moment
# trading actually began). Other thin HL spot pairs showed the same shape
# with 1000%+ jumps (BERA-SPOT, MON-SPOT, TRUMP-SPOT).
# --------------------------------------------------------------------------


def _series(index: pd.DatetimeIndex, values: list[float]) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def test_constant_run_flags_every_bar_in_the_run_including_the_first():
    idx = pd.date_range("2025-02-03", periods=6, freq="1D", tz="UTC")
    # 3 constant placeholder bars, then a real, varying price.
    prices = _series(idx, [6969696.0, 6969696.0, 6969696.0, 97578.0, 97597.0, 96243.0])
    flagged = detect_bad_price_bars(prices)
    assert list(flagged.iloc[:3]) == [True, True, True]
    # The transition bar (97578) is a +1300% jump off the placeholder --
    # also correctly flagged, even though it's the first REAL price.
    assert flagged.iloc[3]
    assert not flagged.iloc[4]
    assert not flagged.iloc[5]


def test_reproduces_live_btc_spot_defect_exact_values():
    """The exact sequence observed live (see module docstring above):
    constant 6969696 for 5 bars, constant 7979573 for 6 bars, then a real
    print. Every placeholder bar must be flagged; the two ordinary bars
    that follow the transition must not be."""
    idx = pd.date_range("2025-02-03", periods=13, freq="1D", tz="UTC")
    values = [6969696.0] * 5 + [7979573.0] * 6 + [97578.0, 97597.0]
    prices = _series(idx, values)
    flagged = detect_bad_price_bars(prices)
    assert flagged.iloc[:11].all()  # both placeholder runs, in full
    assert not flagged.iloc[12]  # 97597 vs 97578: an ordinary ~0.02% move


def test_isolated_single_repeat_is_flagged_not_ignored():
    """A run of length 2 (not "many") is still a constancy hit -- see the
    function's docstring: no magnitude to calibrate, it's a pure equality
    test, so there is no minimum run length below which it's ignored."""
    idx = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC")
    prices = _series(idx, [100.0, 100.0, 101.0])
    flagged = detect_bad_price_bars(prices)
    assert flagged.iloc[0]
    assert flagged.iloc[1]
    assert not flagged.iloc[2]


def test_ordinary_varying_series_is_never_flagged():
    idx = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    prices = _series(idx, [100.0, 101.0, 99.5, 102.0, 101.5])
    flagged = detect_bad_price_bars(prices)
    assert not flagged.any()


def test_jump_at_exactly_the_threshold_is_not_flagged_but_just_over_is():
    idx = pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC")
    at_threshold = _series(idx, [100.0, 100.0 * (1 + MAX_PLAUSIBLE_BAR_MOVE)])
    just_over = _series(idx, [100.0, 100.0 * (1 + MAX_PLAUSIBLE_BAR_MOVE) + 1.0])
    assert not detect_bad_price_bars(at_threshold).iloc[1]
    assert detect_bad_price_bars(just_over).iloc[1]


def test_first_and_last_bar_have_nothing_to_compare_against():
    """A single-element series (or the very first/last bar of a longer one,
    once its one neighbour is itself absent) can never be flagged -- there
    is nothing to detect an anomaly against."""
    idx = pd.date_range("2025-01-01", periods=1, freq="1D", tz="UTC")
    prices = _series(idx, [123456.0])  # would look wild in isolation
    assert not detect_bad_price_bars(prices).any()


def test_gaps_around_a_not_yet_listed_period_are_not_flagged():
    """NaN neighbours (e.g. bars before an instrument's first_seen, once
    reindexed onto the panel's full index) must not spuriously trigger
    either rule -- NaN == NaN is False, and a pct-change against NaN is
    NaN, both correctly treated as "nothing to flag" via `fillna(False)`."""
    idx = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
    prices = _series(idx, [float("nan"), float("nan"), 100.0, 100.5])
    flagged = detect_bad_price_bars(prices)
    assert not flagged.any()


def test_empty_series_returns_empty_result():
    idx = pd.DatetimeIndex([], tz="UTC")
    flagged = detect_bad_price_bars(pd.Series(dtype=float, index=idx))
    assert len(flagged) == 0
