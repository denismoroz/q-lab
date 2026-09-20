"""Tests for MarketPanel: shape validation and the slice/restrict/align helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.data.panel import MarketPanel


def _utc_index(n: int, freq: str = "1h", start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _panel(
    index: pd.DatetimeIndex | None = None,
    columns: list[str] | None = None,
    snapshot_id: str = "deadbeef",
    universe_complete: bool | None = None,
) -> MarketPanel:
    index = index if index is not None else _utc_index(5)
    columns = columns if columns is not None else ["BTC", "ETH"]
    prices = pd.DataFrame(1.0, index=index, columns=columns)
    funding = pd.DataFrame(0.0001, index=index, columns=columns)
    tradeable = pd.DataFrame(True, index=index, columns=columns)
    meta: dict[str, object] = {"venue": "test", "interval": "1h"}
    if universe_complete is not None:
        meta["universe_complete"] = universe_complete
    return MarketPanel(
        snapshot_id=snapshot_id,
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta=meta,
    )


class TestValidation:
    def test_valid_panel_constructs(self):
        panel = _panel()
        assert panel.instruments == ["BTC", "ETH"]

    def test_mismatched_index_rejected(self):
        index = _utc_index(5)
        other_index = _utc_index(5, start="2024-02-01")
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=other_index, columns=["BTC"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC"])
        with pytest.raises(ValueError, match="index"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )

    def test_mismatched_columns_rejected(self):
        index = _utc_index(5)
        prices = pd.DataFrame(1.0, index=index, columns=["BTC", "ETH"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC", "SOL"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC", "ETH"])
        with pytest.raises(ValueError, match="columns"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )

    def test_naive_index_rejected(self):
        index = pd.date_range("2024-01-01", periods=5, freq="1h")  # no tz
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC"])
        with pytest.raises(ValueError, match="timezone-aware"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )

    def test_non_utc_index_rejected(self):
        index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="US/Eastern")
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC"])
        with pytest.raises(ValueError, match="UTC"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )

    def test_non_bool_tradeable_rejected(self):
        index = _utc_index(5)
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC"])
        tradeable = pd.DataFrame(1, index=index, columns=["BTC"])  # int, not bool
        with pytest.raises(ValueError, match="bool"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )

    def test_non_bool_universe_complete_rejected(self):
        index = _utc_index(5)
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC"])
        with pytest.raises(ValueError, match="universe_complete"):
            MarketPanel(
                snapshot_id="x",
                prices=prices,
                funding=funding,
                tradeable=tradeable,
                meta={"universe_complete": "yes"},
            )

    def test_universe_complete_is_optional(self):
        # Not every caller populates meta -- absence must not be an error.
        panel = _panel(universe_complete=None)
        assert "universe_complete" not in panel.meta

    def test_duplicate_timestamps_rejected(self):
        index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC")
        prices = pd.DataFrame(1.0, index=index, columns=["BTC"])
        funding = pd.DataFrame(0.0, index=index, columns=["BTC"])
        tradeable = pd.DataFrame(True, index=index, columns=["BTC"])
        with pytest.raises(ValueError, match="duplicate"):
            MarketPanel(
                snapshot_id="x", prices=prices, funding=funding, tradeable=tradeable, meta={}
            )


class TestSlice:
    def test_slice_restricts_index_inclusive(self):
        panel = _panel(index=_utc_index(10))
        sliced = panel.slice(panel.prices.index[2], panel.prices.index[5])
        assert len(sliced.prices) == 4
        assert sliced.prices.index[0] == panel.prices.index[2]
        assert sliced.prices.index[-1] == panel.prices.index[5]

    def test_slice_keeps_snapshot_id_and_meta(self):
        panel = _panel(index=_utc_index(10), snapshot_id="abc123")
        sliced = panel.slice(panel.prices.index[0], panel.prices.index[-1])
        assert sliced.snapshot_id == "abc123"
        assert sliced.meta == panel.meta

    def test_slice_start_after_end_raises(self):
        panel = _panel(index=_utc_index(10))
        with pytest.raises(ValueError):
            panel.slice(panel.prices.index[5], panel.prices.index[2])

    def test_slice_preserves_universe_complete(self):
        """Narrowing the time range doesn't hand-pick any instrument, so a
        discovered-complete panel stays complete after slicing."""
        panel = _panel(index=_utc_index(10), universe_complete=True)
        sliced = panel.slice(panel.prices.index[0], panel.prices.index[-1])
        assert sliced.meta["universe_complete"] is True


class TestRestrict:
    def test_restrict_subsets_columns_in_order(self):
        panel = _panel(columns=["BTC", "ETH", "SOL"])
        restricted = panel.restrict(["SOL", "BTC"])
        assert restricted.instruments == ["SOL", "BTC"]
        assert list(restricted.prices.columns) == ["SOL", "BTC"]
        assert list(restricted.funding.columns) == ["SOL", "BTC"]
        assert list(restricted.tradeable.columns) == ["SOL", "BTC"]

    def test_restrict_resets_universe_complete_to_false(self):
        panel = _panel(columns=["BTC", "ETH", "SOL"], universe_complete=True)
        restricted = panel.restrict(["BTC", "ETH"])
        assert restricted.meta["universe_complete"] is False

    def test_restrict_to_the_full_set_still_resets_the_flag(self):
        """Naming every instrument by hand is still a manual selection --
        this is literally the survivorship-via-instrument-list bug, and the
        flag exists precisely to catch it even when nothing was dropped."""
        panel = _panel(columns=["BTC", "ETH"], universe_complete=True)
        restricted = panel.restrict(["BTC", "ETH"])
        assert restricted.meta["universe_complete"] is False

    def test_restrict_does_not_mutate_original_meta(self):
        panel = _panel(columns=["BTC", "ETH"], universe_complete=True)
        panel.restrict(["BTC"])
        assert panel.meta["universe_complete"] is True

    def test_restrict_unknown_instrument_raises(self):
        panel = _panel(columns=["BTC", "ETH"])
        with pytest.raises(ValueError, match="DOGE"):
            panel.restrict(["DOGE"])


class TestAlign:
    def test_align_intersects_index_and_columns(self):
        # Build overlapping-but-not-identical indices deterministically.
        idx_a = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        idx_b = pd.date_range("2024-01-01T05:00:00", periods=10, freq="1h", tz="UTC")
        panel_a = _panel(index=idx_a, columns=["BTC", "ETH"], snapshot_id="a")
        panel_b = _panel(index=idx_b, columns=["ETH", "SOL"], snapshot_id="b")

        aligned_a, aligned_b = panel_a.align(panel_b)

        expected_index = idx_a.intersection(idx_b)
        assert aligned_a.prices.index.equals(expected_index)
        assert aligned_b.prices.index.equals(expected_index)
        assert aligned_a.instruments == ["ETH"]
        assert aligned_b.instruments == ["ETH"]
        # provenance is preserved per-panel, not merged
        assert aligned_a.snapshot_id == "a"
        assert aligned_b.snapshot_id == "b"

    def test_align_resets_universe_complete_on_both_sides(self):
        idx = _utc_index(5)
        panel_a = _panel(index=idx, columns=["BTC", "ETH"], universe_complete=True)
        panel_b = _panel(index=idx, columns=["ETH", "SOL"], universe_complete=True)
        aligned_a, aligned_b = panel_a.align(panel_b)
        assert aligned_a.meta["universe_complete"] is False
        assert aligned_b.meta["universe_complete"] is False

    def test_align_no_overlap_raises(self):
        idx_a = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        idx_b = pd.date_range("2030-01-01", periods=5, freq="1h", tz="UTC")
        panel_a = _panel(index=idx_a)
        panel_b = _panel(index=idx_b)
        with pytest.raises(ValueError, match="overlapping timestamps"):
            panel_a.align(panel_b)

    def test_align_no_common_instruments_raises(self):
        idx = _utc_index(5)
        panel_a = _panel(index=idx, columns=["BTC"])
        panel_b = _panel(index=idx, columns=["ETH"])
        with pytest.raises(ValueError, match="overlapping instruments"):
            panel_a.align(panel_b)
