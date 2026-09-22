"""Tests for qlab.harness.capacity: book_daily_volume_usd and capacity_usd.

Every scenario is built with round numbers so the expected result can be
checked by hand from `capacity_usd`'s own docstring arithmetic, not just
asserted against whatever the code happens to produce.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.capacity import CapacityError, book_daily_volume_usd, capacity_usd
from qlab.harness.panel import MarketPanel


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")


def _panel(
    index: pd.DatetimeIndex,
    prices: dict[str, list[float]],
    volume: dict[str, list[float]] | None = None,
) -> MarketPanel:
    columns = list(prices)
    prices_df = pd.DataFrame(prices, index=index)[columns]
    funding = pd.DataFrame(0.0, index=index, columns=columns)
    tradeable = pd.DataFrame(True, index=index, columns=columns)
    volume_df = (
        pd.DataFrame(volume, index=index)[columns]
        if volume is not None
        else pd.DataFrame(float("nan"), index=index, columns=columns)
    )
    return MarketPanel(
        snapshot_id="test",
        prices=prices_df,
        funding=funding,
        tradeable=tradeable,
        meta={},
        volume=volume_df,
    )


def _weights(index: pd.DatetimeIndex, values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=index)[list(values)]


# --------------------------------------------------------------------------
# book_daily_volume_usd
# --------------------------------------------------------------------------


class TestBookDailyVolumeUsd:
    def test_profile_restricted_to_held_bars(self):
        index = _index(5)
        # BTC: price 100 flat, volume 1000 flat -> volume_usd = 100_000 every bar.
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 5},
            volume={"BTC": [1000.0] * 5},
        )
        # Enters at bar 1 (weight 0), held at 0.5 from bar 1 onward.
        weights = _weights(index, {"BTC": [0.0, 0.5, 0.5, 0.5, 0.5]})

        profile = book_daily_volume_usd(weights, panel)

        assert list(profile.index) == ["BTC"]
        row = profile.loc["BTC"]
        assert row["n_bars_held"] == 4
        assert row["avg_abs_weight"] == pytest.approx(0.5)
        assert row["median_daily_volume_usd"] == pytest.approx(100_000.0)
        assert row["min_daily_volume_usd"] == pytest.approx(100_000.0)
        assert row["n_bars_volume_known"] == 4

    def test_never_held_instrument_excluded(self):
        index = _index(3)
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 3, "ETH": [10.0] * 3},
            volume={"BTC": [1000.0] * 3, "ETH": [1000.0] * 3},
        )
        weights = _weights(index, {"BTC": [0.5, 0.5, 0.5], "ETH": [0.0, 0.0, 0.0]})

        profile = book_daily_volume_usd(weights, panel)

        assert list(profile.index) == ["BTC"]

    def test_all_flat_weights_gives_empty_profile(self):
        index = _index(3)
        panel = _panel(index, prices={"BTC": [100.0] * 3}, volume={"BTC": [1000.0] * 3})
        weights = _weights(index, {"BTC": [0.0, 0.0, 0.0]})

        profile = book_daily_volume_usd(weights, panel)

        assert profile.empty

    def test_unknown_volume_excluded_from_median_and_min_not_treated_as_zero(self):
        index = _index(4)
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 4},
            volume={"BTC": [float("nan"), float("nan"), 1000.0, 2000.0]},
        )
        weights = _weights(index, {"BTC": [0.5, 0.5, 0.5, 0.5]})

        profile = book_daily_volume_usd(weights, panel)

        row = profile.loc["BTC"]
        assert row["n_bars_held"] == 4
        assert row["n_bars_volume_known"] == 2
        # Known volume_usd values are 100_000 and 200_000 -- NaN bars must
        # not pull these toward zero.
        assert row["min_daily_volume_usd"] == pytest.approx(100_000.0)
        assert row["median_daily_volume_usd"] == pytest.approx(150_000.0)

    def test_median_and_min_differ_for_a_book_with_a_dry_spell(self):
        """The whole reason both are reported: a leg that is usually liquid
        but has one thin day looks identical to a consistently-thin leg if
        you only look at the median."""
        index = _index(5)
        panel = _panel(
            index,
            prices={"BTC": [1.0] * 5},
            volume={"BTC": [100_000.0, 100_000.0, 1_000.0, 100_000.0, 100_000.0]},
        )
        weights = _weights(index, {"BTC": [0.5] * 5})

        profile = book_daily_volume_usd(weights, panel)

        row = profile.loc["BTC"]
        assert row["median_daily_volume_usd"] == pytest.approx(100_000.0)
        assert row["min_daily_volume_usd"] == pytest.approx(1_000.0)

    def test_shape_mismatch_raises(self):
        index = _index(3)
        panel = _panel(index, prices={"BTC": [1.0] * 3}, volume={"BTC": [1.0] * 3})
        weights = _weights(index, {"ETH": [0.5] * 3})

        with pytest.raises(CapacityError, match="columns"):
            book_daily_volume_usd(weights, panel)


# --------------------------------------------------------------------------
# capacity_usd
# --------------------------------------------------------------------------


class TestCapacityUsd:
    def test_hand_computed_single_leg(self):
        """Matches capacity_usd's own docstring worked example exactly:
        min_daily_volume_usd=$100,000, daily_turnover_fraction=0.125,
        participation=0.05 -> capacity = $40,000."""
        index = _index(5)
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 5},
            volume={"BTC": [1000.0] * 5},
        )
        # Opens at bar 1 with weight 0.5, held flat afterward:
        # turnover on held bars [1,2,3,4] = [0.5, 0, 0, 0] -> mean 0.125.
        weights = _weights(index, {"BTC": [0.0, 0.5, 0.5, 0.5, 0.5]})

        result = capacity_usd(weights, panel, participation=0.05)

        assert result == pytest.approx(40_000.0)

    def test_thinnest_leg_is_binding(self):
        index = _index(5)
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 5, "THIN": [100.0] * 5},
            volume={"BTC": [1000.0] * 5, "THIN": [10.0] * 5},
        )
        weights = _weights(
            index,
            {
                "BTC": [0.0, 0.5, 0.5, 0.5, 0.5],
                "THIN": [0.0, 0.1, 0.1, 0.1, 0.1],
            },
        )
        # BTC: min_daily_volume_usd=100_000, turnover_frac=0.125 -> capacity 40_000.
        # THIN: min_daily_volume_usd=1_000, turnover_frac=0.1/4=0.025 -> capacity 2_000.
        result = capacity_usd(weights, panel, participation=0.05)

        assert result == pytest.approx(2_000.0)

    def test_participation_must_be_positive(self):
        index = _index(3)
        panel = _panel(index, prices={"BTC": [1.0] * 3}, volume={"BTC": [1.0] * 3})
        weights = _weights(index, {"BTC": [0.5] * 3})

        with pytest.raises(CapacityError, match="participation must be > 0"):
            capacity_usd(weights, panel, participation=0.0)

    def test_participation_has_no_default(self):
        import inspect

        sig = inspect.signature(capacity_usd)
        assert sig.parameters["participation"].default is inspect.Parameter.empty

    def test_all_flat_book_raises(self):
        index = _index(3)
        panel = _panel(index, prices={"BTC": [1.0] * 3}, volume={"BTC": [1.0] * 3})
        weights = _weights(index, {"BTC": [0.0] * 3})

        with pytest.raises(CapacityError, match="all-flat book"):
            capacity_usd(weights, panel, participation=0.05)

    def test_unknown_volume_on_held_leg_raises(self):
        index = _index(3)
        panel = _panel(
            index,
            prices={"BTC": [100.0] * 3},
            volume={"BTC": [float("nan")] * 3},
        )
        weights = _weights(index, {"BTC": [0.5, 0.5, 0.5]})

        with pytest.raises(CapacityError, match="unknown"):
            capacity_usd(weights, panel, participation=0.05)

    def test_no_measurable_turnover_raises(self):
        """A position held at a constant nonzero weight for the ENTIRE
        panel never shows a transition in this window (the only possible
        turnover bar, index 0, has no predecessor to diff against) --
        capacity from turnover is genuinely undefined here, not zero."""
        index = _index(4)
        panel = _panel(
            index, prices={"BTC": [100.0] * 4}, volume={"BTC": [1000.0] * 4}
        )
        weights = _weights(index, {"BTC": [0.5, 0.5, 0.5, 0.5]})

        with pytest.raises(CapacityError, match="never traded again"):
            capacity_usd(weights, panel, participation=0.05)

    def test_shape_mismatch_raises(self):
        index = _index(3)
        panel = _panel(index, prices={"BTC": [1.0] * 3}, volume={"BTC": [1.0] * 3})
        weights = _weights(index, {"ETH": [0.5] * 3})

        with pytest.raises(CapacityError, match="columns"):
            capacity_usd(weights, panel, participation=0.05)

    def test_daily_bar_interval_is_a_no_op_bars_per_day(self):
        """Sanity check that the bars_per_day conversion is 1.0 at a daily
        panel (every spec in specs/ today) -- i.e. this test's hand-computed
        expectations elsewhere in this file aren't accidentally relying on
        some other conversion factor."""
        index = _index(3)
        panel = _panel(index, prices={"BTC": [1.0] * 3}, volume={"BTC": [10.0] * 3})
        weights = _weights(index, {"BTC": [0.0, 1.0, 1.0]})
        # turnover restricted to held bars [1, 2] = [1.0, 0.0] -> mean 0.5.
        # min_daily_volume_usd = 10 (price=1 * volume=10).
        # capacity = participation * 10 / 0.5 = participation * 20.
        result = capacity_usd(weights, panel, participation=0.1)
        assert result == pytest.approx(2.0)
