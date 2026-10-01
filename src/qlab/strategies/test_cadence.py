"""Tests for `qlab.strategies.cadence.SplitCadence` (docs/TASKS.md T25)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.cadence import SplitCadence, closes_period


class MomentumToy:
    """Causal toy: half the book long/short on the sign of the last 3-bar move."""

    name = "momentum-toy"
    valid_intervals = ("1h", "1d")

    def target_weights(self, panel, params):
        sign = np.sign(panel.prices.pct_change(periods=3)).fillna(0.0)
        return (0.5 * sign).where(panel.tradeable, 0.0)


class DailyOnlyToy(MomentumToy):
    valid_intervals = ("1d",)


def _panel(n_hours: int = 24 * 10, seed: int = 5) -> MarketPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h", tz="UTC")
    prices = (1 + pd.DataFrame(rng.normal(0, 0.01, (n_hours, 2)), index=index,
                               columns=["A", "B"])).cumprod() * 100
    return MarketPanel(
        snapshot_id="toy",
        prices=prices,
        funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
        tradeable=pd.DataFrame(True, index=index, columns=prices.columns),
        meta={"universe_complete": True},
    )


def _params(entry: str, exit_: str, inner: str = "MomentumToy") -> dict:
    return {
        "inner_code_ref": f"qlab.strategies.test_cadence:{inner}",
        "inner_params": {},
        "entry_every": entry,
        "exit_every": exit_,
    }


def test_closes_period_marks_the_last_hour_of_each_day() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    rows = closes_period(index, pd.Timedelta("1D"))
    assert [index[i].strftime("%d %H:%M") for i in np.flatnonzero(rows)] == [
        "01 23:00",
        "02 23:00",
    ]


def test_same_cadence_as_the_bar_is_the_wrapped_strategy_unchanged() -> None:
    panel = _panel()
    wrapped = SplitCadence().target_weights(panel, _params("1h", "1h"))
    inner = MomentumToy().target_weights(panel, {})
    pd.testing.assert_frame_equal(wrapped, inner, check_names=False)


def test_book_grows_only_at_the_daily_close_and_shrinks_any_hour() -> None:
    panel = _panel()
    weights = SplitCadence().target_weights(panel, _params("1D", "1h"))
    day_close = closes_period(panel.prices.index, pd.Timedelta("1D"))
    size = weights.abs().to_numpy()
    grew = (size[1:] > size[:-1] + 1e-12).any(axis=1)
    shrank = (size[1:] < size[:-1] - 1e-12).any(axis=1)
    assert grew.any() and shrank.any()
    assert day_close[1:][grew].all()  # every increase lands on a daily close
    assert not day_close[1:][shrank].all()  # some reductions happen intra-day


def test_inner_strategy_not_valid_on_the_bar_size_is_refused() -> None:
    with pytest.raises(ValueError, match="not valid on"):
        SplitCadence().target_weights(_panel(), _params("1D", "1h", inner="DailyOnlyToy"))


def test_period_must_be_a_whole_number_of_bars() -> None:
    with pytest.raises(ValueError, match="whole number"):
        SplitCadence().target_weights(_panel(), _params("90min", "1h"))


def test_split_cadence_never_sees_the_future() -> None:
    panel = _panel()
    cut = 24 * 6 + 5
    original = SplitCadence().target_weights(panel, _params("1D", "1h"))
    future = panel.prices.copy()
    rng = np.random.default_rng(1)
    future.iloc[cut + 1 :] *= rng.uniform(0.5, 1.5, size=future.iloc[cut + 1 :].shape)
    tampered = MarketPanel(
        snapshot_id="t", prices=future, funding=panel.funding,
        tradeable=panel.tradeable, meta=panel.meta,
    )
    scrambled = SplitCadence().target_weights(tampered, _params("1D", "1h"))
    pd.testing.assert_frame_equal(original.iloc[: cut + 1], scrambled.iloc[: cut + 1])
