"""The price a spot leg is valued at (qlab.strategies.live._marks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.live._marks import spot_marks

HOURS = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")


def _panel(meta: dict[str, object]) -> MarketPanel:
    prices = pd.DataFrame({
        "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "A-SPOT": [np.nan, 101.5, np.nan, np.nan, 104.5, np.nan],
    }, index=HOURS)
    return MarketPanel(snapshot_id="m", prices=prices,
                       funding=pd.DataFrame(0.0, index=HOURS, columns=prices.columns),
                       tradeable=prices.notna(), meta=meta)


def test_by_default_the_spot_is_valued_at_the_perp() -> None:
    for meta in ({}, {"spot_prices": "perp"}):
        marks = spot_marks(_panel(meta), "A", "A-SPOT")
        assert list(marks) == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]


def test_own_prices_are_carried_forward_and_never_replaced_by_the_perp() -> None:
    marks = spot_marks(_panel({"spot_prices": "own"}), "A", "A-SPOT")
    assert np.isnan(marks[0])  # before the first print nothing is held
    assert list(marks[1:]) == [101.5, 101.5, 101.5, 104.5, 104.5]
    perp = _panel({}).prices["A"].to_numpy()
    assert not np.isin(marks[1:], perp).any()


def test_a_later_print_never_reaches_an_earlier_bar() -> None:
    panel = _panel({"spot_prices": "own"})
    before = spot_marks(panel, "A", "A-SPOT")[:4]
    prices = panel.prices.copy()
    prices.iloc[4:, 1] = 999.0
    after = spot_marks(MarketPanel(snapshot_id="m", prices=prices, funding=panel.funding,
                                   tradeable=panel.tradeable, meta=panel.meta),
                       "A", "A-SPOT")[:4]
    np.testing.assert_array_equal(before, after)
