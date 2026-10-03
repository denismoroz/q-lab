"""A bar without a price is held through, not forced flat (docs/TASKS.md T36)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.gaps import held_through, holdable_gaps, marked_through
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest

IDX = pd.date_range("2023-03-24 09:00", periods=7, freq="1h", tz="UTC")


def _frame(values, name="BTC-SPOT"):
    return pd.DataFrame({name: values}, index=IDX, dtype=float)


def test_a_gap_is_a_missing_price_inside_the_instruments_life() -> None:
    prices = _frame([np.nan, 100.0, 101.0, np.nan, np.nan, 103.0, np.nan])
    funding = _frame([np.nan] * 7)
    gaps = holdable_gaps(prices, funding, no_funding=["BTC-SPOT"])["BTC-SPOT"]
    # Before listing and after the last price: not held through.
    assert list(gaps) == [False, False, False, True, True, False, False]


def test_a_perp_with_unknown_funding_is_not_held_through() -> None:
    prices = pd.DataFrame({"BTC": [100.0, np.nan, 102.0]}, index=IDX[:3])
    unknown = pd.DataFrame({"BTC": [0.0, np.nan, 0.0]}, index=IDX[:3])
    known = pd.DataFrame({"BTC": [0.0, 0.0, 0.0]}, index=IDX[:3])
    assert not holdable_gaps(prices, unknown)["BTC"].iloc[1]
    assert holdable_gaps(prices, known)["BTC"].iloc[1]


def test_weights_are_held_and_prices_marked_across_the_gap() -> None:
    gaps = _frame([0, 0, 0, 1, 1, 0, 0]).astype(bool)
    weights = _frame([0.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.2])
    assert list(held_through(weights, gaps)["BTC-SPOT"]) == [0.0, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
    prices = _frame([99.0, 100.0, 101.0, np.nan, np.nan, 103.0, 104.0])
    marked = marked_through(prices, gaps)["BTC-SPOT"]
    assert list(marked.iloc[2:6]) == [101.0, 101.0, 101.0, 103.0]


def test_the_harness_holds_the_book_through_an_hour_without_trades() -> None:
    """Binance spot BTC had no trades 2023-03-24 12:00 UTC: a book long
    before it earns the move from the last print to the next one, pays no
    cost for the gap, and the strategy itself is still flat there."""
    prices = _frame([100.0, 100.0, 101.0, np.nan, 103.0, 103.0, 103.0])
    funding = _frame([np.nan] * 7)
    tradeable = _frame([1, 1, 1, 0, 1, 1, 1]).astype(bool)
    panel = MarketPanel(snapshot_id="s", prices=prices, funding=funding, tradeable=tradeable,
                        meta={"no_funding_instruments": ["BTC-SPOT"]})
    weights = _frame([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])  # flat where it may not trade
    result = run_backtest(panel, weights, CostModel(taker_fee_bps=10.0, slippage_bps=0.0),
                          panel.funding)
    growth = float((1 + result.gross_return).prod())
    assert growth == pytest.approx(103.0 / 100.0)
    assert result.turnover.iloc[1:].sum() == 0.0  # no exit and re-entry around the gap


def test_the_bar_before_a_gap_is_tradeable_in_the_panel() -> None:
    from qlab.data import snapshot as snap
    from qlab.data.test_snapshot import _hist

    prices = pd.Series([100.0, 100.0, 101.0, np.nan, 103.0, 103.0, 103.0], index=IDX)
    spot = dataclasses.replace(
        _hist("BTC-SPOT", IDX, is_delisted=False, prices=prices, has_funding=False),
        trade_count=pd.Series([5, 5, 5, 0, 5, 5, 5], index=IDX, dtype="int64"))
    out_prices, _f, tradeable, _v = snap._build_frames_from_histories(
        {"BTC-SPOT": spot}, IDX, pd.Timedelta(hours=1))
    assert np.isnan(out_prices["BTC-SPOT"].iloc[3])
    assert list(tradeable["BTC-SPOT"]) == [True, True, True, False, True, True, True]


def test_the_complete_book_window_is_not_broken_by_a_held_gap() -> None:
    from qlab.pipeline.evaluate import complete_book_window

    prices = pd.DataFrame({"BTC": [100.0] * 7,
                           "BTC-SPOT": [100.0, 100.0, 101.0, np.nan, 103.0, 103.0, 103.0]},
                          index=IDX)
    funding = pd.DataFrame({"BTC": [0.0] * 7, "BTC-SPOT": [np.nan] * 7}, index=IDX)
    tradeable = prices.notna()
    panel = MarketPanel(snapshot_id="s", prices=prices, funding=funding, tradeable=tradeable,
                        meta={"no_funding_instruments": ["BTC-SPOT"]})
    assert complete_book_window(panel, ["BTC", "BTC-SPOT"]).window == (0, 6)
