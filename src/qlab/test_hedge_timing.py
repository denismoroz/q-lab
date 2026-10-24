"""The hedge-timing test keeps the spot and the amount of hedging, and moves
only the moments (docs/HEDGE_TIMING.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.hedge_timing import hedge_pairs, hedge_timing_test, shifted_hedges

HOURS = pd.date_range("2026-01-01", periods=24 * 60, freq="1h", tz="UTC")


def _market() -> tuple[MarketPanel, pd.DataFrame]:
    """A coin that rises and falls in five-day blocks, and a book that holds
    half its capital in spot and hedges exactly the falling blocks."""
    falling = (np.arange(len(HOURS)) // 120) % 2 == 1
    step = np.where(falling, -0.002, 0.002)
    price = pd.Series(100 * np.exp(np.cumsum(step)), index=HOURS)
    prices = pd.DataFrame({"A": price, "A-SPOT": price})
    zero = pd.DataFrame(0.0, index=HOURS, columns=prices.columns)
    panel = MarketPanel(snapshot_id="t", prices=prices, funding=zero,
                        tradeable=pd.DataFrame(True, index=HOURS, columns=prices.columns),
                        meta={})
    # Decided at the bar before the block turns: a weight at t earns t -> t+1.
    hedge = pd.Series(falling, index=HOURS).shift(-1, fill_value=False)
    weights = pd.DataFrame({"A": np.where(hedge, -0.5, 0.0), "A-SPOT": 0.5}, index=HOURS)
    return panel, weights


def test_only_the_moments_move() -> None:
    _, weights = _market()
    start = HOURS[24 * 10]
    shifted = shifted_hedges(weights, start, seed=3)
    assert hedge_pairs(weights) == {"A": "A-SPOT"}
    pd.testing.assert_series_equal(shifted["A-SPOT"], weights["A-SPOT"])
    inside = weights.index >= start
    assert (shifted.loc[~inside, "A"] == weights.loc[~inside, "A"]).all()
    assert (shifted.loc[inside, "A"] != 0).sum() == (weights.loc[inside, "A"] != 0).sum()
    assert not shifted.loc[inside, "A"].equals(weights.loc[inside, "A"])


def test_perfect_timing_beats_shifted_timing() -> None:
    panel, weights = _market()
    r = hedge_timing_test(panel, weights, CostModel(taker_fee_bps=0.0, slippage_bps=0.0),
                          HOURS[24 * 10], n=40)
    assert r.percentile >= 0.9
    assert r.own > r.spot_only and r.own > r.always_hedged
    assert 0.4 < r.hedged_share < 0.6
