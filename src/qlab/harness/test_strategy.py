"""Tests for the Strategy interface and validate_weights.

Also demonstrates the three shapes `Strategy`'s docstring claims the
interface expresses without special-casing: cross-sectional long/short with
periodic rebalance, carry with limited slots and a minimum size, and spot
holding with a conditionally applied hedge leg. Each example below is a
minimal, illustrative implementation (not part of the framework itself) used
only to prove the interface accommodates the shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import WeightValidationError, validate_weights


def _panel(
    prices: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    tradeable: pd.DataFrame | None = None,
) -> MarketPanel:
    if funding is None:
        funding = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if tradeable is None:
        tradeable = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    return MarketPanel(
        snapshot_id="snap-strategy",
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta={},
    )


def _flat_prices(periods: int, cols: list[str]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(100.0, index=index, columns=cols)


# --- validate_weights: structural checks -----------------------------------


def test_validate_weights_accepts_well_formed_weights() -> None:
    prices = _flat_prices(3, ["A", "B"])
    panel = _panel(prices)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    validate_weights(panel, weights)  # must not raise


def test_validate_weights_rejects_index_mismatch() -> None:
    prices = _flat_prices(3, ["A", "B"])
    panel = _panel(prices)
    bad_index = pd.date_range("2027-01-01", periods=3, freq="D", tz="UTC")
    weights = pd.DataFrame(0.5, index=bad_index, columns=prices.columns)
    with pytest.raises(WeightValidationError, match="index"):
        validate_weights(panel, weights)


def test_validate_weights_rejects_columns_mismatch() -> None:
    prices = _flat_prices(3, ["A", "B"])
    panel = _panel(prices)
    weights = pd.DataFrame(0.5, index=prices.index, columns=["A", "C"])
    with pytest.raises(WeightValidationError, match="columns"):
        validate_weights(panel, weights)


def test_validate_weights_rejects_nan() -> None:
    prices = _flat_prices(3, ["A", "B"])
    panel = _panel(prices)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    weights.iloc[1, 0] = np.nan
    with pytest.raises(WeightValidationError, match="NaN"):
        validate_weights(panel, weights)


def test_validate_weights_rejects_exposure_on_non_tradeable_instrument() -> None:
    prices = _flat_prices(3, ["A", "B"])
    tradeable = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    tradeable.iloc[2, 1] = False  # B delisted on the last day
    panel = _panel(prices, tradeable=tradeable)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    with pytest.raises(WeightValidationError, match="non-tradeable"):
        validate_weights(panel, weights)


def test_validate_weights_allows_zero_on_non_tradeable_instrument() -> None:
    prices = _flat_prices(3, ["A", "B"])
    tradeable = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    tradeable.iloc[2, 1] = False
    panel = _panel(prices, tradeable=tradeable)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    weights.iloc[2, 1] = 0.0
    validate_weights(panel, weights)  # must not raise


# --- shape 1: cross-sectional long/short, periodic rebalance ---------------


class _ToyCrossSectional:
    """Long the best-scoring instrument, short the worst, every `rebal_every`
    periods; carry-forward the vector between rebalances."""

    name = "toy_xsec"

    def target_weights(self, panel: MarketPanel, params: dict) -> pd.DataFrame:
        rebal_every = params["rebal_every"]
        # A trivial "score": trailing 1-period return, known at t using only
        # prices up to and including t (pct_change() at row t uses price[t]
        # and price[t-1] -- never price[t+1]).
        scores = panel.prices.pct_change().fillna(0.0)
        weights = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        held = pd.Series(0.0, index=panel.prices.columns)
        for i, ts in enumerate(panel.prices.index):
            if i % rebal_every == 0:
                ranked = scores.loc[ts].sort_values(ascending=False)
                held = pd.Series(0.0, index=panel.prices.columns)
                held[ranked.index[0]] = 0.5
                held[ranked.index[-1]] = -0.5
            weights.loc[ts] = held
        return weights


def test_cross_sectional_shape_rebalances_and_carries_forward() -> None:
    cols = ["A", "B", "C"]
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.DataFrame(
        [[100.0, 100.0, 100.0], [110.0, 100.0, 90.0], [90.0, 100.0, 110.0], [90.0, 100.0, 110.0]],
        index=index,
        columns=cols,
    )
    panel = _panel(prices)
    strategy = _ToyCrossSectional()
    weights = strategy.target_weights(panel, {"rebal_every": 2})
    validate_weights(panel, weights)

    # Row 0: no prior info (pct_change NaN -> 0 for all) -> ties broken by
    # sort_values' stable order: first column long, last column short.
    assert weights.loc[index[0], "A"] == pytest.approx(0.5)
    assert weights.loc[index[0], "C"] == pytest.approx(-0.5)
    # Row 1: not a rebalance row (i=1, rebal_every=2) -> carried forward.
    assert (weights.loc[index[1]] == weights.loc[index[0]]).all()
    # Row 2: rebalance. A dropped 100->90 (worst), C rose 90->110 (best).
    assert weights.loc[index[2], "C"] == pytest.approx(0.5)
    assert weights.loc[index[2], "A"] == pytest.approx(-0.5)
    # Row 3: carried forward from row 2.
    assert (weights.loc[index[3]] == weights.loc[index[2]]).all()


# --- shape 2: carry with limited slots and a minimum position size ---------


class _ToyCarry:
    """Rank by |funding|, keep the top `max_slots`, drop (rather than
    under-size) any slot whose equal-weight share is below `min_position_size`."""

    name = "toy_carry"

    def target_weights(self, panel: MarketPanel, params: dict) -> pd.DataFrame:
        max_slots = params["max_slots"]
        min_position_size = params["min_position_size"]
        weights = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        for ts in panel.prices.index:
            funding_row = panel.funding.loc[ts].dropna()
            ranked = funding_row.abs().sort_values(ascending=False)
            slots = ranked.index[:max_slots]
            if len(slots) == 0:
                continue
            share = 1.0 / len(slots)
            if share < min_position_size:
                continue  # whole book too small to size this honestly -- skip
            for col in slots:
                # Short positive funding (pay us to be short), long negative.
                sign = -1.0 if funding_row[col] > 0 else 1.0
                weights.loc[ts, col] = sign * share
        return weights


def test_carry_shape_respects_slots_and_min_size() -> None:
    cols = ["A", "B", "C", "D"]
    prices = _flat_prices(2, cols)
    funding = pd.DataFrame(
        [[0.001, -0.0005, 0.02, 0.0001], [0.001, -0.0005, 0.02, 0.0001]],
        index=prices.index,
        columns=cols,
    )
    panel = _panel(prices, funding=funding)
    strategy = _ToyCarry()

    # max_slots=2 -> only the two largest |funding| survive: C (0.02), A (0.001).
    weights = strategy.target_weights(panel, {"max_slots": 2, "min_position_size": 0.1})
    validate_weights(panel, weights)
    row = weights.iloc[0]
    assert row["C"] == pytest.approx(-0.5)  # positive funding -> short
    assert row["A"] == pytest.approx(-0.5)  # positive funding -> short
    assert row["B"] == 0.0
    assert row["D"] == 0.0

    # A prohibitively high min_position_size drops the whole book to flat.
    weights_flat = strategy.target_weights(panel, {"max_slots": 2, "min_position_size": 0.9})
    validate_weights(panel, weights_flat)
    assert (weights_flat == 0.0).all().all()


# --- shape 3: spot holding with a conditionally applied hedge leg ----------


class _ToySpotHedge:
    """Always hold the spot leg; hedge only while funding on the hedge
    instrument exceeds a threshold."""

    name = "toy_spot_hedge"

    def target_weights(self, panel: MarketPanel, params: dict) -> pd.DataFrame:
        spot, hedge = params["spot"], params["hedge"]
        threshold = params["hedge_funding_threshold"]
        weights = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        weights[spot] = 1.0
        condition = (panel.funding[hedge] > threshold).fillna(False)
        weights.loc[condition, hedge] = -1.0
        return weights


def test_spot_hedge_shape_applies_hedge_conditionally() -> None:
    cols = ["SPOT", "PERP"]
    prices = _flat_prices(4, cols)
    funding = pd.DataFrame(
        {"SPOT": [0.0, 0.0, 0.0, 0.0], "PERP": [0.0001, 0.0009, np.nan, 0.0002]},
        index=prices.index,
    )
    panel = _panel(prices, funding=funding)
    strategy = _ToySpotHedge()
    weights = strategy.target_weights(
        panel, {"spot": "SPOT", "hedge": "PERP", "hedge_funding_threshold": 0.0005}
    )
    validate_weights(panel, weights)

    assert (weights["SPOT"] == 1.0).all()  # spot leg always held
    assert weights.loc[prices.index[0], "PERP"] == 0.0  # below threshold
    assert weights.loc[prices.index[1], "PERP"] == -1.0  # above threshold -> hedged
    assert weights.loc[prices.index[2], "PERP"] == 0.0  # NaN funding -> condition False
    assert weights.loc[prices.index[3], "PERP"] == 0.0  # below threshold again
