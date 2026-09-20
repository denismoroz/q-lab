"""Tests for `qlab.strategies.bv2.BStrategyV2`.

Fixtures are small synthetic daily panels -- no network, no real market
data. `mom_short_days` / `mom_long_days` / `sticky_exit_hours` are given
small test-only values (not the production defaults, which need ~30 days of
warm-up history) so the momentum/sticky-latch arithmetic can be hand-checked
against a handful of rows; the strategy code itself has no hardcoded
defaults, so this only changes which numbers are plugged in, not which code
path runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.bv2 import BStrategyV2

COLUMNS = ["X-SPOT", "X-PERP"]
COIN_COLUMNS = {"spot_columns": {"X": "X-SPOT"}, "hedge_columns": {"X": "X-PERP"}}


def _panel(prices_x: list[float]) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=len(prices_x), freq="D", tz="UTC")
    prices = pd.DataFrame({"X-SPOT": prices_x, "X-PERP": prices_x}, index=index)
    tradeable = pd.DataFrame(True, index=index, columns=COLUMNS)
    funding = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    return MarketPanel(
        snapshot_id="snap-bv2", prices=prices, funding=funding, tradeable=tradeable, meta={}
    )


def _base_params(**overrides: object) -> dict:
    params = {
        **COIN_COLUMNS,
        "spot_share": 1.0,
        "hedge_threshold": 0.0,
        "mom_short_days": 2,
        "mom_long_days": 4,
        "sticky_exit_hours": 48,  # 2 daily bars
        "ratchet_threshold": 0.5,
    }
    params.update(overrides)
    return params


# Days 0-3: flat at 100 -> mom undefined/zero, no hedge.
# Days 4-9: falling (100 -> 80) -> mom14/mom30-equivalents negative -> hedge on.
# Day 9 (i=9): momentum flattens (mom_short back to 0) -- still latched via sticky exit.
# Day 10-11: rising strongly -> both moms positive -> hedge wish off, and by day 11
# the sticky latch (2 bars) has expired -> hedge off.
_HEDGE_PRICES = [100, 100, 100, 100, 100, 90, 85, 80, 80, 80, 95, 110]


def test_bv2_weights_pass_validate_weights() -> None:
    panel = _panel(_HEDGE_PRICES)
    strategy = BStrategyV2()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # must not raise


def test_bv2_hedge_appears_and_disappears_per_momentum_with_sticky_exit() -> None:
    panel = _panel(_HEDGE_PRICES)
    strategy = BStrategyV2()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)

    index = panel.prices.index
    # Hand-derived from mom_short (2d) / mom_long (4d) on `_HEDGE_PRICES`:
    # raw hedge wish is True for days 4-9 (mom<=0 on both windows, or the
    # 4-day window undefined before day 4 forces it False), False otherwise.
    # The sticky-exit latch (2 bars) then keeps it on through day 10 and
    # turns it off at day 11.
    expected_hedge_on = [False] * 4 + [True] * 7 + [False]
    for ts, on in zip(index, expected_hedge_on, strict=True):
        hedge_w = weights.loc[ts, "X-PERP"]
        spot_w = weights.loc[ts, "X-SPOT"]
        if on:
            assert hedge_w == pytest.approx(-spot_w), f"{ts}: hedge should offset spot"
            assert hedge_w != 0.0
        else:
            assert hedge_w == 0.0, f"{ts}: hedge should be off"


def test_bv2_ratchet_trims_spot_at_documented_threshold() -> None:
    """spot_weight_target=1.0 (spot_share=1.0, one coin); ratchet_threshold=0.5
    means the ratchet fires once value exceeds target*(1+0.5)=1.5, trimming
    it straight back to 1.0 rather than letting it float above."""
    prices = [100, 120, 151, 170, 230, 100]
    panel = _panel(prices)
    strategy = BStrategyV2()
    # hedge_threshold so low that "up" is true whenever momentum is defined
    # at all -- the hedge never fires, isolating the ratchet's own behaviour.
    params = _base_params(hedge_threshold=-10.0, mom_short_days=1, mom_long_days=2)
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)

    index = panel.prices.index
    spot = weights["X-SPOT"]
    assert spot.loc[index[0]] == pytest.approx(1.0)
    assert spot.loc[index[1]] == pytest.approx(1.2)  # 120/100, below threshold -- no trim
    assert spot.loc[index[2]] == pytest.approx(1.0)  # 151/100=1.51 > 1.5 -- ratchet resets
    assert spot.loc[index[3]] == pytest.approx(170 / 151)  # growth measured from the new reset
    assert spot.loc[index[4]] == pytest.approx(1.0)  # 230/151=1.523 > 1.5 -- ratchet resets again
    assert spot.loc[index[5]] == pytest.approx(100 / 230)  # falls back, nothing to ratchet
    assert (weights["X-PERP"] == 0.0).all()  # hedge never triggered in this fixture


def test_bv2_no_hedge_when_momentum_never_turns_negative() -> None:
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    panel = _panel(prices)
    strategy = BStrategyV2()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)
    assert (weights["X-PERP"] == 0.0).all()


def test_bv2_is_deterministic() -> None:
    panel = _panel(_HEDGE_PRICES)
    strategy = BStrategyV2()
    params = _base_params()
    weights1 = strategy.target_weights(panel, params)
    weights2 = strategy.target_weights(panel, params)
    pd.testing.assert_frame_equal(weights1, weights2)
