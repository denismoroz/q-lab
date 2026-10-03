"""Tests for the Bv2 adapter over frab's live book (`qlab.strategies.live.bv2`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.live.bv2 import LiveBv2

# The production b2 instance's own params (frab.db strategies.params_json on
# 10.8.0.5, read 2026-10-01), restricted to one coin for a small panel.
PARAMS = {
    "coins": ["BTC"],
    "capital_usd": 257.0,
    "spot_share": 0.5,
    "hedge_threshold": 0.0,
    "sticky_exit_hours": 12,
    "ratchet_threshold": 0.5,
    "carry_enabled": False,
    "margin_enabled": True,
    "short_leverage": {"BTC": 3.0},
    "maint_margin_rate": {"BTC": 0.0125},
    "margin_buffer": 0.1,
    "hedge_margin_headroom": 1.0,
    "rebalance_at_im_share": 0.5,
    "mode": "paper",
}


def _panel(days_up: int = 40, days_down: int = 30, spot_from: int = 0) -> MarketPanel:
    hours = 24 * (days_up + days_down)
    index = pd.date_range("2026-01-01", periods=hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    drift = np.r_[np.full(24 * days_up, 0.0006), np.full(24 * days_down, -0.0009)]
    px = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, hours))
    prices = pd.DataFrame({"BTC": px, "BTC-SPOT": px * 1.0005}, index=index)
    tradeable = pd.DataFrame(True, index=index, columns=prices.columns)
    tradeable.iloc[:spot_from, 1] = False
    return MarketPanel(
        snapshot_id="bv2",
        prices=prices,
        funding=pd.DataFrame(0.00001, index=index, columns=prices.columns),
        tradeable=tradeable,
        meta={"universe_complete": True},
        high=prices * 1.001,
    )


def test_weights_are_valid_and_the_hedge_turns_on_in_the_downtrend() -> None:
    panel = _panel()
    run = LiveBv2().run(panel, PARAMS)
    validate_weights(panel, run.weights)

    up = run.weights.iloc[24 * 35]  # deep in the rise, 30-day momentum positive
    down = run.weights.iloc[-1]  # deep in the fall
    assert up["BTC-SPOT"] > 0 and up["BTC"] == 0.0
    assert down["BTC"] < 0
    # Hedged = delta-neutral: the short covers the spot units.
    assert down["BTC"] == pytest.approx(-down["BTC-SPOT"], rel=0.01)
    assert run.book_equity.iloc[0] == pytest.approx(257.0, rel=0.01)


def test_unknown_bar_high_is_refused_not_replaced_by_the_close() -> None:
    panel = _panel()
    high = panel.high.copy()
    high.iloc[900, 0] = np.nan  # after the book has started (startup history = 788 bars)
    broken = MarketPanel(
        snapshot_id="bv2", prices=panel.prices, funding=panel.funding,
        tradeable=panel.tradeable, meta=panel.meta, high=high,
    )
    with pytest.raises(ValueError, match="bar high unknown"):
        LiveBv2().run(broken, PARAMS)


def test_book_starts_only_with_both_legs_and_the_startup_history() -> None:
    from qlab.strategies.live.bv2 import HISTORY_BARS, WARMUP_BARS

    need = HISTORY_BARS + WARMUP_BARS
    late_spot = 24 * 40  # spot lists after the startup history already exists
    weights = LiveBv2().target_weights(_panel(spot_from=late_spot), PARAMS)
    assert (weights.iloc[:late_spot] == 0.0).all().all()
    assert weights["BTC-SPOT"].iloc[late_spot] > 0

    early = LiveBv2().target_weights(_panel(), PARAMS)  # both legs from bar 0
    assert (early.iloc[:need] == 0.0).all().all()  # waits for the history
    assert early["BTC-SPOT"].iloc[need] > 0


def test_adapter_never_sees_the_future() -> None:
    panel = _panel()
    cut = 24 * 50
    original = LiveBv2().target_weights(panel, PARAMS)
    future = panel.prices.copy()
    noise = np.random.default_rng(3).uniform(0.7, 1.3, size=future.iloc[cut + 1 :].shape)
    future.iloc[cut + 1 :] *= noise
    tampered = MarketPanel(
        snapshot_id="t", prices=future, funding=panel.funding, tradeable=panel.tradeable,
        meta=panel.meta, high=future * 1.001,
    )
    scrambled = LiveBv2().target_weights(tampered, PARAMS)
    pd.testing.assert_frame_equal(original.iloc[: cut + 1], scrambled.iloc[: cut + 1])


def test_missing_column_is_named() -> None:
    panel = _panel()
    with pytest.raises(ValueError, match="ETH"):
        LiveBv2().run(panel, {**PARAMS, "coins": ["BTC", "ETH"],
                              "short_leverage": {"BTC": 3.0, "ETH": 2.0}})


def test_weights_change_only_when_the_book_trades() -> None:
    """The book holds units; as fractions they would drift every bar with
    price. Weights are re-read only on decision bars, so they change rarely --
    on hedge switches, ratchet sells, starts -- not hourly."""
    weights = LiveBv2().target_weights(_panel(), PARAMS)
    changed = (weights.diff().abs().sum(axis=1) > 0).sum()
    assert 0 < changed < 0.05 * len(weights)


def test_an_hour_without_a_spot_price_keeps_both_legs() -> None:
    """docs/TASKS.md T36: the live engine waits on a missing price with both
    legs open. The adapter used to report the coin flat for that hour, which
    dropped the hedge on the perp (still tradeable) and re-opened it after."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest

    panel = _panel()
    gap = len(panel.prices) - 48  # deep in the fall: the hedge is on
    prices, tradeable = panel.prices.copy(), panel.tradeable.copy()
    prices.iloc[gap, 1] = np.nan
    tradeable.iloc[gap, 1] = False
    holed = MarketPanel(snapshot_id="bv2", prices=prices, funding=panel.funding,
                        tradeable=tradeable, meta=panel.meta, high=panel.high)
    weights = LiveBv2().run(holed, PARAMS).weights
    validate_weights(holed, weights)
    assert weights["BTC"].iloc[gap - 1] < 0
    assert weights["BTC"].iloc[gap] == weights["BTC"].iloc[gap - 1]  # the hedge stays
    result = run_backtest(holed, weights, CostModel(taker_fee_bps=4.5, slippage_bps=0.0),
                          holed.funding)
    assert result.turnover.iloc[gap - 1 : gap + 1].sum() == 0.0  # nothing closed or reopened
