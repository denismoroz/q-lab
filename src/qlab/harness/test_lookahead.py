"""Tests for `qlab.harness.lookahead`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.lookahead import lookahead_violation, tampered_after
from qlab.harness.panel import MarketPanel


def _panel() -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=60, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (60, 3)), axis=0),
                          index=index, columns=["A", "B", "C"])
    prices.iloc[:10, 2] = np.nan  # C lists later
    tradeable = prices.notna()
    return MarketPanel(snapshot_id="la", prices=prices,
                       funding=pd.DataFrame(1e-5, index=index, columns=prices.columns),
                       tradeable=tradeable, meta={})


class Momentum:
    def target_weights(self, panel, params):
        sig = np.sign(panel.prices.pct_change(5)).fillna(0.0) * 0.3
        return sig.where(panel.tradeable, 0.0)


class Peek:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def target_weights(self, panel, params):
        sig = np.sign(panel.prices.shift(-self.horizon) - panel.prices).fillna(0.0) * 0.3
        return sig.where(panel.tradeable, 0.0)


def test_tampering_changes_only_rows_after_the_cut_and_keeps_listing() -> None:
    panel = _panel()
    t = tampered_after(panel, 30, seed=1)
    pd.testing.assert_frame_equal(t.prices.iloc[:31], panel.prices.iloc[:31])
    assert not np.allclose(t.prices.iloc[31:].to_numpy(), panel.prices.iloc[31:].to_numpy())
    pd.testing.assert_frame_equal(t.prices.isna(), panel.prices.isna())
    assert t.snapshot_id != panel.snapshot_id


def test_honest_strategy_passes() -> None:
    panel = _panel()
    w = Momentum().target_weights(panel, {})
    assert lookahead_violation(Momentum(), panel, {}, w) is None


def test_peeks_of_one_bar_and_of_many_are_caught() -> None:
    panel = _panel()
    for horizon in (1, 15):
        strategy = Peek(horizon)
        reason = lookahead_violation(strategy, panel, {}, strategy.target_weights(panel, {}))
        assert reason is not None and "look-ahead" in reason


class LateStarter:
    """Flat for most of the window, then peeks -- cuts must land where it trades."""

    def target_weights(self, panel, params):
        sig = np.sign(panel.prices.shift(-1) - panel.prices).fillna(0.0) * 0.3
        sig.iloc[:45] = 0.0
        return sig.where(panel.tradeable, 0.0)


def test_cuts_land_where_the_strategy_holds_positions() -> None:
    panel = _panel()
    strategy = LateStarter()
    assert lookahead_violation(strategy, panel, {}, strategy.target_weights(panel, {})) is not None


class RareTrader:
    """Changes position once every 20 bars, deciding with tomorrow's price."""

    def target_weights(self, panel, params):
        nxt = np.sign(panel.prices.shift(-1) - panel.prices).fillna(0.0) * 0.3
        held = nxt.copy()
        for i in range(len(held.index)):
            if i % 20 != 7:
                held.iloc[i] = held.iloc[i - 1] if i else 0.0
        return held.where(panel.tradeable, 0.0)


def test_a_rare_trader_is_cut_right_before_its_decisions() -> None:
    panel = _panel()
    strategy = RareTrader()
    w = strategy.target_weights(panel, {})
    assert lookahead_violation(strategy, panel, {}, w, cut_fractions=()) is not None
