"""Tests for FrabPerpSpread: short the venue that pays more funding, long the other."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest
from qlab.harness.strategy import profit_reinvested, validate_weights
from qlab.strategies.frab_perp_spread import FrabPerpSpread

A, B = "BTC", "BTC@ASTER"


def _panel(funding: dict, n: int = 6, prices: dict | None = None,
           untradeable: dict | None = None) -> MarketPanel:
    """Hourly panel; `funding` maps column -> a number or a list of n numbers."""
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    columns = list(funding)
    price_frame = pd.DataFrame(100.0, index=index, columns=columns)
    for column, values in (prices or {}).items():
        price_frame[column] = values
    funding_frame = pd.DataFrame(
        {c: np.broadcast_to(np.asarray(v, dtype=float), (n,)) for c, v in funding.items()},
        index=index)
    tradeable = pd.DataFrame(True, index=index, columns=columns)
    for column, rows in (untradeable or {}).items():
        tradeable.iloc[rows, tradeable.columns.get_loc(column)] = False
    return MarketPanel(snapshot_id="snap-spread", prices=price_frame, funding=funding_frame,
                       tradeable=tradeable, meta={})


def _params(**overrides) -> dict:
    params = {
        "venue_a_columns": {"BTC": A}, "venue_b_columns": {"BTC": B},
        "max_slots": 1, "entry_spread_apr": 0.5, "exit_spread_apr": 0.1,
        "signal_window_hours": 2, "leg_share_of_slot": 1.0,
        "min_leg_notional_usd": 12, "book_capital_usd": 100,
    }
    params.update(overrides)
    return params


def _weights(panel: MarketPanel, **overrides) -> pd.DataFrame:
    return FrabPerpSpread().target_weights(panel, _params(**overrides))


def test_shorts_the_venue_that_pays_more():
    w = _weights(_panel({A: 0.0003, B: 0.0001}))
    assert w.iloc[0].tolist() == [0.0, 0.0]  # the window is not full yet
    assert (w[A].iloc[1:] == -1.0).all()
    assert (w[B].iloc[1:] == 1.0).all()


def test_the_mirror_case_takes_the_other_direction():
    w = _weights(_panel({A: 0.0001, B: 0.0003}))
    assert (w[A].iloc[1:] == 1.0).all()
    assert (w[B].iloc[1:] == -1.0).all()


def test_exits_when_the_spread_falls_below_the_exit_bar():
    w = _weights(_panel({A: [.0003, .0003, .0001, .0001, .0001, .0001], B: .0001}))
    assert w[A].tolist() == [0.0, -1.0, -1.0, 0.0, 0.0, 0.0]
    assert w[B].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_holds_between_the_bars_but_does_not_enter_there():
    # Row 2 spreads 1.026, rows 3 to 5 spread 0.30003: under entry (0.5), over exit (0.1).
    held = _weights(_panel({A: [.0003, .0003] + [.00013425] * 4, B: .0001}))
    assert held[A].tolist() == [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    never = _weights(_panel({A: .00013425, B: .0001}))
    assert (never == 0.0).all().all()


def test_a_reversed_spread_flips_on_the_same_bar():
    w = _weights(_panel({A: [.0003, .0003, .0001, .0001], B: [.0001, .0001, .0003, .0003]}, n=4),
                 signal_window_hours=1)
    assert w[A].tolist() == [-1.0, -1.0, 1.0, 1.0]
    assert w[B].tolist() == [1.0, 1.0, -1.0, -1.0]


def test_ranks_by_the_size_of_the_spread_and_keeps_to_the_slots():
    panel = _panel({"BTC": .0003, "BTC@ASTER": .0001, "ETH": .0001, "ETH@ASTER": .0004,
                    "SOL": .0002, "SOL@ASTER": .0001})
    w = _weights(
        panel, max_slots=2,
        venue_a_columns={"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"},
        venue_b_columns={"BTC": "BTC@ASTER", "ETH": "ETH@ASTER", "SOL": "SOL@ASTER"})
    row = w.iloc[1]
    assert row["ETH"] == 0.5 and row["ETH@ASTER"] == -0.5
    assert row["BTC"] == -0.5 and row["BTC@ASTER"] == 0.5
    assert row["SOL"] == 0.0 and row["SOL@ASTER"] == 0.0


def test_a_leg_that_cannot_trade_closes_both():
    panel = _panel({A: 0.0003, B: 0.0001}, untradeable={B: [3]})
    w = _weights(panel)
    assert w.iloc[3].tolist() == [0.0, 0.0]
    assert w.iloc[4].tolist() == [-1.0, 1.0]
    validate_weights(panel, w)


def test_unknown_funding_is_never_read_as_zero():
    w = _weights(_panel({A: 0.0003, B: [.0001, .0001, .0001, np.nan, .0001, .0001]}))
    assert w[A].tolist() == [0.0, -1.0, -1.0, 0.0, 0.0, -1.0]


def test_a_venue_settling_every_8_hours_is_compared_over_whole_periods():
    b = np.zeros(24)
    b[::8] = 0.0008  # one settlement per 8 hours, 0 between
    w = _weights(_panel({A: 0.0001, B: b}, n=24), signal_window_hours=8)
    assert (w == 0.0).all().all()


def test_the_slot_share_and_the_floor():
    panel = _panel({A: 0.0003, B: 0.0001})
    w = _weights(panel, max_slots=2, leg_share_of_slot=0.5)
    assert w[A].iloc[1:].tolist() == [-0.25] * 5
    assert w[B].iloc[1:].tolist() == [0.25] * 5
    flat = _weights(panel, max_slots=5, leg_share_of_slot=1.0, book_capital_usd=50)
    assert (flat == 0.0).all().all()


def test_refuses_params_that_cannot_mean_what_they_say():
    panel = _panel({A: 0.0003, B: 0.0001})
    with pytest.raises(ValueError, match="same coins"):
        _weights(panel, venue_b_columns={"ETH": B})
    with pytest.raises(ValueError, match="both venues"):
        _weights(panel, venue_b_columns={"BTC": A})
    with pytest.raises(ValueError, match="leg_share_of_slot"):
        _weights(panel, leg_share_of_slot=0)
    with pytest.raises(ValueError, match="exit_spread_apr"):
        _weights(panel, exit_spread_apr=0.5, entry_spread_apr=0.5)


def test_earns_the_difference_and_cancels_the_price():
    path = [100.0, 110.0, 121.0, 133.1]
    panel = _panel({A: 0.0003, B: 0.0001}, n=4, prices={A: path, B: path})
    w = _weights(panel, signal_window_hours=1)
    assert (w[A] == -1.0).all() and (w[B] == 1.0).all()
    result = run_backtest(panel, w, CostModel(taker_fee_bps=0.0, slippage_bps=0.0), panel.funding)
    assert len(result.net_return) == 3
    assert result.net_return.tolist() == pytest.approx([0.0002] * 3)


def test_a_future_bar_does_not_change_a_past_decision():
    base = {A: 0.0003, B: 0.0001}
    changed = {A: 0.0003, B: [.0001] * 4 + [.01] * 4}
    before = _weights(_panel(base, n=8))
    after = _weights(_panel(changed, n=8))
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])


def test_declares_a_constant_budget():
    assert profit_reinvested(FrabPerpSpread(), {}) is False
