"""Tests for `qlab.strategies.live.trend.LiveTrendTSMOMEnsemble`.

The look-ahead guard test (`test_book_vol_scale_never_sees_bar_ts_own_return`)
is the important one -- see its docstring and the module docstring of
`qlab.strategies.live.trend` for why. It was verified to actually catch the
bug it claims to catch: with the driver's equity-curve update temporarily
moved to run BEFORE `book_vol_scale`/`target_weights` instead of after (a
one-line reordering, simulating exactly the "shift the equity window
forward by one bar" mistake `docs/TASKS.md` T29 warns about), this test --
and only this test, in this file or `qlab.strategies.trend.test_trend` --
failed. The driver was then restored to the version below.
"""

from __future__ import annotations

import pandas as pd
import pytest

import qlab.strategies.live.trend as live_trend
from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.live.trend import LiveTrendTSMOMEnsemble

COLUMNS = ["A", "B"]


def _small_params(**overrides: object) -> dict:
    params = {
        "coins": ["A", "B"],
        "lookbacks": [1],
        "vol_window": 2,
        "min_history_days": 3,
        "vol_target_daily": 0.02,
        "leverage_cap": 100.0,
        "risk_scale": 1.0,
        "book_vol_target_ann": 0.1,
        "book_vol_window_days": 3,
        "book_vol_min_days": 1,
        "book_vol_scale_min": 0.01,
        "book_vol_scale_max": 100.0,
        "book_vol_prior_ann": 0.3,
        "capital_usd": 1000.0,
        "min_order_usd": 1.0,
    }
    params.update(overrides)
    return params


def _panel(
    prices_a: list[float], prices_b: list[float], tradeable_a: list[bool] | None = None
) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=len(prices_a), freq="D", tz="UTC")
    prices = pd.DataFrame({"A": prices_a, "B": prices_b}, index=index)
    funding = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    tradeable = pd.DataFrame(True, index=index, columns=COLUMNS)
    if tradeable_a is not None:
        tradeable["A"] = tradeable_a
    return MarketPanel(
        snapshot_id="snap-live-trend", prices=prices, funding=funding, tradeable=tradeable, meta={}
    )


def test_weights_pass_validate_weights() -> None:
    panel = _panel(
        prices_a=[100.0, 101.0, 102.0, 101.5, 103.0, 104.0],
        prices_b=[50.0, 49.5, 49.0, 49.2, 48.8, 48.5],
    )
    strategy = LiveTrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _small_params())
    validate_weights(panel, weights)  # must not raise
    assert list(weights.columns) == COLUMNS


def test_no_position_before_min_history() -> None:
    """`min_history_days=2` -- the very first bar (a single close) must be
    flat on every coin, mirroring frab's own `len(closes) < min_history_days
    -> 0.0` rule."""
    panel = _panel(
        prices_a=[100.0, 101.0, 102.0],
        prices_b=[50.0, 49.5, 49.0],
    )
    strategy = LiveTrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _small_params())
    row0 = panel.prices.index[0]
    assert weights.loc[row0, "A"] == 0.0
    assert weights.loc[row0, "B"] == 0.0


def test_respects_tradeable_mask() -> None:
    """A identical price path could size like B, but is marked non-tradeable
    at the last row -- it must read exactly 0 there, never a stale size."""
    prices_a = [100.0, 101.0, 102.0, 103.0]
    tradeable_a = [True, True, True, False]
    panel = _panel(prices_a=prices_a, prices_b=list(prices_a), tradeable_a=tradeable_a)
    strategy = LiveTrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _small_params())
    validate_weights(panel, weights)  # would raise if A carried exposure while non-tradeable
    last = panel.prices.index[-1]
    assert weights.loc[last, "A"] == 0.0
    assert weights.loc[last, "B"] != 0.0


def test_coin_named_in_params_but_absent_from_panel_is_ignored() -> None:
    """`params.coins` naming a coin the panel has no column for must not
    raise -- it simply never trades (see module docstring)."""
    panel = _panel(
        prices_a=[100.0, 101.0, 102.0],
        prices_b=[50.0, 49.5, 49.0],
    )
    strategy = LiveTrendTSMOMEnsemble()
    params = _small_params(coins=["A", "B", "ZZZ"])
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)
    assert list(weights.columns) == COLUMNS  # no phantom "ZZZ" column


def test_book_vol_scale_never_sees_bar_ts_own_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial guard: `book_vol_scale`'s equity argument, when deciding
    bar `t`, must never include the return that completes AT `t` -- only up
    to `t-1`'s, at the latest (`qlab.strategies.live.trend` module
    docstring).

    Construction: coin "A" oscillates gently for several bars, then has one
    huge move (100 -> 1000, +900%) landing exactly at row 5. That move is
    the realised return for the interval (row4, row5], earned by the weight
    already decided at row4 -- it becomes KNOWABLE the moment row5's price
    is read, which is also the moment this driver is asked to decide row5's
    OWN weight. A look-ahead bug would fold it into the equity curve before
    that decision; the correct driver folds it in only once, starting at
    row6's decision.

    We intercept every call to the real `book_vol_scale` (keeping its real
    behaviour -- this is a spy, not a stub) and check the equity argument's
    own deviation from the flat baseline at each bar. Before the shock is
    knowable (bars 0..5) it must stay small; from the bar after the shock
    lands (bar 6 onward) it must be large -- this failed, and only this
    test failed, when the driver's equity-curve update was deliberately
    moved before the `book_vol_scale`/`target_weights` calls (see module
    docstring) instead of after, then was confirmed to pass again once
    reverted.
    """
    prices_a = [100.0, 101.0, 100.0, 101.0, 100.0, 1000.0, 1000.0, 1000.0]
    prices_b = [50.0, 50.2, 50.0, 50.3, 50.1, 50.2, 50.0, 50.3]  # calm, unrelated coin
    panel = _panel(prices_a=prices_a, prices_b=prices_b)

    captured: list[list[float]] = []
    original = live_trend._signals.book_vol_scale

    def spy(daily_equity, params):  # noqa: ANN001 - test spy, mirrors frab's own signature
        captured.append(list(daily_equity))
        return original(daily_equity, params)

    monkeypatch.setattr(live_trend._signals, "book_vol_scale", spy)
    strategy = LiveTrendTSMOMEnsemble()
    strategy.target_weights(panel, _small_params())

    assert len(captured) == len(prices_a)

    # Structural check: the equity history handed to book_vol_scale for bar
    # i has exactly `max(i, 1)` entries -- it grows by exactly one entry per
    # bar, and the entry for bar i's OWN return is never present yet.
    for i, equity in enumerate(captured):
        assert len(equity) == max(i, 1), f"bar {i}: unexpected equity history length {len(equity)}"

    def max_deviation(equity: list[float]) -> float:
        return max(abs(e - 1.0) for e in equity)

    # Bars 0..5: the shock (row4 -> row5) is not yet reflected -- deviation
    # from the flat baseline stays small (the calm oscillation of A and B
    # before the shock is a few percent at most).
    for i in range(0, 6):
        assert max_deviation(captured[i]) < 0.5, (
            f"bar {i}: equity history already shows a large move "
            f"({captured[i]!r}) -- book_vol_scale saw bar {i}'s own return early"
        )

    # Bars 6, 7: the shock is now in the past and must be visible.
    for i in range(6, len(captured)):
        assert max_deviation(captured[i]) > 1.0, (
            f"bar {i}: equity history does not show the shock "
            f"({captured[i]!r}) -- book_vol_scale should see it by now"
        )
