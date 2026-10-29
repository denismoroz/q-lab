"""Bv2's orders: what the book sent to the venue, on the strategy's own money
(docs/MIN_CAPITAL.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.metrics import min_capital_usd
from qlab.harness.strategy import PlacedOrders, own_orders
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import B2Params, LiveBv2

needs_graded = pytest.mark.skipif("graded_hedge_depth" not in B2Params.__dataclass_fields__,
                                  reason="this frab checkout has no graded hedge")


def _wave_market():
    """The price swings slowly around its start, so the 14- and 30-day returns cross the hedge
    threshold again and again and a graded hedge keeps changing its size."""
    t = np.arange(kit.HOURS)
    path = 100.0 * np.exp(0.06 * np.sin(2 * np.pi * t / (24 * 45)) + 0.0002 * np.sin(t / 7))
    return kit.market(coins=("A", "B"), paths={"A": path, "B": path})


def _rising_market():
    path = 100.0 * np.exp(np.linspace(0.0, np.log(2.5), kit.HOURS))
    return kit.market(coins=("A", "B"), paths={"A": path, "B": path})


def test_the_orders_are_dollars_over_the_money_the_strategy_started_with() -> None:
    panel = _wave_market()
    params = kit.params(coins=("A", "B"), capital=300.0)
    orders = own_orders(LiveBv2(), panel, params)
    assert isinstance(orders, PlacedOrders)
    assert orders.sizes.index.equals(panel.prices.index)
    assert list(orders.sizes.columns) == list(panel.prices.columns)
    run = LiveBv2().run(panel, params)
    pd.testing.assert_frame_equal(orders.sizes * 300.0, run.order_usd)
    first = orders.sizes.iloc[(orders.sizes.sum(axis=1) > 0).to_numpy().argmax()]
    # each coin's first purchase is the same share of the money, and it is a share of 300
    assert first["A-SPOT"] == pytest.approx(first["B-SPOT"])
    assert 0.1 < first["A-SPOT"] < 0.5
    assert (orders.sizes >= 0).all().all()


def test_only_orders_that_open_a_leg_are_kept() -> None:
    panel = _wave_market()
    run = LiveBv2().run(panel, kit.params(coins=("A", "B"), capital=300.0))
    # a hedge short is on the perp's column, the spot purchase on the spot's
    opened = run.order_usd
    assert opened[["A-SPOT", "B-SPOT"]].gt(0).sum().sum() == 2  # the two first purchases
    assert opened[["A", "B"]].gt(0).sum().sum() == run.counts["hedge_open"]
    # closes, cuts and refills are not opening orders
    assert run.counts["hedge_close"] > 0
    assert opened.gt(0).sum().sum() == 2 + run.counts["hedge_open"]


def test_the_orders_do_not_depend_on_what_the_balance_grew_to() -> None:
    """The weights are shares of the book's whole balance, which holds the cash the ratchet set
    aside; an order is a number of dollars."""
    panel = _rising_market()
    params = kit.params(coins=("A", "B"), capital=300.0)
    strategy = LiveBv2()
    run = strategy.run(panel, params)
    assert run.book_equity.iloc[-1] > 1.1 * 300.0, "the balance outgrew the money at work"
    by_weights = min_capital_usd(strategy.target_weights(panel, params), 10.0)
    by_orders = min_capital_usd(strategy.placed_orders(panel, params).sizes, 10.0)
    assert by_weights > 1.1 * by_orders
    # the smallest order is a first purchase (the hedge is never opened in this market), so the
    # floor is the minimum over its share of the money
    first_purchase = strategy.placed_orders(panel, params).sizes["A-SPOT"].max()
    assert by_orders == pytest.approx(10.0 / first_purchase)


def test_the_floor_by_orders_is_the_same_at_another_capital() -> None:
    panel = _wave_market()
    floors = [min_capital_usd(LiveBv2().placed_orders(
        panel, kit.params(coins=("A", "B"), capital=capital)).sizes, 10.0)
        for capital in (300.0, 600.0)]
    assert floors[1] == pytest.approx(floors[0], rel=0.15)


@needs_graded
def test_a_graded_hedge_leaves_dust_in_the_weights_and_none_in_the_orders() -> None:
    """A cut can leave a tiny short that is held, not traded: the weights read it as a leg that
    must be bought and put the floor at thousands of dollars; the book never sent such an order
    and counts the ones it left unsent."""
    panel = _wave_market()
    params = kit.params(coins=("A", "B"), capital=300.0, graded_hedge_depth=0.1)
    strategy = LiveBv2()
    run = strategy.run(panel, params)
    orders = strategy.placed_orders(panel, params)
    by_weights = min_capital_usd(strategy.target_weights(panel, params), 10.0)
    by_orders = min_capital_usd(orders.sizes, 10.0)
    assert by_weights > 5.0 * by_orders
    # every order the book did send is at least the minimum, so the floor is the capital's own
    # order of magnitude, not a multiple of it
    sent = (orders.sizes * 300.0).to_numpy()
    assert sent[sent > 0].min() >= 10.0 - 1e-9
    assert by_orders <= 300.0 * 1.05
    assert orders.skipped.sum() == run.counts["order_unsent_open"] + run.counts[
        "order_unsent_resize"]
    assert orders.skipped.sum() > 0


def test_a_plain_hedge_never_leaves_an_order_unsent_at_a_sum_that_fits_it() -> None:
    panel = _wave_market()
    orders = LiveBv2().placed_orders(panel, kit.params(coins=("A", "B"), capital=300.0))
    assert orders.skipped.sum() == 0
    assert orders.skipped.index.equals(panel.prices.index)


def test_a_hedge_smaller_than_the_minimum_is_counted_as_unsent() -> None:
    """At $20 the spot of one coin is about $6: the hedge it wants is under $10, the book leaves
    it unsent and the hedge is never opened."""
    panel = _wave_market()
    params = kit.params(coins=("A", "B"), capital=20.0)
    run = LiveBv2().run(panel, params)
    assert run.counts.get("hedge_open", 0) == 0
    assert run.counts["order_unsent_open"] > 0
    assert run.skipped.sum() == run.counts["order_unsent_open"]
