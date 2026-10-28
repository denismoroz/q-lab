"""Bv2's base: the book's equity over the capital the strategy started with
(docs/CAPITAL_BASE.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.metrics import capital_metrics, compute_metrics
from qlab.harness.run import run_backtest
from qlab.harness.strategy import CapitalBaseRefused, capital_setup, own_base, profit_reinvested
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import LiveBv2


def _rising_market():
    """Prices that double: the ratchet sells growth above 50% over target and sets it aside."""
    hours = kit.HOURS
    path = 100.0 * np.exp(np.linspace(0.0, np.log(2.5), hours))
    return kit.market(coins=("A", "B"), paths={"A": path, "B": path}, funding=0.00001)


def test_the_base_is_the_book_equity_over_the_capital() -> None:
    panel = _rising_market()
    params = kit.params(coins=("A", "B"), capital=300.0)
    base = LiveBv2().own_base(panel, params)
    run = LiveBv2().run(panel, params)
    assert base.index.equals(panel.prices.index)
    assert base.to_numpy() == pytest.approx((run.book_equity / 300.0).to_numpy())
    assert base.iloc[0] == pytest.approx(1.0, abs=0.01), "the book starts with the capital"
    assert base.iloc[-1] > 1.1, "the book kept the growth, the positions did not"
    assert own_base(LiveBv2(), panel, params).equals(base)


def test_the_base_is_the_capital_of_the_spec_not_a_fixed_number() -> None:
    panel = _rising_market()
    small = LiveBv2().own_base(panel, kit.params(coins=("A", "B"), capital=150.0))
    large = LiveBv2().own_base(panel, kit.params(coins=("A", "B"), capital=600.0))
    # the same relative book, so the same base; with another capital the dollars scale with it
    assert small.iloc[0] == pytest.approx(1.0, abs=0.01)
    assert large.iloc[0] == pytest.approx(1.0, abs=0.01)


def test_the_capital_return_is_the_stand_s_share_times_the_base() -> None:
    panel = _rising_market()
    params = kit.params(coins=("A", "B"), capital=300.0)
    strategy = LiveBv2()
    weights = strategy.target_weights(panel, params)
    setup = capital_setup(strategy, panel, params)
    assert setup.reinvested is False and setup.refused is None
    costs = CostModel(taker_fee_bps=4.5, slippage_bps=0.0)
    result = run_backtest(panel, weights, costs, panel.funding, **setup.kwargs)
    expected = result.net_return * setup.base.reindex(result.net_return.index)
    assert result.capital_return.to_numpy() == pytest.approx(expected.to_numpy())
    stand, capital = compute_metrics(result), capital_metrics(result)
    assert capital["cap_reinvested"] == 0.0
    # the same dollars over a smaller number: the capital's figure is the larger one
    assert capital["cap_total_return"] > stand["total_return_net"]


def test_bv2_declares_that_it_does_not_reinvest_its_profit() -> None:
    assert LiveBv2.reinvests_profit is False
    assert profit_reinvested(LiveBv2(), {}) is False


def test_the_base_is_refused_for_a_book_with_deposits() -> None:
    panel = _rising_market()
    params = kit.params(coins=("A", "B")) | {"deposit_usd": 30.0, "deposit_every": "7d"}
    with pytest.raises(ValueError, match="deposit"):
        LiveBv2().own_base(panel, params)
    with pytest.raises(CapitalBaseRefused):
        own_base(LiveBv2(), panel, params)
    setup = capital_setup(LiveBv2(), panel, params)
    assert setup.base is None and setup.refused and "deposit" in setup.refused
    assert setup.reinvested is False
    # the book itself still runs: only the base is refused
    assert LiveBv2().run(panel, params).counts["deposit"] > 0


def test_a_second_asker_gets_the_first_run_not_a_second_pass_of_the_driver() -> None:
    panel = _rising_market()
    params = kit.params(coins=("A", "B"))
    first = LiveBv2().run(panel, params)
    assert LiveBv2().run(panel, params) is first
    assert LiveBv2().run(panel, kit.params(coins=("A", "B"), capital=301.0)) is not first
    weights = LiveBv2().target_weights(panel, params)
    weights.iloc[:, :] = 0.0  # a caller may do what it likes with the weights it was handed
    assert LiveBv2().run(panel, params).weights.abs().sum().sum() > 0
    assert isinstance(first.book_equity, pd.Series)
