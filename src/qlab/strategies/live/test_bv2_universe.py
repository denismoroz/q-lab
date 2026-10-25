"""A coin list that changes over time (`params.universe`, `qlab.strategies.live.bv2` docstring):
who is held at each rebalance, what happens to the money when a coin leaves and another joins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.lookahead import lookahead_violation
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest
from qlab.harness.strategy import validate_weights
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import HISTORY_BARS, WARMUP_BARS, LiveBv2

FOUR, FIRST, HOURS = kit.FOUR, kit.FIRST_REBALANCE, kit.HOURS
market, universe_params, spike_from = kit.market, kit.universe_params, kit.spike_from


def _held(run, bar: int) -> list[str]:
    return [c for c in FOUR if c in run.in_universe and run.in_universe[c].iloc[bar]]


def test_the_universe_holds_the_n_most_liquid_coins_chosen_from_data_before_the_rebalance() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": spike_from(1200)})
    run = LiveBv2().run(panel, universe_params())
    validate_weights(panel, run.weights)
    assert _held(run, FIRST - 1) == []
    assert _held(run, FIRST) == ["A", "B"]  # the two with the largest dollar volume
    assert _held(run, 1343) == ["A", "B"]
    # D's volume rose at bar 1200; the rebalance at 1176 could not know, the one at 1344 does.
    assert _held(run, 1344) == ["A", "D"]
    assert run.weights["C-SPOT"].abs().sum() == 0.0
    assert run.weights["B-SPOT"].iloc[1344:].abs().sum() == 0.0
    assert run.weights["D-SPOT"].iloc[:1344].abs().sum() == 0.0 < run.weights["D-SPOT"].iloc[1344]
    assert run.counts["universe_join"] == 3 and run.counts["universe_leave"] == 1


def test_volume_that_appears_at_the_rebalance_bar_is_not_known_at_it() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": spike_from(1344)})
    run = LiveBv2().run(panel, universe_params())
    assert _held(run, 1344) == ["A", "B"], "the rebalance at bar 1344 reads bars up to 1343"
    assert _held(run, 1512) == ["A", "D"]


def test_a_changing_universe_passes_the_look_ahead_guard() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": spike_from(1200)})
    params = universe_params()
    weights = LiveBv2().target_weights(panel, params)
    assert weights.abs().to_numpy().sum() > 0
    assert lookahead_violation(LiveBv2(), panel, params, weights) is None


def test_a_leaving_book_is_closed_at_frabs_costs_and_its_money_goes_to_the_joiner() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": spike_from(1200)})
    run = LiveBv2().run(panel, universe_params(capital=300.0))
    joins = [e for e in run.log if e["kind"] == "join"]
    leave = next(e for e in run.log if e["kind"] == "leave")
    assert [e["capital"] for e in joins[:2]] == [150.0, 150.0]
    assert leave["coin"] == "B" and leave["time"] == panel.prices.index[1344]
    fees = leave["equity_before"] - leave["proceeds"]
    # hedge bought back and spot sold: at most the spot and perp takers plus slippage on the book
    assert 0 < fees < 0.0025 * leave["equity_before"]
    assert joins[2]["coin"] == "D" and joins[2]["capital"] == pytest.approx(leave["proceeds"])
    # The stayer's book is not touched: it is not closed, and it joined once, at the start.
    assert [e["coin"] for e in run.log if e["kind"] == "leave"] == ["B"]
    assert [e["coin"] for e in joins].count("A") == 1


def test_before_the_first_rebalance_the_money_waits_as_cash() -> None:
    run = LiveBv2().run(market(coins=FOUR), universe_params(capital=300.0))
    assert (run.weights.iloc[:FIRST] == 0.0).all().all()
    assert (run.book_equity.iloc[:FIRST] == 300.0).all()


def test_a_slot_nobody_fills_stays_cash() -> None:
    # Only two coins have both legs: n = 3 leaves one slot empty, and its share stays idle.
    panel = market(coins=FOUR, spot_from={"C": HOURS + 1, "D": HOURS + 1})
    run = LiveBv2().run(panel, universe_params(universe={"n": 3}, capital=300.0))
    assert _held(run, 2000) == ["A", "B"]
    first = run.weights.iloc[FIRST]
    assert first["A-SPOT"] == pytest.approx(first["B-SPOT"], rel=0.05)
    assert first["A-SPOT"] + first["B-SPOT"] < 0.7, "a third of the money is in no coin"
    assert [e["capital"] for e in run.log if e["kind"] == "join"] == [100.0, 100.0]


def test_a_coin_without_a_perp_is_left_out_never_held_unhedged() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": 9000.0}, perp_from={"D": HOURS + 1})
    run = LiveBv2().run(panel, universe_params())
    assert "D" not in {e["coin"] for e in run.log if e["kind"] == "join"}
    assert run.weights[["D", "D-SPOT"]].abs().to_numpy().sum() == 0.0
    assert _held(run, 2000) == ["A", "B"]


def test_a_young_coin_waits_for_the_startup_history() -> None:
    listed = 1000
    panel = market(coins=FOUR, dollar_volume={"D": 9000.0}, listed_from={"D": listed})
    run = LiveBv2().run(panel, universe_params())
    joined = next(e["time"] for e in run.log if e["kind"] == "join" and e["coin"] == "D")
    ready = listed + HISTORY_BARS + WARMUP_BARS
    assert ready < panel.prices.index.get_loc(joined) <= ready + 168


def test_a_universe_that_never_changes_is_the_fixed_coin_list() -> None:
    """Three coins, n = 3, books startable only from the first rebalance: the same strategy as a
    fixed list of those three -- weights and equity equal to the last bit."""
    # priced from bar 52: the startup history (788 bars) is there at bar 840, not a bar before
    panel = market(coins=("A", "B", "C"), listed_from={c: FIRST - 788 for c in ("A", "B", "C")})
    fixed = LiveBv2().run(panel, kit.params(capital=300.0))
    moving = LiveBv2().run(panel, universe_params(coins=("A", "B", "C"), capital=300.0,
                                                  universe={"n": 3}))
    pd.testing.assert_frame_equal(moving.weights, fixed.weights, check_exact=True)
    pd.testing.assert_series_equal(moving.book_equity, fixed.book_equity, check_exact=True)
    assert moving.counts.get("universe_leave", 0) == 0


def test_candidates_panel_is_every_coin_with_both_legs() -> None:
    panel = market(coins=FOUR)
    named = LiveBv2().run(panel, universe_params())
    discovered = LiveBv2().run(panel, universe_params(universe={"candidates": "panel"}))
    pd.testing.assert_frame_equal(named.weights, discovered.weights, check_exact=True)


def test_the_volume_measure_is_the_one_the_spec_names() -> None:
    # The perp of C trades the most, the spot of A does.
    panel = market(coins=("A", "B", "C"), dollar_volume={"A": 1000.0, "B": 800.0, "C": 5000.0},
                   spot_volume={"A": 3000.0, "B": 800.0, "C": 600.0})
    coins = ("A", "B", "C")
    by_spot = LiveBv2().run(panel, universe_params(coins, universe={"n": 1}))
    by_perp = LiveBv2().run(
        panel, universe_params(coins, universe={"n": 1, "by": "perp_dollar_volume"}))
    assert _held(by_spot, 1000) == ["A"] and _held(by_perp, 1000) == ["C"]


def test_a_delisted_coin_is_closed_at_its_last_price_at_the_next_rebalance() -> None:
    gone = 1300
    panel = market(coins=FOUR)
    prices, tradeable = panel.prices.copy(), panel.tradeable.copy()
    prices.loc[prices.index[gone:], ["B", "B-SPOT"]] = np.nan
    tradeable.loc[tradeable.index[gone:], ["B", "B-SPOT"]] = False
    delisted = MarketPanel(snapshot_id="m", prices=prices, funding=panel.funding,
                           tradeable=tradeable, meta=panel.meta, volume=panel.volume,
                           high=panel.high)
    run = LiveBv2().run(delisted, universe_params())
    leave = next(e for e in run.log if e["kind"] == "leave")
    assert leave["coin"] == "B" and leave["time"] == panel.prices.index[1344]
    assert leave["price"] == pytest.approx(panel.prices["B"].iloc[gone - 1])
    assert _held(run, 1344) == ["A", "C"], "the next most liquid coin takes the slot"
    assert run.weights[["B", "B-SPOT"]].iloc[gone:].abs().to_numpy().sum() == 0.0


def test_the_harness_trades_only_the_leaver_and_the_joiner_at_a_swap() -> None:
    panel = market(coins=FOUR, dollar_volume={"D": spike_from(1200)})
    weights = LiveBv2().target_weights(panel, universe_params())
    result = run_backtest(panel, weights, CostModel(taker_fee_bps=4.5, slippage_bps=0.0),
                          panel.funding)
    assert result.turnover.iloc[1343:1346].sum() > 0
    # the stayer A trades nowhere near the swap, and the quiet days before it nothing happens
    assert weights["A-SPOT"].iloc[1342:1348].nunique() <= 2
