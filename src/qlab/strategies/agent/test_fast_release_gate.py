"""The fast-release block with a coin list that code chooses (`universe`): the K coins with the
largest traded value each day, the leaver sold and unhedged, the skip for a missing spot counted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.strategies.agent.fast_release_gate import FastReleaseGate
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import LiveBv2

FOUR, FIRST, HOURS = kit.FOUR, kit.FIRST_REBALANCE, kit.HOURS
FIRST_DAY = 792  # the first midnight with the startup history (788 bars) behind it
GATE = {"fast_window_hours": 12, "fast_entry": 0}


def _params(**kw) -> dict:
    universe = {"rebalance": "1d", "by": "perp_dollar_volume", **kw.pop("universe", {})}
    return {**GATE, **kit.universe_params(universe=universe, **kw)}


def _held(run, bar: int) -> list[str]:
    return [c for c in FOUR if c in run.in_universe and run.in_universe[c].iloc[bar]]


def test_the_k_coins_with_the_largest_traded_value_are_held_each_day() -> None:
    panel = kit.market(coins=FOUR, dollar_volume={"D": kit.spike_from(1200)})
    run = FastReleaseGate().run(panel, _params())
    assert _held(run, FIRST_DAY - 1) == []
    assert _held(run, FIRST_DAY) == ["A", "B"]
    # D's volume rose at bar 1200; its 3-day mean passes B's only at the rebalance of bar 1248
    # (a rank reads the bars before the rebalance and never the bar itself).
    assert _held(run, 1247) == ["A", "B"]
    assert _held(run, 1248) == ["A", "D"]
    assert run.counts["universe_join"] == 3


def test_a_coin_that_leaves_the_first_k_is_sold_and_unhedged_at_the_next_rebalance() -> None:
    panel = kit.market(coins=FOUR, dollar_volume={"D": kit.spike_from(1200)})
    run = FastReleaseGate().run(panel, _params())
    leave = next(e for e in run.log if e["kind"] == "leave")
    assert leave["coin"] == "B"
    bar = panel.prices.index.get_loc(leave["time"])
    assert bar == 1248 and leave["time"].hour == 0, "the daily rebalance is at the day's first bar"
    assert run.weights[["B", "B-SPOT"]].iloc[bar:].abs().to_numpy().sum() == 0.0
    assert run.weights["B-SPOT"].iloc[bar - 1] != 0.0, "B was held the bar before"
    # the leaver's money goes to the joiner, up to a slot's money (300 / 2), who starts at once
    joiner = [e for e in run.log if e["kind"] == "join"][-1]
    assert joiner["coin"] == "D"
    assert joiner["capital"] == pytest.approx(min(leave["proceeds"], 150.0))


def test_a_joiner_never_works_with_more_than_a_slot_s_money() -> None:
    """bv3-research-topk-coins, 2026-11-04: a K = 4 joiner got up to 254 of a 64 slot, the
    leaver's profit and set-aside cash put back to work although the book declares it does not
    reinvest (docs/research/bv3-topk-2026-11-04.md). The gain stays cash."""
    base = kit.market(coins=FOUR)
    paths = {c: base.prices[c].to_numpy() for c in FOUR}
    paths["B"] = 100.0 * np.cumprod(np.full(HOURS, 1.0007))
    panel = kit.market(coins=FOUR, dollar_volume={"D": kit.spike_from(1200)}, paths=paths)
    run = FastReleaseGate().run(panel, _params())
    leave = next(e for e in run.log if e["kind"] == "leave")
    joiner = [e for e in run.log if e["kind"] == "join"][-1]
    assert leave["coin"] == "B" and leave["proceeds"] > 160.0
    assert joiner["coin"] == "D" and joiner["capital"] == 150.0


def test_the_money_is_split_equally_among_the_k_coins_at_the_start() -> None:
    panel = kit.market(coins=FOUR)
    run = FastReleaseGate().run(panel, _params(capital=300.0, universe={"n": 3}))
    joins = [e for e in run.log if e["kind"] == "join"][:3]
    assert [e["capital"] for e in joins] == [100.0, 100.0, 100.0]


def test_a_top_coin_without_a_spot_leg_is_skipped_and_counted() -> None:
    # C has by far the largest perp turnover but no spot leg for the whole window.
    panel = kit.market(coins=FOUR, dollar_volume={"C": 9000.0}, spot_from={"C": HOURS + 1})
    run = FastReleaseGate().run(panel, _params())
    assert "C" not in {e["coin"] for e in run.log if e["kind"] == "join"}
    assert run.weights[["C", "C-SPOT"]].abs().to_numpy().sum() == 0.0
    assert _held(run, 2000) == ["A", "B"]
    assert run.counts["universe_top_without_spot"] > 0


def test_nothing_is_counted_as_skipped_when_every_coin_has_a_spot_leg() -> None:
    run = FastReleaseGate().run(kit.market(coins=FOUR), _params())
    assert run.counts["universe_top_without_spot"] == 0


def test_the_block_off_with_a_universe_is_the_plain_universe_book() -> None:
    panel = kit.market(coins=FOUR, dollar_volume={"D": kit.spike_from(1200)})
    off = FastReleaseGate().run(panel, _params(fast_window_hours=0))
    plain = LiveBv2().run(panel, kit.universe_params(
        universe={"rebalance": "1d", "by": "perp_dollar_volume"}))
    pd.testing.assert_frame_equal(off.weights, plain.weights, check_exact=True)


def test_a_universe_does_not_run_together_with_coins_or_hedge_by() -> None:
    panel = kit.market(coins=FOUR)
    with pytest.raises(ValueError, match="both name the coins"):
        FastReleaseGate().run(panel, {**_params(), "coins": list(FOUR)})
    with pytest.raises(ValueError, match="cannot run together"):
        FastReleaseGate().run(panel, {**_params(), "hedge_by": {"meta": "x"}})
