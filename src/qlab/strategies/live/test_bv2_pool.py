"""One margin pool for all coins and top-ups that arrive late, through the adapter
(`shared_margin`, `topup_delay_hours`; the rules themselves are frab's, tested there:
`frab/strategy/b2/tests/test_book_pool.py`).

Needs a frab checkout that has them (branch research/bv2-pool-topups); without one these tests
are skipped and `test_bv2_mechanisms.py` checks the adapter refuses the params instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.lookahead import lookahead_violation
from qlab.harness.strategy import validate_weights
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import LiveBv2

pytestmark = pytest.mark.skipif(
    not kit.HAS_POOL,
    reason="the frab checkout in use has no shared margin pool "
           "(QLAB_FRAB_SRC -> branch research/bv2-pool-topups)",
)

FALL_START, REBOUND = 24 * 40, 24 * 75


def _paths(mult: float, ramp: int) -> tuple[np.ndarray, np.ndarray]:
    """Two coins fall for 35 days after a rise (the hedge goes on); then A jumps `mult` times over
    `ramp` hours while B stays flat."""
    h = kit.HOURS
    calm = np.r_[np.linspace(100, 160, FALL_START), np.linspace(160, 100, REBOUND - FALL_START),
                 np.full(h - REBOUND, 100.0)]
    pumped = calm.copy()
    pumped[REBOUND + ramp:] *= mult
    pumped[REBOUND:REBOUND + ramp] *= np.linspace(1.0, mult, ramp)
    return pumped, calm


def _two(mult: float, ramp: int = 5, **kw):
    pumped, calm = _paths(mult, ramp)
    panel = kit.market(coins=("A", "B"), paths={"A": pumped, "B": calm})
    return panel, kit.params(coins=("A", "B"), **kw)


def _liquidated(run) -> list[tuple[str, pd.Timestamp]]:
    return [(e["coin"], e["time"]) for e in run.log if e["kind"] == "liquidation"]


def test_a_pool_of_one_coin_is_that_coins_own_book() -> None:
    pumped, _ = _paths(2.6, 24)
    panel = kit.market(coins=("A",), paths={"A": pumped})
    for delay in (0, 3):  # no liquidation: identical to the last bit
        alone = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=delay))
        pooled = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=delay,
                                                 shared_margin=True))
        pd.testing.assert_frame_equal(pooled.weights, alone.weights, check_exact=True)
        pd.testing.assert_series_equal(pooled.book_equity, alone.book_equity, check_exact=True)
        assert pooled.liquidations == alone.liquidations == 0
    # With a liquidation the pool prices it from the last close towards the high, the book from
    # its liquidation level: the same bar, a slightly different price.
    alone = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=6))
    pooled = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=6, shared_margin=True))
    assert _liquidated(pooled) == _liquidated(alone) != []
    assert pooled.book_equity.iloc[-1] == pytest.approx(alone.book_equity.iloc[-1], rel=0.02)


def test_the_other_coins_usdc_carries_a_short_that_would_be_liquidated_alone() -> None:
    panel, params = _two(2.8)
    apart = LiveBv2().run(panel, params)
    pooled = LiveBv2().run(panel, {**params, "shared_margin": True})
    assert [c for c, _ in _liquidated(apart)] == ["A"]
    assert _liquidated(pooled) == []


def test_a_liquidated_pool_closes_every_short_of_the_pool_the_innocent_one_included() -> None:
    panel, params = _two(8.0)
    apart = LiveBv2().run(panel, params)
    pooled = LiveBv2().run(panel, {**params, "shared_margin": True})
    assert [c for c, _ in _liquidated(apart)] == ["A"], "B's own book is not touched"
    hit = _liquidated(pooled)
    assert sorted(c for c, _ in hit) == ["A", "B"] and len({t for _, t in hit}) == 1
    # The pool lasted longer than A's own pool would have: the price it fell at is higher.
    price = {(e["coin"]): e["price"] for e in pooled.log if e["kind"] == "liquidation"}
    price_apart = next(e["price"] for e in apart.log if e["kind"] == "liquidation")
    assert price["A"] > price_apart
    assert price["B"] == pytest.approx(100.0, rel=0.01), "B's short is closed where B trades"


def test_a_top_up_that_arrives_late_loses_the_short_that_an_instant_one_saves() -> None:
    pumped, _ = _paths(2.6, 24)
    panel = kit.market(coins=("A",), paths={"A": pumped})
    for shared in (False, True):
        kw = {"shared_margin": shared}
        instant = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=0, **kw))
        late = LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=6, **kw))
        assert instant.liquidations == 0 and instant.counts["margin_rebalance"] == 1
        assert late.liquidations == 1 and late.counts.get("margin_rebalance", 0) == 0, (
            "liquidated inside the delay: the top-up that was on its way never happens")
        short = {d: LiveBv2().run(panel, kit.params(coins=("A",), topup_delay_hours=d, **kw))
                 for d in (1, 3)}
        assert instant.book_equity.iloc[-1] > short[1].book_equity.iloc[-1] \
            > short[3].book_equity.iloc[-1] > late.book_equity.iloc[-1]


def test_pool_and_delay_weights_are_valid_and_pass_the_look_ahead_guard() -> None:
    panel, params = _two(2.8)
    params = {**params, "shared_margin": True, "topup_delay_hours": 3}
    run = LiveBv2().run(panel, params)
    validate_weights(panel, run.weights)
    assert lookahead_violation(LiveBv2(), panel, params, run.weights) is None


def test_a_pool_works_with_a_changing_coin_list() -> None:
    four = kit.FOUR
    panel = kit.market(coins=four, dollar_volume={"D": kit.spike_from(1200)})
    params = kit.universe_params(shared_margin=True)
    run = LiveBv2().run(panel, params)
    validate_weights(panel, run.weights)
    assert run.counts["universe_leave"] == 1 and run.weights["D-SPOT"].iloc[1344] > 0
