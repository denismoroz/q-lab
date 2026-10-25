"""New money paid into the books (`params.deposit_usd`, `params.deposit_every`): a deposit changes
the strategy's capital, never its return (CLAUDE.md: returns are on the strategy's capital)."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.run import run_backtest
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import LiveBv2


def _twr(run) -> float:
    return float((1.0 + run.book_return.dropna()).prod() - 1.0)


def _flat_market(coins=("A", "B")):
    h = kit.HOURS
    return kit.market(coins=coins, paths={c: np.full(h, 100.0) for c in coins}, funding=0.0)


def test_a_deposit_is_capital_not_return() -> None:
    """Prices never move and funding is zero: the only thing that can make money is a deposit,
    and the return must be nothing but the fees paid to invest it."""
    panel = _flat_market()
    base = kit.params(coins=("A", "B"))
    plain = LiveBv2().run(panel, base)
    paid = LiveBv2().run(panel, {**base, "deposit_usd": 100.0, "deposit_every": "7d"})
    n = paid.counts["deposit"]
    assert n == 14 and paid.deposits.sum() == pytest.approx(100.0 * n)
    assert paid.book_equity.iloc[-1] == pytest.approx(300.0 + 100.0 * n, rel=0.01)
    naive = paid.book_equity.iloc[-1] / paid.book_equity.iloc[0] - 1.0
    assert naive > 4.0, "the equity curve alone would call the deposits a +400% return"
    assert abs(_twr(paid)) < 5e-3, "the time-weighted return sees only the fees"
    assert abs(_twr(plain)) < 5e-3
    assert (paid.book_return.dropna().abs() < 2e-3).all(), "no hour books a deposit as a gain"


def test_the_harness_return_is_time_weighted_whatever_is_paid_in() -> None:
    panel = kit.market(coins=("A", "B"))
    base = kit.params(coins=("A", "B"))
    plain = LiveBv2().run(panel, base)
    paid = LiveBv2().run(panel, {**base, "deposit_usd": 30.0, "deposit_every": "7d"})
    costs = CostModel(taker_fee_bps=4.5, slippage_bps=0.0)
    totals = [float((1.0 + run_backtest(panel, r.weights, costs, panel.funding).net_return).prod()
                    - 1.0) for r in (plain, paid)]
    assert paid.book_equity.iloc[-1] > 2.0 * plain.book_equity.iloc[-1]
    assert totals[1] == pytest.approx(totals[0], abs=0.01), "420 dollars in did not become return"
    assert _twr(paid) == pytest.approx(_twr(plain), abs=0.01)


def test_a_deposit_into_a_hedged_book_extends_the_hedge_with_the_spot() -> None:
    panel = _flat_market()
    base = kit.params(coins=("A", "B"))
    paid = LiveBv2().run(panel, {**base, "deposit_usd": 200.0, "deposit_every": "7d"})
    later = [e for e in paid.log if e["kind"] == "deposit" and e.get("spot_bought_usd", 0) > 0
             and e["time"] > paid.weights.index[kit.HOURS // 2]]
    assert later and all(e["hedged_units"] > 0 for e in later)
    last = paid.weights.iloc[-1]
    for coin in ("A", "B"):
        assert last[coin] == pytest.approx(-last[f"{coin}-SPOT"], rel=0.02), "still delta neutral"


def test_a_deposit_too_small_to_trade_waits_as_cash_until_it_is_not() -> None:
    panel = _flat_market(coins=("A",))
    paid = LiveBv2().run(panel, kit.params(coins=("A",), capital=150.0, deposit_usd=6.0,
                                           deposit_every="7d"))
    buys = [e["spot_bought_usd"] for e in paid.log
            if e["kind"] == "deposit" and "spot_bought_usd" in e]
    assert buys[0] == 0.0 and any(b >= 10.0 for b in buys), "below the minimum order, then enough"


def test_money_paid_in_before_a_book_starts_goes_into_its_capital() -> None:
    panel = _flat_market(coins=("A", "B"))
    base = kit.params(coins=("A", "B"), capital=300.0)
    paid = LiveBv2().run(panel, {**base, "deposit_usd": 40.0, "deposit_every": "3d"})
    early = [e for e in paid.log if e["kind"] == "deposit" and e["time"] < paid.weights.index[788]]
    assert early and all(e["to"] == "waiting" for e in early)
    started = paid.book_equity.iloc[790]
    assert started == pytest.approx(300.0 + paid.deposits.iloc[:790].sum(), rel=0.002)


def test_in_a_changing_universe_money_paid_in_before_the_first_selection_waits_as_idle() -> None:
    panel = kit.market(coins=kit.FOUR)
    paid = LiveBv2().run(panel, kit.universe_params(capital=300.0, deposit_usd=20.0,
                                                    deposit_every="7d"))
    before = paid.deposits.iloc[:kit.FIRST_REBALANCE].sum()
    assert before == 20.0 * 4 and all(e["to"] == "idle" for e in paid.log
                                      if e["kind"] == "deposit" and e["time"]
                                      < paid.weights.index[kit.FIRST_REBALANCE])
    joins = [e["capital"] for e in paid.log if e["kind"] == "join"][:2]
    assert joins == [pytest.approx((300.0 + before) / 2)] * 2


def test_deposits_ride_on_a_shared_pool_too() -> None:
    if not kit.HAS_POOL:
        pytest.skip("no shared pool in this frab checkout")
    panel = _flat_market()
    run = LiveBv2().run(panel, {**kit.params(coins=("A", "B"), shared_margin=True),
                                "deposit_usd": 100.0, "deposit_every": "7d"})
    assert abs(_twr(run)) < 5e-3 and run.book_equity.iloc[-1] > 1000.0
