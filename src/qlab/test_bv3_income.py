"""Bv3 income: the decomposition adds up to the stand's result; the carry, weakest-coin and
rebalancing pieces do what their docstrings say."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import bv3_income as bi
from qlab.harness.costs import CostModel
from qlab.harness.run import run_backtest
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import B2Params, LiveBv2
from qlab.test_hedge_study import COINS, _market


def test_the_parts_add_up_to_the_book_s_result_on_the_strategy_s_money() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    run = LiveBv2().run(panel, params)
    capital = params["capital_usd"]
    base = run.book_equity / capital
    res = run_backtest(panel, run.weights, CostModel(7.5, 0.9), panel.funding,
                       capital_base=base, capital_reinvested=False)
    p = B2Params.from_dict(params)
    targets = {c: p.book_sizes(c)[0] for c in COINS}
    parts = bi.decompose(panel, run.weights, run.book_equity, capital, res.cost, targets)
    assert np.allclose(parts.sum(axis=1).to_numpy(), res.capital_return.to_numpy(), atol=1e-12)
    assert (parts["fees"] <= 0).all()
    # under a full hedge the spot's move and the hedge's price roughly cancel
    s = bi.summarise_parts(parts, panel.prices.index[len(panel.prices) // 2])
    assert s["first"]["total"] == pytest.approx(sum(v for k, v in s["first"].items()
                                                    if k != "total"))


def test_the_reserve_carry_is_recorded_and_the_run_is_the_live_book_s() -> None:
    from qlab import hedge_study as hs

    panel = _market()
    params = kit.params(coins=COINS, carry_enabled=True, carry_fraction=0.6,
                        carry_entry_apr=0.05, carry_exit_hours=24)
    rec = hs.run_book(panel, params)
    pd.testing.assert_frame_equal(rec.run.weights, LiveBv2().run(panel, params).weights)
    for c in COINS:
        carry = rec.parts["carry_usd"][c]
        assert (carry > 0).mean() > 0.5  # funding 8.8% a year > 5%: the sleeve is mostly on
        assert rec.parts["carry_funding"][c].dropna().iloc[-1] > 0
        assert (rec.parts["liq_dist"][c].dropna() > 0).all()  # never past liquidation


def test_carry_wish_adds_a_hedge_while_funding_pays() -> None:
    base = np.array([0, 0, 0, 0, 1, 0, np.nan, 0], dtype=float)
    enter = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    stay = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    out = bi.carry_wish(base, enter, stay)
    assert out[:6].tolist() == [0, 1, 1, 0, 1, 0]
    assert np.isnan(out[6])


def test_the_weakest_coins_are_hedged_among_the_day_s_members() -> None:
    days = pd.date_range("2024-01-01", periods=2, freq="D")
    r = pd.DataFrame({"A": [0.1, -0.2], "B": [-0.1, 0.3], "C": [0.0, 0.1], "D": [-0.5, 0.0]},
                     index=days)
    members = pd.DataFrame(True, index=days, columns=r.columns)
    members.loc[days[0], "D"] = False
    out = bi.weakest_wish(r, members, 0.5)
    assert out.loc[days[0]].tolist() == [False, True, False, False]  # 1 of 3 members
    assert out.loc[days[1]].tolist() == [True, False, False, True]


def test_a_rebalanced_basket_sells_the_winner_back_to_its_slot() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    prices = pd.DataFrame({"A": [100, 100, 150, 150, 150, 150.0],
                           "B": [100, 100, 100, 100, 100, 100.0]}, index=idx)
    funding = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    members = pd.DataFrame(True, index=idx, columns=prices.columns)
    r = bi.rebalanced_book(prices, funding, members, 10.0, 100.0, band=0.2, weekly=False,
                           tail=None, spot_cost=0.0, perp_cost=0.0)
    assert r.sum() == pytest.approx(0.05)  # A's 10 -> 15 earned once, then sold back to 10
    keep = bi.rebalanced_book(prices, funding, members, 10.0, 100.0, band=None, weekly=False,
                              tail=None, spot_cost=0.0, perp_cost=0.0)
    assert keep.sum() == pytest.approx(0.05)
    costly = bi.rebalanced_book(prices, funding, members, 10.0, 100.0, band=0.2, weekly=False,
                                tail=None, spot_cost=0.01, perp_cost=0.0)
    assert costly.sum() == pytest.approx(0.05 - (20 + 5) * 0.01 / 100)
