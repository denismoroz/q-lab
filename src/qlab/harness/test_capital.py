"""The strategy's own money as the base of a percent (docs/CAPITAL_BASE.md): the `cap_` summaries
beside the stand's, the `capital_return` series of a run, and the strategy-side hooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.costs import CostModel
from qlab.harness.metrics import (
    as_stand_keys,
    cap_key,
    capital_metrics,
    compute_metrics,
    stand_key_of,
    year_breakdown,
)
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest
from qlab.harness.strategy import (
    CapitalBaseRefused,
    capital_setup,
    own_base,
    profit_reinvested,
)


def _index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC")


def _result(net: pd.Series, *, capital: pd.Series | None = None, reinvested: bool = True
            ) -> RunResult:
    zero = pd.Series(0.0, index=net.index)
    return RunResult("s", net, net, zero, zero, zero, capital_return=capital,
                     capital_reinvested=reinvested)


def _panel(n: int = 6) -> MarketPanel:
    idx = _index(n)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 100.0, 103.0, 104.0][:n]}, index=idx)
    return MarketPanel(
        snapshot_id="p", prices=prices, funding=pd.DataFrame(0.0, index=idx, columns=["A"]),
        tradeable=pd.DataFrame(True, index=idx, columns=["A"]), meta={})


# ---- the RunResult field and run_backtest --------------------------------------------------


def test_run_result_capital_return_defaults_to_the_stand_s_net_return() -> None:
    net = pd.Series([0.01, -0.02, 0.03], index=_index(3), name="net_return")
    result = _result(net)
    assert result.capital_return.tolist() == net.tolist()
    assert result.capital_reinvested is True


def test_run_backtest_multiplies_the_stand_s_share_by_the_base() -> None:
    panel = _panel()
    weights = pd.DataFrame({"A": 0.5}, index=panel.prices.index)
    costs = CostModel(taker_fee_bps=0.0, slippage_bps=0.0)
    plain = run_backtest(panel, weights, costs, panel.funding)
    assert plain.capital_return.tolist() == plain.net_return.tolist()

    base = pd.Series(np.linspace(1.0, 3.0, len(panel.prices)), index=panel.prices.index)
    based = run_backtest(panel, weights, costs, panel.funding, capital_base=base,
                         capital_reinvested=False)
    expected = plain.net_return * base.reindex(plain.net_return.index)
    assert based.net_return.tolist() == plain.net_return.tolist(), "the stand's share is unchanged"
    assert based.capital_return.tolist() == pytest.approx(expected.tolist())
    assert based.capital_reinvested is False


def test_a_base_with_holes_takes_the_nearest_known_bar() -> None:
    panel = _panel()
    weights = pd.DataFrame({"A": 1.0}, index=panel.prices.index)
    costs = CostModel(taker_fee_bps=0.0, slippage_bps=0.0)
    base = pd.Series([np.nan, 2.0, np.nan, np.nan, 4.0, np.nan], index=panel.prices.index)
    result = run_backtest(panel, weights, costs, panel.funding, capital_base=base)
    ratio = (result.capital_return / result.net_return).dropna()
    assert ratio.tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0, 4.0])


# ---- sum versus compound -------------------------------------------------------------------


def test_summed_and_compounded_totals_differ_exactly_by_the_compounding() -> None:
    net = pd.Series([0.10, 0.10, 0.10, -0.05], index=_index(4))
    summed = capital_metrics(_result(net, reinvested=False))
    compounded = capital_metrics(_result(net, reinvested=True))
    assert summed["cap_total_return"] == pytest.approx(0.25)
    assert compounded["cap_total_return"] == pytest.approx(1.1 * 1.1 * 1.1 * 0.95 - 1.0)
    assert summed["cap_reinvested"] == 0.0 and compounded["cap_reinvested"] == 1.0
    ppy = 365.25
    assert summed["cap_ann_return"] == pytest.approx(0.25 * ppy / 4)
    assert compounded["cap_ann_return"] == pytest.approx((1.1 * 1.1 * 1.1 * 0.95) ** (ppy / 4) - 1)


def test_the_summed_fall_is_in_points_of_the_capital_from_the_running_peak() -> None:
    net = pd.Series([0.10, -0.05, -0.08, 0.20], index=_index(4))
    out = capital_metrics(_result(net, reinvested=False))
    # running profit 0.10, 0.05, -0.03, 0.17: the deepest fall is 0.10 -> -0.03
    assert out["cap_max_dd"] == pytest.approx(-0.13)
    # a series that starts by losing falls from the start (the curve begins at 0)
    out = capital_metrics(_result(pd.Series([-0.02, -0.03, 0.01], index=_index(3)),
                                  reinvested=False))
    assert out["cap_max_dd"] == pytest.approx(-0.05)


def test_the_best_one_percent_is_dropped_from_a_sum_and_the_rest_summed() -> None:
    rng = np.random.default_rng(1)
    net = pd.Series(rng.normal(0.0004, 0.004, 400), index=_index(400))
    net.iloc[10] = 0.5
    out = capital_metrics(_result(net, reinvested=False))
    keep = net.drop(net.nlargest(4).index)  # round(400 * 0.01) = 4
    assert out["cap_ann_return_ex_best_1pct"] == pytest.approx(keep.sum() * 365.25 / len(keep))


def test_years_on_the_capital_are_sums_with_the_stands_key_names_behind_the_cap_prefix() -> None:
    idx = _index(1000, "2022-01-01")
    net = pd.Series(0.0005, index=idx)
    net.loc["2023-06-01":"2023-09-01"] = -0.001
    out = capital_metrics(_result(net, reinvested=False))
    y2022 = net[net.index.year == 2022].sum()
    assert out["cap_year_2022_return"] == pytest.approx(y2022)
    assert out["cap_year_2022_days"] == 365
    assert out["cap_years_counted"] == 3
    best = max((net[net.index.year == y].sum(), y) for y in (2022, 2023, 2024))[1]
    rest = net[net.index.year != best]
    assert out["cap_ann_return_ex_best_year"] == pytest.approx(rest.sum() * 365.25 / len(rest))
    assert out["cap_best_year_share_of_profit"] == pytest.approx(
        net[net.index.year == best].sum() / net.sum())
    assert as_stand_keys(out)["year_2022_return"] == out["cap_year_2022_return"]


def test_year_breakdown_additive_flag_changes_only_the_arithmetic() -> None:
    net = pd.Series(0.001, index=_index(730, "2022-01-01"))
    compounded = year_breakdown(net)
    summed = year_breakdown(net, additive=True)
    assert set(compounded) == set(summed)
    assert summed["year_2022_return"] == pytest.approx(0.365)
    assert compounded["year_2022_return"] > summed["year_2022_return"]
    assert summed["year_2022_max_dd"] == 0.0


# ---- invariance on a base of 1 -------------------------------------------------------------


def test_on_a_base_of_one_and_a_reinvesting_book_every_cap_figure_is_the_stand_s() -> None:
    rng = np.random.default_rng(7)
    net = pd.Series(rng.normal(0.0003, 0.01, 900), index=_index(900, "2022-01-01"))
    stand = compute_metrics(_result(net))
    stand.update(year_breakdown(net))
    capital = capital_metrics(_result(net))
    pairs = 0
    for key, value in capital.items():
        twin = stand_key_of(key)
        if twin is None:
            continue
        assert stand[twin] == pytest.approx(value, nan_ok=True), key
        pairs += 1
    assert pairs > 15
    assert cap_key("ann_return_net") == "cap_ann_return"


# ---- the main test: money set aside --------------------------------------------------------


def test_a_book_that_sets_money_aside_and_one_that_does_not_agree_on_the_capital() -> None:
    """The same dollars every day. Book A lets its balance grow with the cash it sets aside, so
    its shares (dollars over the balance) shrink; book B keeps its balance at the capital. Today's
    numbers differ by the balance; on the capital they are the same number."""
    n, capital = 500, 100.0
    idx = _index(n, "2024-01-01")
    dollars = pd.Series(np.tile([0.30, -0.10, 0.20, 0.05, -0.15], n // 5), index=idx)
    balance_a = capital + dollars.cumsum().shift(1).fillna(0.0)  # grows with the profit
    base_a = balance_a / capital
    net_a = dollars / balance_a  # the stand's share of the book that kept the cash
    net_b = dollars / capital  # a book whose balance stays the capital
    result_a = _result(net_a, capital=net_a * base_a, reinvested=False)
    result_b = _result(net_b, reinvested=False)

    stand_a, stand_b = compute_metrics(result_a), compute_metrics(result_b)
    assert stand_a["ann_return_net"] != pytest.approx(stand_b["ann_return_net"], rel=0.05)

    cap_a, cap_b = capital_metrics(result_a), capital_metrics(result_b)
    for key in ("cap_ann_return", "cap_total_return", "cap_sharpe", "cap_max_dd",
                "cap_ann_return_ex_best_1pct"):
        assert cap_a[key] == pytest.approx(cap_b[key], rel=1e-9), key
    assert cap_a["cap_total_return"] == pytest.approx(dollars.sum() / capital)


# ---- the hooks -----------------------------------------------------------------------------


class _Plain:
    name = "plain"

    def target_weights(self, panel, params):
        return panel.prices * 0.0


class _Sums:
    name = "sums"
    reinvests_profit = False

    def target_weights(self, panel, params):
        return panel.prices * 0.0

    def own_base(self, panel, params):
        return pd.Series(2.0, index=panel.prices.index)


class _Refuses(_Sums):
    def own_base(self, panel, params):
        raise CapitalBaseRefused("deposits")


def test_an_undeclared_strategy_is_reinvesting_with_no_base_as_the_stand_always_was() -> None:
    panel = _panel()
    assert own_base(_Plain(), panel, {}) is None
    assert profit_reinvested(_Plain(), {}) is True
    setup = capital_setup(_Plain(), panel, {})
    assert setup.base is None and setup.reinvested is True and setup.refused is None
    assert setup.kwargs == {"capital_base": None, "capital_reinvested": True}


def test_a_declared_strategy_gives_its_base_and_its_choice() -> None:
    panel = _panel()
    setup = capital_setup(_Sums(), panel, {})
    assert setup.reinvested is False and setup.refused is None
    assert (setup.base == 2.0).all()


def test_a_refused_base_is_reported_not_swallowed_into_the_stand_s_numbers() -> None:
    setup = capital_setup(_Refuses(), _panel(), {})
    assert setup.base is None and setup.refused == "deposits"
    assert issubclass(CapitalBaseRefused, ValueError)
