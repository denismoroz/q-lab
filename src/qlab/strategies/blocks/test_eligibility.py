"""Tests for `eligible_coins` and the fewest coins a portfolio rule needs
(`PORTFOLIO_MIN_COINS`), the questions the second stage asks of a market before a
run (docs/TWO_STAGE.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks import Composed
from qlab.strategies.blocks.composed import eligible_coins, rebalance_rows
from qlab.strategies.blocks.portfolio import PORTFOLIO_MIN_COINS, PORTFOLIOS, min_coins

COLUMNS = list("ABCDEF")


def _panel(n: int = 120, seed: int = 0) -> MarketPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0, 0.02, (n, len(COLUMNS))), axis=0),
        index=index, columns=COLUMNS,
    )
    prices.iloc[:15, 5] = np.nan  # the last coin is listed late
    volume = pd.DataFrame(rng.uniform(1e3, 1e5, (n, len(COLUMNS))), index=index, columns=COLUMNS)
    funding = pd.DataFrame(rng.normal(1e-4, 1e-4, (n, len(COLUMNS))), index=index,
                           columns=COLUMNS)
    return MarketPanel(snapshot_id="walk", prices=prices, funding=funding,
                       tradeable=prices.notna(), meta={}, volume=volume.where(prices.notna()))


def _params(**overrides: object) -> dict:
    params: dict = {
        "universe": {"rule": "all"},
        "signal": {"block": "momentum", "lookback_days": 5},
        "portfolio": {"block": "long_short", "top_n": 2},
        "sizing": {"block": "equal", "gross": 1.0},
        "rebalance_days": 7,
        "min_history_days": 10,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize("portfolio", [
    {"block": "long_short", "top_n": 2},
    {"block": "long_only", "top_n": 2},
    {"block": "long_short_quantile", "quantiles": 3},
    {"block": "long_short_fraction", "fraction": 0.3},
    {"block": "time_series_sign"},
])
def test_eligible_coins_is_the_mask_the_weights_are_built_on(portfolio: dict) -> None:
    panel = _panel()
    params = _params(portfolio=portfolio)
    eligible = eligible_coins(panel, params)
    weights = Composed().target_weights(panel, params)
    assert list(eligible.columns) == COLUMNS and eligible.index.equals(panel.prices.index)
    # No weight on a coin the mask does not admit, on a rebalance bar (between rebalances a
    # weight is carried, so only those rows are compared).
    rows = rebalance_rows(panel.prices.index, params["rebalance_days"])
    assert not (weights.loc[rows, COLUMNS].abs().gt(0) & ~eligible.loc[rows]).any().any()
    # Nothing is admitted before a coin has its history; the late-listed coin waits for its own.
    assert not eligible.iloc[:9].any().any()
    assert not eligible["F"].iloc[:24].any()


def test_eligible_coins_follows_the_universe_and_the_history_gate() -> None:
    panel = _panel()
    narrow = eligible_coins(panel, _params(universe={"rule": "top_volume", "n": 2,
                                                       "window_days": 5}))
    assert narrow.sum(axis=1).max() <= 2
    gate = eligible_coins(panel, _params(min_history_days=60))
    assert not gate.iloc[:59].any().any() and gate.iloc[60:].any().any()


def test_eligible_coins_rejects_a_bad_spec_like_the_strategy_does() -> None:
    with pytest.raises(ValueError, match="unknown"):
        eligible_coins(_panel(), _params(portfolio={"block": "nope"}))


def test_every_portfolio_block_declares_the_fewest_coins_it_needs() -> None:
    assert set(PORTFOLIO_MIN_COINS) == set(PORTFOLIOS)


@pytest.mark.parametrize("portfolio,fewest", [
    ({"block": "long_short", "top_n": 2}, 4),
    ({"block": "long_only", "top_n": 3}, 3),
    ({"block": "long_short_quantile", "quantiles": 5}, 5),
    ({"block": "long_only_quantile", "quantiles": 4}, 4),
    ({"block": "long_short_fraction", "fraction": 0.3}, 4),  # floor(4*0.3)=1, floor(3*0.3)=0
    ({"block": "long_only_fraction", "fraction": 0.25}, 4),
    ({"block": "time_series_sign"}, 1),
])
def test_fewest_coins_is_where_the_block_first_holds_its_defined_book(
        portfolio: dict, fewest: int) -> None:
    assert min_coins(portfolio) == fewest
    block = PORTFOLIOS[portfolio["block"]]
    params = {k: v for k, v in portfolio.items() if k != "block"}

    def sides(n: int) -> tuple[int, int]:
        scores = pd.DataFrame([np.arange(1.0, n + 1)], columns=list("ABCDEFGHIJ")[:n])
        weights = block.fn(scores, scores.notna(), **params).iloc[0]
        return int((weights > 0).sum()), int((weights < 0).sum())

    name = portfolio["block"]
    if name in ("long_short", "long_only"):
        top_n = portfolio["top_n"]
        assert sides(fewest)[0] == top_n  # the book has the size the parameters name ...
        assert sides(fewest - 1)[0] < top_n  # ... and one coin fewer shrinks it
    elif name == "time_series_sign":
        assert sum(sides(fewest)) >= 1
    else:
        assert sides(fewest)[0] >= 1  # a group of at least one coin
        assert sides(fewest - 1) == (0, 0)  # one coin fewer and there is no position
