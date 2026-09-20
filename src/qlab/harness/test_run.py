"""Adversarial tests for run_backtest.

Covers: hand-computed turnover/cost, hand-computed accrual on a flat-price
book, the accrual-required contract (omitted / NO_ACCRUAL / NaN-while-held),
and a demonstration that a structurally look-ahead strategy is NOT caught by
validate_weights but IS visible as an implausible Sharpe -- see the last
test's docstring for why that limitation is deliberate and documented rather
than "fixed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.accrual import NO_ACCRUAL, AccrualError
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest


def _panel(
    prices: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    tradeable: pd.DataFrame | None = None,
) -> MarketPanel:
    if funding is None:
        funding = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if tradeable is None:
        tradeable = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    return MarketPanel(
        snapshot_id="snap-run", prices=prices, funding=funding, tradeable=tradeable, meta={}
    )


# --- turnover / cost: hand-computed two-instrument example ------------------


def test_turnover_and_cost_match_hand_computation() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    cols = ["A", "B"]
    prices = pd.DataFrame(100.0, index=idx, columns=cols)  # flat: isolate cost/turnover

    weights = pd.DataFrame(
        [
            [1.0, -1.0],   # t0: open from flat -> turnover 2
            [-1.0, 1.0],   # t1: full flip -> turnover 4
            [-1.0, 1.0],   # t2: unchanged -> turnover 0
            [-1.0, 1.0],   # t3: dropped from output (no realised fwd return)
        ],
        index=idx,
        columns=cols,
    )
    panel = _panel(prices)
    costs = CostModel(taker_fee_bps=3.0, slippage_bps=2.0)  # total 5 bps

    result = run_backtest(panel, weights, costs, NO_ACCRUAL)

    assert list(result.turnover.round(10)) == [2.0, 4.0, 0.0]
    assert list(result.cost.round(10)) == [0.001, 0.002, 0.0]
    # Flat prices -> gross is 0 everywhere; net == -cost.
    assert list(result.net_return.round(10)) == [-0.001, -0.002, 0.0]
    assert list(result.net_return.index) == list(idx[:-1])


# --- accrual: constant weight on flat prices, known funding rate -----------


def test_constant_weight_flat_price_accrual_matches_hand_computation() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.DataFrame(100.0, index=idx, columns=["X"])
    rate = 0.0003
    funding = pd.DataFrame(rate, index=idx, columns=["X"])
    weights = pd.DataFrame(1.0, index=idx, columns=["X"])  # always long 1 unit
    panel = _panel(prices, funding=funding)
    costs = CostModel(taker_fee_bps=5.0, slippage_bps=0.0)  # total 5 bps

    result = run_backtest(panel, weights, costs, funding)

    # accrual is EXACTLY `rate` every output period: weight is constant 1.0
    # and funding is constant `rate`, so held * funding_fwd == rate always.
    assert list(result.accrual.round(10)) == [rate, rate, rate]
    # turnover: only period 0 opens the position from flat (turnover=1);
    # cost0 = 1 * 5/1e4 = 0.0005, cost1 = cost2 = 0 (unchanged weight).
    assert list(result.turnover.round(10)) == [1.0, 0.0, 0.0]
    assert list(result.cost.round(10)) == [0.0005, 0.0, 0.0]
    # gross is 0 (flat prices); net = gross + accrual - cost.
    expected_net = [rate - 0.0005, rate, rate]
    assert list(result.net_return.round(10)) == [round(v, 10) for v in expected_net]


# --- accrual contract: omitted / NO_ACCRUAL / NaN-while-held ---------------


def test_omitting_accrual_raises_type_error() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    prices = pd.DataFrame(100.0, index=idx, columns=["X"])
    weights = pd.DataFrame(1.0, index=idx, columns=["X"])
    panel = _panel(prices)
    costs = CostModel(taker_fee_bps=1.0, slippage_bps=1.0)
    with pytest.raises(TypeError, match="required"):
        run_backtest(panel, weights, costs, None)


def test_no_accrual_sentinel_runs_as_pure_gross_minus_cost() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    prices = pd.DataFrame(100.0, index=idx, columns=["X"])
    weights = pd.DataFrame(1.0, index=idx, columns=["X"])
    panel = _panel(prices)
    costs = CostModel(taker_fee_bps=1.0, slippage_bps=1.0)
    result = run_backtest(panel, weights, costs, NO_ACCRUAL)
    assert (result.accrual == 0.0).all()
    assert (result.net_return == result.gross_return - result.cost).all()


def test_holding_through_nan_funding_raises() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    prices = pd.DataFrame(100.0, index=idx, columns=["X"])
    weights = pd.DataFrame(1.0, index=idx, columns=["X"])  # held throughout
    # funding at label idx[1] is NaN -> funding_fwd at idx[0] (= funding[idx[1]])
    # is NaN, and weights[idx[0]] is non-zero -> must raise.
    funding = pd.DataFrame({"X": [0.001, np.nan, 0.001]}, index=idx)
    panel = _panel(prices, funding=funding)
    costs = CostModel(taker_fee_bps=1.0, slippage_bps=1.0)
    with pytest.raises(AccrualError, match="NaN"):
        run_backtest(panel, weights, costs, funding)


# --- look-ahead: structurally uncatchable, but visible as an absurd Sharpe -


class _LookAheadCheat:
    """Illegitimate strategy: decides weights.loc[t] from the return REALISED
    at t -> t+1, i.e. reads panel.prices "in the future" relative to t. This
    is exactly the defect class qlab.harness.strategy.Strategy's alignment
    rule exists to prevent."""

    name = "look_ahead_cheat"

    def target_weights(self, panel: MarketPanel, params: dict) -> pd.DataFrame:
        future_return = panel.prices.pct_change().shift(-1)  # LOOK-AHEAD
        return np.sign(future_return).fillna(0.0)


class _HonestMomentum:
    """Legitimate comparison: decides weights.loc[t] from the return realised
    UP TO t (pct_change()'s own convention), never from t+1 onward."""

    name = "honest_momentum"

    def target_weights(self, panel: MarketPanel, params: dict) -> pd.DataFrame:
        past_return = panel.prices.pct_change().fillna(0.0)
        return np.sign(past_return)


def test_look_ahead_strategy_passes_validate_weights_but_shows_absurd_sharpe() -> None:
    """validate_weights cannot detect look-ahead (see Strategy's docstring):
    it only inspects the SHAPE of the final weights frame, which is
    identical whether target_weights peeked at the future or not. What IS
    visible, and is demonstrated here, is that a look-ahead strategy's
    Sharpe on ordinary noisy data is wildly higher than an honest strategy's
    -- a reviewer's smell test, not a harness guarantee. Catching look-ahead
    for real is the Adversary's job (docs/PLAN.md), not this module's.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-01-01", periods=250, freq="D", tz="UTC")
    daily_returns = rng.normal(loc=0.0, scale=0.01, size=250)
    prices = pd.DataFrame({"X": 100.0 * np.cumprod(1.0 + daily_returns)}, index=idx)
    panel = _panel(prices)
    costs = CostModel(taker_fee_bps=0.0, slippage_bps=0.0)  # isolate the foresight effect

    cheat_weights = _LookAheadCheat().target_weights(panel, {})
    honest_weights = _HonestMomentum().target_weights(panel, {})

    # Both pass the structural check -- look-ahead is invisible to it.
    from qlab.harness.strategy import validate_weights

    validate_weights(panel, cheat_weights)
    validate_weights(panel, honest_weights)

    cheat_result = run_backtest(panel, cheat_weights, costs, NO_ACCRUAL)
    honest_result = run_backtest(panel, honest_weights, costs, NO_ACCRUAL)

    cheat_sharpe = compute_metrics(cheat_result)["sharpe_net"]
    honest_sharpe = compute_metrics(honest_result)["sharpe_net"]

    # Perfect foresight on noisy data produces a Sharpe no honest strategy
    # sustains; an honest trailing-momentum strategy on pure noise does not.
    assert cheat_sharpe > 10.0
    assert cheat_sharpe > honest_sharpe + 5.0
