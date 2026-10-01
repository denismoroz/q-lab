"""Tests for `qlab.strategies.bab` -- known-answer checks on synthetic data,
plus the adversarial look-ahead test every strategy in this project owes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.bab import BettingAgainstBeta, dimson_beta, formation_rows, rank_weights

PARAMS = {
    "market": "MKT",
    "window_bars": 365,
    "min_observations": 200,
    "dimson_lags": 5,
    "shrink_weight": 0.5,
    "shrink_target": 1.0,
}
TRUE_BETAS = {"A": 0.3, "B": 0.6, "C": 0.9, "D": 1.2, "E": 1.5, "F": 1.8}


def _panel(n_days: int = 500, seed: int = 7) -> MarketPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n_days, freq="1D", tz="UTC")
    market = rng.normal(0.0005, 0.01, n_days)
    rets = {"MKT": market}
    for name, beta in TRUE_BETAS.items():
        rets[name] = beta * market + rng.normal(0.0, 0.004, n_days)
    prices = (1.0 + pd.DataFrame(rets, index=index)).cumprod() * 100.0
    return MarketPanel(
        snapshot_id="synthetic",
        prices=prices,
        funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
        tradeable=pd.DataFrame(True, index=index, columns=prices.columns),
        meta={"universe_complete": True},
    )


def test_dimson_beta_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(1)
    market = rng.normal(0.0, 0.01, 400)
    asset = 1.4 * market + rng.normal(0.0, 0.002, 400)
    assert dimson_beta(asset, market, lags=5) == pytest.approx(1.4, abs=0.05)


def test_dimson_beta_picks_up_a_lagged_response() -> None:
    """The reason for the lags: an asset that reacts one bar late has its
    beta in the lag-1 slope, which a same-bar regression would miss."""
    rng = np.random.default_rng(2)
    market = rng.normal(0.0, 0.01, 400)
    asset = np.empty_like(market)
    asset[0] = 0.0
    asset[1:] = 1.0 * market[:-1]
    asset += rng.normal(0.0, 0.001, 400)
    assert dimson_beta(asset, market, lags=0) == pytest.approx(0.0, abs=0.1)
    assert dimson_beta(asset, market, lags=5) == pytest.approx(1.0, abs=0.05)


def test_rank_weights_each_side_sums_to_one_and_favours_extremes() -> None:
    betas = pd.Series({"a": 0.2, "b": 0.5, "c": 0.9, "d": 1.3, "e": 2.0})
    low, high = rank_weights(betas)
    assert low.sum() == pytest.approx(1.0)
    assert high.sum() == pytest.approx(1.0)
    assert low["a"] > low["b"] > 0 and low["d"] == 0 and low["e"] == 0
    assert high["e"] > high["d"] > 0 and high["a"] == 0 and high["b"] == 0
    assert low["c"] == 0 and high["c"] == 0  # the median rank sits on neither side


def test_formation_rows_are_month_ends() -> None:
    index = pd.date_range("2025-01-30", "2025-03-02", freq="1D", tz="UTC")
    rows = formation_rows(index)
    assert [index[i].date().isoformat() for i in rows] == [
        "2025-01-31",
        "2025-02-28",
        "2025-03-02",
    ]


def test_book_is_beta_neutral_at_formation_and_longs_low_beta() -> None:
    """Equation 17: each side is rescaled to beta one using the betas the
    strategy itself estimated, so on those betas the book is exactly
    neutral. The estimates must also track the true betas."""
    panel = _panel()
    weights = BettingAgainstBeta().target_weights(panel, PARAMS)
    t = formation_rows(panel.prices.index)[-1]
    held = weights.iloc[t]
    assert held["MKT"] == 0.0
    assert held["A"] > 0 and held["F"] < 0

    returns = panel.prices.pct_change(fill_method=None)
    lo = t - PARAMS["window_bars"] + 1 - PARAMS["dimson_lags"]
    market = returns["MKT"].iloc[lo : t + 1].to_numpy()
    estimated = {}
    for name, true_beta in TRUE_BETAS.items():
        raw = dimson_beta(returns[name].iloc[lo : t + 1].to_numpy(), market, PARAMS["dimson_lags"])
        estimated[name] = 0.5 * raw + 0.5
        # Six summed slopes (lags 0..5), each with standard error ~0.02 here
        # (noise 0.004 / (market vol 0.01 * sqrt(365))), so the sum's is
        # ~0.05; three standard errors.
        assert raw == pytest.approx(true_beta, abs=0.15)

    long_beta = sum(held[n] * b for n, b in estimated.items() if held[n] > 0)
    short_beta = sum(held[n] * b for n, b in estimated.items() if held[n] < 0)
    assert long_beta == pytest.approx(1.0)
    assert short_beta == pytest.approx(-1.0)


def test_no_position_before_enough_history() -> None:
    panel = _panel()
    weights = BettingAgainstBeta().target_weights(panel, PARAMS)
    first_month_ends = formation_rows(panel.prices.index)[:6]  # Jan..Jun, < 200 obs
    for row in first_month_ends:
        assert (weights.iloc[row] == 0.0).all()


def test_weights_never_see_the_future() -> None:
    """Scramble every price AFTER a formation row; every weight up to and
    including that row must be unchanged. A strategy that read ahead -- a
    centred window, a full-sample beta, a forward fill from the end -- fails
    here. Re-broken deliberately while writing this test (beta estimated on
    `returns.iloc[lo:]` instead of `returns.iloc[lo:t+1]`): it failed."""
    panel = _panel()
    cut = formation_rows(panel.prices.index)[-3]
    original = BettingAgainstBeta().target_weights(panel, PARAMS)

    rng = np.random.default_rng(99)
    future = panel.prices.copy()
    future.iloc[cut + 1 :] *= rng.uniform(0.5, 1.5, size=future.iloc[cut + 1 :].shape)
    tampered = MarketPanel(
        snapshot_id="tampered",
        prices=future,
        funding=panel.funding,
        tradeable=panel.tradeable,
        meta=panel.meta,
    )
    scrambled = BettingAgainstBeta().target_weights(tampered, PARAMS)

    pd.testing.assert_frame_equal(original.iloc[: cut + 1], scrambled.iloc[: cut + 1])
    assert not original.iloc[cut + 1 :].equals(scrambled.iloc[cut + 1 :])


def test_instrument_that_stops_trading_drops_to_zero_mid_month() -> None:
    panel = _panel()
    tradeable = panel.tradeable.copy()
    rows = formation_rows(panel.prices.index)
    stop = rows[-2] + 5
    tradeable.iloc[stop:, tradeable.columns.get_loc("A")] = False
    halted = MarketPanel(
        snapshot_id="halted",
        prices=panel.prices,
        funding=panel.funding,
        tradeable=tradeable,
        meta=panel.meta,
    )
    weights = BettingAgainstBeta().target_weights(halted, PARAMS)
    assert weights["A"].iloc[stop - 1] > 0
    assert (weights["A"].iloc[stop:] == 0.0).all()


def test_every_param_is_required() -> None:
    panel = _panel()
    for key in PARAMS:
        partial = {k: v for k, v in PARAMS.items() if k != key}
        with pytest.raises(ValueError, match=key):
            BettingAgainstBeta().target_weights(panel, partial)
