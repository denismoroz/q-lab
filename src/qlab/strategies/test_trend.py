"""Tests for `qlab.strategies.trend.TrendTSMOMEnsemble`.

Fixtures are small synthetic daily panels -- no network, no real market
data. `lookbacks_days`/`vol_window_days`/`min_history_days` use small
test-only values (not the production 30/60/90/120-day config, which needs
months of warm-up history) so the ensemble/vol arithmetic can be hand-
checked; the strategy code has no hardcoded defaults, so this only changes
which numbers are plugged in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.trend import TrendTSMOMEnsemble

COLUMNS = ["S", "T", "U"]


def _panel(tradeable_u_row3: bool = False) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    # S and U are identical price paths (both lookback signs +1 at row 3).
    # T shares S's LAST move (so the 1-day signal agrees) but disagrees on
    # the 2-day signal -- its ensemble average is exactly 0 at row 3.
    prices = pd.DataFrame(
        {
            "S": [100.0, 90.0, 95.0, 100.0],
            "T": [100.0, 110.0, 95.0, 100.0],
            "U": [100.0, 90.0, 95.0, 100.0],
        },
        index=index,
    )
    tradeable = pd.DataFrame(True, index=index, columns=COLUMNS)
    tradeable.loc[index[3], "U"] = tradeable_u_row3
    funding = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    return MarketPanel(
        snapshot_id="snap-trend", prices=prices, funding=funding, tradeable=tradeable, meta={}
    )


def _base_params(**overrides: object) -> dict:
    params = {
        "lookbacks_days": [1, 2],
        "vol_window_days": 2,
        "vol_target_daily": 0.02,
        "leverage_cap": 3.0,
        "risk_scale": 1.0,
        "min_history_days": 3,
    }
    params.update(overrides)
    return params


def test_trend_weights_pass_validate_weights() -> None:
    panel = _panel()
    strategy = TrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # must not raise


def test_trend_ensemble_averages_disagreeing_lookbacks_to_zero() -> None:
    """At row 3: S's 1-day AND 2-day returns are both positive (ensemble
    sign = mean(+1, +1) = +1); T's 1-day return is positive but its 2-day
    return is negative (ensemble = mean(+1, -1) = 0). A zero ensemble must
    produce EXACTLY zero weight, regardless of realised vol -- this is only
    true if lookbacks are actually averaged, not e.g. OR'd or summed."""
    panel = _panel(tradeable_u_row3=True)
    strategy = TrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)

    row3 = panel.prices.index[3]
    assert weights.loc[row3, "T"] == 0.0
    assert weights.loc[row3, "S"] != 0.0
    assert weights.loc[row3, "S"] > 0.0  # unanimous positive momentum -> long


def test_trend_no_position_when_not_tradeable() -> None:
    """U has the IDENTICAL price path to S (same signal, same vol), but is
    marked non-tradeable at row 3 -- it must get zero weight there even
    though its signal alone would size it exactly like S."""
    panel = _panel(tradeable_u_row3=False)
    strategy = TrendTSMOMEnsemble()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # would raise if U carried exposure while non-tradeable

    row3 = panel.prices.index[3]
    assert weights.loc[row3, "U"] == 0.0
    assert weights.loc[row3, "S"] != 0.0


def test_trend_leverage_cap_bounds_gross_exposure() -> None:
    """With only S contributing nonzero raw exposure at row 3 (T's ensemble
    cancels to 0, U is non-tradeable), the leverage cap must bring S's
    weight down to EXACTLY leverage_cap * risk_scale (a single-instrument
    book saturates the cap one-for-one)."""
    panel = _panel(tradeable_u_row3=False)
    strategy = TrendTSMOMEnsemble()
    params = _base_params(leverage_cap=3.0, risk_scale=1.0)
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)

    row3 = panel.prices.index[3]
    assert weights.loc[row3, "S"] == pytest.approx(3.0)

    gross = weights.abs().sum(axis=1)
    assert (gross <= 3.0 + 1e-9).all()


def test_trend_is_deterministic() -> None:
    panel = _panel()
    strategy = TrendTSMOMEnsemble()
    params = _base_params()
    weights1 = strategy.target_weights(panel, params)
    weights2 = strategy.target_weights(panel, params)
    pd.testing.assert_frame_equal(weights1, weights2)


def _gbm_panels(seed: int = 3) -> tuple[MarketPanel, MarketPanel]:
    """The same random walk seen on 1h bars and on 1d bars (each daily close
    is the last hourly close of that day)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-01", periods=24 * 400, freq="1h", tz="UTC")
    cols = ["X", "Y", "Z"]
    hourly_ret = rng.normal(0.0, 0.004, size=(len(hours), len(cols)))
    hourly = (1.0 + pd.DataFrame(hourly_ret, index=hours, columns=cols)).cumprod() * 100.0
    daily = hourly.resample("1D").last()

    def panel(prices: pd.DataFrame) -> MarketPanel:
        return MarketPanel(
            snapshot_id="gbm",
            prices=prices,
            funding=pd.DataFrame(0.0, index=prices.index, columns=cols),
            tradeable=pd.DataFrame(True, index=prices.index, columns=cols),
            meta={"universe_complete": True},
        )

    return panel(hourly), panel(daily)


def test_position_size_does_not_depend_on_bar_size() -> None:
    """T25: the same params on 1h and 1d bars must size positions alike. The
    target is a DAILY volatility; before `vol` was put in daily units, hourly
    positions came out about sqrt(24) ~ 4.9 times larger."""
    hourly, daily = _gbm_panels()
    params = {
        "lookbacks_days": [30, 60],
        "vol_window_days": 30,
        "vol_target_daily": 0.02,
        "leverage_cap": 1e9,  # never binds: the test is about raw sizing
        "risk_scale": 1.0,
        "min_history_days": 61,
    }
    w_hourly = TrendTSMOMEnsemble().target_weights(hourly, params)
    w_daily = TrendTSMOMEnsemble().target_weights(daily, params)

    size_hourly = w_hourly.abs().sum(axis=1)
    size_hourly = size_hourly[size_hourly > 0].median()
    size_daily = w_daily.abs().sum(axis=1)
    size_daily = size_daily[size_daily > 0].median()
    assert size_hourly / size_daily == pytest.approx(1.0, abs=0.15)


def test_trend_declares_the_intervals_it_was_checked_on() -> None:
    assert TrendTSMOMEnsemble.valid_intervals == ("1d", "4h", "1h")
