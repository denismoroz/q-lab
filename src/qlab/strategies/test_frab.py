"""Tests for `qlab.strategies.frab.FrabFundingHarvest`.

Fixtures are small synthetic daily panels -- no network, no real market
data. `signal_window_hours=24` is chosen throughout so `periods_for` maps it
to exactly 1 daily bar, meaning the smoothed signal equals that bar's raw
funding rate with no warm-up delay -- this keeps the arithmetic hand-
checkable without changing anything about the strategy's own logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.frab import FrabFundingHarvest

COLUMNS = ["A-SPOT", "A-PERP", "B-SPOT", "B-PERP", "C-SPOT", "C-PERP", "D-SPOT", "D-PERP"]
COIN_COLUMNS = {
    "spot_columns": {"A": "A-SPOT", "B": "B-SPOT", "C": "C-SPOT", "D": "D-SPOT"},
    "perp_columns": {"A": "A-PERP", "B": "B-PERP", "C": "C-PERP", "D": "D-PERP"},
}


def _panel(funding: pd.DataFrame, n: int = 6) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.DataFrame(100.0, index=index, columns=COLUMNS)
    tradeable = pd.DataFrame(True, index=index, columns=COLUMNS)
    full_funding = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    full_funding.update(funding)
    return MarketPanel(
        snapshot_id="snap-frab", prices=prices, funding=full_funding, tradeable=tradeable, meta={}
    )


def _base_params(**overrides: object) -> dict:
    params = {
        **COIN_COLUMNS,
        "max_slots": 2,
        "entry_threshold_apr": 0.05,
        "exit_threshold_apr": -1.0,  # never triggers unless a test wants it to
        "signal_window_hours": 24,
        "min_leg_notional_usd": 12.0,
        "book_capital_usd": 1000.0,
    }
    params.update(overrides)
    return params


def test_frab_weights_pass_validate_weights() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    funding = pd.DataFrame(
        {"A-PERP": 0.001, "B-PERP": 0.0001, "C-PERP": 0.005, "D-PERP": 0.004},
        index=index,
    )
    panel = _panel(funding)
    strategy = FrabFundingHarvest()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # must not raise


def test_frab_never_exceeds_max_slots_and_ranks_by_funding() -> None:
    """3 coins clear entry_threshold_apr but max_slots=2 -- only the top 2 by
    annualised funding (C, D) get sized; A (qualifies but ranks 3rd) stays
    flat, and B (never clears the entry bar) stays flat too."""
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    funding = pd.DataFrame(
        {
            "A-PERP": 0.001,  # APR ~0.365 -- qualifies, but ranks 3rd
            "B-PERP": 0.0001,  # APR ~0.037 -- below entry_threshold_apr=0.05
            "C-PERP": 0.005,  # APR ~1.826 -- highest
            "D-PERP": 0.004,  # APR ~1.461 -- 2nd highest
        },
        index=index,
    )
    panel = _panel(funding)
    strategy = FrabFundingHarvest()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)

    share = 0.5  # 1 / max_slots
    for ts in index:
        row = weights.loc[ts]
        assert row["C-SPOT"] == pytest.approx(share)
        assert row["C-PERP"] == pytest.approx(-share)
        assert row["D-SPOT"] == pytest.approx(share)
        assert row["D-PERP"] == pytest.approx(-share)
        assert row["A-SPOT"] == 0.0 and row["A-PERP"] == 0.0
        assert row["B-SPOT"] == 0.0 and row["B-PERP"] == 0.0
        # never more than max_slots=2 non-zero spot legs
        held = sum(1 for c in ("A", "B", "C", "D") if row[f"{c}-SPOT"] != 0.0)
        assert held <= 2


def test_frab_drops_slot_instead_of_undersizing() -> None:
    """share = 1/max_slots is fixed and identical for every slot (the live
    engine's `compute_footprint`). If that fixed share can't clear the
    venue's minimum leg, EVERY slot must stay closed -- never opened at a
    smaller, unexecutable size."""
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    funding = pd.DataFrame({"A-PERP": 0.01, "B-PERP": 0.01, "C-PERP": 0.01}, index=index)
    panel = _panel(funding, n=4)
    strategy = FrabFundingHarvest()

    # share = 1/3 = 0.333; min_position_size = 12 / 20 = 0.6 > share.
    params = _base_params(max_slots=3, book_capital_usd=20.0)
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)
    assert (weights == 0.0).all().all()

    # Same funding, but a book large enough that the fixed share clears the
    # floor (1/3 * 1000 = 333.33 >> 12) -- slots open normally.
    params_ok = _base_params(max_slots=3, book_capital_usd=1000.0)
    weights_ok = strategy.target_weights(panel, params_ok)
    validate_weights(panel, weights_ok)
    assert (weights_ok != 0.0).any().any()


def test_frab_rotates_as_funding_changes() -> None:
    """A starts qualified and B doesn't; funding flips partway through so A
    drops (below exit_threshold_apr) and B takes the freed slot (above
    entry_threshold_apr) -- with max_slots=1, holdings must rotate rather
    than both being held or both being dropped."""
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    funding = pd.DataFrame(
        {
            "A-PERP": [0.01, 0.01, 0.01, -0.01, -0.01, -0.01],
            "B-PERP": [-0.001, -0.001, -0.001, 0.01, 0.01, 0.01],
        },
        index=index,
    )
    panel = _panel(funding)
    strategy = FrabFundingHarvest()
    params = _base_params(max_slots=1, entry_threshold_apr=0.05, exit_threshold_apr=0.0)
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)

    for ts in index[:3]:
        row = weights.loc[ts]
        assert row["A-SPOT"] == pytest.approx(1.0)
        assert row["A-PERP"] == pytest.approx(-1.0)
        assert row["B-SPOT"] == 0.0 and row["B-PERP"] == 0.0
    for ts in index[3:]:
        row = weights.loc[ts]
        assert row["B-SPOT"] == pytest.approx(1.0)
        assert row["B-PERP"] == pytest.approx(-1.0)
        assert row["A-SPOT"] == 0.0 and row["A-PERP"] == 0.0


def test_frab_drops_position_on_non_tradeable_leg() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    funding = pd.DataFrame({"A-PERP": 0.01}, index=index)
    panel = _panel(funding, n=4)
    tradeable = panel.tradeable.copy()
    tradeable.loc[index[2:], "A-PERP"] = False  # A's perp leg delists on day 2
    panel = MarketPanel(
        snapshot_id=panel.snapshot_id,
        prices=panel.prices,
        funding=panel.funding,
        tradeable=tradeable,
        meta={},
    )
    strategy = FrabFundingHarvest()
    weights = strategy.target_weights(panel, _base_params(max_slots=1))
    validate_weights(panel, weights)  # confirms no exposure while non-tradeable
    assert weights.loc[index[0], "A-SPOT"] == pytest.approx(1.0)
    assert weights.loc[index[2], "A-SPOT"] == 0.0
    assert weights.loc[index[3], "A-SPOT"] == 0.0


def test_frab_is_deterministic() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    funding = pd.DataFrame(
        {"A-PERP": 0.001, "B-PERP": 0.0001, "C-PERP": 0.005, "D-PERP": 0.004},
        index=index,
    )
    panel = _panel(funding)
    strategy = FrabFundingHarvest()
    params = _base_params()
    weights1 = strategy.target_weights(panel, params)
    weights2 = strategy.target_weights(panel, params)
    pd.testing.assert_frame_equal(weights1, weights2)
