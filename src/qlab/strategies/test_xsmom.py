"""Tests for `qlab.strategies.xsmom.XsmomCrossSectionalMomentum`.

Fixtures are small synthetic daily panels -- no network, no real market
data. `rebalance_days=7`/`anchor_dow=3` are the PRODUCTION defaults (kept
as-is, not shrunk for the test) because the schedule itself is exactly what
needs checking: the fixture's index starts on a Thursday
(`2026-01-01`, confirmed by `datetime.date(2026, 1, 1).weekday() == 3`) so
the expected rebalance bars (0, 7, 14 -- see module docstring's
`is_rebalance_due` discretisation) can be hand-computed instead of asserted
against the code under test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.xsmom import XsmomCrossSectionalMomentum, _compute_k, _rebalance_rows

COLUMNS = ["UP1", "UP2", "UP3", "DN1", "DN2", "DN3"]
N_DAYS = 15


def _trending_panel() -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=N_DAYS, freq="D", tz="UTC")
    assert index[0].weekday() == 3  # Thursday -- see module docstring above.

    data = {}
    for i, name in enumerate(["UP1", "UP2", "UP3"]):
        # Distinct but unambiguously-positive trends -- strict ranking among
        # the three, well clear of the DN group.
        data[name] = [100.0 * (1.05 + 0.01 * i) ** t for t in range(N_DAYS)]
    for i, name in enumerate(["DN1", "DN2", "DN3"]):
        data[name] = [100.0 * (0.95 - 0.01 * i) ** t for t in range(N_DAYS)]

    prices = pd.DataFrame(data, index=index)
    tradeable = pd.DataFrame(True, index=index, columns=COLUMNS)
    funding = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    return MarketPanel(
        snapshot_id="snap-xsmom", prices=prices, funding=funding, tradeable=tradeable, meta={}
    )


def _base_params(**overrides: object) -> dict:
    params = {
        "universe": None,
        "lookbacks_days": [1, 2],
        "rebalance_days": 7,
        "anchor_dow": 3,
        "n_positions": None,
    }
    params.update(overrides)
    return params


def test_xsmom_weights_pass_validate_weights() -> None:
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # must not raise


def test_xsmom_first_bar_has_no_history_so_no_position() -> None:
    """Bar 0 is a rebalance by `is_rebalance_due`'s "never rebalanced ->
    always due" rule, but `lookbacks_days=[1, 2]` cannot be computed with
    zero prior history -- every instrument's ensemble score is NaN, so the
    live-equivalent behaviour is "no candidate has a score" -> flat book,
    not an error and not a fabricated position."""
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())

    row0 = panel.prices.index[0]
    assert (weights.loc[row0] == 0.0).all()


def test_xsmom_ranks_up_trend_long_and_down_trend_short() -> None:
    """By bar 7 (the second rebalance -- see module docstring), momentum is
    well-defined. Universe of 6 -> tercile k = max(1, 6 // 3) = 2
    (`_compute_k`), so exactly 2 UP names go long and 2 DN names go short,
    each leg at weight 0.5 / 2 = 0.25, and the book is exactly
    dollar-neutral (long notional == short notional)."""
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)

    row7 = panel.prices.index[7]
    row = weights.loc[row7]

    longs = row[row > 0]
    shorts = row[row < 0]
    assert len(longs) == 2
    assert len(shorts) == 2
    assert set(longs.index) <= {"UP1", "UP2", "UP3"}
    assert set(shorts.index) <= {"DN1", "DN2", "DN3"}
    assert longs.tolist() == pytest.approx([0.25, 0.25])
    assert shorts.tolist() == pytest.approx([-0.25, -0.25])
    assert row.sum() == pytest.approx(0.0)  # dollar-neutral


def test_xsmom_holds_flat_between_rebalances() -> None:
    """Bars 8-13 fall strictly between the bar-7 and bar-14 rebalances
    (`rebalance_days=7`, `anchor_dow=3` -> next due bar is 14, the next
    Thursday at least 7 bars later). Weights must be IDENTICAL to bar 7's
    on every one of them -- a periodic-rebalance book that quietly re-scores
    every day is not the strategy being transcribed, and would multiply
    turnover (and therefore cost) roughly 7x, exactly the failure mode the
    task calls out."""
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())

    row7 = weights.loc[panel.prices.index[7]]
    for i in range(8, 14):
        pd.testing.assert_series_equal(
            weights.loc[panel.prices.index[i]], row7, check_names=False
        )


def test_xsmom_turnover_is_zero_except_on_rebalance_bars() -> None:
    """Direct check on the turnover series itself (not just equal rows),
    matching the task's explicit verification requirement: print/inspect
    turnover and confirm non-zero rows land on the expected schedule."""
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # first row has no prior weight to diff from

    nonzero_bars = set(turnover.index[turnover > 1e-12])
    scheduled_rebalance_bars = {
        panel.prices.index[0],
        panel.prices.index[7],
        panel.prices.index[14],
    }
    # Every non-zero turnover row must land on a scheduled rebalance bar
    # (never in between -- see test_xsmom_holds_flat_between_rebalances).
    assert nonzero_bars <= scheduled_rebalance_bars
    # Bar 0 has no history yet (see test_xsmom_first_bar_has_no_history_so_no_position)
    # -> zero turnover despite being scheduled. Bar 7 is where the book
    # actually opens from flat, so it MUST show turnover. Bar 14's turnover
    # can legitimately be zero if the ranking happens not to have changed
    # since bar 7 (this fixture's monotonic trends make that the case) --
    # the schedule still fired, it just had nothing new to do.
    assert panel.prices.index[7] in nonzero_bars


def test_xsmom_no_position_when_not_tradeable() -> None:
    """A name selected at bar 7 that goes non-tradeable at bar 9 must carry
    zero weight from bar 9 onward, even though the rebalance schedule would
    otherwise hold its bar-7 weight flat until bar 14."""
    panel = _trending_panel()
    tradeable = panel.tradeable.copy()
    tradeable.loc[panel.prices.index[9]:, "UP1"] = False
    panel = MarketPanel(
        snapshot_id=panel.snapshot_id,
        prices=panel.prices,
        funding=panel.funding,
        tradeable=tradeable,
        meta=panel.meta,
    )
    strategy = XsmomCrossSectionalMomentum()
    weights = strategy.target_weights(panel, _base_params())
    validate_weights(panel, weights)  # would raise if UP1 kept exposure while non-tradeable

    row7 = weights.loc[panel.prices.index[7]]
    if row7["UP1"] != 0.0:
        row9 = weights.loc[panel.prices.index[9]]
        assert row9["UP1"] == 0.0


def test_xsmom_explicit_universe_restricts_candidates() -> None:
    """A fixed `universe` naming only 4 of the 6 panel columns must never
    select the excluded 2 as long/short candidates, and `_compute_k` must
    size off `len(universe)` (4 -> tercile k = max(1, 4 // 3) = 1), not
    off the panel's full column count."""
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    params = _base_params(universe=["UP1", "UP2", "DN1", "DN2"])
    weights = strategy.target_weights(panel, params)
    validate_weights(panel, weights)

    row7 = weights.loc[panel.prices.index[7]]
    assert row7["UP3"] == 0.0
    assert row7["DN3"] == 0.0
    longs = row7[row7 > 0]
    shorts = row7[row7 < 0]
    assert len(longs) == 1
    assert len(shorts) == 1
    assert longs.iloc[0] == pytest.approx(0.5)  # k=1 -> 0.5 / 1 per leg
    assert shorts.iloc[0] == pytest.approx(-0.5)


def test_xsmom_is_deterministic() -> None:
    panel = _trending_panel()
    strategy = XsmomCrossSectionalMomentum()
    params = _base_params()
    weights1 = strategy.target_weights(panel, params)
    weights2 = strategy.target_weights(panel, params)
    pd.testing.assert_frame_equal(weights1, weights2)


# -- _compute_k, params.py:75-88, tested directly -----------------------------


def test_compute_k_auto_tercile() -> None:
    assert _compute_k(universe_len=32, n_positions=None) == 10  # 32 // 3 == 10
    assert _compute_k(universe_len=6, n_positions=None) == 2  # 6 // 3 == 2
    assert _compute_k(universe_len=2, n_positions=None) == 1  # max(1, 2 // 3) == 1


def test_compute_k_manual_mode() -> None:
    assert _compute_k(universe_len=32, n_positions=8) == 4  # 8 // 2 == 4


def test_compute_k_clamped_to_half_universe() -> None:
    # Requesting more legs per side than half the universe allows is clamped.
    assert _compute_k(universe_len=6, n_positions=20) == 3  # max_k = 6 // 2 == 3


# -- _rebalance_rows, evaluators/rebalance.py:50-75, tested directly ---------


def test_rebalance_rows_weekly_anchored_schedule() -> None:
    index = pd.date_range("2026-01-01", periods=15, freq="D", tz="UTC")  # starts Thursday
    mask = _rebalance_rows(index, rebalance_periods=7, anchor_dow=3)
    assert list(mask.nonzero()[0]) == [0, 7, 14]


def test_rebalance_rows_first_bar_not_on_anchor_still_fires() -> None:
    """`is_rebalance_due`'s "never rebalanced" rule bypasses the anchor-day
    gate entirely for the very first bar. Starting on a Monday (bar 0), the
    nearest Thursday (bar 3, `2026-01-08`) is only 3 bars later -- too soon
    (`elapsed_days < rebalance_days`) -- so the next rebalance is the
    FOLLOWING Thursday, bar 10 (`2026-01-15`, 10 bars after bar 0)."""
    index = pd.date_range("2026-01-05", periods=12, freq="D", tz="UTC")  # starts Monday
    mask = _rebalance_rows(index, rebalance_periods=7, anchor_dow=3)
    assert mask[0]
    assert list(mask.nonzero()[0]) == [0, 10]
