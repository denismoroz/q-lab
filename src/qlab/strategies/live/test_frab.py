"""FRAB on the stand decides with frab's own two-phase functions
(qlab.strategies.live.frab)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.live.frab import FrabParams, LiveFrab, size_for

HOURS = pd.date_range("2026-01-01", periods=24 * 60, freq="1h", tz="UTC")
PARAMS = {
    "entry_threshold_apr": 0.1, "phase2_exit_threshold": -0.1, "base_min_hold_hours": 24,
    "cap_min_hold_hours": 720, "safety_mult": 5, "signal_window_hours": 8,
    "concurrency_cap": 2, "budget_cap_usdc": 90, "margin_buffer_factor": 3,
    "phase1_negative_patience": 24, "phase1_breakeven_cap_hours": 720,
    "neg_stop_threshold_apr": -0.15, "neg_stop_patience_hours": 6,
    "phase2_giveback_fee_mult": 1.0,
    "universe": ["A", "B", "C"], "leverage": {"A": 10, "B": 10, "C": 3},
}


def _panel(rates: dict[str, np.ndarray]) -> MarketPanel:
    cols = [c for coin in rates for c in (coin, f"{coin}-SPOT")]
    prices = pd.DataFrame(100.0, index=HOURS, columns=cols)
    funding = pd.DataFrame(0.0, index=HOURS, columns=cols)
    for coin, r in rates.items():
        funding[coin] = r
    return MarketPanel(snapshot_id="t", prices=prices, funding=funding,
                       tradeable=pd.DataFrame(True, index=HOURS, columns=cols), meta={})


def test_the_strongest_coins_take_the_slots_with_equal_legs() -> None:
    n = len(HOURS)
    panel = _panel({"A": np.full(n, 0.00005), "B": np.full(n, 0.0001), "C": np.zeros(n)})
    run = LiveFrab().run(panel, PARAMS)
    w = run.weights
    validate_weights(panel, w)
    first = w.index[7]  # eight hourly rates make the first signal
    assert w.loc[w.index[6]].abs().sum() == 0
    assert w.loc[first, "B"] < 0 and w.loc[first, "A"] < 0 and w.loc[first, "C"] == 0
    assert w.loc[first, "B"] == -w.loc[first, "B-SPOT"]
    p = FrabParams.from_dict(PARAMS)
    assert w.loc[first, "B-SPOT"] * 90 == size_for(p, 10)


def test_a_position_past_breakeven_leaves_when_funding_turns_and_gives_back_the_fees() -> None:
    r = np.full(len(HOURS), 0.0002)
    turn = 24 * 20
    r[turn:] = -0.0002
    panel = _panel({"A": r, "B": np.zeros(len(HOURS)), "C": np.zeros(len(HOURS))})
    run = LiveFrab().run(panel, PARAMS)
    [pos] = [q for q in run.positions if q.coin == "A"][:1]
    assert pos.post
    assert pos.closed is not None and pos.closed > turn
    assert pos.exit.endswith("CLOSE_POST_BE")
    assert pos.peak_funding - pos.gross_funding >= pos.total_fees


def test_decisions_do_not_see_the_future() -> None:
    r = np.where((np.arange(len(HOURS)) // 100) % 2 == 0, 0.0002, -0.0002)
    panel = _panel({"A": r, "B": r[::-1].copy(), "C": np.zeros(len(HOURS))})
    base = LiveFrab().run(panel, PARAMS).weights
    cut = HOURS[700]
    changed = _panel({"A": np.where(HOURS > cut, 0.001, r), "B": r[::-1].copy(),
                      "C": np.zeros(len(HOURS))})
    moved = LiveFrab().run(changed, PARAMS).weights
    pd.testing.assert_frame_equal(base[base.index <= cut], moved[moved.index <= cut])
