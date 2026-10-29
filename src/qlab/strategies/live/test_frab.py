"""FRAB on the stand decides with frab's own two-phase functions
(qlab.strategies.live.frab)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def _own_panel(rates: dict[str, np.ndarray], basis: float = 1.02,
               spot_gap: tuple[str, slice] | None = None) -> MarketPanel:
    """A panel whose perps drift upward and whose spots trade at `basis` times
    their perp, under `spot_prices: own`; `spot_gap` blanks one spot's prints."""
    cols = [c for coin in rates for c in (coin, f"{coin}-SPOT")]
    drift = 100.0 * (1.0 + 0.001 * np.arange(len(HOURS)))
    prices = pd.DataFrame({c: drift * (basis if c.endswith("-SPOT") else 1.0) for c in cols},
                          index=HOURS)
    tradeable = pd.DataFrame(True, index=HOURS, columns=cols)
    if spot_gap is not None:
        coin, rows = spot_gap
        prices.iloc[rows, prices.columns.get_loc(f"{coin}-SPOT")] = np.nan
        tradeable.iloc[rows, tradeable.columns.get_loc(f"{coin}-SPOT")] = False
    funding = pd.DataFrame(0.0, index=HOURS, columns=cols)
    for coin, r in rates.items():
        funding[coin] = r
    return MarketPanel(snapshot_id="own", prices=prices, funding=funding, tradeable=tradeable,
                       meta={"spot_prices": "own"})


def _flat_rates(n: int) -> dict[str, np.ndarray]:
    return {"A": np.full(n, 0.0001), "B": np.zeros(n), "C": np.zeros(n)}


def test_the_spot_leg_is_bought_and_valued_at_the_spots_own_price() -> None:
    panel = _own_panel(_flat_rates(len(HOURS)), basis=1.02)
    w = LiveFrab().run(panel, PARAMS).weights
    validate_weights(panel, w)
    first = w.index[7]
    p = FrabParams.from_dict(PARAMS)
    # The slot is spent on the spot leg: its notional is the sized amount, and
    # the short in the same units is worth 1/1.02 of it.
    assert np.isclose(w.loc[first, "A-SPOT"] * p.budget_cap_usdc, size_for(p, 10))
    assert np.isclose(w.loc[first, "A"], -w.loc[first, "A-SPOT"] / 1.02)


def test_by_default_a_spot_column_with_another_price_is_not_read() -> None:
    n = len(HOURS)
    panel = _own_panel(_flat_rates(n), basis=1.02)
    by_default = MarketPanel(snapshot_id="d", prices=panel.prices, funding=panel.funding,
                             tradeable=panel.tradeable, meta={})
    marked_at_perp = _own_panel(_flat_rates(n), basis=1.0)
    marked_at_perp = MarketPanel(snapshot_id="p", prices=marked_at_perp.prices,
                                 funding=panel.funding, tradeable=panel.tradeable, meta={})
    a = LiveFrab().run(by_default, PARAMS).weights
    b = LiveFrab().run(marked_at_perp, PARAMS).weights
    pd.testing.assert_frame_equal(a, b)
    assert (a["A-SPOT"] == -a["A"]).all()


def test_a_spot_with_no_print_is_valued_at_its_last_print_not_at_the_perp() -> None:
    """While A's spot printed nothing, B opens; the book is re-read then, and
    A's spot leg is worth its last close, not the perp's price of that bar."""
    n = len(HOURS)
    rates = _flat_rates(n)
    rates["B"] = np.where(np.arange(n) >= 30, 0.0001, 0.0)
    gap = slice(24, 40)
    panel = _own_panel(rates, basis=1.02, spot_gap=("A", gap))
    run = LiveFrab().run(panel, PARAMS)
    [pos_a] = [q for q in run.positions if q.coin == "A"]
    [pos_b] = [q for q in run.positions if q.coin == "B"]
    assert gap.start <= pos_b.opened < gap.stop  # B opens while A's spot is silent
    w = run.weights
    last_print = panel.prices["A-SPOT"].iloc[gap.start - 1]
    perp_now = panel.prices["A"].iloc[pos_b.opened]
    assert np.isclose(w.iloc[pos_b.opened]["A-SPOT"] * 90, pos_a.units * last_print)
    assert not np.isclose(w.iloc[pos_b.opened]["A-SPOT"] * 90, pos_a.units * perp_now)


def test_own_spot_prices_pass_the_look_ahead_guard() -> None:
    from qlab.harness.lookahead import lookahead_violation

    n = len(HOURS)
    r = np.where((np.arange(n) // 100) % 2 == 0, 0.0002, -0.0002)
    panel = _own_panel({"A": r, "B": r[::-1].copy(), "C": np.zeros(n)},
                       spot_gap=("A", slice(300, 320)))
    weights = LiveFrab().target_weights(panel, PARAMS)
    assert weights.abs().to_numpy().sum() > 0
    assert lookahead_violation(LiveFrab(), panel, PARAMS, weights) is None


def test_the_orders_are_the_sizes_a_position_opened_with() -> None:
    """The opening orders are the slot's size on both legs, as a share of the budget; nothing
    is left unsent because this driver models no minimum order."""
    from qlab.harness.strategy import own_orders

    n = len(HOURS)
    panel = _panel({"A": np.full(n, 0.00005), "B": np.full(n, 0.0001), "C": np.zeros(n)})
    orders = own_orders(LiveFrab(), panel, PARAMS)
    assert orders is not None and orders.skipped.sum() == 0
    p = FrabParams.from_dict(PARAMS)
    run = LiveFrab().run(panel, PARAMS)
    placed = orders.sizes[(orders.sizes > 0).any(axis=1)]
    assert len(placed) == 1  # both positions open on the same hour
    row = placed.iloc[0]
    assert row["B-SPOT"] * 90 == pytest.approx(size_for(p, 10))
    assert row["B"] == pytest.approx(row["B-SPOT"])  # equal prices: the short is the same dollars
    assert row["C"] == 0 and row["C-SPOT"] == 0
    # a price move after the open changes the weights, never the order that was sent
    assert run.opened.equals(orders.sizes)
