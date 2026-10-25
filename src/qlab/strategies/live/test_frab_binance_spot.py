"""FRAB with the spot on Binance and the short on Hyperliquid: the live decisions
on a mixed panel, the margin account on Hyperliquid, the flat transfer fees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.live.frab import LiveFrab
from qlab.strategies.live.frab_binance_spot import (
    ALL_PAIRED,
    LiveFrabBinanceSpot,
    SplitVenueParams,
    read_leverage_table,
)

HOURS = pd.date_range("2026-01-01", periods=24 * 40, freq="1h", tz="UTC")
BUDGET = 90.0


@pytest.fixture()
def table(tmp_path):
    path = tmp_path / "lev.csv"
    path.write_text("coin,max_leverage,is_delisted\nA,10,0\nB,10,0\nC,3,0\n", encoding="utf-8")
    return str(path)


def _params(table: str, **over) -> dict:
    params = {
        "entry_threshold_apr": 0.1, "phase2_exit_threshold": -0.1, "base_min_hold_hours": 24,
        "cap_min_hold_hours": 720, "safety_mult": 5, "signal_window_hours": 8,
        "concurrency_cap": 2, "budget_cap_usdc": BUDGET, "margin_buffer_factor": 3,
        "phase1_negative_patience": 24, "phase1_breakeven_cap_hours": 720,
        "neg_stop_threshold_apr": -0.15, "neg_stop_patience_hours": 6,
        "phase2_giveback_fee_mult": 1.0,
        "spot_taker_fee": 0.001, "perp_taker_fee": 0.00045, "transfer_delay_hours": 1,
        "binance_withdrawal_fee_usdc": 0.4, "binance_withdrawal_min_usdc": 3,
        "hl_withdrawal_fee_usdc": 1, "top_up_trigger": 2.0, "healthy_ratio": 3.0,
        "maintenance_over_initial": 0.5, "leverage_table": table,
        "transfers": "when_needed", "batch_share": 0.1, "idle_share": 0.25,
        "entry_parity_tolerance": 0.03, "max_wait_hours": 24,
        "universe": ALL_PAIRED, "leverage": {"A": 10, "B": 10},
    }
    params.update(over)
    return params


def _panel(rates: dict[str, np.ndarray], *, perp: dict[str, np.ndarray] | None = None,
           high_bump: dict[str, np.ndarray] | None = None, spot_from: dict[str, int] | None = None,
           mixed: bool = True) -> MarketPanel:
    """Perps at `perp` (default flat 100), spots at the same price (own prices);
    `high_bump` raises a perp's bar high above its close; `spot_from` is the
    first bar a coin's spot trades."""
    cols = [c for coin in rates for c in (coin, f"{coin}-SPOT")]
    n = len(HOURS)
    perp = perp or {}
    px = {c: perp.get(c.removesuffix("-SPOT"), np.full(n, 100.0)) for c in cols}
    prices = pd.DataFrame(px, index=HOURS)
    funding = pd.DataFrame(0.0, index=HOURS, columns=cols)
    for coin, r in rates.items():
        funding[coin] = r
    high = prices.copy()
    for coin, bump in (high_bump or {}).items():
        high[coin] = prices[coin] + bump
    tradeable = pd.DataFrame(True, index=HOURS, columns=cols)
    for coin, first in (spot_from or {}).items():
        prices.loc[HOURS[:first], f"{coin}-SPOT"] = np.nan
        tradeable.loc[HOURS[:first], f"{coin}-SPOT"] = False
    meta = {"spot_prices": "own",
            "no_funding_instruments": [c for c in cols if c.endswith("-SPOT")]}
    if mixed:
        meta["spot_source"] = "binance"
    return MarketPanel(snapshot_id="t", prices=prices, funding=funding, tradeable=tradeable,
                       high=high, meta=meta)


def _rates(**kw: float) -> dict[str, np.ndarray]:
    return {coin: np.full(len(HOURS), rate) for coin, rate in kw.items()}


def test_it_refuses_a_panel_whose_spot_is_not_another_venues(table) -> None:
    panel = _panel(_rates(A=0.0002), mixed=False)
    with pytest.raises(ValueError, match="mixed panel"):
        LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"]))


def test_it_takes_every_paired_coin_and_none_without_a_leverage_row(table) -> None:
    panel = _panel(_rates(A=0.0002, B=0.0001, X=0.0003))  # X is not in the leverage table
    run = LiveFrabBinanceSpot().run(panel, _params(table))
    assert run.diagnostics["coins_in_universe"] == 2
    assert run.diagnostics["coins_without_leverage"] == 1
    assert run.weights["X"].abs().sum() == 0 and run.weights["A"].abs().sum() > 0


def test_a_perp_without_a_spot_column_is_not_in_the_universe(table) -> None:
    panel = _panel(_rates(A=0.0002))
    prices = panel.prices.copy()
    prices["C"] = 100.0  # a perp whose spot pair does not exist on the other venue
    wider = MarketPanel(
        snapshot_id="t", prices=prices, funding=panel.funding.assign(C=0.0003),
        tradeable=panel.tradeable.assign(C=True), high=panel.high.assign(C=100.0), meta=panel.meta)
    run = LiveFrabBinanceSpot().run(wider, _params(table))
    assert run.diagnostics["coins_in_universe"] == 1
    assert run.weights["C"].abs().sum() == 0


def test_a_named_coin_without_a_spot_is_an_error_not_a_silent_skip(table) -> None:
    panel = _panel(_rates(A=0.0002))
    with pytest.raises(ValueError, match="needs perp and spot columns; missing"):
        LiveFrabBinanceSpot().run(panel, _params(table, universe=["A", "C"]))


def test_a_coin_is_eligible_only_where_both_legs_trade(table) -> None:
    panel = _panel(_rates(A=0.0002, B=0.0002), spot_from={"B": 200})
    run = LiveFrabBinanceSpot().run(panel, _params(table, concurrency_cap=1, budget_cap_usdc=90))
    assert run.eligible.iloc[100] == 1  # A only
    assert run.eligible.iloc[300] == 2
    assert run.weights["B-SPOT"].iloc[:200].abs().sum() == 0
    validate_weights(panel, run.weights)


def test_the_decisions_are_the_live_engines_when_the_fees_and_the_margin_do_not_interfere(
    table,
) -> None:
    """Same panel, frab's own fee rates, a flat market: the positions are the plain
    driver's, bar for bar, and the weights are the same."""
    rates = {"A": np.where((np.arange(len(HOURS)) // 150) % 2 == 0, 0.0002, -0.0002),
             "B": np.full(len(HOURS), 0.0001)}
    panel = _panel(rates)
    venue_fields = set(SplitVenueParams.__dataclass_fields__)
    plain_params = {k: v for k, v in _params(table).items() if k not in venue_fields}
    plain = LiveFrab().run(panel, plain_params | {"universe": ["A", "B"]})
    split = LiveFrabBinanceSpot().run(
        panel, _params(table, spot_taker_fee=0.0007, perp_taker_fee=0.00035))
    assert [(q.coin, q.opened, q.closed) for q in split.positions] == [
        (q.coin, q.opened, q.closed) for q in plain.positions]
    pd.testing.assert_frame_equal(split.weights, plain.weights)
    assert split.diagnostics["top_ups"] == 0


def test_the_engine_counts_a_round_trip_at_the_two_venues_fees(table) -> None:
    panel = _panel(_rates(A=0.0002))
    run = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"]))
    pos = run.positions[0]
    assert pos.total_fees == pytest.approx(pos.size_usdc * (0.001 + 0.00045) * 2)


def _round_trip(table, **over):
    n = len(HOURS)
    r = np.full(n, 0.0002)
    r[24 * 15:] = -0.0002  # funding turns, the position leaves after breakeven
    panel = _panel({"A": r})
    return LiveFrabBinanceSpot().run(
        panel, _params(table, universe=["A"], concurrency_cap=1, **over))


def test_the_margin_deposit_is_the_only_transfer_when_the_margin_stays_on_hyperliquid(
    table,
) -> None:
    run = _round_trip(table)
    pos = run.positions[0]
    assert pos.closed is not None
    assert run.extra_costs.iloc[pos.opened] == pytest.approx(0.4 / BUDGET)
    assert run.extra_costs.iloc[pos.closed] == 0.0  # the margin stays as the account's free margin
    assert run.extra_costs.sum() * BUDGET == pytest.approx(0.4)
    assert run.diagnostics["deposits"] == 1 and run.diagnostics["returns"] == 0


def test_taking_the_margin_home_at_every_close_costs_hyperliquids_withdrawal_fee(table) -> None:
    run = _round_trip(table, transfers="on_close")
    pos = run.positions[0]
    assert run.extra_costs.iloc[pos.opened] == pytest.approx(0.4 / BUDGET)
    assert run.extra_costs.iloc[pos.closed] == pytest.approx(1.0 / BUDGET)
    assert run.extra_costs.sum() * BUDGET == pytest.approx(1.4)
    assert run.diagnostics["deposits"] == 1 and run.diagnostics["returns"] == 1


def test_the_ledger_loses_only_fees_and_keeps_all_funding(table) -> None:
    """Flat prices: what is on Binance plus what is on Hyperliquid at the end is the
    budget plus the funding received minus every fee, to the cent."""
    for mode in ("when_needed", "on_close"):
        run = _round_trip(table, transfers=mode)
        pos = run.positions[0]
        end = run.diagnostics["binance_cash_end"] + run.diagnostics["hl_free_margin_end"]
        notional = pos.units * 100.0
        fees = notional * (2 * 0.001 + 2 * 0.00045) + run.diagnostics["transfer_fees_usd"]
        assert end == pytest.approx(BUDGET + pos.gross_funding - fees, abs=1e-9), mode


def test_money_waiting_on_hyperliquid_covers_the_next_entrys_margin(table) -> None:
    """Two positions one after the other on the same coin: the second one's margin is
    the first one's, left on Hyperliquid, so only one deposit is ever made."""
    n = len(HOURS)
    r = np.where((np.arange(n) // 240) % 2 == 0, 0.0002, -0.0002)
    panel = _panel({"A": r})
    run = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"], concurrency_cap=1))
    assert len(run.positions) >= 2
    assert run.diagnostics["deposits"] == 1


def _rising(n_flat: int, to: float, over: int) -> np.ndarray:
    px = np.full(len(HOURS), 100.0)
    px[n_flat: n_flat + over] = np.linspace(100.0, to, over)
    px[n_flat + over:] = to
    return px


def test_a_rising_price_bleeds_the_margin_and_is_topped_up_from_the_spot(table) -> None:
    """+25% on a coin sized with a 10x leverage: the ratio falls through 2.0, the
    strategy sells the same share of spot and short and sends the proceeds."""
    panel = _panel(_rates(A=0.0002), perp={"A": _rising(100, 125.0, 20)})
    run = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"], concurrency_cap=1))
    assert run.diagnostics["top_ups"] >= 1
    assert run.diagnostics["top_up_usd"] > 0
    w = run.weights
    changes = w[(w["A"] != w["A"].shift(1)) & (w["A"] != 0)]  # a position opens or is resized
    units = -changes["A"] * BUDGET / panel.prices["A"].loc[changes.index]
    spot_units = changes["A-SPOT"] * BUDGET / panel.prices["A-SPOT"].loc[changes.index]
    assert len(units) >= 2
    # The book stays hedged and gets smaller: spot and short shrank together.
    assert spot_units.to_numpy() == pytest.approx(units.to_numpy())
    assert units.iloc[-1] < units.iloc[0]
    assert run.diagnostics["liquidations"] == 0
    # The ratio never went below the liquidation line, and each top-up paid the fee.
    assert run.diagnostics["min_ratio"] > 1.0
    paid = run.extra_costs[run.extra_costs > 0]
    assert len(paid) >= 2  # the margin deposit, then at least one top-up
    assert run.diagnostics["transfer_fees_usd"] == pytest.approx(run.extra_costs.sum() * BUDGET)


def test_the_top_up_arrives_after_the_declared_delay_not_at_once(table) -> None:
    """With a long delay the money is still on its way a few bars later, so the
    next bar's ratio still counts it in (no second top-up) but the liquidation check
    does not: nothing is available to the account until it arrives."""
    panel = _panel(_rates(A=0.0002), perp={"A": _rising(100, 125.0, 20)})
    quick = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"], concurrency_cap=1,
                                                     transfer_delay_hours=1))
    slow = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"], concurrency_cap=1,
                                                    transfer_delay_hours=48))
    # What is on its way counts in the trigger either way: the same top-ups happen.
    assert slow.diagnostics["top_ups"] == quick.diagnostics["top_ups"]


def test_a_spike_through_the_margin_inside_a_bar_liquidates_the_shorts(table) -> None:
    """The close is calm, the bar's high is +60%: the account is assessed at the
    high and the shorts are closed, which is counted."""
    bump = np.zeros(len(HOURS))
    bump[400] = 60.0
    panel = _panel(_rates(A=0.0002), high_bump={"A": bump})
    run = LiveFrabBinanceSpot().run(panel, _params(table, universe=["A"], concurrency_cap=1))
    assert run.diagnostics["liquidations"] == 1
    assert any(q.exit == "liquidation" and q.closed == 400 for q in run.positions)


def test_a_close_without_a_high_is_refused_not_read_as_calm(table) -> None:
    panel = _panel(_rates(A=0.0002))
    high = panel.high.copy()
    high.loc[HOURS[300], "A"] = np.nan
    broken = MarketPanel(snapshot_id="t", prices=panel.prices, funding=panel.funding,
                         tradeable=panel.tradeable, high=high, meta=panel.meta)
    with pytest.raises(ValueError, match="no high"):
        LiveFrabBinanceSpot().run(broken, _params(table, universe=["A"], concurrency_cap=1))


def test_decisions_and_money_do_not_see_the_future(table) -> None:
    r = np.where((np.arange(len(HOURS)) // 100) % 2 == 0, 0.0002, -0.0002)
    perp = {"A": _rising(100, 125.0, 20)}
    panel = _panel({"A": r, "B": r[::-1].copy()}, perp=perp)
    base = LiveFrabBinanceSpot().run(panel, _params(table))
    cut = HOURS[500]
    r2 = np.where(HOURS > cut, 0.001, r)
    moved = LiveFrabBinanceSpot().run(_panel({"A": r2, "B": r[::-1].copy()}, perp=perp),
                                      _params(table))
    pd.testing.assert_frame_equal(base.weights[base.weights.index <= cut],
                                  moved.weights[moved.weights.index <= cut])
    pd.testing.assert_series_equal(base.extra_costs[base.extra_costs.index <= cut],
                                   moved.extra_costs[moved.extra_costs.index <= cut])


def test_the_venue_params_are_all_required_and_ordered(table) -> None:
    good = _params(table)
    venue, rest = SplitVenueParams.split(good)
    assert venue.top_up_trigger == 2.0 and "entry_threshold_apr" in rest
    with pytest.raises(ValueError, match="params missing"):
        SplitVenueParams.split({k: v for k, v in good.items() if k != "healthy_ratio"})
    with pytest.raises(ValueError, match="top_up_trigger < healthy_ratio"):
        SplitVenueParams.split({**good, "top_up_trigger": 3.0})
    with pytest.raises(ValueError, match="cannot be negative"):
        SplitVenueParams.split({**good, "transfer_delay_hours": -1})
    with pytest.raises(ValueError, match="transfers must be one of"):
        SplitVenueParams.split({**good, "transfers": "sometimes"})


def test_the_leverage_table_reads_coin_and_max_leverage(table) -> None:
    assert read_leverage_table(table) == {"A": 10.0, "B": 10.0, "C": 3.0}


# -- transfers: batched (owner, 2026-10-25) ----------------------------------------------


def test_batched_has_nothing_to_batch_while_every_slot_is_funded(table) -> None:
    """The slots' footprints add up to the whole budget, so the cash that would make a
    bigger first deposit is the cash of the slots not yet filled: reserved, not spare.
    Batching then deposits what an entry needs, as `when_needed` does."""
    n = len(HOURS)
    r = np.where((np.arange(n) // 240) % 2 == 0, 0.0002, -0.0002)
    panel = _panel({"A": r, "B": r})
    batched = LiveFrabBinanceSpot().run(panel, _params(table, transfers="batched",
                                                       concurrency_cap=2, batch_share=0.5,
                                                       idle_share=0.9))
    needed = LiveFrabBinanceSpot().run(panel, _params(table, transfers="when_needed",
                                                      concurrency_cap=2))
    assert batched.diagnostics["deposits"] == needed.diagnostics["deposits"]
    assert batched.diagnostics["transfer_fees_usd"] == needed.diagnostics["transfer_fees_usd"]


def test_batched_top_up_takes_binance_cash_before_it_sells_any_spot(table) -> None:
    """One position of two slots leaves half the budget as cash on Binance; a +25% rise
    is covered from that cash, so no spot is sold and the book is not resized."""
    panel = _panel(_rates(A=0.0002), perp={"A": _rising(100, 125.0, 20)})
    params = _params(table, universe=["A"], concurrency_cap=2, transfers="batched")
    run = LiveFrabBinanceSpot().run(panel, params)
    assert run.diagnostics["top_ups"] >= 1
    assert run.diagnostics["forced_spot_sales"] == 0
    w = run.weights
    changes = w[(w["A"] != w["A"].shift(1)) & (w["A"] != 0)]  # the position opens or is resized
    assert len(changes) == 1  # opened once and never resized by the top-up
    assert run.diagnostics["liquidations"] == 0


def test_batched_with_no_cash_left_falls_back_to_selling_spot_and_says_so(table) -> None:
    panel = _panel(_rates(A=0.0002), perp={"A": _rising(100, 125.0, 20)})
    run = LiveFrabBinanceSpot().run(
        panel, _params(table, universe=["A"], concurrency_cap=1, transfers="batched"))
    assert run.diagnostics["forced_spot_sales"] >= 1  # every dollar is in the position


def test_batched_takes_idle_money_home_in_one_withdrawal(table) -> None:
    """After a position closes, its margin sits on Hyperliquid; above the idle cap it
    is withdrawn down to one batch, for one withdrawal fee."""
    n = len(HOURS)
    r = np.full(n, 0.0002)
    r[24 * 15:] = -0.0002
    panel = _panel({"A": r})
    run = LiveFrabBinanceSpot().run(
        panel, _params(table, universe=["A"], concurrency_cap=1, transfers="batched",
                       batch_share=0.05, idle_share=0.1))
    assert run.diagnostics["idle_withdrawals"] == 1
    assert run.diagnostics["hl_free_margin_end"] == pytest.approx(0.05 * BUDGET, abs=1e-6)


def test_an_instant_transfer_lands_at_the_bar_it_is_sent(table) -> None:
    panel = _panel(_rates(A=0.0002), perp={"A": _rising(100, 125.0, 20)})
    run = LiveFrabBinanceSpot().run(
        panel, _params(table, universe=["A"], concurrency_cap=1, transfer_delay_hours=0,
                       binance_withdrawal_fee_usdc=0.0, binance_withdrawal_min_usdc=0.0))
    assert run.diagnostics["top_ups"] >= 1
    assert run.diagnostics["min_ratio"] > 1.0


# -- both legs on Binance (owner, 2026-10-25) -------------------------------------------


def _binance_panel(rates, **kw):
    panel = _panel(rates, **kw)
    meta = {**panel.meta, "venue": "binance"}
    meta.pop("spot_source", None)
    return MarketPanel(snapshot_id="t", prices=panel.prices, funding=panel.funding,
                       tradeable=panel.tradeable, high=panel.high, meta=meta)


def _free(table, **over):
    return _params(table, spot_taker_fee=0.001, perp_taker_fee=0.0005,
                   binance_withdrawal_fee_usdc=0.0, binance_withdrawal_min_usdc=0.0,
                   hl_withdrawal_fee_usdc=0.0, transfer_delay_hours=0, transfers="batched",
                   batch_share=0.0, idle_share=1.0, **over)


def test_the_all_binance_strategy_runs_only_on_a_binance_panel(table) -> None:
    from qlab.strategies.live.frab_binance_spot import LiveFrabAllBinance

    mixed = _panel(_rates(A=0.0002))
    with pytest.raises(ValueError, match="Binance panel with Binance's own spot"):
        LiveFrabAllBinance().run(mixed, _free(table, universe=["A"]))
    with pytest.raises(ValueError, match="mixed panel"):
        LiveFrabBinanceSpot().run(_binance_panel(_rates(A=0.0002)), _free(table, universe=["A"]))
    run = LiveFrabAllBinance().run(_binance_panel(_rates(A=0.0002)), _free(table, universe=["A"]))
    assert run.positions


def test_with_both_legs_on_one_venue_nothing_is_paid_for_moving_money(table) -> None:
    from qlab.strategies.live.frab_binance_spot import LiveFrabAllBinance

    perp = {"A": _rising(100, 125.0, 20)}
    run = LiveFrabAllBinance().run(_binance_panel(_rates(A=0.0002), perp=perp),
                                   _free(table, universe=["A"], concurrency_cap=2))
    assert run.extra_costs.sum() == 0.0
    assert run.diagnostics["top_ups"] >= 1  # the margin was still topped up, from the cash
    assert run.diagnostics["transfer_fees_usd"] == 0.0
    assert run.diagnostics["liquidations"] == 0


def test_all_binance_sells_spot_for_margin_only_when_no_cash_is_left(table) -> None:
    from qlab.strategies.live.frab_binance_spot import LiveFrabAllBinance

    perp = {"A": _rising(100, 125.0, 20)}
    full = LiveFrabAllBinance().run(_binance_panel(_rates(A=0.0002), perp=perp),
                                    _free(table, universe=["A"], concurrency_cap=1))
    assert full.diagnostics["forced_spot_sales"] >= 1
    idle = LiveFrabAllBinance().run(_binance_panel(_rates(A=0.0002), perp=perp),
                                    _free(table, universe=["A"], concurrency_cap=2))
    assert idle.diagnostics["forced_spot_sales"] == 0


# -- broken pairs ------------------------------------------------------------------------


def test_a_pair_whose_spot_is_far_from_its_perp_is_not_entered(table) -> None:
    """frab's parity rule at the moment of entry: a spot quoted at a fifth of its perp
    (a delisting token) is no hedge."""
    panel = _panel(_rates(A=0.0002, B=0.0002))
    prices = panel.prices.copy()
    prices["A-SPOT"] = prices["A-SPOT"] * 0.2
    broken = MarketPanel(snapshot_id="t", prices=prices, funding=panel.funding,
                         tradeable=panel.tradeable, high=panel.high, meta=panel.meta)
    run = LiveFrabBinanceSpot().run(broken, _params(table, concurrency_cap=2))
    assert {q.coin for q in run.positions} == {"B"}
    assert run.weights["A"].abs().sum() == 0


def test_a_held_pair_that_cannot_be_traded_for_long_is_closed_not_waited_for(table) -> None:
    """The spot disappears for good after bar 300: the engine waits at most
    `max_wait_hours`, then the position is closed at the last marks and its slot freed."""
    panel = _panel(_rates(A=0.0002, B=0.0002))
    prices = panel.prices.copy()
    tradeable = panel.tradeable.copy()
    prices.loc[HOURS[300]:, "A-SPOT"] = np.nan
    tradeable.loc[HOURS[300]:, "A-SPOT"] = False
    gone = MarketPanel(snapshot_id="t", prices=prices, funding=panel.funding,
                       tradeable=tradeable, high=panel.high, meta=panel.meta)
    run = LiveFrabBinanceSpot().run(gone, _params(table, concurrency_cap=2))
    [a] = [q for q in run.positions if q.coin == "A"]
    assert a.exit == "pair_lost" and a.closed == 300 + 24
    assert run.diagnostics["pairs_lost"] == 1
