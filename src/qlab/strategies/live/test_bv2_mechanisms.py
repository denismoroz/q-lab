"""Tests for the mechanisms the Bv2 adapter adds to frab's book: a coin list that changes over
time, one margin pool, delayed top-ups, deposits (`qlab.strategies.live.bv2`, docstring).

First the proof that, with all of them off, the stand's Bv2 is the Bv2 it was: the current driver
and the frozen pre-switch driver (`_bv2_reference`) must agree to the last bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live import bv2 as bv2_module
from qlab.strategies.live._bv2_mechanisms import parse_universe, period_starts
from qlab.strategies.live._bv2_reference import reference_run
from qlab.strategies.live.bv2 import LiveBv2

HAS_POOL, HOURS = kit.HAS_POOL, kit.HOURS
_market, _params = kit.market, kit.params


def _same_as_reference(panel: MarketPanel, params: dict) -> None:
    now = LiveBv2().run(panel, params)
    ref = reference_run(panel, params)
    pd.testing.assert_frame_equal(now.weights, ref.weights, check_exact=True)
    pd.testing.assert_series_equal(now.book_equity, ref.book_equity, check_exact=True)
    assert now.liquidations == ref.liquidations
    assert now.weights.abs().to_numpy().sum() > 0


# ── with every mechanism off, the driver is the one that was ────────────────────────────────

def test_off_is_the_reference_on_a_fixed_coin_list() -> None:
    _same_as_reference(_market(), _params())


def test_off_is_the_reference_with_a_late_spot_listing_and_a_data_hole() -> None:
    panel = _market(spot_from={"B": 24 * 38})
    prices, tradeable = panel.prices.copy(), panel.tradeable.copy()
    hole = HOURS - 60
    prices.iloc[hole, prices.columns.get_loc("A-SPOT")] = np.nan  # no print: both legs are held
    tradeable.iloc[hole, tradeable.columns.get_loc("A-SPOT")] = False
    holed = MarketPanel(snapshot_id="m", prices=prices, funding=panel.funding, tradeable=tradeable,
                        meta=panel.meta, volume=panel.volume, high=panel.high)
    _same_as_reference(holed, _params())


def test_off_is_the_reference_when_the_pool_is_stressed_into_top_ups_and_liquidations() -> None:
    rng = np.random.default_rng(1)
    hours = HOURS
    # a long fall (the hedge goes on) and then violent rebounds
    steps = np.r_[np.full(24 * 40, 0.0004), np.full(24 * 25, -0.0012), np.full(24 * 35, 0.0)]
    path = 100.0 * np.cumprod(1.0 + steps + rng.normal(0, 0.004, hours))
    path[24 * 65:] *= np.r_[np.linspace(1.0, 1.9, 24 * 10), np.full(hours - 24 * 75, 1.9)] \
        [: hours - 24 * 65]
    panel = _market(coins=("A", "B"), paths={"A": path, "B": path * 0.5})
    params = _params(coins=("A", "B"), hedge_margin_headroom=0.3, default_leverage=6.0,
                     short_leverage={"A": 6.0, "B": 6.0})
    now = LiveBv2().run(panel, params)
    assert now.counts.get("margin_rebalance", 0) > 0, "the scenario must reach top-ups"
    _same_as_reference(panel, params)


def test_off_is_the_reference_without_margin_and_with_the_carry_sleeve() -> None:
    panel = _market(coins=("A", "B"))
    _same_as_reference(panel, _params(coins=("A", "B"), margin_enabled=False, carry_enabled=True,
                                      min_order_usd=0.0))


def test_off_is_the_reference_under_an_external_hedge(monkeypatch) -> None:
    import qlab.regime_detect as rd

    panel = _market(coins=("A", "B"))
    days = pd.date_range(panel.prices.index[0].normalize(), periods=110, freq="1D", tz="UTC")
    stored = pd.DataFrame({"label": ["bear", "bull"] * 55}, index=days)
    monkeypatch.setattr(rd, "load_predictions", lambda name: stored)
    params = {**_params(coins=("A", "B")), "hedge_by": {"predictions": "x", "hedge_when": ["bear"]}}
    _same_as_reference(panel, params)


def test_stating_the_switches_at_their_defaults_changes_nothing() -> None:
    panel = _market(coins=("A", "B"))
    base = _params(coins=("A", "B"))
    plain = LiveBv2().run(panel, base)
    explicit_params = dict(base)
    if HAS_POOL:
        explicit_params.update(shared_margin=False, topup_delay_hours=0)
    explicit = LiveBv2().run(panel, explicit_params)
    pd.testing.assert_frame_equal(plain.weights, explicit.weights, check_exact=True)
    pd.testing.assert_series_equal(plain.book_equity, explicit.book_equity, check_exact=True)
    assert (plain.deposits == 0).all() and plain.counts.get("universe_join", 0) == 0


# ── guards ──────────────────────────────────────────────────────────────────────────────────

def test_a_frab_without_the_mechanisms_is_refused_not_silently_ignored(monkeypatch) -> None:
    class OldParams:
        __dataclass_fields__ = {"coins": None}

    monkeypatch.setattr(bv2_module, "B2Params", OldParams)
    for key in ("shared_margin", "topup_delay_hours"):
        with pytest.raises(ValueError, match="frab checkout"):
            bv2_module._require_frab_support({key: 1})
    bv2_module._require_frab_support({"coins": ["A"]})  # nothing asked, nothing refused


def test_deposits_need_an_amount_and_a_period_together() -> None:
    panel = _market(coins=("A",))
    with pytest.raises(ValueError, match="go together"):
        LiveBv2().run(panel, _params(coins=("A",), deposit_usd=10.0))
    with pytest.raises(ValueError, match="go together"):
        LiveBv2().run(panel, _params(coins=("A",), deposit_every="7d"))
    with pytest.raises(ValueError, match="positive"):
        LiveBv2().run(panel, _params(coins=("A",), deposit_usd=-5.0, deposit_every="7d"))


def test_a_changing_coin_list_refuses_the_carry_sleeve_and_a_second_coin_list() -> None:
    panel = _market()
    universe = {"rule": "top_liquid", "n": 2, "rebalance": "7d", "by": "spot_dollar_volume",
                "window_days": 3, "candidates": "panel"}
    params = _params(universe=universe)
    with pytest.raises(ValueError, match="universe.candidates"):
        LiveBv2().run(panel, params)  # `coins` and `universe` both name the coins
    params.pop("coins")
    with pytest.raises(ValueError, match="carry"):
        LiveBv2().run(panel, {**params, "carry_enabled": True})


# ── periods and the universe's own parameters ───────────────────────────────────────────────

def test_period_starts_mark_the_first_bar_of_each_calendar_period() -> None:
    index = pd.date_range("2026-01-29", periods=24 * 20, freq="1h", tz="UTC")
    monthly = np.flatnonzero(period_starts(index, "monthly"))
    assert [str(index[i]) for i in monthly] == ["2026-02-01 00:00:00+00:00"]
    weekly = np.flatnonzero(period_starts(index, "weekly"))
    assert all(index[i].weekday() == 0 and index[i].hour == 0 for i in weekly) and len(weekly) == 3
    every3 = np.flatnonzero(period_starts(index, "3d"))
    assert list(every3) == [72, 144, 216, 288, 360, 432]
    assert not period_starts(index, "monthly")[0]
    quarterly = np.flatnonzero(period_starts(pd.date_range("2026-03-30", periods=96, freq="1h",
                                                           tz="UTC"), "quarterly"))
    assert len(quarterly) == 1
    with pytest.raises(ValueError, match="period"):
        period_starts(index, "fortnightly")
    with pytest.raises(ValueError, match="period"):
        period_starts(index, "0d")


def test_the_universe_rule_has_no_defaults() -> None:
    full = {"rule": "top_liquid", "n": 10, "rebalance": "monthly", "by": "spot_dollar_volume",
            "window_days": 30, "candidates": ["A", "B"]}
    assert parse_universe(full).n == 10
    for key in full:
        with pytest.raises(ValueError, match="needs"):
            parse_universe({k: v for k, v in full.items() if k != key})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_universe({**full, "weights": "equal"})
    for bad in ({"n": 0}, {"n": 2.5}, {"by": "market_cap"}, {"window_days": 0},
                {"rebalance": "sometimes"}, {"rule": "top_cap"}, {"candidates": []},
                {"candidates": ["A-SPOT"]}, {"candidates": ["A", "A"]}):
        with pytest.raises(ValueError):
            parse_universe({**full, **bad})
