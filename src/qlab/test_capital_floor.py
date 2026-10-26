"""Tests for `qlab.capital_floor`: the least money a strategy works on, measured on its own
code (owner, 2026-10-26)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.capital_floor import (
    MIN_CORRELATION,
    REFERENCE_USD,
    Floor,
    Step,
    cached,
    floor_from,
    judge,
    measure,
    with_money,
)
from qlab.harness.panel import MarketPanel
from qlab.pipeline.spec import StrategySpec


def test_money_is_set_at_any_depth() -> None:
    params, found = with_money({"base_params": {"capital_usd": 3000.0, "lookbacks": [30]},
                                "budget_cap_usdc": 90, "other": 1}, 150)
    assert found and params["base_params"]["capital_usd"] == 150.0
    assert params["budget_cap_usdc"] == 150.0 and params["other"] == 1
    assert with_money({"coins": ["BTC"]}, 150) == ({"coins": ["BTC"]}, False)


def test_the_floor_is_where_every_larger_step_works_too() -> None:
    def s(usd, works):
        return Step(usd=usd, dropped_positions=0.0, below_minimum=0.0, correlation=1.0,
                    net_return=0.1, works=works)

    assert floor_from([s(50, True), s(100, False), s(150, True), s(10_000, True)]) == 150
    assert floor_from([s(50, True), s(10_000, False)]) is None


def _frame(values, index, columns):
    return pd.DataFrame(values, index=index, columns=columns)


def test_judge_names_every_failed_condition() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    ref = _frame(np.full((48, 2), 0.25), index, ["A", "B"])
    rng = np.random.default_rng(1)
    ref_net = pd.Series(rng.normal(0.001, 0.01, 48), index=index)
    from qlab.capital_floor import daily

    small = ref.copy()
    small["B"] = 0.0  # B dropped at the small sum
    step = judge(20.0, small, -ref_net.abs(), ref, daily(ref_net), 10.0)
    assert not step.works and step.dropped_positions == 0.5
    assert step.below_minimum == 1.0  # 0.25 x $20 = $5 < $10
    assert any("minimum order" in w for w in step.why_not)
    assert any("not held" in w for w in step.why_not)
    assert any("net return" in w for w in step.why_not)
    same = judge(REFERENCE_USD, ref, ref_net.abs(), ref, daily(ref_net.abs()), 10.0)
    assert same.works and same.correlation >= MIN_CORRELATION


class SizedBook:
    """Two coins; at a small sum the second one's leg is below the $10 minimum and is not
    held, the way frab's own code skips such an order."""

    name = "sized"

    def target_weights(self, panel, params):
        capital = float(params["capital_usd"])
        w = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        w["A"] = 0.6
        w["B"] = 0.4 if 0.1 * capital >= 10.0 else 0.0  # its smallest order is a tenth
        return w


def _panel() -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=120, freq="1D", tz="UTC")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.003, 0.02, (120, 2)), axis=0),
                          index=index, columns=["A", "B"])
    return MarketPanel(prices=prices, funding=pd.DataFrame(0.0, index=index, columns=["A", "B"]),
                       tradeable=pd.DataFrame(True, index=index, columns=["A", "B"]),
                       snapshot_id="snap", meta={})


def _spec(params) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "sized", "title": "sized", "code_ref": "qlab.test_capital_floor:SizedBook",
        "params": params, "data": {"source": "hyperliquid", "interval": "1d",
                                   "start": "2026-01-01", "end": "2026-04-30"},
        "costs": {"taker_fee_bps": 4.5, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "simultaneous_legs": 1, "unexpressed_mechanisms": []})


def test_a_book_that_drops_a_leg_below_the_minimum_works_from_where_it_holds_it() -> None:
    floor = measure(_spec({"capital_usd": 257.0}), _panel(),
                    ladder=(50, 75, 100, 150, REFERENCE_USD))
    assert floor.money_in_params
    assert floor.floor_usd == 100  # B (a tenth of the money) needs $100 to be a $10 order
    by = {s.usd: s for s in floor.steps}
    assert by[75].dropped_positions > 0 and not by[75].works
    assert floor.reasons_below()["positions"] == 100


def test_a_book_without_money_in_its_params_has_only_the_position_floor() -> None:
    spec = _spec({"capital_usd": 257.0}).model_copy(update={"params": {"x": 1}})

    class Fixed:
        name = "fixed"

        def target_weights(self, panel, params):
            return pd.DataFrame(0.05, index=panel.prices.index, columns=panel.prices.columns)

    import qlab.test_capital_floor as me

    me.Fixed = Fixed
    floor = measure(spec.model_copy(update={"code_ref": "qlab.test_capital_floor:Fixed"}),
                    _panel())
    assert not floor.money_in_params and floor.floor_usd == 200.0  # $10 / 0.05


def test_a_floor_is_reused_for_the_same_spec_text_only(tmp_path, monkeypatch) -> None:
    import qlab.capital_floor as module

    calls = []

    def fake(session, spec):
        calls.append(spec.idea_id)
        return Floor(idea_id=spec.idea_id, floor_usd=150.0, position_floor_usd=99.0,
                     money_in_params=True, window=("a", "b"), steps=[], note="")

    monkeypatch.setattr(module, "measure_spec", fake)
    monkeypatch.setattr(module, "load_spec", lambda path: _spec({"capital_usd": 1.0}))
    spec_file = tmp_path / "s.yaml"
    spec_file.write_text("idea_id: sized\n", encoding="utf-8")
    assert cached(None, spec_file, store=tmp_path / "store").floor_usd == 150.0
    assert cached(None, spec_file, store=tmp_path / "store").floor_usd == 150.0
    spec_file.write_text("idea_id: sized\n# changed\n", encoding="utf-8")
    cached(None, spec_file, store=tmp_path / "store")
    assert calls == ["sized", "sized"]
