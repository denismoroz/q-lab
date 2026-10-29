"""A stored run taken apart by coin and day, and the counterfactual rules that only ever remove
exposure (qlab.anomalies.rerun). The toy book (`qlab.anomalies.toy`) has known answers; one test
re-runs a real stored trial of the evaluation pipeline."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from qlab.anomalies import toy
from qlab.anomalies.rerun import (
    Rule,
    RunUnavailable,
    apply_rule,
    daily,
    decompose,
    load_run,
)
from qlab.harness.run import run_backtest
from qlab.pipeline import test_evaluate as base
from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.test_forward_store import RULESET, _causal_spec

# The registry and panel of `test_evaluate`, as fixtures of this module.
engine, session = base.engine, base.session
_idea_row, _no_network = base._idea_row, base._no_network


def test_the_taken_apart_book_reconciles_with_the_stands_own_run() -> None:
    run = toy.toy_run()

    dec = decompose(run)

    assert dec.reconciles and dec.max_gap < 1e-12
    expected = run_backtest(*run.part(), run.costs, run.part()[0].funding).net_return
    pd.testing.assert_series_equal(dec.book_net.rename("net_return"), expected,
                                   check_freq=False, atol=1e-12)


def test_a_single_cell_is_price_plus_funding_minus_cost() -> None:
    dec = decompose(toy.toy_run())
    pump_day = dec.net.index[toy.PUMP_BAR]

    # +200% on a 0.1 weight; funding is minus the weight times the rate; no turnover that day.
    assert dec.price.loc[pump_day, "PUMP"] == pytest.approx(0.1 * 2.0)
    assert dec.funding.loc[pump_day, "PUMP"] == pytest.approx(-0.1 * 0.0001)
    assert dec.cost.loc[pump_day, "PUMP"] == 0.0
    assert dec.net.loc[pump_day, "PUMP"] == pytest.approx(0.2 - 0.00001)
    # Entering costs the whole weight times fee plus slippage on the first bar.
    assert dec.cost.iloc[0]["A"] == pytest.approx(0.1 * 10e-4)
    assert dec.positions.iloc[0] == 6 and dec.gross.iloc[0] == pytest.approx(0.6)


def test_a_coin_that_stopped_trading_is_known_by_its_last_price() -> None:
    dec = decompose(toy.toy_run())

    assert dec.last_priced["DEAD"] == dec.net.index[toy.DEAD_LAST_BAR]
    assert dec.last_priced["A"] == dec.panel_end
    # The last priced bar has no next price: nothing is earned or lost on it.
    assert dec.price.iloc[toy.DEAD_LAST_BAR]["DEAD"] == 0.0


def test_daily_groups_bars_into_utc_days() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="12h", tz="UTC")
    returns = pd.Series([0.01, 0.01, -0.02, 0.0], index=index)

    assert daily(returns, "sum").tolist() == pytest.approx([0.02, -0.02])
    assert daily(returns, "compound").tolist() == pytest.approx([1.01 * 1.01 - 1, -0.02])
    assert daily(returns, "mean").tolist() == pytest.approx([0.01, -0.01])


# --- the rules ----------------------------------------------------------------------------


def _held(weights: pd.DataFrame, name: str) -> int:
    return int((weights[name].abs() > 1e-9).sum())


def test_thin_coin_days_are_removed_and_unknown_volume_is_kept_and_counted() -> None:
    run = toy.toy_run()

    effect = apply_rule(run, Rule(min_volume_usd=300_000))

    # PUMP trades 200 thousand dollars up to and including bar 20; DEAD 500 thousand (kept).
    assert _held(effect.weights, "PUMP") == len(run.weights) - (toy.PUMP_BAR + 1)
    assert _held(effect.weights, "DEAD") == _held(run.weights, "DEAD")
    assert effect.removed_bars == toy.PUMP_BAR + 1
    assert effect.unknown_bars == 2  # A has no volume figure on two bars; it stays held
    assert _held(effect.weights, "A") == _held(run.weights, "A")
    assert _held(run.weights, "PUMP") == len(run.weights)  # the stored book is untouched


def test_the_days_before_a_coin_stops_trading_are_removed() -> None:
    run = toy.toy_run()

    effect = apply_rule(run, Rule(delist_days=3))

    kept = effect.weights["DEAD"]
    assert (kept.iloc[toy.DEAD_LAST_BAR - 3: toy.DEAD_LAST_BAR + 1] == 0.0).all()
    assert kept.iloc[toy.DEAD_LAST_BAR - 4] == 0.1
    assert effect.removed_bars == 4
    assert _held(effect.weights, "PUMP") == len(run.weights)  # a coin still trading is not touched


def test_an_instrument_is_capped_at_a_share_of_the_gross_exposure() -> None:
    run = toy.toy_run()

    effect = apply_rule(run, Rule(max_instrument_share=0.1))  # 0.1 of a gross of 0.6 = 0.06

    first = effect.weights.iloc[0]
    assert first.abs().max() == pytest.approx(0.06)
    assert (first.abs().sum()) == pytest.approx(0.36)  # the excess is cash, not moved elsewhere


def test_only_the_other_venues_listed_coins_stay_from_its_first_day() -> None:
    run = toy.toy_run()
    listed = pd.DataFrame(True, index=run.panel.prices.index, columns=run.panel.prices.columns)
    listed["PUMP"] = False
    listed.iloc[:5] = False  # the other venue starts on bar 5

    effect = apply_rule(run, Rule(only_listed=listed))

    assert effect.first_row == 5
    assert _held(effect.weights, "PUMP") == 0
    assert _held(effect.weights, "A") == len(run.weights) - 5


def test_removing_the_pump_coins_exposure_removes_its_profit() -> None:
    run = toy.toy_run()
    base_total = run.backtest().net_return.sum()

    thin = run.backtest(apply_rule(run, Rule(min_volume_usd=300_000)).weights).net_return.sum()

    assert base_total - thin == pytest.approx(0.2, abs=0.01)


# --- a real stored run --------------------------------------------------------------------


def _write_spec(spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{spec.idea_id}.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")


def test_a_stored_trial_is_re_run_to_the_numbers_it_recorded(session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    run = load_run(session, result.trial_id, spec_root=tmp_path / "specs")
    dec = decompose(run)

    assert dec.reconciles
    total = float((1.0 + run.backtest().net_return).prod() - 1.0)
    assert total == pytest.approx(result.metrics["total_return_net"])
    assert run.idea_id == spec.idea_id and run.route == "paper"
    assert set(dec.net.columns) == {"BTC", "ETH"}


def test_a_trial_that_cannot_be_re_run_says_why(session, tmp_path):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)

    with pytest.raises(RunUnavailable, match="spec file"):
        load_run(session, result.trial_id, spec_root=tmp_path / "no-specs")
    with pytest.raises(RunUnavailable, match="no trial 999"):
        load_run(session, 999)
