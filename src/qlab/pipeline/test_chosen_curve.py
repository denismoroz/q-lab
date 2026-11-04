"""The selection part of a run that chooses among declared values shows the chosen value's own
money, not a flat warm-up (`qlab.chosen_curve`, docs/FORWARD_CHART.md). Reuses the toy strategy,
fixture panel and registry of `test_evaluate`."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from qlab import forward_series, idea_standard, twin
from qlab.api.serializers import trial_json
from qlab.data.snapshot import load_snapshot
from qlab.harness.costs import CostModel
from qlab.harness.run import run_backtest
from qlab.pipeline import test_evaluate as base
from qlab.pipeline.evaluate import _slice_rows, evaluate_spec
from qlab.pipeline.spec import Choice, StrategySpec
from qlab.registry.models import Spec as SpecRow
from qlab.registry.models import Trial

engine, session = base.engine, base.session
_idea_row, _no_network = base._idea_row, base._no_network

RULESET = base._by_data_ruleset([base.CAPITAL_FIT_GENEROUS])


def _run(session, tmp_path, ruleset=RULESET):
    spec = base._month(session, tmp_path, **base._choice_check())
    result = evaluate_spec(spec, session=session, ruleset=ruleset, check_lookahead=False,
                           check_sources=False)
    assert result.error is None, result.error
    return spec, result


def _chosen_weight(session, trial_id) -> float:
    """The weight the single choice picked, read from the registry's own diagnostics."""
    trial = session.get(Trial, trial_id)
    from qlab.pipeline.evaluate import resolve_strategy
    from qlab.strategies.retune import Retune

    panel = load_snapshot(trial.snapshot_id, session=session)
    strategy = resolve_strategy("qlab.strategies.retune:Retune")
    assert isinstance(strategy, Retune)
    choices = strategy.run(panel, trial.params).choices
    return float(base._choice_check()["params"]["grid"]["weight"][choices.iloc[-1]]["weight"])


def test_the_selection_part_is_the_chosen_values_own_money_not_a_flat_line(session, tmp_path):
    spec, result = _run(session, tmp_path)
    trial = session.get(Trial, result.trial_id)

    stored = forward_series.load(result.trial_id)

    parts = {p["part"]: p for p in stored["whole"]["parts"]}
    assert "warmup" not in parts and parts["selection"]["chosen_in_sample"] is True
    assert parts["selection"]["total_return"] != 0.0
    # the same money as the chosen value run alone over those rows, with the same costs
    panel = load_snapshot(trial.snapshot_id, session=session)
    weight = _chosen_weight(session, result.trial_id)
    cut = int(panel.prices.index.searchsorted(pd.Timestamp(trial.params["choose_at"], tz="UTC")))
    sliced = _slice_rows(panel, 0, cut)
    weights = base.ToyStrategy().target_weights(panel, {"weight": weight}).iloc[:cut + 1]
    alone = run_backtest(sliced, weights, CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                                                     slippage_bps=spec.costs.slippage_bps),
                         sliced.funding)
    assert parts["selection"]["total_return"] == pytest.approx(
        float((1 + alone.capital_return).prod() - 1), rel=1e-6)
    assert parts["selection"]["chosen"] in spec.params["grid"]["weight"]


def test_the_forward_part_and_its_totals_are_what_they_were(session, tmp_path):
    _spec, result = _run(session, tmp_path)

    stored = forward_series.load(result.trial_id)

    forward = {p["part"]: p for p in stored["whole"]["parts"]}["forward"]
    assert stored["judged_on_forward"] is True
    assert stored["total_return"] == pytest.approx(result.metrics["total_return_net"])
    assert forward["total_return"] == pytest.approx(stored["total_return"], rel=1e-9)
    assert stored["whole"]["split_date"] == forward["start"] == stored["start"]


def test_the_selection_figures_are_kept_beside_the_forward_ones_and_decide_nothing(
        session, tmp_path):
    ruleset = base._by_data_ruleset([base.CAPITAL_FIT_GENEROUS])
    _spec, result = _run(session, tmp_path, ruleset)
    m = result.metrics

    assert m["fit_chosen_in_sample"] == 1.0
    for key in ("fit_cap_ann_return", "fit_cap_sharpe", "fit_cap_max_dd",
                "fit_cap_ann_return_ex_best_1pct"):
        assert isinstance(m[key], float), key
    # no selection period was judged: nothing was rejected on it, the forward part decided
    assert "selection_rules_failed" not in m and m["selection_days"] == 0.0
    assert result.routing.route == "paper", result.routing.reason
    # the same figures stand in the series file, for runs whose metrics were written without them
    assert forward_series.load(result.trial_id)["whole"]["selection_numbers"][
        "fit_cap_ann_return"] == pytest.approx(m["fit_cap_ann_return"])


def test_the_idea_standard_sees_two_parts_for_such_a_run(session, tmp_path):
    _spec, result = _run(session, tmp_path)
    trial = session.get(Trial, result.trial_id)

    facts = idea_standard.Facts(
        idea_id="toy", status="open", has_run=True, outcome={"trial": trial_json(trial)},
        rules_version="2026-11-04.1")

    item = idea_standard._numbers(facts)
    assert item.state == idea_standard.STATE_HAVE, item.reason


def test_a_stored_run_replays_into_the_same_selection_part(session, tmp_path):
    spec, result = _run(session, tmp_path)
    live = forward_series.load(result.trial_id)
    folder = tmp_path / "specs"
    folder.mkdir()
    (folder / f"{spec.idea_id}.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")

    replayed = forward_series.replay(session, result.trial_id, spec_root=folder)

    assert replayed.payload is not None, replayed.reason
    again = {p["part"]: p for p in replayed.payload["whole"]["parts"]}
    first = {p["part"]: p for p in live["whole"]["parts"]}
    assert again["selection"]["chosen_in_sample"] is True
    assert again["selection"]["total_return"] == pytest.approx(first["selection"]["total_return"])
    assert again["forward"]["total_return"] == pytest.approx(first["forward"]["total_return"])
    assert replayed.payload["whole"]["selection_numbers"] == pytest.approx(
        live["whole"]["selection_numbers"])


def test_a_series_built_before_the_revision_with_a_warmup_is_built_again(session, tmp_path):
    _spec, result = _run(session, tmp_path)
    payload = forward_series.load(result.trial_id)
    assert forward_series.is_current(payload)

    old = {**payload, "whole": {**payload["whole"], "revision": None, "parts": [
        {**p, "part": "warmup"} if p["part"] == "selection" else p
        for p in payload["whole"]["parts"]]}}
    old["whole"].pop("revision")

    assert not forward_series.is_current(old)
    assert forward_series.is_current({**old, "whole": {**old["whole"], "parts": [
        p for p in old["whole"]["parts"] if p["part"] != "warmup"]}})


def test_a_run_of_an_idea_with_a_choice_is_found_by_its_check_form(session, tmp_path,
                                                                    monkeypatch):
    """The registry keeps the check form (a Retune with `choose_at: split`) while the file holds
    the idea with its `choice`: the form is rebuilt from the file and matched to the row, the
    split day aside (it is put in when a run starts)."""
    form, result = _run(session, tmp_path)
    trial = session.get(Trial, result.trial_id)
    assert session.get(SpecRow, trial.spec_id).params["choose_at"] == "split"
    declared = base._month(session, tmp_path, params={"weight": 0.4}).model_copy(update={
        "choice": Choice(axes={"weight": [0.4, 0.2]}, source="night/own_numbers.yaml")})
    folder = tmp_path / "choice-specs"
    folder.mkdir()
    (folder / f"{declared.idea_id}.yaml").write_text(
        yaml.safe_dump(declared.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setattr("qlab.one_check.check_spec",
                        lambda spec, **kw: twin.TwinResult(form, None))

    assert forward_series.spec_file_for(session, trial.spec_id, folder) is None
    found = forward_series.stored_spec(session, trial.spec_id, folder)

    assert isinstance(found, StrategySpec) and found.params == form.params
    replayed = forward_series.replay(session, result.trial_id, spec_root=folder)
    assert replayed.payload is not None and replayed.payload["matches_trial"] is True
    # a file that is not this idea's check does not match
    other = form.model_copy(update={"params": {**form.params, "window": "30D"}})
    monkeypatch.setattr("qlab.one_check.check_spec",
                        lambda spec, **kw: twin.TwinResult(other, None))
    assert forward_series.stored_spec(session, trial.spec_id, folder) is None


def test_an_older_run_without_the_figures_gets_them_from_its_series(session, tmp_path):
    _spec, result = _run(session, tmp_path)
    trial = session.get(Trial, result.trial_id)
    trial.metrics = {k: v for k, v in trial.metrics.items() if not k.startswith("fit_")}

    shown = trial_json(trial)

    assert shown["headline"]["fit"] is not None  # read from the stored series
    facts = idea_standard.Facts(
        idea_id="toy", status="open", has_run=True, outcome={"trial": shown},
        rules_version="2026-11-04.1")
    assert idea_standard._numbers(facts).state == idea_standard.STATE_HAVE
