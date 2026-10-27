"""Tests for `qlab.waiting`: ideas that wait for data are re-run as the data arrives
(docs/NIGHT.md), on the same spec, once, and only on a day the route can change."""

from __future__ import annotations

import types
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

import qlab.registry.db as db_module
from qlab import night, waiting
from qlab.night import ItemResult
from qlab.pipeline import stage2
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.lifecycle import apply_route
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    Spec,
    TrialRoute,
    TrialStatus,
)
from qlab.registry.stages import NOTES_PREFIX
from qlab.waiting import Basis, days_before_decision, decide_due

END = date(2026, 10, 25)
DAY = date(2026, 10, 26)
SPLIT = {"fit_window_days": 200.0, "forward_days": 10.0, "forward_days_needed": 100.0,
         "regime_bull_days": 30.0, "regime_flat_days": 30.0, "regime_bear_days": 30.0}


def _basis(*, ago: int, route: str = "needs-forward", metrics: dict | None = None,
           **kw) -> Basis:
    return Basis(data_end=END - timedelta(days=ago), route=route,
                 metrics=SPLIT if metrics is None else metrics, **kw)


# ---- when a re-run can change anything (pure) -------------------------------------------


def test_until_the_forward_test_is_long_enough_the_run_decides_nothing_so_it_waits() -> None:
    # 10 of 100 forward days: the route cannot change before 90 more days have come.
    assert days_before_decision(SPLIT, "needs-forward", 30) == 90
    assert decide_due(_basis(ago=89), END, coverage_days=30).due is False
    due = decide_due(_basis(ago=90), END, coverage_days=30)
    assert due.due is True and due.due_end == END


def test_a_day_without_new_data_is_never_due() -> None:
    due = decide_due(_basis(ago=0, metrics={}), END)
    assert due.due is False and due.due_end == END + timedelta(days=1)


def test_every_regime_must_have_its_days_and_each_new_day_fills_only_one() -> None:
    metrics = {**SPLIT, "forward_days": 99.0, "regime_bull_days": 5.0, "regime_flat_days": 30.0,
               "regime_bear_days": 20.0}
    # length needs 1 more day; coverage needs 25 + 0 + 10 = 35 days at the very least
    assert days_before_decision(metrics, "needs-forward", 30) == 35
    assert days_before_decision(metrics, "needs-forward", None) == 1


def test_regime_coverage_bounds_a_forward_test_that_has_no_selection_period_too() -> None:
    # parameters chosen causally: the forward test is the whole window, no length is asked,
    # but a decision still needs every regime for 30 days (15 days short in all)
    causal = {"forward_days": 175.0, "regime_bull_days": 25.0, "regime_flat_days": 30.0,
              "regime_bear_days": 20.0}
    assert days_before_decision(causal, "needs-forward", 30) == 15
    assert days_before_decision({**causal, "forward_days": 0.0}, "needs-forward", 30) == 0


def test_a_run_with_nothing_to_wait_for_is_due_after_one_new_day() -> None:
    # needs-more-data, no forward test yet, a forward test without a selection period
    # (it can reject at any length): the first day that arrives may change the route
    assert days_before_decision(SPLIT, "needs-more-data", 30) == 0
    assert days_before_decision({**SPLIT, "forward_days": 0.0}, "needs-forward", 30) == 0
    no_selection = {k: v for k, v in SPLIT.items() if k != "fit_window_days"}
    assert days_before_decision(no_selection, "needs-forward", 30) == 0
    assert decide_due(_basis(ago=1, route="needs-more-data"), END).due is True


def test_a_missing_metric_that_two_runs_on_different_data_did_not_supply_waits_for_new_rules(
) -> None:
    stuck = _basis(ago=1, route="needs-more-data", stuck=True, rules_version="a")
    due = decide_due(stuck, END, current_rules="a")
    assert due.due is False and "смены правил" in due.why
    assert decide_due(stuck, END, current_rules="b").due is True  # a new rule set may compute it


def test_a_crashed_run_a_missing_run_and_a_new_rule_set_are_due_at_once() -> None:
    assert decide_due(_basis(ago=1, failed=True), END).due is True
    assert decide_due(Basis(None, None), END).due is True
    changed = decide_due(_basis(ago=1, rules_version="a"), END, current_rules="b")
    assert changed.due is True and "набор правил" in changed.why
    assert decide_due(_basis(ago=1, rules_version="a"), END, current_rules="a").due is False


# ---- the registry fixture -----------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    Base.metadata.create_all(db_module.get_engine())
    yield db_module
    db_module._engine = None
    db_module._SessionLocal = None


def _idea(session, idea_id: str, status: IdeaStatus, *, parent: str | None = None,
          notes: str | None = None) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     notes=notes)
    session.get(Idea, idea_id).parent_id = parent


def _trial(session, idea_id: str, route: str | None, *, ago: int,
           metrics: dict | None = None, status: TrialStatus = TrialStatus.OK,
           reason: str | None = None):
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={}, data_requirements={}, rebalance="1d",
        costs_model={}, code_ref="a:b")
    end = END - timedelta(days=ago)
    snapshot = f"snap-{end}"
    if session.get(DataSnapshot, snapshot) is None:
        repo.add_data_snapshot(session, id=snapshot, source="hyperliquid", instruments={},
                               range_start=date(2025, 1, 1), range_end=end, path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                          status=status, snapshot_id=snapshot,
                          metrics=SPLIT if metrics is None else metrics,
                          route=TrialRoute(route) if route else None, route_reason=reason)


def _spec(idea_id: str, tmp_path: Path, *, source: str = "hyperliquid") -> Path:
    spec = StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": source, "interval": "1d", "start": "2025-01-01", "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})
    path = tmp_path / "specs" / f"{idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")
    return path


STAGE2_CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))


def _world(tmp_path, registry) -> None:
    """Ideas in every waiting state the stage tells apart."""
    with registry.session_scope() as s:
        # waits for data, ripe: 10 of 100 forward days at END - 90
        _idea(s, "ripe", IdeaStatus.BENCH)
        _trial(s, "ripe", "needs-forward", ago=90)
        # waits for data, one new day only: far from its day
        _idea(s, "fresh", IdeaStatus.BENCH)
        _trial(s, "fresh", "needs-forward", ago=1)
        # a metric was missing: the next day may fix it
        _idea(s, "no-metric", IdeaStatus.CANDIDATE)
        _trial(s, "no-metric", "needs-more-data", ago=1, metrics={})
        # already run on the last closed day (the researcher declared and ran it tonight)
        _idea(s, "same-day", IdeaStatus.BENCH)
        _trial(s, "same-day", "needs-forward", ago=0)
        # the watch list runs this one
        _idea(s, "on-the-watch-list", IdeaStatus.BENCH)
        _trial(s, "on-the-watch-list", "needs-forward", ago=90)
        # decided: waits for capital, for the owner, or was thrown out
        _idea(s, "shelved", IdeaStatus.BENCH)
        _trial(s, "shelved", "shelf", ago=90)
        _idea(s, "passed", IdeaStatus.VALIDATED)
        _trial(s, "passed", "paper", ago=90)
        _idea(s, "thrown-out", IdeaStatus.REJECTED)
        _trial(s, "thrown-out", "needs-forward", ago=90)
        # nobody can re-run it: a bench idea without a spec file; and a noise book
        _idea(s, "no-spec", IdeaStatus.BENCH)
        _trial(s, "no-spec", "needs-forward", ago=90)
        _idea(s, "noise-book", IdeaStatus.CANDIDATE)
        _trial(s, "noise-book", "needs-forward", ago=90)
    for name in ("ripe", "fresh", "no-metric", "same-day", "on-the-watch-list", "shelved"):
        _spec(name, tmp_path)


def _plans(tmp_path, registry, **kw):
    with registry.session_scope() as s:
        return {(p.idea_id, p.step): p for p in waiting.plan_waiting(
            s, end=END, watched={"on-the-watch-list"}, specs=waiting.spec_files(tmp_path / "specs"),
            coverage_days=30, **kw)}


def test_only_ideas_that_wait_for_data_and_are_nobodys_job_yet_are_planned(
        tmp_path, registry) -> None:
    _world(tmp_path, registry)
    plans = _plans(tmp_path, registry)
    assert sorted(i for i, _ in plans) == ["fresh", "no-metric", "no-spec", "ripe", "same-day"]
    assert plans[("ripe", "idea")].due and plans[("no-metric", "idea")].due
    assert not plans[("fresh", "idea")].due and not plans[("same-day", "idea")].due
    assert plans[("fresh", "idea")].due_end == END - timedelta(days=1) + timedelta(days=90)
    assert plans[("no-spec", "idea")].skip == "нет файла спека" and not plans[
        ("no-spec", "idea")].due


def test_stuck_is_read_from_two_runs_with_the_same_reason_on_different_data(
        tmp_path, registry) -> None:
    reason = "metric(s) could not be computed: atomic_execution"
    with registry.session_scope() as s:
        for idea, first in (("stuck", reason), ("moving", "another reason")):
            _idea(s, idea, IdeaStatus.CANDIDATE)
            _trial(s, idea, "needs-more-data", ago=5, metrics={}, reason=first)
            _trial(s, idea, "needs-more-data", ago=3, metrics={}, reason=reason)
        _idea(s, "same-data", IdeaStatus.CANDIDATE)  # run again on the very same data
        _trial(s, "same-data", "needs-more-data", ago=3, metrics={}, reason=reason)
        _trial(s, "same-data", "needs-more-data", ago=3, metrics={}, reason=reason)
        _idea(s, "first-run", IdeaStatus.CANDIDATE)
        _trial(s, "first-run", "needs-more-data", ago=3, metrics={}, reason=reason)
    for name in ("stuck", "moving", "same-data", "first-run"):
        _spec(name, tmp_path)
    plans = _plans(tmp_path, registry)
    assert not plans[("stuck", "idea")].due
    assert plans[("moving", "idea")].due and plans[("same-data", "idea")].due
    assert plans[("first-run", "idea")].due


# ---- the stage -------------------------------------------------------------------------


@pytest.fixture()
def stand(monkeypatch):
    """The stand and the second stage replaced by recorders; the rule set by a stub."""
    calls: list[tuple] = []
    results: dict[str, ItemResult] = {}

    def fake_eval(kind, spec, *, end, noise_trials, deployable_capital_usd):
        calls.append(("idea", spec.idea_id, end, spec.params))
        return results.get(spec.idea_id) or ItemResult(
            kind=kind, name=spec.idea_id, trial_id=900, route="needs-forward",
            previous_route="needs-forward",
            metrics={"forward_days": 100.0, "forward_days_needed": 100.0})

    s2: dict[str, stage2.Stage2Result] = {}

    def fake_stage2(spec, *, stage1_route, stage1_trial_id, noise_trials, deployable_capital_usd,
                    config=None, **kw):
        calls.append(("stage2", spec.idea_id, spec.data.end, stage1_route, stage1_trial_id))
        return s2.get(spec.idea_id) or stage2.Stage2Result(
            parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl", status="evaluated",
            route="needs-forward", trial_id=950, metrics={"forward_days": 20.0,
                                                          "forward_days_needed": 60.0})

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    monkeypatch.setattr(stage2, "run_stage2", fake_stage2)
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    monkeypatch.setattr("qlab.rules.loader.load_latest",
                        lambda: types.SimpleNamespace(regime_coverage_days=30, version="r1"))
    return types.SimpleNamespace(calls=calls, results=results, stage2=s2)


def _stage(tmp_path, state=None) -> dict:
    state = state if state is not None else {"done": {}}
    on_the_watch_list = tmp_path / "specs" / "on-the-watch-list.yaml"
    waiting.waiting_stage(DAY, END, state, noise_trials=1, spec_dir=tmp_path / "specs",
                          watch=[str(on_the_watch_list)] if on_the_watch_list.is_file() else [])
    return state


def test_an_idea_that_gained_its_days_is_re_run_on_the_fresh_tail_with_the_same_spec(
        tmp_path, registry, stand) -> None:
    _world(tmp_path, registry)
    state = _stage(tmp_path)
    ran = [c for c in stand.calls if c[0] == "idea"]
    assert sorted(c[1] for c in ran) == ["no-metric", "ripe"]
    assert all(c[3] == {"lookback": 14} for c in ran)  # the spec is the file's, untouched
    rows = {e["idea"]: e for e in state["waiting"]}
    assert rows["ripe"]["action"] == "rerun" and rows["ripe"]["forward_days"] == 100.0
    assert rows["ripe"]["forward_days_needed"] == 100.0


def test_an_idea_with_too_few_new_days_is_skipped_and_its_days_come_from_the_calendar(
        tmp_path, registry, stand) -> None:
    _world(tmp_path, registry)
    state = _stage(tmp_path)
    assert "fresh" not in [c[1] for c in stand.calls]
    row = next(e for e in state["waiting"] if e["idea"] == "fresh")
    assert row["action"] == "wait" and row["forward_days"] == 11.0  # 10 + the one new day
    assert row["due_end"] == (END - timedelta(days=1) + timedelta(days=90)).isoformat()
    text = "\n".join(waiting.report_lines(state["waiting"]))
    assert "11 (по календарю) из 100" in text and "ещё 89 дн." in text


def test_the_watch_list_and_a_run_of_the_same_night_are_not_run_twice(
        tmp_path, registry, stand) -> None:
    _world(tmp_path, registry)
    state = _stage(tmp_path)
    first = list(stand.calls)
    assert "on-the-watch-list" not in [c[1] for c in first]  # the watch list's own run
    assert "same-day" not in [c[1] for c in first]  # its latest trial reaches END already
    _stage(tmp_path, state)  # the night is restarted: nothing is re-run
    assert stand.calls == first


def test_a_failed_rerun_is_tried_again_and_the_finished_one_is_not(
        tmp_path, registry, stand) -> None:
    _world(tmp_path, registry)
    stand.results["ripe"] = ItemResult(kind="waiting", name="ripe", error="Boom: no data")
    state = _stage(tmp_path)
    assert state["done"].keys() == {"waiting:no-metric:idea"}
    stand.calls.clear()
    del stand.results["ripe"]
    _stage(tmp_path, state)
    assert [c[1] for c in stand.calls] == ["ripe"]


def test_a_trial_the_stand_routed_error_is_a_failure_tried_again(tmp_path, registry, stand) -> None:
    # 2026-10-27: the stand caught a strategy's exception and routed its trial "error"; the
    # night counted it re-run and done, so it was never tried again.
    _world(tmp_path, registry)
    stand.results["ripe"] = ItemResult(kind="waiting", name="ripe", trial_id=9, route="error",
                                       reason="int() argument must be ... not 'list'")
    state = _stage(tmp_path)
    row = next(e for e in state["waiting"] if e["idea"] == "ripe")
    assert row["action"] == "error" and "not 'list'" in row["error"]
    assert "waiting:ripe:idea" not in state["done"]


def test_a_route_change_is_applied_by_code_and_the_report_names_it(
        tmp_path, registry, stand, monkeypatch) -> None:
    _world(tmp_path, registry)

    def reject(kind, spec, *, end, noise_trials, deployable_capital_usd):
        # What the stand does on a run: the trial, then the lifecycle moves the status.
        with registry.session_scope() as s:
            trial = _trial(s, spec.idea_id, "reject", ago=0)
            apply_route(s, idea_id=spec.idea_id, route=TrialRoute.REJECT, route_reason="x",
                        trial_id=trial.id, rules_version="r1", run_data_end=end)
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=trial.id, route="reject",
                          previous_route="needs-forward", reason="forward test failed",
                          metrics={"forward_days": 100.0, "forward_days_needed": 100.0})

    monkeypatch.setattr(night, "evaluate_item", reject)
    state = _stage(tmp_path)
    row = next(e for e in state["waiting"] if e["idea"] == "ripe")
    assert (row["route_before"], row["route_after"], row["status"]) == (
        "needs-forward", "reject", "rejected")
    text = "\n".join(waiting.report_lines(state["waiting"]))
    assert "needs-forward → reject **(изменился)**" in text
    with registry.session_scope() as s:
        assert s.get(Idea, "ripe").status == IdeaStatus.REJECTED
    # a rejected idea waits for nothing: the next night does not plan it
    assert ("ripe", "idea") not in _plans(tmp_path, registry)


# ---- generated ideas and their second stage -------------------------------------------------


def _generated(tmp_path, registry, *, stage1: str, child: str | None, child_ago: int = 60,
               stage1_ago: int = 60, status: IdeaStatus = IdeaStatus.BENCH) -> int:
    """A generated idea with stage 1's latest trial (its id is returned) and, maybe, a child."""
    with registry.session_scope() as s:
        _idea(s, "gen-x", status)
        stage1_trial = _trial(s, "gen-x", stage1, ago=stage1_ago).id
        if child is not None:
            _idea(s, "gen-x-hl", IdeaStatus.BENCH, parent="gen-x",
                  notes=f"{NOTES_PREFIX}gen-x (docs/TWO_STAGE.md)")
            _trial(s, "gen-x-hl", child, ago=child_ago,
                   metrics={**SPLIT, "forward_days": 40.0, "forward_days_needed": 60.0})
    _spec("gen-x", tmp_path, source="binance")
    return stage1_trial


def test_a_second_stage_waiting_for_hyperliquid_data_is_re_run_on_the_parents_spec(
        tmp_path, registry, stand) -> None:
    stage1_trial = _generated(tmp_path, registry, stage1="paper", child="needs-forward")
    stand.stage2["gen-x"] = stage2.Stage2Result(
        parent_id="gen-x", idea_id="gen-x-hl", status="evaluated", route="reject", trial_id=951,
        combined_route="reject", parent_status="rejected", data_end=END.isoformat(),
        metrics={"forward_days": 60.0, "forward_days_needed": 60.0})
    state = _stage(tmp_path)
    # stage 1 passed (`paper`): it is not re-run; stage 2 is, on the spec extended to END
    # and with stage 1's outcome and trial from the registry
    assert [c[0] for c in stand.calls] == ["stage2"]
    assert stand.calls[0][1:] == ("gen-x", END, "paper", stage1_trial)
    row = state["waiting"][0]
    assert (row["step"], row["route_before"], row["route_after"]) == ("stage2", "needs-forward",
                                                                      "reject")
    assert row["combined"] == "reject" and row["status"] == "rejected"
    text = "\n".join(waiting.report_lines(state["waiting"]))
    assert "второй этап (Hyperliquid)" in text and "(изменился)" in text
    assert "по двум этапам reject" not in text  # the combined route equals the stage 2 one


def test_a_second_stage_with_too_few_new_days_waits(tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="paper", child="needs-forward", child_ago=1)
    state = _stage(tmp_path)
    assert stand.calls == []
    row = state["waiting"][0]
    assert (row["step"], row["action"], row["forward_days"]) == ("stage2", "wait", 41.0)


def test_a_first_stage_that_still_waits_is_re_run_and_then_its_second_stage(
        tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="needs-forward", child=None, stage1_ago=90)
    stand.results["gen-x"] = ItemResult(kind="waiting", name="gen-x", trial_id=777, route="paper",
                                        previous_route="needs-forward",
                                        metrics={"forward_days": 100.0,
                                                 "forward_days_needed": 100.0})
    state = _stage(tmp_path)
    assert [c[:2] for c in stand.calls] == [("idea", "gen-x"), ("stage2", "gen-x")]
    assert stand.calls[1][3:] == ("paper", 777)  # stage 2 judges the fresh stage 1 outcome
    assert {e["step"] for e in state["waiting"]} == {"idea", "stage2"}


def test_a_second_stage_is_not_run_while_stage_1_decided_nothing(
        tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="needs-more-data", child=None)
    stand.results["gen-x"] = ItemResult(kind="waiting", name="gen-x", trial_id=900,
                                        route="needs-more-data", previous_route="needs-more-data")
    _stage(tmp_path)
    assert [c[0] for c in stand.calls] == ["idea"]  # only the first stage is retried


def test_an_idea_whose_rerun_makes_it_a_survivor_gets_its_second_stage_the_same_night(
        tmp_path, registry, stand) -> None:
    # stuck on a missing metric, the re-run supplies it and the idea survives stage 1: the plan
    # was made from the old route, but its second stage does not wait for another night
    _generated(tmp_path, registry, stage1="needs-more-data", child=None)
    stand.results["gen-x"] = ItemResult(kind="waiting", name="gen-x", trial_id=901,
                                        route="paper", previous_route="needs-more-data")
    state = _stage(tmp_path)
    assert [c[0] for c in stand.calls] == ["idea", "stage2"]
    assert stand.calls[1][3:] == ("paper", 901)  # told the route and trial of tonight's run
    assert [(e["idea"], e["step"]) for e in state["waiting"]] == [("gen-x", "idea"),
                                                                  ("gen-x", "stage2")]
    assert "waiting:gen-x:stage2" in state["done"]


def test_a_second_stage_switched_off_in_its_config_is_not_run(
        tmp_path, registry, stand, monkeypatch) -> None:
    _generated(tmp_path, registry, stage1="paper", child="needs-forward")
    monkeypatch.setattr(stage2, "load_config", lambda: stage2.Stage2Config(
        **{**STAGE2_CONFIG.__dict__, "enabled": False}))
    _stage(tmp_path)
    assert stand.calls == []


def test_a_stage_two_that_never_finished_on_a_validated_idea_is_run(
        tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="paper", child=None, status=IdeaStatus.VALIDATED)
    state = _stage(tmp_path)
    assert [c[0] for c in stand.calls] == ["stage2"]
    assert state["waiting"][0]["why"] == "второй этап ещё не проведён"


def test_a_stage_two_error_is_tried_again_on_the_next_run(tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="paper", child="needs-forward")
    stand.stage2["gen-x"] = stage2.Stage2Result(parent_id="gen-x", idea_id="gen-x-hl",
                                                status="error", error="Hyperliquid is down")
    state = _stage(tmp_path)
    assert state["waiting"][0]["action"] == "error" and state["done"] == {}
    del stand.stage2["gen-x"]
    _stage(tmp_path, state)
    assert [c[0] for c in stand.calls] == ["stage2", "stage2"]


def test_a_crash_in_one_rerun_or_in_the_planning_is_a_report_line_not_a_failed_night(
        tmp_path, registry, stand, monkeypatch) -> None:
    _world(tmp_path, registry)

    def boom(kind, spec, **kw):
        if spec.idea_id == "ripe":
            raise RuntimeError("disk full")
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=1, route="needs-forward")

    monkeypatch.setattr(night, "evaluate_item", boom)
    state = _stage(tmp_path)
    rows = {e["idea"]: e for e in state["waiting"]}
    assert rows["ripe"]["action"] == "error" and "disk full" in rows["ripe"]["error"]
    assert rows["no-metric"]["action"] == "rerun"  # the others went on
    assert "waiting:ripe:idea" not in state["done"]

    def no_plan(*a, **kw):
        raise OSError("registry is locked")

    monkeypatch.setattr(waiting, "plan_waiting", no_plan)
    broken = _stage(tmp_path, {"done": {}})
    assert "registry is locked" in broken["waiting_notes"][0] and "waiting" not in broken
    assert "registry is locked" in "\n".join(waiting.report_lines([], broken["waiting_notes"]))


def test_a_second_stage_that_stage_1_no_longer_lets_through_waits_with_the_reason(
        tmp_path, registry, stand) -> None:
    _generated(tmp_path, registry, stage1="paper", child="needs-forward")
    stand.stage2["gen-x"] = stage2.Stage2Result(
        parent_id="gen-x", idea_id="gen-x-hl",
        skipped="stage 1 routed reject: only survivors go on")
    state = _stage(tmp_path)
    row = state["waiting"][0]
    assert row["action"] == "wait" and "only survivors go on" in row["why"]


# ---- the night --------------------------------------------------------------------------


def test_the_morning_report_lists_what_was_re_run_and_leaves_followed_ideas_off_the_idle_list(
        tmp_path, registry, stand, monkeypatch) -> None:
    _world(tmp_path, registry)
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: [specs/on-the-watch-list.yaml]\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "unwatched_bench", lambda w: ["ripe", "no-spec", "fresh"])
    monkeypatch.chdir(tmp_path)  # the stage reads `specs/` from the working directory
    path = night.run(DAY, graveyard=False, paper=False, implement=False, search=False,
                     detectors=False, assess=False, research=False)
    text = path.read_text(encoding="utf-8")
    assert f"## {waiting.SECTION_TITLE}" in text
    assert "| ripe | описание на свежем хвосте |" in text and "| fresh |" in text
    idle = text.split("## На скамейке, но не в списке наблюдения")[1]
    assert "- no-spec" in idle and "- ripe" not in idle and "- fresh" not in idle
    # the watch list's own idea once (by the watch list), the waiting ones by this stage
    assert sorted(c[1] for c in stand.calls) == ["no-metric", "on-the-watch-list", "ripe"]


def test_the_stage_spends_no_tokens() -> None:
    import inspect

    source = inspect.getsource(waiting)
    assert "qlab.budget" not in source and "qlab.agents" not in source


def test_an_idea_whose_last_run_crashed_stays_in_the_plan_and_is_due(tmp_path, registry) -> None:
    _world(tmp_path, registry)
    with registry.session_scope() as s:
        _trial(s, "fresh", "error", ago=0, metrics=None, status=TrialStatus.ERROR)
    plan = _plans(tmp_path, registry)[("fresh", "idea")]
    assert plan.due and plan.why == "прошлый прогон не дал исхода"


# ---- experiments on Binance's history, twins, a hand run ---------------------------------------


def _experiment(tmp_path: Path, idea_id: str, source: str, folder: str = "research/x") -> Path:
    """A spec file of `idea_id` on `source`, in a folder of `specs/`."""
    path = tmp_path / "specs" / folder / f"{idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": source, "interval": "1d", "start": "2025-01-01", "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")
    return path


def test_a_researchers_experiment_on_binance_that_survived_stage_1_gets_its_second_stage(
        tmp_path, registry) -> None:
    with registry.session_scope() as s:
        for idea in ("exp-binance", "exp-hl", "plain-binance", "exp-binance-failed"):
            _idea(s, idea, IdeaStatus.BENCH)
            _trial(s, idea, "needs-forward" if idea != "exp-binance-failed" else "needs-more-data",
                   ago=90)
    _experiment(tmp_path, "exp-binance", "binance")
    _experiment(tmp_path, "exp-hl", "hyperliquid")
    _experiment(tmp_path, "exp-binance-failed", "binance")
    _experiment(tmp_path, "plain-binance", "binance", folder=".")
    plans = _plans(tmp_path, registry, stage2_config=STAGE2_CONFIG)
    assert ("exp-binance", "stage2") in plans
    assert plans[("exp-binance", "stage2")].due
    assert plans[("exp-binance", "stage2")].why == "второй этап ещё не проведён"
    assert ("exp-hl", "stage2") not in plans  # it already runs on Hyperliquid
    assert ("plain-binance", "stage2") not in plans  # not a researcher's experiment
    assert ("exp-binance-failed", "stage2") not in plans  # stage 1 decided nothing
    assert ("exp-binance", "stage2") not in _plans(tmp_path, registry)  # stage 2 switched off


def test_a_twin_is_an_ordinary_idea_for_the_waiting_plan(tmp_path, registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "exp-causal", IdeaStatus.BENCH)
        _trial(s, "exp-causal", "needs-forward", ago=90)
    _spec("exp-causal", tmp_path)
    plans = _plans(tmp_path, registry)
    assert plans[("exp-causal", "idea")].due and plans[("exp-causal", "idea")].skip is None


def test_a_hand_run_limits_the_plan_to_the_named_ideas_and_forces_them(
        tmp_path, registry, stand) -> None:
    _world(tmp_path, registry)
    state = {"done": {}}
    waiting.waiting_stage(DAY, END, state, noise_trials=1, spec_dir=tmp_path / "specs", watch=[],
                          only={"fresh", "on-the-watch-list"}, force=True)
    assert sorted(c[1] for c in stand.calls if c[0] == "idea") == ["fresh", "on-the-watch-list"]
    rows = {e["idea"]: e for e in state["waiting"]}
    assert set(rows) == {"fresh", "on-the-watch-list"} and rows["fresh"]["action"] == "rerun"
    # without force the day that can change the route has not come for `fresh`
    stand.calls.clear()
    other = {"done": {}}
    waiting.waiting_stage(DAY, END, other, noise_trials=1, spec_dir=tmp_path / "specs", watch=[],
                          only={"fresh"})
    assert stand.calls == [] and other["waiting"][0]["action"] == "wait"
