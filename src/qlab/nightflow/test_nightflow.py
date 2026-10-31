"""The night as steps (docs/PIPELINE.md): the table, the plan, the runner, the journal, the lock,
the rebuilt history and the page's routes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import qlab.night as night
from qlab.nightflow import backfill, control, journal
from qlab.nightflow.plan import make_plan
from qlab.nightflow.runner import ItemTracker, Runner, StepImpl, failed_items
from qlab.nightflow.table import TableError, load_table, parse
from qlab.registry.models import Base, PipelineItem, PipelineRun, PipelineStepRun, TokenSpend


@pytest.fixture()
def factory():
    eng = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)

    @event.listens_for(eng, "connect")
    def _fk(dbapi, _record):
        cursor = dbapi.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, expire_on_commit=False)
    eng.dispose()


@pytest.fixture()
def jr(factory):
    @contextmanager
    def scope():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return journal.Journal(scope)


def _small() -> dict:
    """a -> b (data) -> c (data); a -> d (fresh); e alone; f depends on b by budget."""
    def step(id_, parents=(), **extra):
        return {"id": id_, "title": id_.upper(), "description": f"step {id_}", "kind": "code",
                "tokens": False, "depends_on": [
                    {"step": p, "kind": k, "why": f"{id_} needs {p}"} for p, k in parents], **extra}
    return {"steps": [step("a"), step("b", [("a", "data")]), step("c", [("b", "data")]),
                      step("d", [("a", "fresh")]), step("e"), step("f", [("b", "budget")])]}


# ---------------------------------------------------------------------------------- the table


def test_the_shipped_table_is_valid_and_every_step_has_an_implementation() -> None:
    table = load_table()
    assert set(table.ids) == set(night.night_steps())
    assert table.ids[0] == "detectors" and table.ids[-1] == "report"
    for step in table.steps:
        assert step.title and step.description
        for edge in step.depends_on:
            assert edge.why and table.ids.index(edge.step) < table.ids.index(step.id)
    # the report reads everything and nothing reads it
    assert not table.children("report")
    # only `data` edges block; the agents' order is a budget order, not data
    assert {e.kind for e in table.by_id["programmer"].depends_on} == {"budget"}
    assert table.by_id["recheck"].hard_parents == ("detectors",)


def test_the_shipped_table_says_which_steps_spend_tokens() -> None:
    table = load_table()
    spenders = {s.id for s in table.steps if s.tokens}
    assert {"assess", "programmer", "director", "research", "implement", "search", "catalogue",
            "cards", "generate", "anomalies-investigate"} == spenders
    assert all(s.kind == "agent" for s in table.steps if s.tokens)


def test_table_rejects_a_parent_below_its_child_an_unknown_one_and_a_repeat() -> None:
    doc = _small()
    doc["steps"][0]["depends_on"] = [{"step": "c", "kind": "data", "why": "x"}]
    with pytest.raises(TableError, match="not declared above"):
        parse(doc)
    doc = _small()
    doc["steps"][1]["depends_on"][0]["step"] = "zzz"
    with pytest.raises(TableError, match="zzz"):
        parse(doc)
    doc = _small()
    doc["steps"].append(dict(doc["steps"][0]))
    with pytest.raises(TableError, match="twice"):
        parse(doc)
    doc = _small()
    doc["steps"][1]["depends_on"][0]["kind"] = "maybe"
    with pytest.raises(TableError, match="maybe"):
        parse(doc)
    doc = _small()
    doc["steps"][1]["depends_on"][0]["why"] = " "
    with pytest.raises(TableError, match="no reason"):
        parse(doc)


def test_descendants_waves_and_who_can_run_together() -> None:
    table = parse(_small())
    assert table.descendants("a", ("data",)) == ["b", "c"]
    assert table.descendants("a") == ["b", "c", "d", "f"]
    assert table.layers() == {"a": 0, "b": 1, "c": 2, "d": 1, "e": 0, "f": 2}
    assert table.layers(("data", "fresh")) == {"a": 0, "b": 1, "c": 2, "d": 1, "e": 0, "f": 0}
    assert table.can_run_with("c") == ["d", "e", "f"]
    assert table.can_run_with("e") == ["a", "b", "c", "d", "f"]
    # a restart runs the step and what depends on it by data, never the budget order's followers
    assert table.restart_set("b") == ["b", "c"]


def test_restart_adds_the_steps_that_only_read() -> None:
    table = load_table()
    assert table.restart_set("recheck") == ["recheck", "assess", "ladder", "report"]
    assert table.restart_set("research") == ["research", "twins-research", "ladder", "report"]


def test_reads_all_makes_a_fresh_edge_to_every_step_above() -> None:
    report = load_table().by_id["report"]
    assert {e.kind for e in report.depends_on} == {"fresh"}
    assert len(report.depends_on) == len(load_table().ids) - 1


# ---------------------------------------------------------------------------------- the plan


def test_a_switched_off_step_blocks_its_data_children_unless_it_was_done_today() -> None:
    table = parse(_small())
    plan = {p.step: p for p in make_plan(table, off={"a"})}
    assert plan["a"].action == "off"
    assert plan["b"].action == "blocked" and plan["b"].blocked_by == "a"
    assert plan["c"].action == "blocked" and plan["c"].blocked_by == "a"  # the root cause
    assert plan["d"].action == "run" and plan["e"].action == "run" and plan["f"].action == "run"
    done = {p.step: p for p in make_plan(table, off={"a"}, done_today={"a"})}
    assert done["b"].action == "run" and done["c"].action == "run"


def test_a_partial_run_leaves_the_rest_outside_and_does_not_block_on_it() -> None:
    table = parse(_small())
    plan = {p.step: p.action for p in make_plan(table, only={"b", "c"})}
    assert plan == {"a": "outside", "b": "run", "c": "run", "d": "outside", "e": "outside",
                    "f": "outside"}
    with pytest.raises(ValueError, match="unknown"):
        make_plan(table, off={"nope"})


# ---------------------------------------------------------------------------------- the runner


class Ctx:
    def __init__(self) -> None:
        self.state: dict = {"done": {}}
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _impls(table, ran, *, fail=(), skip=(), cut=(), boom=BaseException) -> dict:
    def make(step_id):
        def run(ctx):
            ran.append(step_id)
            if step_id in fail:
                raise RuntimeError(f"{step_id} broke")
        return StepImpl(
            run, step_id, skip_reason=(lambda c: "не сегодня") if step_id in skip else None,
            is_cut=(lambda c: True) if step_id in cut else None)
    return {s: make(s) for s in table.ids}


def _states(factory, run_id):
    session = factory()
    rows = session.query(PipelineStepRun).filter_by(run_id=run_id).all()
    return {r.step_id: r for r in rows}


def test_a_failed_step_blocks_only_its_data_children_and_the_rest_run(jr, factory) -> None:
    table = parse(_small())
    ran: list[str] = []
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=1, pid=1)
    out = Runner(table, _impls(table, ran, fail={"a"}), Ctx(), journal=jr, run_id=run_id,
                 log=lambda t: None).execute()
    assert ran == ["a", "d", "e", "f"]  # b and c blocked; d reads a freshly but is not blocked
    assert out["a"] == "failed" and out["b"] == "blocked" and out["c"] == "blocked"
    assert out["d"] == "ok" and out["e"] == "ok"
    assert out["f"] == "ok"  # a budget-only edge never blocks (its parent b was blocked, not data)
    rows = _states(factory, run_id)
    assert rows["b"].blocked_by == "a" and rows["c"].blocked_by == "a"
    assert rows["a"].error.startswith("RuntimeError: a broke (test_nightflow.py:")
    assert rows["a"].duration_s is not None and rows["e"].state == "ok"


def test_a_switched_off_step_is_recorded_and_blocks_like_a_failure(jr, factory) -> None:
    table = parse(_small())
    ran: list[str] = []
    run_id = jr.start_run(date(2026, 10, 31), trigger="button", steps_off=["a"], only=None,
                          restart=None, concurrency=1, pid=1)
    out = Runner(table, _impls(table, ran), Ctx(), journal=jr, run_id=run_id, off={"a"},
                 log=lambda t: None).execute()
    assert ran == ["d", "e", "f"] and out["a"] == "off" and out["b"] == "blocked"
    assert _states(factory, run_id)["b"].blocked_by == "a"


def test_skipped_and_cut_steps_do_not_block_anything(jr, factory) -> None:
    table = parse(_small())
    ran: list[str] = []
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=1, pid=1)
    out = Runner(table, _impls(table, ran, skip={"a"}, cut={"e"}), Ctx(), journal=jr,
                 run_id=run_id, log=lambda t: None).execute()
    assert out["a"] == "skipped" and out["e"] == "cut" and out["b"] == "ok" and out["c"] == "ok"
    assert _states(factory, run_id)["a"].note == "не сегодня"


def test_a_stop_of_the_night_is_a_row_and_still_a_stop(jr, factory) -> None:
    table = parse(_small())
    impls = _impls(table, [])

    def stop(ctx):
        raise KeyboardInterrupt

    impls["b"] = StepImpl(stop, "b")
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=1, pid=1)
    with pytest.raises(KeyboardInterrupt):
        Runner(table, impls, Ctx(), journal=jr, run_id=run_id, log=lambda t: None).execute()
    assert _states(factory, run_id)["b"].state == "interrupted"


def test_the_row_exists_while_the_step_runs(jr, factory) -> None:
    """Written live: a step in progress is a `running` row with a start time."""
    table = parse(_small())
    seen = {}
    impls = _impls(table, [])

    def look(ctx):
        seen.update({k: (v.state, v.started_at is not None, v.finished_at)
                     for k, v in _states(factory, run_id).items()})

    impls["b"] = StepImpl(look, "b")
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=1, pid=1)
    Runner(table, impls, Ctx(), journal=jr, run_id=run_id, log=lambda t: None).execute()
    assert seen["b"] == ("running", True, None) and seen["a"][0] == "ok"


def test_the_scheduler_for_more_than_one_step_keeps_every_edge(jr) -> None:
    table = parse(_small())
    order: list[str] = []
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=3, pid=1)
    out = Runner(table, _impls(table, order, fail={"a"}), Ctx(), journal=jr, run_id=run_id,
                 concurrency=3, log=lambda t: None).execute()
    assert out["b"] == "blocked" and out["c"] == "blocked" and out["e"] == "ok"
    assert order.index("a") < order.index("d") and "b" not in order


def test_items_come_from_the_done_markers_as_the_state_is_saved(jr, factory) -> None:
    table = parse({"steps": [{"id": "w", "title": "W", "description": "d", "kind": "code",
                              "tokens": False, "done_prefixes": ["watch:"]}]})
    ctx = Ctx()
    step_run = jr.start_step(jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[],
                                          only=None, restart=None, concurrency=1, pid=1), "w")
    tracker = ItemTracker(table, jr)
    ctx.state["done"]["watch:old.yaml"] = "2026-10-31T00:00:00+00:00"  # earlier work: not counted
    tracker.begin(table.by_id["w"], step_run, ctx.state)
    start = datetime.now(UTC)
    ctx.state["done"]["watch:a.yaml"] = (start + timedelta(seconds=30)).isoformat()
    ctx.state["done"]["other:x"] = start.isoformat()
    tracker.observe(ctx.state)
    ctx.state["done"]["watch:b.yaml"] = (start + timedelta(seconds=90)).isoformat()
    tracker.end("w", ctx.state)
    items = journal.step_items(factory(), step_run)
    assert [i["name"] for i in items] == ["a.yaml", "b.yaml"]
    assert 29 < items[0]["duration_s"] < 31 and 59 < items[1]["duration_s"] < 61


def test_failed_items_are_found_in_the_step_state() -> None:
    step = parse({"steps": [{"id": "w", "title": "W", "description": "d", "kind": "code",
                             "tokens": False, "state_keys": ["results"]}]}).by_id["w"]
    state = {"results": [{"name": "a.yaml", "error": None}, {"name": "b.yaml", "error": "Boom: x"}],
             "other": [{"name": "c", "error": "not mine"}]}
    assert failed_items(step, state) == [("b.yaml", "Boom: x")]


# ---------------------------------------------------------------------------------- the night


def _night_env(tmp_path, monkeypatch, specs="[]"):
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text(f"specs: {specs}\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")


FLAGS = dict(graveyard=False, paper=False, implement=False, detectors=False, assess=False,
             waiting=False, director=False, programmer=False, search=False, research=True)


def test_a_failing_step_blocks_its_descendants_and_the_journal_shows_it(
        tmp_path, monkeypatch, jr, factory) -> None:
    _night_env(tmp_path, monkeypatch)
    ran: list[str] = []

    def boom(*a, **kw):
        raise KeyError("strategy")

    monkeypatch.setattr(night, "research_stage", boom)
    monkeypatch.setattr("qlab.twin_stage.twins_stage", lambda *a, **k: ran.append("twins"))
    monkeypatch.setattr(night, "generate_stage", lambda *a, **kw: ran.append("generate"))
    day = date(2026, 10, 27)
    report = night.run(day, journal=jr, trigger="button", **{**FLAGS, "twins": True,
                                                              "generate": True})
    states = {s.step_id: s for s in journal.night_states(factory(), day).values()}
    assert states["research"].state == "failed" and "KeyError" in states["research"].error
    assert states["twins-research"].state == "blocked"
    assert states["twins-research"].blocked_by == "research"
    assert states["report"].state == "ok" and states["ladder"].state in ("ok", "failed")
    assert ran == ["twins", "generate"]  # unrelated steps still ran; twins-research did not
    state = night.load_state(day)
    assert state["stage_errors"]["research"].startswith("KeyError: 'strategy'")
    run = journal.latest_run(factory())
    assert run.state == "failed" and run.trigger == "button" and run.finished_at is not None
    assert report.exists()


def test_off_steps_block_children_and_a_step_done_earlier_today_does_not(
        tmp_path, monkeypatch, jr, factory) -> None:
    _night_env(tmp_path, monkeypatch)
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: None)
    calls: list[str] = []
    monkeypatch.setattr("qlab.twin_stage.twins_stage", lambda *a, **k: calls.append("twins"))
    day = date(2026, 10, 28)
    night.run(day, journal=jr, off=["research"], **{**FLAGS, "twins": True})
    states = {s.step_id: s for s in journal.night_states(factory(), day).values()}
    assert states["research"].state == "off"
    assert states["twins-research"].state == "blocked" and "twins" not in calls[1:]
    # research done earlier tonight: switching it off no longer blocks its child
    night.run(day, journal=jr, **{**FLAGS, "twins": True})
    assert journal.night_states(factory(), day)["research"].state == "ok"
    night.run(day, journal=jr, off=["research"], **{**FLAGS, "twins": True})
    assert journal.night_states(factory(), day)["twins-research"].state == "ok"


def test_a_restart_clears_the_steps_markers_and_runs_it_with_what_depends_on_it(
        tmp_path, monkeypatch, jr, factory) -> None:
    _night_env(tmp_path, monkeypatch, specs="[a.yaml]")
    from qlab.pipeline.spec import StrategySpec
    spec = StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": {},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20"}, "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0},
        "min_leg_notional": 10.0, "unexpressed_mechanisms": []})
    monkeypatch.setattr(night, "load_spec", lambda p: spec)
    calls: list[str] = []

    def fake_eval(kind, spec, **kw):
        calls.append(spec.idea_id)
        return night.ItemResult(kind=kind, name=spec.idea_id, route="needs-forward")

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    flags = {**FLAGS, "research": False}
    day = date(2026, 10, 29)
    night.run(day, journal=jr, **flags)
    night.run(day, journal=jr, **flags)
    assert calls == ["x"]  # the second run found it done
    table = load_table()
    night.run(day, journal=jr, only=table.restart_set("recheck"), restart="recheck", **flags)
    assert calls == ["x", "x"]
    runs = factory().query(PipelineRun).order_by(PipelineRun.id).all()
    assert runs[-1].restart == "recheck" and "recheck" in runs[-1].only
    assert "detectors" not in runs[-1].only
    last = factory().query(PipelineStepRun).filter_by(run_id=runs[-1].id).all()
    assert {r.step_id for r in last} == {"recheck", "assess", "ladder", "report"}
    items = factory().query(PipelineItem).join(PipelineStepRun).filter(
        PipelineStepRun.run_id == runs[-1].id, PipelineStepRun.step_id == "recheck").all()
    assert [i.name for i in items] == ["a.yaml"] and items[0].outcome == "ok"


def test_without_a_journal_the_night_is_what_it_was(tmp_path, monkeypatch) -> None:
    _night_env(tmp_path, monkeypatch)
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: None)
    report = night.run(date(2026, 10, 30), **{**FLAGS, "research": False})
    assert report.exists() and "stage_errors" not in night.load_state(date(2026, 10, 30))


def test_a_stage_that_swallows_its_own_failure_is_a_failed_step(
        tmp_path, monkeypatch, jr, factory) -> None:
    _night_env(tmp_path, monkeypatch)

    def decay_fails(day, end, state):
        state["decay"] = {"lines": ["не посчитано: KeyError: 'x'"], "results": []}

    monkeypatch.setattr(night, "decay_stage", decay_fails)
    monkeypatch.setattr(night, "paper_reconciliation", lambda day: ["ok"])
    monkeypatch.setattr("qlab.pool_allocation.night_stage", lambda *a, **k: None)
    day = date(2026, 10, 26)
    night.run(day, journal=jr, **{**FLAGS, "paper": True, "research": False})
    states = {s.step_id: s for s in journal.night_states(factory(), day).values()}
    assert states["decay"].state == "failed" and "не посчитано" in states["decay"].error
    assert states["allocation"].state == "blocked" and states["allocation"].blocked_by == "decay"
    assert states["paper"].state == "ok"


def test_tokens_of_a_step_are_those_spent_between_its_start_and_its_end(jr, factory) -> None:
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=1, pid=1)
    step_run = jr.start_step(run_id, "assess")
    session = factory()
    inside = session.get(PipelineStepRun, step_run).started_at.replace(tzinfo=UTC)
    session.add(TokenSpend(at=inside, stage="recheck", agent="assessor", tokens_in=100,
                           tokens_out=50, usd_est=0.0, tokens_cache_read=1000))
    session.add(TokenSpend(at=datetime.now(UTC) - timedelta(hours=2), stage="x", agent="a",
                           tokens_in=9999, tokens_out=0, usd_est=0.0))
    session.commit()
    jr.finish_step(step_run, "ok", tokens=True)
    assert factory().get(PipelineStepRun, step_run).tokens == 1150


def test_a_journal_that_fails_never_fails_the_night(tmp_path, monkeypatch, capsys) -> None:
    _night_env(tmp_path, monkeypatch)
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: None)

    class Broken(journal.Journal):
        def __init__(self) -> None:
            pass

        def start_run(self, *a, **k):
            raise RuntimeError("no such table: pipeline_run")

    safe = journal.SafeJournal(Broken())
    report = night.run(date(2026, 10, 25), journal=safe, **{**FLAGS, "research": False})
    assert report.exists() and safe.failures and "pipeline_run" in capsys.readouterr().err


# ---------------------------------------------------------------------------------- statistics


def test_step_statistics(jr, factory) -> None:
    session = factory()
    run = PipelineRun(night_date=date(2026, 10, 30), trigger="schedule", state="ok",
                      started_at=datetime.now(UTC), concurrency=1, steps_off=[], approximate=False)
    session.add(run)
    session.flush()
    for state, seconds, error, tokens in (("ok", 60, None, 100), ("ok", 120, None, 300),
                                          ("failed", 300, "Boom: x", None),
                                          ("blocked", 0, None, None)):
        session.add(PipelineStepRun(run_id=run.id, step_id="s", state=state, duration_s=seconds,
                                    items_done=0, items_failed=0, error=error, tokens=tokens,
                                    finished_at=datetime.now(UTC), approximate=False))
    session.add(PipelineStepRun(run_id=run.id, step_id="s", state="ok", duration_s=5,
                                items_done=0, items_failed=0, approximate=True))
    session.commit()
    stats = journal.step_stats(factory(), "s")
    assert (stats["runs"], stats["ok"], stats["failed"], stats["blocked"]) == (3, 2, 1, 1)
    assert stats["success_share"] == pytest.approx(0.6667, abs=1e-3)
    assert stats["avg_s"] == 90 and stats["median_s"] == 90 and stats["max_s"] == 300
    assert stats["last_error"] == "Boom: x" and stats["tokens_avg"] == 200
    assert journal.step_stats(factory(), "s", approximate=True)["runs"] == 1  # kept apart


# ---------------------------------------------------------------------------------- the lock


def test_the_lock_is_taken_by_one_process_and_freed_when_it_ends(tmp_path) -> None:
    path = tmp_path / "run.lock"
    assert control.lock_holder(path) is None
    first = control.RunLock(path)
    assert first.acquire({"trigger": "schedule", "token": "t1"})
    holder = control.lock_holder(path)
    assert holder["trigger"] == "schedule" and holder["pid"] == os.getpid()
    assert not control.RunLock(path).acquire({"trigger": "button"}, wait=0.2)
    first.release()
    assert control.lock_holder(path) is None and control.RunLock(path).acquire({}, wait=0.2)


def test_a_dead_process_leaves_no_lock_behind(tmp_path) -> None:
    path = tmp_path / "run.lock"
    code = ("import sys; from qlab.nightflow.control import RunLock; "
            f"l = RunLock(__import__('pathlib').Path({str(path)!r})); "
            "assert l.acquire({'trigger': 'hand'}); print('held', flush=True); "
            "import os; os._exit(0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "held" in done.stdout
    assert control.lock_holder(path) is None  # the file stays, the lock is gone


def test_a_refused_start_says_why(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(control.LOCK_ENV, str(tmp_path / "run.lock"))
    monkeypatch.setattr(control, "STARTING_FILE", tmp_path / "run.starting")
    assert control.refusal() is None
    lock = control.RunLock()
    lock.acquire({"trigger": "schedule"})
    text = control.refusal()
    assert "Уже идёт запуск" in text and "по расписанию" in text and str(os.getpid()) in text
    assert control.start_detached([])["started"] is False
    lock.release()


def test_preview_names_what_runs_what_spends_tokens_and_what_is_blocked() -> None:
    table = load_table()
    shown = control.preview(table, done_today=set(), off={"research"})
    assert "research" in shown["off"]
    assert [(b["step"], b["blocked_by"]) for b in shown["blocked"]] == [
        ("twins-research", "research")]
    assert "выключен и сегодня не выполнен" in shown["blocked"][0]["reason"]
    assert "generate" in shown["token_steps"] and "research" not in shown["token_steps"]
    assert "detectors" in shown["run"]
    again = control.preview(table, done_today={"research"}, off={"research"})
    assert again["blocked"] == [] and "twins-research" in again["run"]
    restart = control.preview(table, done_today=set(), restart="recheck")
    assert restart["run"] == ["recheck", "assess", "ladder", "report"]
    assert restart["token_steps"] == ["assess"] and "detectors" in restart["outside"]
    assert "twins" in restart["reads_after"]  # read it fresh; not restarted


def test_the_arguments_of_a_button_run() -> None:
    assert control.run_arguments(off=["b", "a"], restart="recheck") == [
        "--off", "a", "--off", "b", "--restart", "recheck"]


# ---------------------------------------------------------------------------------- the history


def test_the_history_is_rebuilt_from_the_saved_state_and_marked_approximate(
        tmp_path, factory) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "2026-10-29.log").write_text(
        "=== 2026-10-29T00:20:05Z start\nnoise\n=== 2026-10-29T04:55:40Z end (exit 0)\n")
    (tmp_path / "2026-10-29.json").write_text(json.dumps({
        "done": {"watch:specs/a.yaml": "2026-10-29T01:00:00+00:00",
                 "watch:specs/b.yaml": "2026-10-29T01:10:00+00:00",
                 "assess:1": "2026-10-29T02:00:00+00:00"},
        "results": [], "stage_errors": {"research": "KeyError: 'x'"}, "decay": {"lines": []}}))
    session = factory()
    assert backfill.backfill(session, tmp_path) == ["2026-10-29"]
    assert backfill.backfill(session, tmp_path) == []  # idempotent
    session.commit()
    run = session.query(PipelineRun).one()
    assert run.approximate and run.trigger == "schedule" and run.state == "failed"
    rows = {r.step_id: r for r in session.query(PipelineStepRun)}
    assert rows["recheck"].duration_s == 600 and rows["recheck"].items_done == 2
    assert rows["research"].state == "failed" and rows["decay"].state == "ok"
    assert all(r.approximate for r in rows.values())
    assert journal.step_stats(session, "recheck")["runs"] == 0  # never mixed into measured stats
    assert journal.step_stats(session, "recheck", approximate=True)["runs"] == 1


def test_the_real_nights_can_be_rebuilt() -> None:
    """The shape of the saved nights we have is what the rebuild reads."""
    root = Path(__file__).resolve().parents[3] / "data" / "night"
    files = sorted(root.glob("2026-10-3*.json"))
    if not files:
        pytest.skip("no saved nights on this machine")
    state = json.loads(files[-1].read_text())
    built = backfill.rebuild_night(load_table(), date.fromisoformat(files[-1].stem), state,
                                   root / "logs" / f"{files[-1].stem}.log")
    assert built is not None and any(r["step"] == "recheck" for r in built["steps"])


# ---------------------------------------------------------------------------------- the routes


@pytest.fixture()
def api(factory, tmp_path, monkeypatch):
    from qlab.api.app import create_app
    from qlab.api.deps import get_session

    monkeypatch.setenv(control.LOCK_ENV, str(tmp_path / "run.lock"))
    monkeypatch.setattr(control, "STARTING_FILE", tmp_path / "run.starting")
    app = create_app()

    def _override():
        session = factory()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_session] = _override
    with TestClient(app, base_url="http://127.0.0.1:8890", client=("127.0.0.1", 50000)) as c:
        yield c


def _seed(factory):
    session = factory()
    run = PipelineRun(night_date=date(2026, 10, 31), trigger="schedule", state="ok",
                      started_at=datetime(2026, 10, 31, 0, 20, tzinfo=UTC), concurrency=1,
                      steps_off=[], approximate=False)
    session.add(run)
    session.flush()
    for step, state, seconds, error in (("detectors", "ok", 30, None), ("recheck", "failed", 600,
                                                                         "Boom: x"),
                                        ("assess", "blocked", 0, None)):
        row = PipelineStepRun(run_id=run.id, step_id=step, state=state, duration_s=seconds,
                              started_at=datetime(2026, 10, 31, 0, 21, tzinfo=UTC),
                              finished_at=datetime(2026, 10, 31, 0, 31, tzinfo=UTC), items_done=1,
                              items_failed=0, error=error, blocked_by="recheck" if state ==
                              "blocked" else None, approximate=False)
        session.add(row)
        session.flush()
        session.add(PipelineItem(step_run_id=row.id, name="item-1", outcome="ok", duration_s=12.5))
    session.commit()


def test_the_page_reads_steps_states_stats_and_history(api, factory) -> None:
    _seed(factory)
    body = api.get("/api/pipeline").json()
    steps = {s["id"]: s for s in body["steps"]}
    assert len(steps) == 25 and steps["recheck"]["depends_on"][0]["step"] == "detectors"
    assert steps["recheck"]["children"] and steps["recheck"]["wave"] == 1
    assert "waiting" in steps["recheck"]["can_run_with"]
    assert body["states"]["recheck"]["state"] == "failed" and body["running"] is False
    assert body["states"]["assess"]["blocked_by"] == "recheck"
    assert steps["recheck"]["stats"]["runs"] == 1 and steps["recheck"]["last"]["state"] == "failed"
    one = api.get("/api/pipeline/steps/recheck").json()
    assert one["history"][0]["error"] == "Boom: x" and one["items"][0]["name"] == "item-1"
    assert one["stats"]["last_error"] == "Boom: x" and one["neighbour_states"]["detectors"] == "ok"
    assert api.get("/api/pipeline/steps/nope").status_code == 404
    runs = api.get("/api/pipeline/runs").json()["runs"]
    assert runs[0]["trigger"] == "schedule" and runs[0]["state"] == "ok"


def test_a_run_row_with_nobody_holding_the_lock_reads_as_interrupted(api, factory) -> None:
    session = factory()
    run = PipelineRun(night_date=date(2026, 10, 31), trigger="button", state="running",
                      started_at=datetime(2026, 10, 31, 9, 0, tzinfo=UTC), concurrency=1,
                      steps_off=[], approximate=False)
    session.add(run)
    session.flush()
    session.add(PipelineStepRun(run_id=run.id, step_id="recheck", state="running",
                                started_at=datetime(2026, 10, 31, 9, 1, tzinfo=UTC),
                                items_done=0, items_failed=0, approximate=False))
    session.commit()
    body = api.get("/api/pipeline").json()
    assert body["run"]["state"] == "interrupted" and body["states"]["recheck"]["state"] == \
        "interrupted"


def test_the_press_of_a_button_answers_only_the_local_console(api, factory, monkeypatch) -> None:
    started = []
    monkeypatch.setattr(control, "start_detached",
                        lambda args, **k: started.append(args) or {"started": True, "pid": 7})
    body = {"off": ["research"]}
    assert api.post("/api/pipeline/preview", json=body).status_code == 403  # no console header
    ok = {"x-qlab-console": "1"}
    assert api.post("/api/pipeline/preview", json=body, headers={
        **ok, "origin": "https://evil.example"}).status_code == 403
    assert api.post("/api/pipeline/preview", json=body, headers={
        **ok, "host": "evil.example"}).status_code == 403
    shown = api.post("/api/pipeline/preview", json=body, headers=ok).json()
    assert shown["blocked"][0]["step"] == "twins-research" and not started
    assert api.post("/api/pipeline/run", json={"off": ["nope"]}, headers=ok).status_code == 422
    sent = api.post("/api/pipeline/run", json=body, headers=ok)
    assert sent.status_code == 200 and started == [["--off", "research"]]
    again = api.post("/api/pipeline/restart", json={"step": "recheck"}, headers=ok)
    assert again.status_code == 200
    assert started[-1] == ["--restart", "recheck"]
    stranger = TestClient(api.app, base_url="http://127.0.0.1:8890", client=("10.0.0.7", 1))
    assert stranger.post("/api/pipeline/run", json=body, headers=ok).status_code == 403


def test_a_second_start_is_refused_with_the_reason(api, tmp_path) -> None:
    lock = control.RunLock()
    assert lock.acquire({"trigger": "schedule"})
    try:
        refused = api.post("/api/pipeline/run", json={"off": []}, headers={"x-qlab-console": "1"})
        assert refused.status_code == 409 and "Уже идёт запуск" in refused.json()["detail"]
        assert api.get("/api/pipeline").json()["running"] is True
    finally:
        lock.release()


def test_the_yaml_is_the_only_place_that_names_the_steps() -> None:
    doc = yaml.safe_load((Path(__file__).resolve().parents[3] / "night" / "pipeline.yaml")
                         .read_text(encoding="utf-8"))
    assert [s["id"] for s in doc["steps"]] == load_table().ids
