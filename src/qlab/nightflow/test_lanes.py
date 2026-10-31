"""Steps running together (docs/PIPELINE.md, «Дорожки»): the lanes of the table, the scheduler that
keeps one step per lane, the state hub that gives every step its own copy of the night's state, and
the registry writes that two steps make at once."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import qlab.night as night
from qlab import tree_writes
from qlab.boundedcache import BoundedCache
from qlab.nightflow import journal
from qlab.nightflow.runner import ItemTracker, Runner, StepImpl
from qlab.nightflow.statehub import ForeignStateWrite, NightStateHub
from qlab.nightflow.table import LANES, TableError, load_table, parse
from qlab.registry.models import Base


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


def _step(id_, parents=(), lane="", keys=(), prefixes=(), **extra) -> dict:
    return {"id": id_, "title": id_.upper(), "description": f"step {id_}", "kind": "code",
            "tokens": False, "lane": lane, "state_keys": list(keys),
            "done_prefixes": list(prefixes),
            "depends_on": [{"step": p, "kind": k, "why": f"{id_} needs {p}"} for p, k in parents],
            **extra}


# ---------------------------------------------------------------------------------- the table


def test_the_shipped_table_puts_every_step_in_a_lane_and_the_agents_in_one() -> None:
    table = load_table()
    assert all(s.lane in LANES for s in table.steps)
    lanes = table.lanes()
    assert set(lanes) == {"calc", "checks", "agents", "final"}
    assert {s.lane for s in table.steps if s.tokens} == {"agents"}
    assert lanes["final"] == ["ladder", "report"]
    assert lanes["checks"] == ["paper", "decay", "allocation"]
    # the owner's order of the calculations lane, and the steps that write ideas stay in it
    assert lanes["calc"][:3] == ["detectors", "recheck", "waiting"]
    assert {"twins", "twins-research", "restamp", "overlays", "second_venue",
            "anomalies"} <= set(lanes["calc"])


def test_the_ladder_and_the_report_wait_for_every_other_step() -> None:
    table = load_table()
    for last in ("ladder", "report"):
        assert set(table.parents(last)) >= set(table.ids) - {last, "ladder", "report"}
    assert "ladder" in table.parents("report")


def test_decay_keeps_its_edge_to_the_watch_list_and_the_agents_keep_their_order() -> None:
    table = load_table()
    kinds = {e.step: e.kind for e in table.by_id["decay"].depends_on}
    assert kinds["recheck"] == "fresh" and kinds["paper"] == "data"
    order = ["assess", "anomalies-investigate", "programmer", "director", "research", "implement",
            "search", "catalogue", "cards", "generate"]
    for before, after in zip(order, order[1:], strict=False):
        if after == "anomalies-investigate":
            continue
        assert before in table.ancestors(after) or before == "assess" and \
            "assess" in table.ancestors(after)
    assert "anomalies-investigate" in table.ancestors("programmer")


def test_two_steps_may_not_own_one_key_unless_the_table_orders_them() -> None:
    doc = {"steps": [_step("a", keys=["x"]), _step("b", keys=["x"])]}
    with pytest.raises(TableError, match="both own"):
        parse(doc)
    ordered = {"steps": [_step("a", keys=["x"]), _step("b", [("a", "data")], keys=["x"])]}
    assert parse(ordered).ordered("a", "b")
    clash = {"steps": [_step("a", prefixes=["p:"]), _step("b", prefixes=["p:"])]}
    with pytest.raises(TableError, match="done-marker prefix"):
        parse(clash)
    with pytest.raises(TableError, match="keeps"):
        parse({"steps": [_step("a", keys=["stage_errors"])]})


def test_a_lane_must_be_a_known_one_and_the_token_spenders_share_one() -> None:
    with pytest.raises(TableError, match="lane"):
        parse({"steps": [_step("a", lane="fast")]})
    two = [_step("a", lane="calc", tokens=True, kind="agent"),
           _step("b", lane="checks", tokens=True, kind="agent")]
    with pytest.raises(TableError, match="share one lane"):
        parse({"steps": two})


# ---------------------------------------------------------------------------------- the state hub


def _hub(tmp_path, steps, master=None, strict=True):
    table = parse({"steps": steps})
    written: list[str] = []
    seen: list[dict] = []
    hub = NightStateHub(master if master is not None else {"done": {}}, table,
                        write=lambda text: (written.append(text),
                                            (tmp_path / "state.json").write_text(text)),
                        observers=lambda: [seen.append], strict=strict, log=lambda t: None)
    return hub, written, seen


def test_a_view_is_a_private_copy_and_only_the_steps_own_keys_are_merged(tmp_path) -> None:
    hub, written, seen = _hub(tmp_path, [_step("a", keys=["x"], prefixes=["a:"]),
                                         _step("b", keys=["y"])],
                              {"done": {"a:old": "t0"}, "x": [1], "y": [2]})
    view = hub.view("a")
    view["x"].append(10)
    view["done"]["a:new"] = "t1"
    assert hub.master["x"] == [1]  # nothing reaches the master before a save
    view.hub.save(view)
    assert hub.master["x"] == [1, 10] and hub.master["done"] == {"a:old": "t0", "a:new": "t1"}
    assert json.loads((tmp_path / "state.json").read_text())["x"] == [1, 10]
    assert seen and seen[-1]["x"] == [1, 10]
    # another step's view, taken earlier, does not see it and does not overwrite it
    earlier = hub.view("b")
    view["x"].append(11)
    hub.save(view)
    earlier["y"].append(3)
    hub.save(earlier)
    assert hub.master["x"] == [1, 10, 11] and hub.master["y"] == [2, 3]
    assert written and not hub.violations


def test_a_key_the_step_does_not_own_is_a_violation_and_strict_refuses_it(tmp_path) -> None:
    hub, _, _ = _hub(tmp_path, [_step("a", keys=["x"]), _step("b", keys=["y"])],
                     {"done": {}, "y": [0]})
    view = hub.view("a")
    view["y"] = [99]
    with pytest.raises(ForeignStateWrite, match="'y'"):
        hub.save(view)
    assert hub.master["y"] == [0]
    soft, _, _ = _hub(tmp_path, [_step("a", keys=["x"]), _step("b", keys=["y"])],
                      {"done": {}, "y": [0]}, strict=False)
    view = soft.view("a")
    view["y"] = [99]
    view["done"]["zzz"] = "t"
    soft.save(view)
    assert soft.master["y"] == [0] and len(soft.violations) == 1  # the done-marker has no prefix
    assert soft.master["done"]["zzz"] == "t"


def test_done_markers_of_a_prefix_are_merged_and_a_removed_one_is_removed(tmp_path) -> None:
    hub, _, _ = _hub(tmp_path, [_step("a", prefixes=["a:"]), _step("b", prefixes=["b:"])],
                     {"done": {"a:1": "t", "b:1": "t"}})
    mine, other = hub.view("a"), hub.view("b")
    mine["done"]["a:2"] = "t2"
    del mine["done"]["a:1"]
    other["done"]["b:2"] = "t3"
    hub.save(mine)
    hub.save(other)
    assert hub.master["done"] == {"b:1": "t", "a:2": "t2", "b:2": "t3"}


def test_many_threads_saving_at_once_lose_nothing_and_leave_a_valid_file(tmp_path) -> None:
    steps = [_step(f"s{i}", keys=[f"k{i}"], prefixes=[f"s{i}:"]) for i in range(6)]
    hub, _, _ = _hub(tmp_path, steps)
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            view = hub.view(f"s{i}")
            block = view.setdefault(f"k{i}", {"n": 0, "rows": []})
            for n in range(150):
                block["n"] = n + 1
                block["rows"].append({"i": i, "n": n})
                view["done"][f"s{i}:{n}"] = datetime.now(UTC).isoformat()
                night.save_state(date(2026, 10, 31), view)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    saved = json.loads((tmp_path / "state.json").read_text())
    assert all(saved[f"k{i}"]["n"] == 150 and len(saved[f"k{i}"]["rows"]) == 150
               for i in range(6))
    assert len(saved["done"]) == 6 * 150 and not hub.violations


def test_past_stop_and_errors_of_a_step_go_to_the_hubs_master(tmp_path) -> None:
    hub, _, _ = _hub(tmp_path, [_step("a", keys=["x"])])
    view = hub.view("a")
    past = datetime(2020, 1, 1, tzinfo=UTC)
    assert night.past_stop(past, view, "recheck:foo") is True
    assert hub.master["stopped_at"]["cut"] == ["recheck"]
    night.guarded(date(2026, 10, 31), view, "boom", lambda: 1 / 0)
    assert "ZeroDivisionError" in hub.master["stage_errors"]["boom"]
    assert "stopped_at" not in view and not hub.violations
    hub.save(view)  # a view never carries the hub's keys back


# ---------------------------------------------------------------------------------- the scheduler


class Probe:
    """A context that makes one view per step, like `NightContext`, over a hub."""

    def __init__(self, hub) -> None:
        self.hub = hub
        self.state = hub.master
        self.day = date(2026, 10, 31)

    def for_step(self, step):
        from qlab.night import StepContext
        return StepContext(self, self.hub.view(step.id))

    def snapshot(self):
        return self.hub.snapshot()


class Clock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, str, float]] = []
        self.live: set[str] = set()
        self.peak = 0

    def enter(self, name: str) -> None:
        with self.lock:
            self.live.add(name)
            self.peak = max(self.peak, len(self.live))
            self.events.append(("start", name, time.monotonic()))

    def leave(self, name: str) -> None:
        with self.lock:
            self.live.discard(name)
            self.events.append(("end", name, time.monotonic()))

    def span(self, name: str) -> tuple[float, float]:
        start = next(t for k, n, t in self.events if k == "start" and n == name)
        end = next(t for k, n, t in self.events if k == "end" and n == name)
        return start, end


def _lanes_run(jr, tmp_path, steps, work, *, concurrency=3, fail=(), off=()):
    """Run the table with `work[step_id]` seconds of sleeping per step; returns (clock, outcome,
    hub)."""
    table = parse({"steps": steps})
    hub, _, _ = _hub(tmp_path, steps)
    clock = Clock()

    def make(step_id):
        def run(ctx):
            clock.enter(step_id)
            try:
                for key in ctx.state.hub.owned_keys(step_id):
                    ctx.state.setdefault(key, []).append(step_id)
                ctx.save()
                time.sleep(work.get(step_id, 0.05))
                if step_id in fail:
                    raise RuntimeError(f"{step_id} broke")
            finally:
                clock.leave(step_id)
        return StepImpl(run, step_id)

    impls = {s: make(s) for s in table.ids}
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=concurrency, pid=1)
    runner = Runner(table, impls, Probe(hub), journal=jr, run_id=run_id, concurrency=concurrency,
                    off=off, log=lambda t: None)
    return clock, runner.execute(), hub, run_id


def _three_lanes() -> list[dict]:
    return [
        _step("c1", lane="calc", keys=["c1"]),
        _step("c2", [("c1", "data")], lane="calc", keys=["c2"]),
        _step("c3", [("c2", "data")], lane="calc", keys=["c3"]),
        _step("k1", [("c1", "fresh")], lane="checks", keys=["k1"]),
        _step("k2", [("k1", "data")], lane="checks", keys=["k2"]),
        _step("g1", lane="agents", keys=["g1"]),
        _step("g2", [("g1", "budget")], lane="agents", keys=["g2"]),
        _step("z", [(s, "fresh") for s in ("c3", "k2", "g2")], lane="final", keys=["z"]),
    ]


def test_lanes_run_together_one_step_at_a_time_in_each(jr, tmp_path) -> None:
    work = {"c1": 0.1, "c2": 0.4, "c3": 0.2, "k1": 0.3, "k2": 0.2, "g1": 0.5, "g2": 0.3}
    clock, outcome, hub, _ = _lanes_run(jr, tmp_path, _three_lanes(), work)
    assert set(outcome.values()) == {"ok"}
    assert clock.peak == 3  # a step of each lane at once, never two of one lane
    for a, b in (("c1", "c2"), ("c2", "c3"), ("k1", "k2"), ("g1", "g2")):
        assert clock.span(a)[1] <= clock.span(b)[0] + 1e-3
    # the lanes overlap: the agents' first step runs while the calculations lane does
    assert clock.span("g1")[0] < clock.span("c2")[1] and clock.span("k1")[0] < clock.span("c2")[1]
    # the last lane waits for all of them
    assert clock.span("z")[0] >= max(clock.span(s)[1] for s in ("c3", "k2", "g2")) - 1e-3
    # every step's result reached the master, from its own thread's view
    assert all(hub.master[k] == [k] for k in ("c1", "c2", "c3", "k1", "k2", "g1", "g2", "z"))
    assert not hub.violations


def test_a_step_waiting_for_another_lane_does_not_hold_up_the_steps_behind_it(jr, tmp_path) -> None:
    steps = [_step("slow", lane="agents", keys=["slow"]),
             _step("late", [("slow", "data")], lane="calc", keys=["late"]),
             _step("free", lane="calc", keys=["free"])]
    clock, outcome, _, _ = _lanes_run(jr, tmp_path, steps, {"slow": 0.6, "free": 0.1})
    assert set(outcome.values()) == {"ok"}
    # `late` is first in the table's order in its lane but waits for `slow`; `free` goes at once
    assert clock.span("free")[1] < clock.span("slow")[1]
    assert clock.span("late")[0] >= clock.span("slow")[1] - 1e-3


def test_concurrency_caps_the_steps_running_together(jr, tmp_path) -> None:
    clock, _, _, _ = _lanes_run(jr, tmp_path, _three_lanes(),
                                {"c1": 0.2, "g1": 0.2, "k1": 0.2}, concurrency=2)
    assert clock.peak == 2


def test_a_failed_step_blocks_only_its_data_descendants_in_any_lane(jr, tmp_path) -> None:
    clock, outcome, hub, _ = _lanes_run(jr, tmp_path, _three_lanes(), {"c1": 0.1},
                                        fail={"c1"})
    assert outcome["c1"] == "failed"
    assert outcome["c2"] == "blocked" and outcome["c3"] == "blocked"
    assert outcome["k1"] == "ok" and outcome["k2"] == "ok"  # a fresh edge does not block
    assert outcome["g1"] == "ok" and outcome["g2"] == "ok" and outcome["z"] == "ok"
    assert hub.master["c1"] == ["c1"]  # what the failed step had saved is kept


def test_no_step_loses_its_row_and_the_journal_shows_the_overlap(jr, tmp_path, factory) -> None:
    from qlab.registry.models import PipelineStepRun

    work = {"c1": 0.1, "c2": 0.4, "c3": 0.2, "k1": 0.3, "k2": 0.2, "g1": 0.5, "g2": 0.3}
    _, outcome, _, run_id = _lanes_run(jr, tmp_path, _three_lanes(), work, fail={"k1"})
    session = factory()
    rows = {r.step_id: r for r in session.query(PipelineStepRun).filter_by(run_id=run_id)}
    assert set(rows) == set(outcome)
    assert all(r.state == outcome[r.step_id] for r in rows.values())
    assert rows["k1"].state == "failed" and "k1 broke" in rows["k1"].error
    assert rows["k2"].state == "blocked" and rows["k2"].blocked_by == "k1"
    assert all(r.finished_at is not None and r.duration_s is not None for r in rows.values())
    spans = {k: (r.started_at, r.finished_at) for k, r in rows.items()}
    assert spans["g1"][0] < spans["c2"][1] and spans["c2"][0] < spans["g1"][1]  # they overlapped


def test_a_stop_of_the_night_marks_the_running_steps_interrupted(jr, tmp_path, factory) -> None:
    from qlab.registry.models import PipelineStepRun

    steps = [_step("a", lane="calc", keys=["a"]), _step("b", lane="agents", keys=["b"])]
    table = parse({"steps": steps})
    hub, _, _ = _hub(tmp_path, steps)
    started = threading.Event()

    def sleeper(ctx):
        started.set()
        time.sleep(30)

    def stopper(ctx):
        started.wait(5)
        raise KeyboardInterrupt

    impls = {"a": StepImpl(sleeper, "a"), "b": StepImpl(stopper, "b")}
    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=3, pid=1)
    runner = Runner(table, impls, Probe(hub), journal=jr, run_id=run_id, concurrency=3,
                    log=lambda t: None)
    started_at = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        runner.execute()
    assert time.monotonic() - started_at < 10  # the sleeping step did not hold the stop up
    rows = {r.step_id: r.state for r in factory().query(PipelineStepRun).filter_by(run_id=run_id)}
    assert rows == {"a": "interrupted", "b": "interrupted"}


def test_sequential_and_lane_runs_leave_the_same_state(jr, tmp_path) -> None:
    work = {"c1": 0.01, "c2": 0.02, "c3": 0.01, "k1": 0.02, "k2": 0.01, "g1": 0.02, "g2": 0.01}
    _, _, seq, _ = _lanes_run(jr, tmp_path, _three_lanes(), work, concurrency=1)
    _, _, par, _ = _lanes_run(jr, tmp_path, _three_lanes(), work, concurrency=3)
    assert seq.master == par.master


def test_the_item_tracker_sees_the_markers_of_steps_running_together(
        jr, tmp_path, factory) -> None:
    from qlab.registry.models import PipelineItem

    steps = [_step("a", lane="calc", keys=["a"], prefixes=["a:"]),
             _step("b", lane="agents", keys=["b"], prefixes=["b:"])]
    table = parse({"steps": steps})
    tracker = ItemTracker(table, jr)
    hub, _, _ = _hub(tmp_path, steps)
    hub._observers = lambda: [tracker.observe]

    def make(name):
        def run(ctx):
            for n in range(5):
                ctx.state["done"][f"{name}:{n}"] = datetime.now(UTC).isoformat()
                ctx.save()
                time.sleep(0.02)
        return StepImpl(run, name)

    run_id = jr.start_run(date(2026, 10, 31), trigger="hand", steps_off=[], only=None,
                          restart=None, concurrency=3, pid=1)
    Runner(table, {"a": make("a"), "b": make("b")}, Probe(hub), journal=jr, run_id=run_id,
           concurrency=3, tracker=tracker, log=lambda t: None).execute()
    names = sorted(r.name for r in factory().query(PipelineItem))
    assert names == sorted(str(n) for _ in "ab" for n in range(5))  # five markers of each step


# ---------------------------------------------------------------------------------- the registry


@pytest.fixture()
def file_registry(tmp_path, monkeypatch):
    """A registry file shared by threads, each with its own connection (as in the night)."""
    from qlab.registry import db as db_module
    from qlab.registry.models import Base

    path = tmp_path / "reg.db"
    monkeypatch.setenv("QLAB_DB", str(path))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    yield db_module
    db_module.get_engine().dispose()


def _spec(idea="idea-x", **params):
    from qlab.pipeline.spec import StrategySpec

    return StrategySpec.model_validate({
        "idea_id": idea, "title": "t", "code_ref": "a:b", "params": {"lookback": 20, **params},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def test_two_threads_adding_spec_versions_of_one_idea_get_distinct_numbers(file_registry) -> None:
    from qlab.pipeline import evaluate
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Profile, SourceType
    from qlab.registry.models import Spec as SpecRow

    with session_scope() as session:
        repo.upsert_idea(session, id="idea-x", title="t", source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, notes="")
    barrier = threading.Barrier(8)
    got: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            barrier.wait()
            with session_scope() as session:
                row = evaluate._get_or_create_spec_row(session, _spec(lookback=100 + i % 4))
                got.append((i % 4, row.version))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    with session_scope() as session:
        versions = sorted(r.version for r in session.query(SpecRow).filter_by(idea_id="idea-x"))
    assert versions == [1, 2, 3, 4]  # four contents, eight callers: one row each, no gap
    assert len({v for _, v in got}) == 4 and all(
        len({v for k, v in got if k == c}) == 1 for c in range(4))


def test_a_collision_with_another_process_is_read_again(file_registry, monkeypatch) -> None:
    """The unique key refuses a second row with the same number; the numbering is read again."""
    from qlab.pipeline import evaluate
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Profile, SourceType
    from qlab.registry.models import Spec as SpecRow

    with session_scope() as session:
        repo.upsert_idea(session, id="idea-x", title="t", source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, notes="")
    real = evaluate._next_version
    calls = {"n": 0}

    def stale_once(session, idea_id):
        calls["n"] += 1
        if calls["n"] == 1:  # another process took version 1 between the reading and the insert
            with session_scope() as other:
                evaluate._add_spec_row(other, _spec(lookback=7), 1, *evaluate._spec_row_content(
                    _spec(lookback=7)))
            return 1
        return real(session, idea_id)

    monkeypatch.setattr(evaluate, "_next_version", stale_once)
    with session_scope() as session:
        row = evaluate._get_or_create_spec_row(session, _spec(lookback=9))
        assert row.version == 2
    with session_scope() as session:
        assert session.query(SpecRow).filter_by(idea_id="idea-x").count() == 2


def test_noise_spec_rows_numbered_by_two_callers_at_once_do_not_collide(file_registry) -> None:
    from qlab.calibration.run import (
        GENERATOR_NAMES,
        build_noise_spec,
        ensure_noise_idea,
        load_reference_spec,
        noise_idea_id,
    )
    from qlab.pipeline.evaluate import ensure_spec_rows_committed
    from qlab.registry.db import session_scope
    from qlab.registry.models import Spec as SpecRow

    reference = load_reference_spec()
    with session_scope() as session:
        for generator in GENERATOR_NAMES:
            ensure_noise_idea(session, series="dollar_neutral", generator=generator)
    barrier = threading.Barrier(4)
    out: dict[int, list[int]] = {}
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            specs = [build_noise_spec(series="dollar_neutral", generator=GENERATOR_NAMES[0],
                                      seed=i * 100 + n, reference=reference) for n in range(10)]
            barrier.wait()
            out[i] = ensure_spec_rows_committed(specs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    with session_scope() as session:
        versions = sorted(r.version for r in session.query(SpecRow).filter_by(
            idea_id=noise_idea_id("dollar_neutral", GENERATOR_NAMES[0])))
    assert versions == list(range(1, 41))
    assert len({i for ids in out.values() for i in ids}) == 40


def test_a_journal_call_is_tried_again_when_the_registry_is_locked(factory) -> None:
    from contextlib import contextmanager

    from sqlalchemy.exc import OperationalError

    calls = {"n": 0}

    @contextmanager
    def scope():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("insert", {}, Exception("database is locked"))
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    original = journal.time.sleep
    journal.time.sleep = lambda s: None
    try:
        run_id = journal.Journal(scope).start_run(
            date(2026, 10, 31), trigger="hand", steps_off=[], only=None, restart=None,
            concurrency=3, pid=1)
    finally:
        journal.time.sleep = original
    assert run_id and calls["n"] == 3


def test_a_session_that_commits_before_a_long_wait_lets_another_writer_in(file_registry) -> None:
    """The registry has one write lock; a session holding it across a fetch or an agent call makes
    every other writer wait out its busy timeout. After a commit a second connection writes at
    once."""
    from sqlalchemy import text

    from qlab.registry.db import get_engine

    engine = get_engine()
    with engine.connect() as slow:
        slow.execute(text("create table t (x integer)"))
        slow.commit()
        slow.execute(text("insert into t values (1)"))
        slow.commit()  # what evaluate_spec and run_agent now do before the long part
        started = time.monotonic()
        with engine.connect() as other:
            other.execute(text("insert into t values (2)"))
            other.commit()
        assert time.monotonic() - started < 2


# ---------------------------------------------------------------------------------- small parts


def test_the_bounded_cache_is_safe_for_threads_and_keeps_its_size() -> None:
    cache: BoundedCache[int] = BoundedCache(4)
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for n in range(2000):
                cache.put((i, n % 9), n)
                cache.get((i, (n + 3) % 9))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(cache) <= 4
    cache.put("a", 1)
    assert cache.get("a") == 1 and "a" in cache and cache.get("zzz") is None


def test_files_a_code_step_writes_are_not_taken_for_an_agents(tmp_path, monkeypatch) -> None:
    from qlab.agents import implementer

    root = Path(tree_writes.ROOT)
    noted = root / "specs" / "overlays" / "unit-test-base" / "unit-test-variant.yaml"
    tree_writes.note(noted)
    tree_writes.note(tmp_path / "outside.yaml")  # not in the repository: not kept
    assert "specs/overlays/unit-test-base/unit-test-variant.yaml" in tree_writes.written()
    assert not any("outside.yaml" in w for w in tree_writes.written())
    changes = ["src/qlab/strategies/agent/x.py",
               "specs/overlays/unit-test-base/unit-test-variant.yaml",
               "specs/overlays/unit-test-base/"]
    assert tree_writes.without_code_writes(changes) == ["src/qlab/strategies/agent/x.py"]

    class Done:
        stdout = "?? specs/overlays/unit-test-base/unit-test-variant.yaml\n?? src/a.py\n"

    monkeypatch.setattr(implementer.subprocess, "run", lambda *a, **k: Done())
    assert implementer._changed_paths() == ["src/a.py"]


def test_the_noise_pool_is_made_once_by_threads_asking_together(monkeypatch) -> None:
    import qlab.calibration.run as run_module

    made: list[object] = []

    class FakePool:
        _broken = False

        def __init__(self, **kwargs) -> None:
            made.append(self)
            time.sleep(0.05)

        def shutdown(self, **kwargs) -> None:
            pass

    monkeypatch.setattr(run_module, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(run_module, "_POOL", None)
    monkeypatch.setattr(run_module, "_POOL_KEY", None)
    got: list[object] = []
    threads = [threading.Thread(target=lambda: got.append(run_module.shared_pool(3)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and len({id(p) for p in got}) == 1
    run_module._POOL.__class__ = FakePool
    run_module.discard_broken_pool()  # a sound pool is left alone
    assert run_module._POOL is made[0]
    made[0]._broken = True
    run_module.discard_broken_pool()
    assert run_module._POOL is None
