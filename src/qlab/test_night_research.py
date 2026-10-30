"""Tests for the nightly researcher: cadence, once per night, the four strategies,
the drawdown evidence in its context and DROP proposals (docs/NIGHT.md)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import qlab.agents.researcher as researcher
import qlab.night as night
from qlab.harness.panel import MarketPanel

REPO = Path(__file__).resolve().parents[2]


# ---- night/research.yaml ----------------------------------------------------------------


def test_the_research_plan_has_the_five_strategies_and_runs_nightly() -> None:
    from qlab.pipeline.spec import load_spec

    plan = yaml.safe_load((REPO / "night/research.yaml").read_text(encoding="utf-8"))
    assert plan["cadence"] == "nightly"
    assert set(plan["strategies"]) == {"bv2", "trend", "frab", "xsmom", "frab-binance-spot"}
    for name, entry in plan["strategies"].items():
        assert entry["ideas"] and all(isinstance(i, str) for i in entry["ideas"]), name
        assert entry["specs"] and entry["docs"], name
        spec_ideas = {load_spec(REPO / p).idea_id for p in entry["specs"]}
        assert spec_ideas <= set(entry["ideas"]) | {"strategy-b-v2"}, (name, spec_ideas)
        for doc in entry["docs"]:
            assert (REPO / doc).is_file(), doc
    xsmom = plan["strategies"]["xsmom"]
    assert xsmom["ideas"][0] == "xsmom"
    assert {load_spec(REPO / p).idea_id for p in xsmom["specs"]} == set(xsmom["ideas"])


# ---- cadence and once-per-night ---------------------------------------------------------


def test_nightly_runs_every_night_weekly_on_the_sweep_day_or_after_a_stop(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    plan = {"strategies": {"a": {}, "b": {}}}
    day = date(2026, 10, 27)
    assert night.research_due(day, {**plan, "cadence": "nightly"}, sweep=False)
    assert not night.research_due(day, plan, sweep=False)  # weekly is the default
    assert night.research_due(day, plan, sweep=True)
    assert not night.research_due(day, {"cadence": "nightly"}, sweep=True)  # nothing to research
    night.save_state(date(2026, 10, 26), {"done": {"research:a": "t"}, "research": [{}]})
    assert night.unfinished_research(day, plan) == ["b"]
    assert night.research_due(day, plan, sweep=False)  # b was left unfinished yesterday
    with pytest.raises(ValueError, match="cadence"):
        night.research_due(day, {**plan, "cadence": "monthly"}, sweep=False)


class _Budget:
    spent_tokens = 7


class _Reading:
    class report:  # noqa: N801 - mimics the probe's result shape
        reading = None


def _stage_env(tmp_path, monkeypatch, *, exhausted_on: str | None = None):
    """A research_stage environment: empty registry, fake budget, recorded `research` calls."""
    import qlab.budget as budget_module
    import qlab.registry.db as db_module
    from qlab.budget import NightExhausted
    from qlab.registry.models import Base

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    Base.metadata.create_all(db_module.get_engine())
    calls: list[str] = []

    class _Stage:
        def candidate(self, name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, _stage, candidates):
            return _Stage()

    def fake_research(name, **kw):
        if name == exhausted_on:
            raise NightExhausted("the night's budget is spent")
        calls.append(name)
        return researcher.ResearchOutcome(memo=None, specs=[], problems=[])

    monkeypatch.setattr(researcher, "research", fake_research)
    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe", lambda _session: _Reading)
    config = tmp_path / "research.yaml"
    config.write_text("cadence: nightly\nmax_experiments: 2\nstrategies:\n"
                      "  s1: {ideas: [i1], specs: []}\n  s2: {ideas: [i2], specs: []}\n",
                      encoding="utf-8")
    return calls, config, db_module


def test_a_strategy_is_researched_once_per_night(tmp_path, monkeypatch) -> None:
    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    state = {"done": {}}
    night.research_stage(date(2026, 10, 26), state, noise_trials=1, config=config)
    night.research_stage(date(2026, 10, 26), state, noise_trials=1, config=config)
    assert calls == ["s1", "s2"]
    assert set(state["done"]) == {"research:s1", "research:s2"}
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_stopped_run_continues_first_on_the_next_night(tmp_path, monkeypatch) -> None:
    calls, config, db_module = _stage_env(tmp_path, monkeypatch, exhausted_on="s2")
    monday = {"done": {}}
    night.research_stage(date(2026, 10, 26), monday, noise_trials=1, config=config)
    assert calls == ["s1"] and "research:s2" not in monday["done"]
    assert monday["research"][-1]["strategy"] == "s2" and monday["research"][-1]["stopped"]
    night.save_state(date(2026, 10, 26), monday)
    calls.clear()
    monkeypatch.setattr(researcher, "research", lambda name, **kw: (
        calls.append(name) or researcher.ResearchOutcome(memo=None)))
    tuesday = {"done": {}}
    night.research_stage(date(2026, 10, 27), tuesday, noise_trials=1, config=config)
    assert calls == ["s2", "s1"]  # the unfinished one goes first
    db_module._engine = None
    db_module._SessionLocal = None


# ---- the rotation: one strategy a night (owner, 2026-10-30) ------------------------------


def _rotation_env(tmp_path, monkeypatch, *, strategies=("s1", "s2", "s3"), high: str | None = None):
    """A research_stage environment with a rotation of one strategy a night over `strategies`;
    `high` is the one the research director marks high importance."""
    import types

    import qlab.directions as directions_module

    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    config.write_text("cadence: nightly\nrotation: {per_night: 1}\nstrategies:\n" + "".join(
        f"  {n}: {{ideas: [i-{n}], specs: []}}\n" for n in strategies), encoding="utf-8")
    fixed = None if high is None else types.SimpleNamespace(
        date=date(2026, 10, 25),
        researcher={high: types.SimpleNamespace(priority="high", note=None)})
    monkeypatch.setattr(directions_module, "active_directions", lambda *a, **kw: fixed)
    return calls, config, db_module


def _night(config, day: date) -> dict:
    state = night.load_state(day)
    night.research_stage(day, state, noise_trials=1, config=config)
    night.save_state(day, state)
    return state


def test_a_rotation_takes_one_strategy_a_night_and_keeps_the_turns(tmp_path, monkeypatch) -> None:
    calls, config, db_module = _rotation_env(tmp_path, monkeypatch)
    for n in range(4):
        _night(config, date(2026, 10, 26 + n))
    assert calls == ["s1", "s2", "s3", "s1"]  # in turn, the longest-waiting first
    state = _night(config, date(2026, 10, 30))
    assert calls[-1] == "s2" and set(state["done"]) == {"research:s2"}
    _night(config, date(2026, 10, 30))  # the same night again: its one place is taken
    assert calls[-1] == "s2" and calls.count("s2") == 2
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_high_importance_strategy_goes_first_but_not_two_nights_running(
        tmp_path, monkeypatch) -> None:
    calls, config, db_module = _rotation_env(tmp_path, monkeypatch, high="s3")
    for n in range(5):
        _night(config, date(2026, 10, 26 + n))
    assert calls == ["s3", "s1", "s3", "s2", "s3"]  # every second night, the others in between
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_strategy_a_stop_cut_goes_first_the_next_night_not_all_the_others(
        tmp_path, monkeypatch) -> None:
    calls, config, db_module = _rotation_env(tmp_path, monkeypatch)
    _night(config, date(2026, 10, 26))
    assert calls == ["s1"]
    from qlab.budget import NightExhausted

    def stopped(name, **kw):
        raise NightExhausted("the night's budget is spent")

    monkeypatch.setattr(researcher, "research", stopped)
    state = _night(config, date(2026, 10, 27))  # s2's turn, cut by the budget
    assert state["research"][-1]["strategy"] == "s2" and state["research"][-1]["stopped"]
    plan = night.load_research_plan(config)
    assert night.unfinished_research(date(2026, 10, 28), plan) == ["s2"]  # s3 was just not its turn
    monkeypatch.setattr(researcher, "research", lambda name, **kw: (
        calls.append(name) or researcher.ResearchOutcome(memo=None)))
    _night(config, date(2026, 10, 28))
    assert calls == ["s1", "s2"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_research_plan_rotates_one_strategy_a_night_and_the_slots_follow_it() -> None:
    plan = yaml.safe_load((REPO / "night/research.yaml").read_text(encoding="utf-8"))
    assert plan["rotation"] == {"per_night": 1} and 0 < plan["night_share"] <= 1
    assert night.research_slots(plan) == 1
    assert night.research_slots({k: v for k, v in plan.items() if k != "rotation"}) == 5


# ---- DROP proposals ---------------------------------------------------------------------


def test_drop_lines_are_parsed_from_the_memo() -> None:
    memo = "\n".join([
        "# Записка", "Текст.",
        "DROP: bv2-a — слабее соседа на любых данных",
        "- DROP: `bv2-b` -- дублирует bv2-a",
        "**DROP:** bv2-c - no edge over noise",
        "DROP: bv2-a — second mention is ignored",
        "Мы бы сделали DROP: not-a-line — inside a sentence",
        "DROP: missing-reason —",
        "DROP:",
    ])
    assert researcher.parse_drop_proposals(memo) == [
        researcher.DropProposal("bv2-a", "слабее соседа на любых данных"),
        researcher.DropProposal("bv2-b", "дублирует bv2-a"),
        researcher.DropProposal("bv2-c", "no edge over noise"),
    ]
    assert researcher.parse_drop_proposals("no proposals here") == []


def test_research_reads_drop_lines_after_the_run(tmp_path, monkeypatch) -> None:
    import subprocess
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from qlab.budget import ExplicitNoBudget
    from qlab.budget.agent import CLAUDE_BIN_ENV
    from qlab.registry.models import Base

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=tmp_path, check=True)
    monkeypatch.setattr(researcher, "QLAB_ROOT", tmp_path)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", tmp_path)
    today = datetime.now(UTC).date().isoformat()
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.2,
                                                              "resetsAt": reset}}}}) + "\n"
        + json.dumps({"type": "result", "is_error": False, "result": "done",
                      "total_cost_usd": 0.0, "usage": {"input_tokens": 1, "output_tokens": 1},
                      "modelUsage": {"claude-opus": {}}}) + "\n")
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\n"
                      "printf '# memo\\nDROP: v1 — no better than v2\\n' "
                      f"> docs/research/t-{today}.md\n"
                      f'cat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    outcome = researcher.research("t", idea_ids=[], specs=[], docs=[],
                                  budget=ExplicitNoBudget("test"), session=sessionmaker(
                                      bind=engine)())
    assert outcome.drops == [researcher.DropProposal("v1", "no better than v2")]


def test_drops_are_sorted_for_the_owner_and_change_no_status(tmp_path, monkeypatch) -> None:
    """2026-10-26: all six DROP lines were reported as "not accepted"; four named
    ideas already closed, two named bench specs with no idea in the registry."""
    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.test_night import _add_ideas

    with session_scope() as session:
        _add_ideas(session, [("i1", "paper", None), ("i1-v", "bench", "i1"),
                             ("i1-old", "rejected", "i1"), ("i2", "paper", None),
                             # a retired branch head that stays a variant through its active
                             # child (2026-10-29: proposed again a day after the owner retired it)
                             ("i1-gone", "retired", "i1"), ("i1-gone-b", "bench", "i1-gone")])
    config.write_text("cadence: nightly\nstrategies:\n  s1: {ideas: [i1], specs: []}\n",
                      encoding="utf-8")
    (tmp_path / "specs" / "owner").mkdir(parents=True)
    (tmp_path / "specs" / "owner" / "i1-m.yaml").write_text("idea_id: i1-m\n", encoding="utf-8")
    focus: list[str] = []

    def fake_research(name, **kw):
        calls.append(name)
        focus.append(kw["focus"])
        return researcher.ResearchOutcome(
            memo=None, drops=[researcher.DropProposal("i1-v", "weak"),
                              researcher.DropProposal("i1-old", "already out"),
                              researcher.DropProposal("i1-gone", "retired yesterday"),
                              researcher.DropProposal("i1-m", "measures nothing new"),
                              researcher.DropProposal("i2", "another strategy"),
                              researcher.DropProposal("ghost", "no such thing"),
                              researcher.DropProposal("i1", "the root")])

    monkeypatch.setattr(researcher, "research", fake_research)
    state = {"done": {}}
    night.research_stage(date(2026, 10, 26), state, noise_trials=1, config=config)
    entry = state["research"][0]
    assert entry["drops"] == [{"idea": "i1-v", "why": "weak"},
                              {"idea": "i1-m", "why": "measures nothing new",
                               "spec": "specs/owner/i1-m.yaml"}]
    assert entry["closed"] == [{"idea": "i1-old", "status": "rejected"},
                               {"idea": "i1-gone", "status": "retired"}]
    problems = " ".join(entry["problems"])
    assert "`i2`, an idea of strategy i2" in problems
    assert "`ghost`, which is neither" in problems and "`i1`, the strategy itself" in problems
    assert "DROP lines may name" in focus[0] and "i1-v" in focus[0]
    with session_scope() as session:  # the agent's word moved nothing
        assert session.get(Idea, "i1-v").status == IdeaStatus.BENCH
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    report = night.write_report(date(2026, 10, 26), state).read_text(encoding="utf-8")
    assert "предлагает вывести варианты" in report
    assert "- s1: `i1-v` — weak" in report
    assert "спек на скамейке `specs/owner/i1-m.yaml`" in report
    assert "`i1-old`, но он уже закрыт (rejected): делать нечего" in report
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_full_strategy_is_told_to_write_drop_lines() -> None:
    assert "DROP: <idea_id>" in night.full_strategy_focus("bv2", 3, ["a", "b", "c"])


# ---- the drawdown evidence in the researcher's context ----------------------------------


DAYS = pd.date_range("2025-01-01", periods=300, freq="1D", tz="UTC")


def _fake_panel() -> MarketPanel:
    steps = np.where(np.arange(300) % 9 < 4, 1.01, 0.99)
    return MarketPanel(
        snapshot_id="fake", prices=pd.DataFrame({"A": 100.0 * np.cumprod(steps)}, index=DAYS),
        funding=pd.DataFrame(0.0, index=DAYS, columns=["A"]),
        tradeable=pd.DataFrame(True, index=DAYS, columns=["A"]), meta={})


def _write_spec(tmp_path: Path, fixed_at: str | None = None) -> Path:
    spec = {"idea_id": "fake", "title": "fake", "code_ref": "qlab.test_diagnose:_Seen",
            "data": {"source": "binance", "interval": "1d", "start": "2025-01-01",
                     "end": "2025-10-27"},
            "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
            "unexpressed_mechanisms": []}
    if fixed_at:
        spec.update(params_fixed_at=fixed_at, params_fixed_evidence="a test")
    path = tmp_path / "fake.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from qlab.registry.models import Base

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_context_carries_the_drawdown_evidence_and_the_cut_sentence(
        tmp_path, monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate
    import qlab.regimes
    from qlab.rules.loader import load_latest

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: _fake_panel())
    monkeypatch.setattr(qlab.regimes, "load", lambda: None)
    text = researcher.build_context("t", [], _session(), [_write_spec(tmp_path)])
    share = load_latest().holdout_share.share  # read from the rules, not written in the code
    cut = DAYS[int(round(300 * (1 - share)))].date()
    assert (f"До даты отреза {cut}; данные после неё — для проверки исправлений, "
            "на них не смотреть.") in text
    assert "## Drawdown evidence:" in text and "idea fake" in text
    assert "Deepest drawdowns" in text and "Worst calendar months" in text
    assert "no stored regime labels" in text
    assert f"held-out share is read from ruleset {load_latest().version}" in text


def test_context_uses_the_specs_params_fixed_at_as_the_cut(tmp_path, monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate
    import qlab.regimes

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: _fake_panel())
    monkeypatch.setattr(qlab.regimes, "load", lambda: None)
    text = researcher.build_context("t", [], _session(), [_write_spec(tmp_path, "2025-06-01")])
    assert "До даты отреза 2025-06-01;" in text
    assert "params_fixed_at 2025-06-01" in text


def test_context_says_so_when_the_panel_is_not_cached(tmp_path, monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    def never_fetch(*_a, **_k):
        raise AssertionError("the evidence must never fetch data")

    monkeypatch.setattr(evaluate, "resolve_panel", never_fetch)
    monkeypatch.setattr(evaluate, "build_snapshot", never_fetch)
    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)
    text = researcher.build_context("t", [], _session(), [_write_spec(tmp_path)])
    assert "is not cached" in text and "nothing is fetched" in text
    assert "До даты отреза" not in text


def test_a_broken_spec_does_not_stop_the_context(tmp_path) -> None:
    text = researcher.build_context("t", [], _session(), [tmp_path / "missing.yaml"])
    assert "evidence not built" in text


def test_the_instructions_frame_the_work_on_the_owners_terms() -> None:
    text = researcher.INSTRUCTIONS.format(
        name="n", context_path="c", docs="d", specs="s", memo_path="m", spec_dir="sd",
        max_experiments=2, today="2026-10-26", focus="")
    assert "ignore that" in text and "NOT a reason to stop" in text
    assert "DROP: <idea_id>" in text and "LAST THIRD" in text and "preliminary" in text
    assert "data.source: binance" in text
    assert "Never set params_fixed_at to 2026-10-26" in " ".join(text.split())


def test_the_owners_ideas_reach_the_researcher_verbatim() -> None:
    # Owner, 2026-10-25: «закинь идею для researcher подумать о том как включать/отключать
    # хеджирование почаще».
    from qlab import night

    assert night.owner_ideas_focus(None) is None and night.owner_ideas_focus([]) is None
    text = night.owner_ideas_focus(["включать/отключать хеджирование почаще"])
    assert "«включать/отключать хеджирование почаще»" in text and "hypothesis" in text
    plan = night.load_research_plan()
    assert any("хеджирование почаще" in i for i in plan["strategies"]["bv2"]["owner_ideas"])


def test_five_researcher_strategies_leave_one_of_the_six_places_free() -> None:
    """Owner, 2026-10-25: FRAB with the spot on Binance is a new strategy with its
    own place; the five of night/research.yaml fit under night/limits.yaml's six."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from qlab.registry.models import Base

    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        free, cap, active = night.strategy_slots(
            session, night.load_research_plan(REPO / "night/research.yaml"),
            REPO / "night/limits.yaml")
    assert cap == 6 and len(active) == 5 and free == 1
    assert "frab-binance-spot" in active


def test_a_decided_focus_is_passed_as_a_rule_not_a_question() -> None:
    from qlab.night import plan_focus

    assert plan_focus({"focus": "q?"}) == "q?"
    assert plan_focus({}) is None
    ruled = plan_focus({"focus": "q?", "focus_decided": "2026-10-25"})
    assert "«q?»" in ruled and "do not answer" in ruled and "2026-10-25" in ruled


def test_experiments_cut_by_the_stop_time_run_first_on_the_next_night(
        tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    ran: list[str] = []
    monkeypatch.setattr(researcher, "run_experiments", lambda memo, specs, **kw: (
        ran.append(str(memo)) or ["- ok"]))
    monkeypatch.setattr(night, "record_research_note", lambda entry, plan: entry.update(noted=1))
    monday = {"done": {}, "research": [{"strategy": "s1", "memo": "m1.md", "specs": ["x.yaml"]}]}
    late = datetime.now(UTC) - timedelta(minutes=1)
    night.research_stage(date(2026, 10, 26), monday, noise_trials=1, config=config, stop_at=late)
    assert calls == [] and ran == [] and monday["stopped_at"]["before"] == "research:s1"
    night.save_state(date(2026, 10, 26), monday)
    tuesday = {"done": {}}
    night.research_stage(date(2026, 10, 27), tuesday, noise_trials=1, config=config)
    assert ran == ["m1.md"]  # yesterday's memo's experiments, then tonight's agents
    assert tuesday["research"][0]["carried_from"] == "2026-10-26"
    assert tuesday["research"][0]["noted"] == 1
    night.research_stage(date(2026, 10, 27), tuesday, noise_trials=1, config=config)
    assert ran == ["m1.md"]  # carried once
    db_module._engine = None
    db_module._SessionLocal = None
