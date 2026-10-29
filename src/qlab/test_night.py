"""Tests for `qlab.night`: dates, data extension, resume, report."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import qlab.night as night
from qlab.night import ItemResult, last_closed_day, with_end
from qlab.pipeline.spec import StrategySpec


def _spec() -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": {},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def test_last_closed_day_and_extension_never_shortens() -> None:
    assert last_closed_day(datetime(2026, 10, 3, 5, 30, tzinfo=UTC)) == date(2026, 10, 2)
    assert with_end(_spec(), date(2026, 10, 2)).data.end == date(2026, 10, 2)
    assert with_end(_spec(), date(2026, 9, 1)).data.end == date(2026, 9, 20)


def test_a_stopped_night_resumes_where_it_left_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: [a.yaml, b.yaml]\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "load_spec",
                        lambda p: _spec().model_copy(update={"idea_id": str(p)}))
    calls = []

    def fake_eval(kind, spec, **kw):
        calls.append(spec.idea_id)
        if spec.idea_id == "b.yaml" and len(calls) == 2:
            raise KeyboardInterrupt  # the night is stopped mid-run
        return ItemResult(kind=kind, name=spec.idea_id, route="needs-forward",
                          metrics={"ann_return_net": 0.05, "sharpe_net": 1.0})

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    day = date(2026, 10, 3)
    try:
        night.run(day, graveyard=False, paper=False, implement=False, search=False,
                  detectors=False, assess=False, research=False, waiting=False)
    except KeyboardInterrupt:
        pass
    report = night.run(day, graveyard=False, paper=False, implement=False, search=False,
                       detectors=False, assess=False, research=False, waiting=False)
    assert calls == ["a.yaml", "b.yaml", "b.yaml"]  # a was not redone
    text = report.read_text()
    assert "a.yaml" in text and "b.yaml" in text and "Токены" in text


def test_report_keeps_the_forward_test_out_of_yearly_figures() -> None:
    from qlab.night import _report_row

    # Nineteen forward days at +390% a year must not be shown as +390%.
    row = _report_row({"ann_return_net": 3.9, "sharpe_net": 6.4, "forward_days": 19.0,
                       "fit_ann_return_net": 0.068, "fit_sharpe_net": 0.97,
                       "fit_regime_bull_return": 0.31, "regime_flat_return": 0.066})
    yearly, sharpe, days, forward_total, bull, flat, bear = row
    assert (yearly, sharpe, days) == ("+6.8%", "0.97", "19")
    assert forward_total == f"{(1 + 3.9) ** (19 / 365.25) - 1:+.1%}"
    assert (bull, flat, bear) == ("+31.0%", "—", "—")  # the selection period's split only


def test_report_without_a_forward_test_shows_the_whole_window() -> None:
    from qlab.night import _report_row

    row = _report_row({"ann_return_net": 0.076, "sharpe_net": 1.07, "forward_days": 0.0,
                       "regime_bear_return": 0.124})
    assert row == ("+7.6%", "1.07", "0", "—", "—", "—", "+12.4%")


def test_detectors_are_rebuilt_before_the_watch_list_and_reported(tmp_path, monkeypatch) -> None:
    import qlab.detector_builds as db
    from qlab import night

    order = []
    config = tmp_path / "detectors.yaml"
    config.write_text("regimes: false\n")
    monkeypatch.setattr(db, "CONFIG", config)
    monkeypatch.setattr(db, "build_all", lambda end, config=config: order.append("detectors")
                        or [{"name": "level1 x", "ok": True, "note": "2020-01-01..2026-10-03"}])
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    path = night.run(date(2026, 10, 4), graveyard=False, paper=False, implement=False,
                     search=False, assess=False, research=False, waiting=False)
    assert order == ["detectors"]
    assert "level1 x" in path.read_text(encoding="utf-8")


def test_report_with_only_a_forward_test_and_with_a_nan() -> None:
    from qlab.night import _report_row

    row = _report_row({"ann_return_net": 0.1, "sharpe_net": 1.0, "forward_days": 365.25})
    assert row[:4] == ("—", "—", "365", "+10.0%")
    row = _report_row({"ann_return_net": float("nan"), "fit_ann_return_net": 0.05,
                       "fit_sharpe_net": 0.5, "forward_days": 2.0})
    assert row[3] == "—"


def test_bench_ideas_nobody_follows_are_named(tmp_path, monkeypatch) -> None:
    """2026-10-10: seventeen bench ideas were finished experiments whose
    forward tests could never grow. The report says which ideas wait in vain."""
    import qlab.registry.db as db_module
    from qlab import night
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Base, IdeaStatus, Profile, SourceType

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "q.db"))
    db_module._engine = None
    db_module._SessionLocal = None
    Base.metadata.create_all(db_module.get_engine())
    with session_scope() as session:
        for idea_id in ("followed", "forgotten", "forgotten-hl"):
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                             notes="Second stage of forgotten (docs/TWO_STAGE.md)"
                             if idea_id == "forgotten-hl" else None)
            repo.set_status(session, idea_id=idea_id, new_status=IdeaStatus.BENCH)
        session.get(repo.Idea, "forgotten-hl").parent_id = "forgotten"

    class _Spec:
        idea_id = "followed"

    monkeypatch.setattr(night, "load_spec", lambda path: _Spec())
    assert night.unwatched_bench(["specs/followed.yaml"]) == ["forgotten"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_second_opinion_where_the_verdict_changed_or_money_is_at_stake() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from qlab.registry import repo
    from qlab.registry.models import AssetClass, Base, IdeaStatus, Profile, SourceType

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for idea, status in (("paper", IdeaStatus.PAPER), ("bench", IdeaStatus.BENCH),
                         ("moved", IdeaStatus.BENCH)):
        repo.upsert_idea(session, id=idea, title=idea, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                         status=status)
    results = [
        {"kind": "watch", "name": "paper", "trial_id": 1, "route": "needs-forward",
         "previous_route": "needs-forward"},
        {"kind": "watch", "name": "bench", "trial_id": 2, "route": "needs-forward",
         "previous_route": "needs-forward"},
        {"kind": "watch", "name": "moved", "trial_id": 3, "route": "reject",
         "previous_route": "needs-forward"},
        {"kind": "watch", "name": "moved", "trial_id": None, "error": "boom"},
    ]
    assert [r["name"] for r in night.worth_assessing(session, results)] == ["paper", "moved"]


def test_the_weekly_researcher_builds_on_its_last_memo(tmp_path, monkeypatch) -> None:
    memos = tmp_path / "research"
    memos.mkdir()
    for day in ("2026-10-03", "2026-10-24"):
        (memos / f"bv2-{day}.md").write_text("memo")
    (memos / "trend-2026-10-03.md").write_text("memo")
    assert night.latest_memo("bv2", memos) == memos / "bv2-2026-10-24.md"
    assert night.latest_memo("frab", memos) is None


def test_no_research_config_no_research(tmp_path) -> None:
    state = {"done": {}}
    night.research_stage(date(2026, 10, 25), state, noise_trials=1,
                         config=tmp_path / "missing.yaml")
    assert "research" not in state


def test_research_stage_stores_each_memo_as_notes_for_the_strategys_ideas(
        tmp_path, monkeypatch) -> None:
    import qlab.agents.researcher as researcher
    import qlab.budget as budget_module
    import qlab.registry.db as db_module
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Base, Profile, ResearchNote, SourceType

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    Base.metadata.create_all(db_module.get_engine())
    with session_scope() as session:
        for idea_id in ("frab", "frab-2"):
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    config = tmp_path / "research.yaml"
    config.write_text("max_experiments: 1\nstrategies:\n  frab:\n    ideas: [frab, frab-2]\n"
                      "    specs: [specs/frab-live.yaml]\n", encoding="utf-8")

    class _Budget:
        spent_tokens = 4321

    class _Stage:
        def candidate(self, _name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, _stage, candidates):
            return _Stage()

    class _Reading:
        class report:  # noqa: N801 - mimics the probe's result shape
            reading = None

    def fake_research(name, **_kw):
        memo = tmp_path / "docs" / "research" / f"{name}-2026-10-25.md"
        memo.parent.mkdir(parents=True)
        memo.write_text("The memo.\n", encoding="utf-8")
        return researcher.ResearchOutcome(
            memo=Path("docs/research") / memo.name,
            specs=[Path("specs/research/frab/frab-exp.yaml")], problems=["one problem"])

    def fake_run_experiments(memo, spec_paths, *, name, noise_trials):
        lines = ["", "## Эксперименты на стенде (записано кодом, не исследователем)", "",
                 "- `frab-exp` (прогон 99): needs-forward — fine"]
        with memo.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return lines

    monkeypatch.setattr(researcher, "research", fake_research)
    monkeypatch.setattr(researcher, "run_experiments", fake_run_experiments)
    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe", lambda _session: _Reading)

    state = {"done": {}}
    night.research_stage(date(2026, 10, 25), state, noise_trials=1, config=config)
    night.research_stage(date(2026, 10, 25), state, noise_trials=1, config=config)

    assert state["research"][0]["noted"] == 2
    with session_scope() as session:
        rows = session.query(ResearchNote).order_by(ResearchNote.idea_id).all()
        assert [r.idea_id for r in rows] == ["frab", "frab-2"]
        row = rows[0]
        assert row.name == "frab" and row.memo_path == "docs/research/frab-2026-10-25.md"
        assert "The memo." in row.memo and "прогон 99" in row.memo
        assert row.experiments == [{"idea_id": "frab-exp", "trial_id": 99,
                                    "route": "needs-forward",
                                    "spec": "specs/research/frab/frab-exp.yaml"}]
        assert (row.problems, row.tokens, row.model) == (["one problem"], 4321, "opus")
    db_module._engine = None
    db_module._SessionLocal = None


# ---- the idea generator stage ------------------------------------------------


def test_generator_cadence_is_enabled_weekly_with_a_catch_up_after_a_stop(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    wednesday, thursday = date(2026, 10, 21), date(2026, 10, 22)
    plan = {"enabled": True, "cadence": "weekly", "weekday": "wednesday"}
    assert night.generate_due(wednesday, plan)
    assert not night.generate_due(thursday, plan)
    assert not night.generate_due(wednesday, {**plan, "enabled": False})
    assert night.generate_due(thursday, {**plan, "cadence": "daily"})
    assert night.generate_due(wednesday, {"enabled": True, "weekday": 2})
    # Wednesday's run was stopped by the budget before the agent finished: Thursday goes on.
    night.save_state(wednesday, {"done": {}, "generate": {"stopped": "budget"}})
    assert night.generate_due(thursday, plan)
    night.save_state(wednesday, {"done": {"generate:agent": "t"}, "generate": {}})
    assert not night.generate_due(thursday, plan)
    try:
        night.generate_due(wednesday, {"enabled": True, "weekday": "someday"})
    except ValueError as exc:
        assert "weekday" in str(exc)
    else:
        raise AssertionError("an unknown weekday must fail loudly")


def _generator_env(tmp_path, monkeypatch):
    """A throwaway registry, a fake budget guard and a generator config."""
    import qlab.agents.generator as generator
    import qlab.budget as budget_module
    import qlab.registry.db as db_module
    from qlab.registry.models import Base

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    Base.metadata.create_all(db_module.get_engine())
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "MECHANISMS.md").write_text("### m1\n", encoding="utf-8")
    spec_dir = tmp_path / "specs" / "generated"
    spec_dir.mkdir(parents=True)
    spec_text = ("idea_id: gen-a\ntitle: Тест\ncode_ref: qlab.strategies.blocks:Composed\n"
                 "data: {source: binance, interval: 1d, start: 2019-09-08, end: 2026-10-24}\n"
                 "costs: {taker_fee_bps: 3.5, slippage_bps: 0.9}\nmin_leg_notional: 10.0\n"
                 "unexpressed_mechanisms: []\n")
    (spec_dir / "gen-a.yaml").write_text(spec_text, encoding="utf-8")
    config = tmp_path / "generate.yaml"
    config.write_text("enabled: true\ncadence: weekly\nweekday: wednesday\nmax_specs: 2\n",
                      encoding="utf-8")

    class _Budget:
        spent_tokens = 1234

    class _Stage:
        def candidate(self, _name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, stage, candidates):
            _NightBudget.stage_used = (stage, candidates)
            return _Stage()

    class _Reading:
        class report:  # noqa: N801 - mimics the probe's result shape
            reading = None

    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe", lambda _session: _Reading)
    return generator, budget_module, config, _NightBudget


def test_generator_stage_runs_once_resumes_and_reports(tmp_path, monkeypatch) -> None:
    import qlab.registry.db as db_module
    from qlab.budget import Stage

    generator, _, config, fake_night = _generator_env(tmp_path, monkeypatch)
    calls = {"agent": 0, "stand": []}

    def fake_generate(*, budget, session, catalogue_path, max_specs, data_end, registry_idea_ids,
                      max_own=1):
        calls["agent"] += 1
        calls["args"] = (max_specs, data_end, list(registry_idea_ids), catalogue_path)
        item = generator.GeneratedSpec(
            path=Path("specs/generated/gen-a.yaml"),
            spec=night.load_spec(Path("specs/generated/gen-a.yaml")), mechanism="m1")
        return generator.GenerateOutcome(
            accepted=[item], reply="gen-a: m1, params_fixed_at 2019-09-08",
            rejected=[generator.Rejected("specs/generated/gen-b.yaml", ["code_ref is wrong"])],
            problems=["wrote outside its places (reverted): ['X']"])

    def fake_stand(item, *, noise_trials, deployable_capital_usd):
        calls["stand"].append((item.spec.idea_id, noise_trials))
        if len(calls["stand"]) == 1:
            return generator.GeneratedResult(
                idea_id="gen-a", mechanism="m1", spec_path=str(item.path), error="boom")
        return generator.GeneratedResult(
            idea_id="gen-a", mechanism="m1", spec_path=str(item.path), trial_id=9,
            route="needs-forward", metrics={"ann_return_net": 0.05, "sharpe_net": 0.5,
                                            "max_dd": -0.1, "noise_return_percentile": 0.7})

    monkeypatch.setattr(generator, "generate", fake_generate)
    monkeypatch.setattr(generator, "run_generated_one", fake_stand)

    thursday = date(2026, 10, 22)
    state = {"done": {}}
    night.generate_stage(thursday, state, noise_trials=3, config=config)
    assert "generate" not in state and calls["agent"] == 0  # not its day

    wednesday = date(2026, 10, 21)
    night.generate_stage(wednesday, state, noise_trials=3, config=config)
    assert fake_night.stage_used == (Stage.SEARCH, 1)  # the stage the budget cuts first
    assert calls["args"][0] == 2 and calls["args"][1] <= date(2026, 10, 20)
    assert calls["args"][3] == Path("docs/MECHANISMS.md")
    gen_state = state["generate"]
    assert gen_state["tokens"] == 1234 and gen_state["accepted"][0]["idea_id"] == "gen-a"
    assert "generate:agent" in state["done"]
    assert "generate:run:gen-a" not in state["done"]  # the first stand run failed: retried
    assert gen_state["results"][0]["error"] == "boom"

    night.generate_stage(wednesday, state, noise_trials=3, config=config)  # a restart
    assert calls["agent"] == 1  # the agent is not paid for twice
    assert [c[0] for c in calls["stand"]] == ["gen-a", "gen-a"]
    assert "generate:run:gen-a" in state["done"] and len(gen_state["results"]) == 1
    night.generate_stage(wednesday, state, noise_trials=3, config=config)
    assert len(calls["stand"]) == 2  # nothing left to do

    night.save_state(wednesday, state)
    monkeypatch.setattr(night, "WATCHLIST", tmp_path / "watch.yaml")
    (tmp_path / "watch.yaml").write_text("specs: []\n")
    report = night.write_report(wednesday, state).read_text(encoding="utf-8")
    assert "## Генератор идей" in report
    assert "принято стратегий: 1, не принято: 1" in report
    assert "не принято `specs/generated/gen-b.yaml`: code_ref is wrong" in report
    assert "`gen-a` (механизм m1, прогон 9): needs-forward" in report
    db_module._engine = None
    db_module._SessionLocal = None


def test_generator_stage_retries_only_the_second_stage_and_reports_both(
        tmp_path, monkeypatch) -> None:
    """Stage 2 (the same mechanism on Hyperliquid) is code, in the same stage as stage 1.
    When only it failed, a restart runs it again -- not the stage 1 trial and its 200
    noise books a second time -- and the report puts the two stages side by side."""
    import qlab.registry.db as db_module

    generator, _, config, _ = _generator_env(tmp_path, monkeypatch)
    calls = {"stand": 0, "second": 0}

    def fake_generate(*, budget, session, catalogue_path, max_specs, data_end, max_own=1,
                      registry_idea_ids):
        item = generator.GeneratedSpec(
            path=Path("specs/generated/gen-a.yaml"),
            spec=night.load_spec(Path("specs/generated/gen-a.yaml")), mechanism="m1")
        return generator.GenerateOutcome(accepted=[item], reply="ok")

    def fake_stand(item, *, noise_trials, deployable_capital_usd):
        calls["stand"] += 1
        return generator.GeneratedResult(
            idea_id="gen-a", mechanism="m1", spec_path=str(item.path), trial_id=9, route="paper",
            metrics={"ann_return_net": 0.1, "sharpe_net": 0.8, "max_dd": -0.1,
                     "noise_return_percentile": 0.9},
            stage2={"status": "error", "error": "ConnectionError: hyperliquid is unreachable",
                    "stage1_route": "paper"})

    def fake_second(item, result, *, noise_trials, deployable_capital_usd):
        calls["second"] += 1
        result.stage2 = {
            "status": "evaluated", "stage1_route": "paper", "route": "reject", "trial_id": 10,
            "reason": "fails on the selection period: capital_fit", "venue": "hyperliquid",
            "metrics": {"ann_return_net": -0.2, "sharpe_net": -0.5, "max_dd": -0.4,
                        "noise_return_percentile": 0.1},
            "judged_from": "2025-01-30", "needed_coins": 5, "data_end": "2026-10-24",
            "combined_route": "reject", "parent_status": "rejected"}

    monkeypatch.setattr(generator, "generate", fake_generate)
    monkeypatch.setattr(generator, "run_generated_one", fake_stand)
    monkeypatch.setattr(generator, "run_second_stage", fake_second)

    wednesday = date(2026, 10, 21)
    state = {"done": {}}
    night.generate_stage(wednesday, state, noise_trials=3, config=config)
    assert calls == {"stand": 1, "second": 0}
    assert "generate:run:gen-a" not in state["done"]  # stage 2 failed: not finished

    night.generate_stage(wednesday, state, noise_trials=3, config=config)  # a restart
    assert calls == {"stand": 1, "second": 1}  # stage 1 was not run again
    assert "generate:run:gen-a" in state["done"]
    [result] = state["generate"]["results"]
    assert result["stage2"]["trial_id"] == 10 and result["route"] == "paper"

    night.generate_stage(wednesday, state, noise_trials=3, config=config)
    assert calls == {"stand": 1, "second": 1}  # nothing left to do

    night.save_state(wednesday, state)
    monkeypatch.setattr(night, "WATCHLIST", tmp_path / "watch.yaml")
    (tmp_path / "watch.yaml").write_text("specs: []\n")
    report = night.write_report(wednesday, state).read_text(encoding="utf-8")
    assert "`gen-a` (механизм m1, прогон 9): paper" in report
    assert "второй этап, Hyperliquid (тот же механизм и те же числа" in report
    assert "прогон 10: отвергнута" in report
    assert "итог по двум этапам: отвергнута на втором этапе" in report
    db_module._engine = None
    db_module._SessionLocal = None


def test_generator_stage_stopped_by_the_budget_waits_for_the_next_night(
        tmp_path, monkeypatch) -> None:
    import qlab.registry.db as db_module
    from qlab.budget import NightExhausted

    generator, _, config, _ = _generator_env(tmp_path, monkeypatch)

    def exhausted(**_kw):
        raise NightExhausted("weekly utilization 70% reached tonight's target")

    monkeypatch.setattr(generator, "generate", exhausted)
    wednesday = date(2026, 10, 21)
    state = {"done": {}}
    night.generate_stage(wednesday, state, noise_trials=1, config=config)
    assert "target" in state["generate"]["stopped"] and "generate:agent" not in state["done"]
    night.save_state(wednesday, state)
    assert night.generate_due(date(2026, 10, 22), {"enabled": True, "weekday": "wednesday"})
    db_module._engine = None
    db_module._SessionLocal = None


def test_generator_stage_without_a_catalogue_or_when_the_scout_was_stopped(
        tmp_path, monkeypatch) -> None:
    import qlab.registry.db as db_module

    generator, _, config, _ = _generator_env(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(generator, "generate", lambda **kw: called.append(kw))
    day = date(2026, 10, 21)

    state = {"done": {}, "scout": {"reply": None, "stopped": "night budget"}}
    night.generate_stage(day, state, noise_trials=1, config=config)
    assert "ended in the search stage" in state["generate"]["stopped"] and not called

    (tmp_path / "docs" / "MECHANISMS.md").unlink()
    state = {"done": {}}
    night.generate_stage(day, state, noise_trials=1, config=config)
    assert "no catalogue" in state["generate"]["skipped"] and not called

    state = {"done": {}}
    night.generate_stage(day, state, noise_trials=1, config=tmp_path / "missing.yaml")
    assert "generate" not in state  # no config, not forced
    night.generate_stage(day, state, noise_trials=1, config=config, force=False)
    assert "generate" not in state
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_night_without_search_does_not_run_the_generator(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    seen, order = [], []
    monkeypatch.setattr(night, "generate_stage", lambda *a, **kw: (
        seen.append(kw.get("force")), order.append("generate")))
    monkeypatch.setattr(night, "search_stage", lambda *a, **kw: order.append("search"))
    monkeypatch.setattr(night, "catalogue_stage", lambda *a, **kw: order.append("catalogue"))
    flags = dict(graveyard=False, paper=False, implement=False, detectors=False, assess=False,
                 research=False, waiting=False, programmer=False)
    night.run(date(2026, 10, 21), search=False, **flags)
    assert seen == [] and order == []
    night.run(date(2026, 10, 21), search=True, **flags)
    # a mechanism the cataloguer writes tonight can be built by the generator tonight
    assert order == ["search", "catalogue", "generate"]
    night.run(date(2026, 10, 21), search=False, generate=True, **flags)
    assert seen == [None, True]


def _memory_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from qlab.registry.models import Base

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_ideas(session, rows) -> None:
    """rows: (id, status, parent_id or None)."""
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    for idea_id, status, _parent in rows:
        session.add(Idea(id=idea_id, title=idea_id, status=status, parent_id=None,
                         source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                         profile=Profile.OTHER))
    session.flush()
    for idea_id, _status, parent in rows:
        session.get(Idea, idea_id).parent_id = parent
    session.flush()


_PLAN = {"strategies": {
    "bv2": {"ideas": ["strategy-b-v2", "bv2-binance"]},
    "trend": {"ideas": ["trend-live"]},
    "frab": {"ideas": ["frab"]},
    "xsmom": {"ideas": ["xsmom"]},
}}


def test_strategy_slots_count_strategies_not_ideas(tmp_path) -> None:
    # Owner, 2026-10-25: «давай увеличим число до 6», «4 стратегии где он может работать».
    session = _memory_session()
    _add_ideas(session, [
        ("strategy-b-v2", "paper", None), ("bv2-binance", "bench", "strategy-b-v2"),
        ("bv2-research-x", "bench", "strategy-b-v2"), ("trend-live", "live", None),
        ("frab", "live", None), ("xsmom", "bench", None),
        ("gen-a", "candidate", None), ("gen-b", "bench", None),
        ("gen-variant", "bench", "gen-a"),  # a variant of a generated strategy: no place
        ("gen-old", "rejected", None),
        ("gen-under-bv2", "bench", "strategy-b-v2"),  # a generated variant of Bv2: no place
    ])
    limits = tmp_path / "limits.yaml"
    limits.write_text("max_active_strategies: 6\n", encoding="utf-8")
    free, cap, active = night.strategy_slots(session, _PLAN, limits)
    assert (free, cap) == (0, 6)
    assert active == ["bv2", "trend", "frab", "xsmom", "gen-a", "gen-b"]
    limits.write_text("max_active_strategies: 7\n", encoding="utf-8")
    assert night.strategy_slots(session, _PLAN, limits)[0] == 1
    limits.write_text("max_active_strategies: 3\n", encoding="utf-8")
    assert night.strategy_slots(session, _PLAN, limits)[0] == 0
    assert night.strategy_slots(session, _PLAN, tmp_path / "missing.yaml")[0] > 100
    # Variants alone never fill the places: the plan's four strategies leave two.
    limits.write_text("max_active_strategies: 6\n", encoding="utf-8")
    assert night.strategy_slots(session, _PLAN, limits)[0] == 0
    assert night.strategy_slots(session, {"strategies": {}}, limits)[0] == 4


def test_a_generator_idea_that_is_a_researched_strategys_root_is_not_counted_twice(
        tmp_path) -> None:
    session = _memory_session()
    _add_ideas(session, [("gen-root", "bench", None)])
    limits = tmp_path / "limits.yaml"
    limits.write_text("max_active_strategies: 6\n", encoding="utf-8")
    plan = {"strategies": {"gen": {"ideas": ["gen-root"]}}}
    assert night.strategy_slots(session, plan, limits) == (5, 6, ["gen"])


def test_variant_slots_and_the_drop_one_instruction(tmp_path) -> None:
    session = _memory_session()
    _add_ideas(session, [
        ("strategy-b-v2", "paper", None), ("v1", "bench", "strategy-b-v2"),
        ("v2", "candidate", "v1"), ("v3", "validated", "strategy-b-v2"),
        ("v4", "retired", "strategy-b-v2")])
    limits = tmp_path / "limits.yaml"
    limits.write_text("max_active_variants_per_strategy: 3\n", encoding="utf-8")
    entry = {"ideas": ["v2"]}  # any idea of the chain leads to the same root
    # v2 is a variant of v1 (the same idea on other data): v1's branch takes one place.
    assert night.variant_slots(session, entry, limits) == (1, 3, ["v1", "v3"])
    limits.write_text("max_active_variants_per_strategy: 2\n", encoding="utf-8")
    assert night.variant_slots(session, entry, limits)[0] == 0
    assert night.variant_slots(session, entry, tmp_path / "missing.yaml")[0] > 100
    text = night.full_strategy_focus("bv2", 3, ["v1", "v2", "v3"])
    assert "Propose NO new experiment" in text and "v1, v2, v3" in text and "`bv2`" in text


def test_research_stage_limits_experiments_per_strategy_and_says_what_to_drop(
        tmp_path, monkeypatch) -> None:
    import qlab.agents.researcher as researcher
    import qlab.budget as budget_module
    import qlab.registry.db as db_module
    from qlab.registry.db import session_scope
    from qlab.registry.models import Base

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    (tmp_path / "night").mkdir()
    Base.metadata.create_all(db_module.get_engine())
    (tmp_path / "night" / "limits.yaml").write_text(
        "max_active_strategies: 6\nmax_active_variants_per_strategy: 3\n", encoding="utf-8")
    with session_scope() as session:
        _add_ideas(session, [
            ("bv2", "paper", None), ("bv2-a", "bench", "bv2"), ("bv2-b", "bench", "bv2"),
            ("bv2-c", "candidate", "bv2"),  # Bv2 is full
            ("frab", "live", None), ("frab-a", "bench", "frab"),  # FRAB has two free places
            ("trend", "live", None)])
    config = tmp_path / "research.yaml"
    config.write_text(
        "max_experiments: 2\nstrategies:\n  bv2: {ideas: [bv2], specs: []}\n"
        "  frab: {ideas: [frab], specs: []}\n  trend: {ideas: [trend], specs: []}\n",
        encoding="utf-8")

    class _Budget:
        spent_tokens = 1

    class _Stage:
        def candidate(self, _name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, _stage, candidates):
            return _Stage()

    class _Reading:
        class report:  # noqa: N801 - mimics the probe's result shape
            reading = None

    calls = {}

    def fake_research(name, **kw):
        calls[name] = kw
        return researcher.ResearchOutcome(memo=None, specs=[], problems=[])

    monkeypatch.setattr(researcher, "research", fake_research)
    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe", lambda _session: _Reading)
    night.research_stage(date(2026, 10, 26), {"done": {}}, noise_trials=1, config=config)

    assert calls["bv2"]["max_experiments"] == 0
    assert "Propose NO new experiment" in calls["bv2"]["focus"]
    assert "bv2-a, bv2-b, bv2-c" in calls["bv2"]["focus"]
    assert calls["frab"]["max_experiments"] == 2 and "Propose NO" not in (
        calls["frab"]["focus"] or "")
    assert calls["trend"]["max_experiments"] == 2
    db_module._engine = None
    db_module._SessionLocal = None


def test_generator_stage_waits_when_no_place_is_free(tmp_path, monkeypatch) -> None:
    import qlab.registry.db as db_module
    from qlab.registry.models import Idea

    generator, _, config, _ = _generator_env(tmp_path, monkeypatch)
    calls = {"agent": 0}

    def fake_generate(**_kw):
        calls["agent"] += 1
        raise AssertionError("the agent must not be called when no place is free")

    monkeypatch.setattr(generator, "generate", fake_generate)
    (tmp_path / "night").mkdir()
    (tmp_path / "night" / "limits.yaml").write_text("max_active_strategies: 1\n",
                                                    encoding="utf-8")
    with db_module.session_scope() as session:
        session.add(Idea(id="gen-busy", title="t", status="bench", source_type="internal",
                         asset_class="crypto-perp", profile="other"))
    state = {"done": {}}
    night.generate_stage(date(2026, 10, 28), state, noise_trials=3, config=config)
    assert calls["agent"] == 0
    assert "gen-busy" in state["generate"]["skipped"]


def test_the_report_carries_the_agents_funnel_and_the_paper_rule(tmp_path, monkeypatch) -> None:
    """2026-10-25: the owner wants to see whether the agents' loop works (ideas
    found, how far they got, tokens per survivor) and how long each paper
    book has run against the 4-6 weeks rule."""
    import pandas as pd

    import qlab.registry.db as db_module
    from qlab import funnel, paper_check, paper_rule
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import (
        AssetClass,
        Base,
        IdeaStatus,
        Profile,
        SourceType,
        TokenSpend,
        TrialRoute,
        TrialStatus,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    monkeypatch.setattr(funnel, "NIGHT_DIR", tmp_path / "no-nights")
    Base.metadata.create_all(db_module.get_engine())
    day = date(2026, 10, 20)
    with session_scope() as session:
        idea = repo.upsert_idea(
            session, id="x-research-a", title="x", source_type=SourceType.INTERNAL,
            asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
            status=IdeaStatus.VALIDATED, notes="Declared by the researcher (x).")
        idea.created_at = datetime(2026, 10, 19, 5, tzinfo=UTC)
        spec = repo.add_spec(session, idea_id="x-research-a", version=1, params={},
                             data_requirements={}, rebalance="1d", costs_model={}, code_ref="a:b")
        repo.add_data_snapshot(session, id="snap", source="hyperliquid", instruments={},
                               range_start=date(2025, 1, 1), range_end=date(2026, 10, 18),
                               path="x", rows=1)
        repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="x", params={},
                       status=TrialStatus.OK, route=TrialRoute.PAPER, snapshot_id="snap")
        session.add(TokenSpend(at=datetime(2026, 10, 20, 3, tzinfo=UTC), stage="implement",
                               agent="researcher", tokens_in=400_000, tokens_out=100_000,
                               usd_est=0.0))

    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n  x-research-a:\n    book: trend\n    label: Тренд\n"
        "    paper_file: trend_equity.csv\n    strategy_id: 5\n    sum_coins: false\n"
        "    strategy_capital_usd: 3000\n    replay_block: trend\n", encoding="utf-8")
    monkeypatch.setattr(paper_check, "BOOKS", books)
    export = tmp_path / "paper" / "2026-10-20"
    export.mkdir(parents=True)
    start_ms = int(datetime(2026, 9, 14, tzinfo=UTC).timestamp() * 1000)
    ts = [start_ms + h * 3_600_000 for h in range(24 * 36 + 1)]
    pd.DataFrame({"strategy_id": 5, "ts_ms": ts, "equity": 3000.0}).to_csv(
        export / "trend_equity.csv", index=False)
    monkeypatch.setattr(paper_rule, "PAPER_DIR", tmp_path / "paper")

    state = {"done": {}, "paper": ["**trend**", "    paper  2999.29 -> 2963.24 (-1.20%)",
                                   "    replay 2999.29 -> 2962.79 (-1.22%)",
                                   "    daily return correlation 0.9998 (n=41)"]}
    report = night.write_report(day, state).read_text(encoding="utf-8")

    assert "## Воронка агентов" in report
    assert "Агенты придумали идей: 1 (генератор 0, исследователь 1, реализатор 0)" in report
    assert "Дошли до владельца (прошли все правила и ждут его решения): 1 (`x-research-a`)" \
        in report
    assert "цена одного выжившего — 500,000 токенов" in report
    assert "**Правило владельца: бумага не меньше 4–6 недель**" in report
    assert ("Тренд: на бумаге 5 недель из минимум 4–6 (с 14.09.2026); повтор бумаги на стенде "
            "сошёлся; по правилу владельца книга готова к решению "
            "(бумага не судит о доходности — судит стенд)") in report
    db_module._engine = None
    db_module._SessionLocal = None


# ---- stop time --------------------------------------------------------------------------


def test_stop_time_is_today_on_the_local_clock() -> None:
    now = datetime(2026, 10, 26, 0, 25, tzinfo=UTC)
    stop = night.stop_time("07:10", now)
    assert stop.astimezone().strftime("%H:%M") == "07:10"
    assert stop.astimezone().date() == now.astimezone().date()


def test_nothing_starts_after_the_stop_time_and_the_report_says_where(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: [a.yaml, b.yaml]\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "load_spec",
                        lambda p: _spec().model_copy(update={"idea_id": str(p)}))
    calls = []

    def fake_eval(kind, spec, **kw):
        calls.append(spec.idea_id)
        time.sleep(0.3)  # the stop time passes while a started item runs; it still finishes
        return ItemResult(kind=kind, name=spec.idea_id, route="needs-forward")

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    report = night.run(date(2026, 10, 3), graveyard=False, paper=False, implement=False,
                       search=False, detectors=False, assess=False, research=False,
                       waiting=False, stop_at=datetime.now(UTC) + timedelta(seconds=0.2))
    assert calls == ["a.yaml"]
    assert night.load_state(date(2026, 10, 3))["stopped_at"]["before"] == "recheck"
    assert "Ночь остановлена по сроку" in report.read_text()


def test_past_stop_records_only_the_first_cut_stage() -> None:
    state: dict = {}
    late = datetime.now(UTC) - timedelta(minutes=1)
    assert night.past_stop(None, state, "recheck") is False
    assert night.past_stop(datetime.now(UTC) + timedelta(hours=1), state, "recheck") is False
    assert "stopped_at" not in state
    assert night.past_stop(late, state, "waiting") and night.past_stop(late, state, "director")
    assert state["stopped_at"]["before"] == "waiting"


def test_a_failing_second_opinion_is_reported_and_the_night_goes_on(tmp_path, monkeypatch) -> None:
    # The night of 2026-10-27 stopped at 04:51 on one brief that could not be built: research,
    # the generator and the programmer never ran. One candidate's failure is now its row.
    from contextlib import contextmanager
    from types import SimpleNamespace

    import qlab.agents.assessor as assessor
    import qlab.budget as budget
    import qlab.registry.db as db

    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    fake_session = SimpleNamespace(commit=lambda: None, rollback=lambda: None)

    @contextmanager
    def scope():
        yield fake_session

    monkeypatch.setattr(db, "session_scope", scope)
    monkeypatch.setattr(budget, "probe", lambda s: SimpleNamespace(report=SimpleNamespace(
        reading=None)))
    stage = SimpleNamespace(candidate=lambda name: SimpleNamespace(spent_tokens=7))
    monkeypatch.setattr(budget, "NightBudget", SimpleNamespace(open=lambda s, r: SimpleNamespace(
        stage=lambda *a, **k: stage)))
    monkeypatch.setattr(night, "worth_assessing", lambda s, results: results)

    def brief(session, trial_id, paper=None):
        if trial_id == 1:
            raise AttributeError("'NoneType' object has no attribute 'items'")
        return trial_id

    monkeypatch.setattr(assessor, "build_brief", brief)
    monkeypatch.setattr(assessor, "run_assessment", lambda brief, budget, session: SimpleNamespace(
        view="agree", dispute=False, summary="ok"))
    state = {"done": {}, "results": [{"name": "broken", "trial_id": 1},
                                     {"name": "fine", "trial_id": 2}]}
    night.assess_stage(date(2026, 10, 27), state)
    broken, fine = state["assessed"]
    assert broken["stopped"].startswith("сбой: AttributeError")
    assert fine["view"] == "agree"
    assert "assess:1" not in state["done"] and "assess:2" in state["done"]  # retried next run


def test_a_failing_stage_is_a_report_line_and_the_rest_of_the_night_runs(
        tmp_path, monkeypatch) -> None:
    # The night of 2026-10-27 stopped on one stage's exception; the stages after it never ran.
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: [missing/broken.yaml]\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    ran = []

    def boom(*a, **kw):
        raise KeyError("strategy")

    monkeypatch.setattr(night, "research_stage", boom)
    monkeypatch.setattr(night, "search_stage", lambda *a, **kw: ran.append("search"))
    monkeypatch.setattr(night, "generate_stage", lambda *a, **kw: ran.append("generate"))
    flags = dict(graveyard=False, paper=False, implement=False, detectors=False, assess=False,
                 waiting=False, director=False, programmer=False)
    report = night.run(date(2026, 10, 27), research=True, **flags)
    assert ran == ["search", "generate"]
    state = night.load_state(date(2026, 10, 27))
    assert state["stage_errors"]["research"].startswith("KeyError: 'strategy'")
    text = report.read_text(encoding="utf-8")
    assert "## Сбои стадий" in text and "**исследователь** (`research`)" in text
    broken = state["results"][0]  # a broken spec file on the watch list is its own row
    assert broken["name"] == "missing/broken.yaml" and "FileNotFoundError" in broken["error"]


def test_the_graveyard_sweep_walks_the_whole_spec_tree_but_not_stages_or_variants(
        tmp_path, monkeypatch) -> None:
    """Owner, 2026-10-29 (docs/EVALUATION_PATHS.md П2): the sweep saw only the folder's own files;
    a second stage and an overlay variant are left out on purpose."""
    import qlab.registry.db as db_module
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Base, Idea, IdeaStatus, Profile, SourceType

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "g.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    ids = {"flat": IdeaStatus.REJECTED, "deep": IdeaStatus.REJECTED, "kept": IdeaStatus.BENCH,
           "deep-hl": IdeaStatus.REJECTED, "deep-ov-gate": IdeaStatus.REJECTED}
    with session_scope() as session:
        for idea_id, status in ids.items():
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                             status=status,
                             notes="Second stage of deep" if idea_id == "deep-hl" else None)
        session.get(Idea, "deep-hl").parent_id = "deep"
    specs = tmp_path / "specs"
    for rel, idea_id in (("flat.yaml", "flat"), ("research/x/deep.yaml", "deep"),
                         ("kept.yaml", "kept"), ("generated/stage2/deep-hl.yaml", "deep-hl"),
                         ("overlays/deep/deep-ov-gate.yaml", "deep-ov-gate")):
        (specs / rel).parent.mkdir(parents=True, exist_ok=True)
        (specs / rel).write_text(idea_id, encoding="utf-8")
    monkeypatch.setattr(night, "load_spec",
                        lambda p: _spec().model_copy(update={"idea_id": p.read_text()}))

    found = night.graveyard_specs(specs)

    assert [p.relative_to(specs).as_posix() for p in found] == ["flat.yaml", "research/x/deep.yaml"]
