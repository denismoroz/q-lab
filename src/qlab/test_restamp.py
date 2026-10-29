"""Tests for `qlab.restamp`: ideas whose latest run was measured an older way are measured again
(docs/RESTAMP.md). The stand and the second stage are replaced by recorders that write real
trials into a temporary registry, so the selection, the order, the bounds, the ledger and the
report are the real ones; nothing under `specs/` or in the owner's registry is touched."""

from __future__ import annotations

import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import qlab.registry.db as db_module
from qlab import night, restamp, waiting
from qlab.harness.version import STAND_VERSION, is_older, version_key
from qlab.pipeline import stage2
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    Spec,
    Trial,
    TrialRoute,
    TrialSource,
    TrialStatus,
    VerdictStage,
)
from qlab.registry.stages import NOTES_PREFIX
from qlab.rules.loader import load_latest

END = date(2026, 10, 26)
DAY = date(2026, 10, 27)
RULES = load_latest().version
OLD_RULES = "2026-09-20.1"
STAGE2_CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))
_REAL_RERUN_STAGE2 = waiting.rerun_stage2
CURRENT = "current"
"""A trial made now: the stand version the code records by itself."""


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


def _idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.BENCH, *,
          parent: str | None = None, notes: str | None = None) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     notes=notes)
    session.get(Idea, idea_id).parent_id = parent


def _trial(session, idea_id: str, route: str, *, stand: str | None = CURRENT,
           rules: str | None = RULES, metrics: dict | None = None,
           end: date = END) -> Trial:
    """A routed trial made under the stand version `stand` (None: the old trials, which have
    none) and judged under `rules` (None: no verdicts, as a not evaluable run)."""
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={},
        data_requirements={"end": end.isoformat()}, rebalance="1d", costs_model={},
        code_ref="a:b")
    snapshot = f"snap-{end}"
    if session.get(DataSnapshot, snapshot) is None:
        repo.add_data_snapshot(session, id=snapshot, source="hyperliquid", instruments={},
                               range_start=date(2025, 1, 1), range_end=end, path="x", rows=1)
    trial = repo.add_trial(
        session, spec_id=spec.id, config_hash="h", code_sha="c", params={}, status=TrialStatus.OK,
        snapshot_id=snapshot, route=TrialRoute(route), route_reason="r",
        metrics={"ann_return_net": 0.10, "max_dd": -0.20, **(metrics or {})})
    if stand != CURRENT:
        trial.stand_version = stand
    if rules is not None:
        repo.add_verdicts(session, [{
            "idea_id": idea_id, "spec_id": spec.id, "trial_id": trial.id,
            "stage": VerdictStage.EDGE,
            "rule_id": "r", "rules_version": rules, "metric": "m", "value": 1.0,
            "comparator": ">=", "threshold": 0.0, "passed": True,
            "data_range_start": date(2025, 1, 1), "data_range_end": end}])
    return trial


def _write_spec(path: Path, idea_id: str, **data) -> Path:
    spec = StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": "binance", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20", **data},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")
    return path


# ---- the stand version is recorded --------------------------------------------------------


def test_a_new_trial_carries_the_stand_version_and_an_imported_one_none(registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "a")
        made = _trial(s, "a", "paper")
        spec_id = made.spec_id
        imported = repo.add_trial(s, spec_id=spec_id, config_hash="h", code_sha="c", params={},
                                  status=TrialStatus.OK, source=TrialSource.IMPORTED)
        own = repo.add_trial(s, spec_id=spec_id, config_hash="h", code_sha="c", params={},
                             status=TrialStatus.OK, stand_version="2026-01-01.1",
                             snapshot_id=made.snapshot_id)
        assert made.stand_version == STAND_VERSION
        assert imported.stand_version is None
        assert own.stand_version == "2026-01-01.1"


def test_versions_compare_as_date_and_serial_and_a_missing_one_is_oldest() -> None:
    assert version_key("2026-10-28.1") == ("2026-10-28", 1)
    assert version_key("junk") is None and version_key(None) is None
    assert is_older(None, "2026-10-28.1") and is_older("junk", "2026-10-28.1")
    assert is_older("2026-10-27.9", "2026-10-28.1") and is_older("2026-10-28.1", "2026-10-28.2")
    assert not is_older("2026-10-28.2", "2026-10-28.2")
    assert not is_older("2026-10-28.10", "2026-10-28.9")  # 10 is after 9, not before
    assert version_key(STAND_VERSION) is not None


def test_the_migration_adds_the_column_without_touching_old_rows(tmp_path, monkeypatch) -> None:
    import sqlite3

    from alembic import command
    from alembic.config import Config as AlembicConfig

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "migrated.db"))
    root = Path(__file__).resolve().parents[2]
    cfg = AlembicConfig(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "src/qlab/registry/migrations"))
    command.upgrade(cfg, "c5d8a1f3e742")
    con = sqlite3.connect(tmp_path / "migrated.db")
    assert "stand_version" not in {r[1] for r in con.execute("PRAGMA table_info(trial)")}
    con.close()
    command.upgrade(cfg, "head")
    con = sqlite3.connect(tmp_path / "migrated.db")
    assert "stand_version" in {r[1] for r in con.execute("PRAGMA table_info(trial)")}
    con.close()


# ---- the world ----------------------------------------------------------------------------


@pytest.fixture()
def world(tmp_path, registry, monkeypatch):
    """Ideas in every state the selection tells apart, a spec file for each that has one, and
    the stand and the second stage recorded (they write real trials under the present versions)."""
    specs = tmp_path / "specs"
    for name in ("old-pass", "fresh", "old-rules", "old-bench", "parent", "variant-ov-gate",
                 "orphan-parent", "retired-one", "no-run", "pending"):
        _write_spec(specs / f"{name}.yaml", name)
    config = tmp_path / "restamp.yaml"
    config.write_text(yaml.safe_dump({
        "enabled": True, "max_ideas_per_night": 4, "slow_idea_minutes": 30, "max_attempts": 2,
        "passed_statuses": ["live", "paper", "validated"],
        "passed_routes": ["paper", "shelf", "needs-infrastructure"]}), encoding="utf-8")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n", encoding="utf-8")
    with registry.session_scope() as s:
        _idea(s, "old-pass", IdeaStatus.VALIDATED)
        _trial(s, "old-pass", "paper", stand=None, rules=OLD_RULES)
        _idea(s, "fresh", IdeaStatus.VALIDATED)
        _trial(s, "fresh", "paper")
        _idea(s, "old-rules", IdeaStatus.BENCH)
        _trial(s, "old-rules", "needs-forward", rules=OLD_RULES)
        _idea(s, "old-bench", IdeaStatus.BENCH)
        _trial(s, "old-bench", "needs-forward", stand=None, rules=None)
        # a first stage that is up to date with a second stage that is not
        _idea(s, "parent", IdeaStatus.VALIDATED)
        _trial(s, "parent", "paper")
        _idea(s, "parent-hl", IdeaStatus.VALIDATED, parent="parent", notes=f"{NOTES_PREFIX}parent")
        _trial(s, "parent-hl", "paper", stand=None, rules=OLD_RULES)
        # a variant
        _idea(s, "variant-ov-gate", IdeaStatus.BENCH)
        _trial(s, "variant-ov-gate", "needs-infrastructure", stand=None)
        # a second stage whose first stage is closed
        _idea(s, "orphan-parent", IdeaStatus.RETIRED)
        _idea(s, "orphan-parent-hl", IdeaStatus.VALIDATED, parent="orphan-parent",
              notes=f"{NOTES_PREFIX}orphan-parent")
        _trial(s, "orphan-parent-hl", "paper", stand=None)
        # never to be recomputed
        _idea(s, "retired-one", IdeaStatus.RETIRED)
        _trial(s, "retired-one", "reject", stand=None)
        _idea(s, "rejected-one", IdeaStatus.REJECTED)
        _trial(s, "rejected-one", "reject", stand=None)
        _idea(s, "noise-dollar_neutral-random", IdeaStatus.CANDIDATE)
        _trial(s, "noise-dollar_neutral-random", "needs-more-data", stand=None)
        _idea(s, "no-run", IdeaStatus.CANDIDATE)

    calls: list[tuple] = []
    results: dict[str, str] = {}
    outcome = types.SimpleNamespace(fail=set(), hl_route="paper")

    def fake_idea(plan, end, *, noise_trials, deployable_capital_usd):
        calls.append(("idea", plan.idea_id, end, noise_trials))
        if plan.idea_id in outcome.fail:
            return {"idea": plan.idea_id, "step": "idea", "action": "error",
                    "route_before": plan.route, "error": "boom"}
        route = results.get(plan.idea_id, plan.route)
        with registry.session_scope() as s:
            trial = _trial(s, plan.idea_id, route, metrics={"cap_ann_return": 0.05,
                                                            "cap_max_dd": -0.30})
            trial_id = trial.id
            s.get(Idea, plan.idea_id).status = (IdeaStatus.REJECTED if route == "reject"
                                                else s.get(Idea, plan.idea_id).status)
        return {"idea": plan.idea_id, "step": "idea", "action": "rerun",
                "route_before": plan.route, "route_after": route, "trial": trial_id,
                "reason": "r", "status": "bench", "why": plan.why}

    def fake_stage2(plan, end, *, stage1, noise_trials, deployable_capital_usd, config):
        calls.append(("stage2", plan.idea_id, end, stage1))
        if f"{plan.idea_id}-hl" in outcome.fail:
            return {"idea": plan.idea_id, "step": "stage2", "action": "error",
                    "route_before": plan.route, "error": "hl boom"}
        with registry.session_scope() as s:
            trial = _trial(s, f"{plan.idea_id}-hl", outcome.hl_route,
                           metrics={"cap_ann_return": 0.04, "cap_max_dd": -0.25})
            trial_id = trial.id
        return {"idea": plan.idea_id, "step": "stage2", "action": "rerun",
                "route_before": plan.route, "route_after": outcome.hl_route, "trial": trial_id}

    series: list[list[str]] = []

    def fake_rebuild(session, idea_ids=None, **kw):
        series.append(list(idea_ids))
        return [(i, 1, "построен") for i in idea_ids]

    monkeypatch.setattr(waiting, "rerun_idea", fake_idea)
    monkeypatch.setattr(waiting, "rerun_stage2", fake_stage2)
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    monkeypatch.setattr("qlab.forward_series.rebuild", fake_rebuild)
    monkeypatch.setattr("qlab.overlay_compare.compare_specs",
                        lambda *a, **kw: {"start": "a", "end": "b"})
    monkeypatch.setattr("qlab.overlay_compare.save_record", lambda record, *a, **kw: None)
    return types.SimpleNamespace(
        tmp=tmp_path, specs=specs, config=config, watch=watch, calls=calls, results=results,
        outcome=outcome, series=series, ledger=tmp_path / "ledger.json")


def _stage(w, state=None, **kw) -> dict:
    state = state if state is not None else {"done": {}}
    restamp.restamp_stage(DAY, END, state, noise_trials=1, config_path=w.config,
                          spec_dir=w.specs, ledger_path=w.ledger, watch_config=w.watch, **kw)
    return state


def _ids(state) -> list[str]:
    return [e["idea"] for e in state["restamp"]["entries"]]


# ---- which ideas ----------------------------------------------------------------------------


def test_stale_ideas_are_taken_by_their_stand_version_or_their_rules_version(world) -> None:
    state = _stage(world, dry_run=True, no_cap=True)
    assert set(_ids(state)) == {"old-pass", "old-rules", "old-bench", "parent",
                                "variant-ov-gate"}
    by = {e["idea"]: e for e in state["restamp"]["entries"]}
    assert [r["kind"] for r in by["old-pass"]["reasons"]] == ["stand", "rules"]
    assert [r["kind"] for r in by["old-rules"]["reasons"]] == ["rules"]
    # a run with no verdicts (not evaluable) has no rules version: stale by the stand only
    assert [r["kind"] for r in by["old-bench"]["reasons"]] == ["stand"]
    assert by["old-pass"]["reasons"][0]["was"] is None  # an old trial has no version


def test_a_run_whose_noise_bar_lost_books_is_taken_without_a_new_stand_version(
        world, registry) -> None:
    """Owner 2026-10-29, «да, чини учёт»: 197 of 199 books reads 0.98995, not the 0.99 that 198
    of 200 reads. A run measured that way is stale for that reason alone; one measured on 200
    books, and one that records its own counts (even short ones), is not."""
    for name in ("short-books", "full-books", "counted-short"):
        _write_spec(world.specs / f"{name}.yaml", name)
    with registry.session_scope() as s:
        _idea(s, "short-books")
        _trial(s, "short-books", "needs-forward", metrics={"noise_return_percentile": 197 / 199})
        _idea(s, "full-books")
        _trial(s, "full-books", "needs-forward", metrics={"noise_return_percentile": 198 / 200})
        _idea(s, "counted-short")
        _trial(s, "counted-short", "needs-forward", metrics={
            "noise_return_percentile": 150 / 199, "noise_n_requested": 200.0,
            "noise_n_usable": 199.0, "noise_n_replaced": 200.0})
    state = _stage(world, dry_run=True, no_cap=True)
    by = {e["idea"]: e for e in state["restamp"]["entries"]}
    assert [r["kind"] for r in by["short-books"]["reasons"]] == ["noise_books"]
    assert "full-books" not in by and "counted-short" not in by
    assert "меньше 200" in restamp._reasons_ru(by["short-books"]["reasons"])


def test_the_noise_book_test_reads_the_counts_or_else_the_percentile_denominator() -> None:
    assert restamp.noise_books_short({"noise_return_percentile": 197 / 199})
    assert restamp.noise_books_short({"noise_return_percentile": 0.7039473684210527})
    assert not restamp.noise_books_short({"noise_return_percentile": 0.99})
    assert not restamp.noise_books_short({"noise_return_percentile": 1.0})
    assert not restamp.noise_books_short({})
    assert not restamp.noise_books_short(None)
    # a run that records its counts has been through the replacement already
    assert not restamp.noise_books_short({"noise_return_percentile": 197 / 199,
                                          "noise_n_usable": 199.0})


def test_a_run_recomputed_for_its_noise_books_is_not_taken_again(world, registry) -> None:
    _write_spec(world.specs / "short-books.yaml", "short-books")
    with registry.session_scope() as s:
        _idea(s, "short-books")
        _trial(s, "short-books", "needs-forward", metrics={"noise_return_percentile": 197 / 199})
    state = _stage(world, only={"short-books"})
    assert _ids(state) == ["short-books"]
    assert not any(e["idea"] == "short-books" for e in
                   _stage(world, dry_run=True, no_cap=True)["restamp"]["entries"])


def test_a_rules_version_that_only_adds_readings_is_no_reason_to_measure_again() -> None:
    """Rules 2026-10-29.1 added the protection readings and nothing that decides; measuring
    thirty-odd ideas again for it would move no verdict."""
    assert not restamp.rules_changed("2026-10-28.2", "2026-10-29.1")
    assert not restamp.rules_changed("2026-10-28.2", "2026-10-28.2")
    # 2026-10-28.1 -> .2 retired two decisive rules; 2026-09-20.1 is far from the latest
    assert restamp.rules_changed("2026-10-28.1", "2026-10-28.2")
    assert restamp.rules_changed(OLD_RULES, "2026-10-29.1")
    # a version that cannot be loaded counts as changed: a stale run must not hide
    assert restamp.rules_changed("frab-legacy", "2026-10-29.1")
    view = {"stand": STAND_VERSION, "rules": "2026-10-28.2", "noise_short": False}
    assert restamp.stale_reasons(view, STAND_VERSION, "2026-10-29.1") == []
    view["rules"] = "2026-10-28.1"
    assert [r["kind"] for r in restamp.stale_reasons(view, STAND_VERSION, "2026-10-28.2")] == [
        "rules"]


def test_retired_rejected_noise_and_runless_ideas_are_left_alone(world) -> None:
    state = _stage(world, dry_run=True)
    assert not {"retired-one", "rejected-one", "noise-dollar_neutral-random", "no-run",
                "fresh"} & set(_ids(state))


def test_a_stale_second_stage_alone_runs_only_the_second_stage(world) -> None:
    state = _stage(world, dry_run=True)
    entry = next(e for e in state["restamp"]["entries"] if e["idea"] == "parent")
    assert entry["steps"] == ["stage2"]
    _stage(world, state={"done": {}})
    assert [c[0] for c in world.calls if c[1] == "parent"] == ["stage2"]


def test_a_second_stage_is_not_run_when_its_first_stage_no_longer_goes_on(
        world, registry) -> None:
    with registry.session_scope() as s:
        _trial(s, "parent", "reject")
    state = _stage(world, dry_run=True)
    assert "parent" not in _ids(state)


def test_a_second_stage_with_a_closed_first_stage_is_named_not_run(world) -> None:
    state = _stage(world)
    assert "orphan-parent-hl" in state["restamp"]["skipped"]
    assert "orphan-parent-hl" not in _ids(state)
    assert not [c for c in world.calls if "orphan" in c[1]]
    assert "orphan-parent-hl" in "\n".join(restamp.report_lines(state["restamp"]))


def test_naming_a_second_stage_stands_for_its_first_stage(world, registry) -> None:
    with registry.session_scope() as s:
        _trial(s, "fresh", "paper")  # fresh and up to date
    state = _stage(world, only={"parent-hl"}, dry_run=True)
    assert _ids(state) == ["parent"]
    state = _stage(world, only={"fresh"}, dry_run=True)  # named: recomputed though up to date
    assert _ids(state) == ["fresh"]


def test_a_variant_is_an_ordinary_idea_with_its_comparison_rebuilt(world) -> None:
    state = _stage(world, only={"variant-ov-gate"})
    entry = state["restamp"]["entries"][0]
    assert entry["variant"] is True and entry["action"] == "restamped"
    assert world.calls[0][:2] == ("idea", "variant-ov-gate")
    # the base (`variant`) has no spec file here: said, not hidden
    assert any("сравнение с базой не пересчитано" in n for n in entry["notes"])
    _write_spec(world.specs / "variant.yaml", "variant")
    state = _stage(world, state={"done": {}}, only={"variant-ov-gate"})
    assert "сравнение с базой пересчитано" in state["restamp"]["entries"][0]["notes"]


def test_an_idea_with_several_spec_files_takes_the_one_of_its_run_or_the_watch_list(
        world, registry) -> None:
    _write_spec(world.specs / "two-a.yaml", "two")
    _write_spec(world.specs / "two-b.yaml", "two")
    with registry.session_scope() as s:
        _idea(s, "two", IdeaStatus.BENCH)
        _trial(s, "two", "needs-forward", stand=None)
    state = _stage(world, dry_run=True)
    assert "two" in state["restamp"]["skipped"] and "спеков несколько" in state[
        "restamp"]["skipped"]["two"]
    world.watch.write_text("specs:\n  - " + (world.specs / "two-b.yaml").as_posix() + "\n",
                           encoding="utf-8")
    state = _stage(world, dry_run=True)
    assert "two" in _ids(state)


# ---- the order and the bounds -----------------------------------------------------------------


def test_passed_ideas_go_first_then_more_reasons_then_the_newer_run(world) -> None:
    state = _stage(world, dry_run=True, no_cap=True)
    # passed (validated, or a route that passed) first: the two with two reasons, the newer run
    # first, then the variant (one reason); then the bench: the newer run first
    assert _ids(state) == ["parent", "old-pass", "variant-ov-gate", "old-bench", "old-rules"]


def test_at_most_the_nightly_cap_and_the_rest_wait_and_are_named(world) -> None:
    world.config.write_text(world.config.read_text().replace(
        "max_ideas_per_night: 4", "max_ideas_per_night: 2"), encoding="utf-8")
    state = _stage(world)
    assert len(state["restamp"]["entries"]) == 2
    assert len(state["restamp"]["deferred"]) == 3
    assert "ждут следующей ночи" in "\n".join(restamp.report_lines(state["restamp"]))


def test_a_hand_run_with_all_or_with_ideas_named_is_not_capped(world) -> None:
    world.config.write_text(world.config.read_text().replace(
        "max_ideas_per_night: 4", "max_ideas_per_night: 1"), encoding="utf-8")
    assert len(_stage(world, no_cap=True)["restamp"]["entries"]) == 5
    assert len(_stage(world, only={"fresh", "old-pass"})["restamp"]["entries"]) == 2


def test_nothing_starts_after_the_stop_time(world) -> None:
    state = {"done": {}}
    _stage(world, state=state, stop_at=datetime.now(UTC) - timedelta(minutes=1))
    assert not state["restamp"]["entries"] and not world.calls
    assert state["stopped_at"]["before"].startswith("restamp:")


def test_a_slow_idea_is_reported_as_slow(world, monkeypatch) -> None:
    world.config.write_text(world.config.read_text().replace(
        "slow_idea_minutes: 30", "slow_idea_minutes: -1"), encoding="utf-8")
    state = _stage(world, only={"old-pass"})
    assert state["restamp"]["entries"][0]["slow"] is True
    assert "дольше нормы" in "\n".join(restamp.report_lines(state["restamp"]))


def test_a_dry_run_runs_and_writes_nothing(world) -> None:
    state = _stage(world, dry_run=True)
    assert not world.calls and not world.ledger.exists() and not state["done"]
    assert all(e["action"] == "would-restamp" for e in state["restamp"]["entries"])
    assert "был бы пересчитан" in "\n".join(restamp.report_lines(state["restamp"]))


# ---- the run and the second stage ---------------------------------------------------------------


def test_an_idea_is_run_to_the_last_day_its_latest_run_saw_not_to_the_nights(
        world, registry) -> None:
    _write_spec(world.specs / "early.yaml", "early")
    with registry.session_scope() as s:
        _idea(s, "early", IdeaStatus.BENCH)
        _trial(s, "early", "needs-forward", stand=None, end=END - timedelta(days=9))
    _stage(world, only={"early"})
    assert world.calls[0] == ("idea", "early", END - timedelta(days=9), 1)


def test_a_first_stage_that_survives_sends_the_idea_on_to_the_second_stage(
        world, registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "old-pass-hl", IdeaStatus.VALIDATED, parent="old-pass",
              notes=f"{NOTES_PREFIX}old-pass")
        _trial(s, "old-pass-hl", "paper", stand=None)
    state = _stage(world, only={"old-pass"})
    kinds = [c[0] for c in world.calls]
    assert kinds == ["idea", "stage2"]
    first = world.calls[1][3]
    assert first["route_after"] == "paper" and first["trial"]  # stage 2 gets stage 1's new run
    entry = state["restamp"]["entries"][0]
    assert entry["steps"] == ["idea", "stage2"]
    assert entry["after"]["stage2"]["stand"] == STAND_VERSION
    assert world.series == [["old-pass"]]  # the forward series is rebuilt


def test_a_first_stage_that_no_longer_goes_on_leaves_the_second_stage_and_says_so(
        world, registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "old-pass-hl", IdeaStatus.VALIDATED, parent="old-pass",
              notes=f"{NOTES_PREFIX}old-pass")
        _trial(s, "old-pass-hl", "paper", stand=None)
    world.results["old-pass"] = "reject"
    state = _stage(world, only={"old-pass"})
    assert [c[0] for c in world.calls] == ["idea"]
    entry = state["restamp"]["entries"][0]
    assert any("второй этап не пересчитан" in n for n in entry["notes"])
    assert entry["after"]["idea"]["route"] == "reject"
    assert entry["after"]["idea"]["status"] == "rejected"


def test_the_real_stage_2_path_is_used_for_a_second_stage_alone(world, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(waiting, "rerun_stage2", _REAL_RERUN_STAGE2)  # the real step
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    monkeypatch.setattr(
        stage2, "run_stage2",
        lambda spec, **kw: seen.append((spec.idea_id, kw["stage1_route"], spec.data.end))
        or stage2.Stage2Result(parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl",
                               status="evaluated", route="paper", trial_id=1))
    monkeypatch.setattr("qlab.forward_series.rebuild", lambda *a, **kw: [])
    _stage(world, only={"parent"})
    assert seen and seen[0][:2] == ("parent", "paper")


# ---- failures and idempotence ---------------------------------------------------------


def test_a_recomputed_idea_is_not_taken_again(world) -> None:
    first = _stage(world)
    done = {e["idea"] for e in first["restamp"]["entries"]}
    assert done
    second = _stage(world, state={"done": {}})
    assert not set(_ids(second)) & done


def test_the_whole_stale_set_is_finished_in_a_few_nights_and_then_nothing_is_left(world) -> None:
    taken: set[str] = set()
    for _ in range(5):
        state = _stage(world, state={"done": {}})
        ids = _ids(state)
        assert not set(ids) & taken
        taken |= set(ids)
    assert taken == {"old-pass", "old-rules", "old-bench", "parent", "variant-ov-gate"}
    assert _ids(_stage(world, state={"done": {}}, dry_run=True)) == []


def test_a_done_marker_keeps_a_resumed_night_from_repeating_an_idea(world) -> None:
    state = _stage(world, only={"old-pass"})
    calls = len(world.calls)
    _stage(world, state=state, only={"old-pass"})
    assert len(world.calls) == calls


def test_a_failure_is_a_row_and_is_tried_again_until_the_attempts_are_used(world) -> None:
    world.outcome.fail = {"old-bench"}
    state = _stage(world, only={"old-bench"})
    entry = state["restamp"]["entries"][0]
    assert entry["action"] == "error" and "boom" in entry["error"]
    assert "old-bench" not in state["done"] and "сбой" in "\n".join(
        restamp.report_lines(state["restamp"]))
    ledger = restamp.load_ledger(world.ledger)
    assert ledger["old-bench"]["status"] == "error" and ledger["old-bench"]["attempts"] == 1
    _stage(world, state={"done": {}}, only={"old-bench"})
    assert restamp.load_ledger(world.ledger)["old-bench"]["attempts"] == 2
    # two failures under the same versions: it is left alone and named
    state = _stage(world, state={"done": {}})
    assert "old-bench" not in _ids(state)
    assert "2 раз подряд" in state["restamp"]["skipped"]["old-bench"]


def test_a_failed_idea_whose_run_left_a_current_trial_is_still_retried(world, registry) -> None:
    with registry.session_scope() as s:
        _trial(s, "old-bench", "error", rules=None)  # the failed run wrote a current error trial
    restamp.save_ledger({"old-bench": {"status": "error", "attempts": 1, "stand": STAND_VERSION,
                                       "rules": RULES, "error": "x"}}, world.ledger)
    state = _stage(world, dry_run=True)
    entry = next(e for e in state["restamp"]["entries"] if e["idea"] == "old-bench")
    assert entry["reasons"][0]["kind"] == "retry"


def test_a_new_version_gives_a_failed_idea_a_fresh_start(world) -> None:
    restamp.save_ledger({"old-bench": {"status": "error", "attempts": 5, "stand": "2026-01-01.1",
                                       "rules": RULES, "error": "x"}}, world.ledger)
    assert "old-bench" in _ids(_stage(world, dry_run=True))


def test_one_ideas_failure_does_not_stop_the_others(world) -> None:
    world.outcome.fail = {"old-pass"}
    state = _stage(world, no_cap=True)
    actions = {e["idea"]: e["action"] for e in state["restamp"]["entries"]}
    assert actions["old-pass"] == "error" and actions["old-rules"] == "restamped"


def test_a_second_stage_failure_is_the_ideas_failure_and_the_first_stage_stands(
        world, registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "old-pass-hl", IdeaStatus.VALIDATED, parent="old-pass",
              notes=f"{NOTES_PREFIX}old-pass")
        _trial(s, "old-pass-hl", "paper", stand=None)
    world.outcome.fail = {"old-pass-hl"}
    state = _stage(world, only={"old-pass"})
    entry = state["restamp"]["entries"][0]
    assert entry["action"] == "error" and "второй этап" in entry["error"]
    assert entry["after"]["idea"]["stand"] == STAND_VERSION  # stage 1 is recomputed


# ---- the stage's switch and the shipped config -------------------------------------------


def test_the_stage_is_off_without_its_config_and_when_switched_off(world, tmp_path) -> None:
    state = {"done": {}}
    restamp.restamp_stage(DAY, END, state, config_path=tmp_path / "missing.yaml")
    assert "restamp" not in state
    world.config.write_text(world.config.read_text().replace("enabled: true", "enabled: false"),
                            encoding="utf-8")
    assert "restamp" not in _stage(world)


def test_the_shipped_config_is_read() -> None:
    cfg = restamp.load_config(Path(__file__).resolve().parents[2] / "night" / "restamp.yaml")
    assert cfg is not None and cfg.enabled and cfg.max_ideas_per_night >= 1
    assert "validated" in cfg.passed_statuses and "paper" in cfg.passed_routes


def test_a_config_with_a_missing_key_is_an_error(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        restamp.load_config(path)


def test_the_stage_spends_no_tokens() -> None:
    text = Path(restamp.__file__).read_text(encoding="utf-8")
    assert "budget" not in text.lower().replace("budget guard", "")
    assert "claude" not in text.replace("Claude", "").lower().replace("claude's", "")


# ---- the report ---------------------------------------------------------------------------------


def test_the_report_gives_route_return_and_fall_before_and_after_and_a_changed_status(
        world) -> None:
    world.results["old-pass"] = "reject"
    state = _stage(world, only={"old-pass"})
    text = "\n".join(restamp.report_lines(state["restamp"]))
    assert "**old-pass**" in text and "пересчитана" in text
    assert "способ измерения сменился (версии не было записано → " in text
    assert "правила сменились" in text
    # before: the old stand's figures; after: on the strategy's money
    assert "в год на деньгах стратегии +10.0% → +5.0%" in text
    assert "худшее падение −20.0% → −30.0%" in text
    assert "раньше считалось по счёту стенда" in text
    assert "статус validated → rejected" in text and "исход изменился у 1" in text


def test_a_night_with_nothing_out_of_date_has_no_section_but_a_hand_run_names_what_is_skipped(
        world) -> None:
    _stage(world, no_cap=True)  # everything out of date is recomputed
    state = _stage(world, state={"done": {}})
    assert not state["restamp"]["entries"]
    assert restamp.report_lines(state["restamp"]) == []
    # the second stage with a closed first stage stays skipped for good: named only on request
    assert "orphan-parent-hl" in "\n".join(
        restamp.report_lines(state["restamp"], even_if_quiet=True))


# ---- the night ----------------------------------------------------------------------------------


def _night_env(tmp_path, monkeypatch, order: list[str]):
    watch = tmp_path / "nwatch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "unwatched_bench", lambda w: [])
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "nstate")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "nreports")
    monkeypatch.setattr("qlab.waiting.waiting_stage", lambda *a, **kw: order.append("waiting"))
    monkeypatch.setattr("qlab.twin_stage.twins_stage", lambda *a, **kw: order.append("twins"))
    monkeypatch.setattr(night, "anomalies_stage", lambda *a, **kw: order.append("anomalies"))
    monkeypatch.setattr(night, "overlays_stage", lambda *a, **kw: order.append("overlays"))
    monkeypatch.setattr(night, "paper_reconciliation", lambda day: order.append("paper") or [])

    def fake_restamp(day, end, state, **kw):
        order.append("restamp")
        state["restamp"] = {"entries": [{
            "idea": "x-idea", "action": "restamped", "steps": ["idea"],
            "reasons": [{"kind": "stand", "was": None, "now": STAND_VERSION}],
            "before": {"idea": None, "stage2": None}, "after": {"idea": None, "stage2": None},
            "notes": []}], "skipped": {}, "deferred": [], "notes": [], "stand": STAND_VERSION,
            "rules": RULES}

    monkeypatch.setattr(night, "restamp_stage", fake_restamp)


def _run_night(**kw):
    flags = dict(graveyard=False, implement=False, search=False, detectors=False, assess=False,
                 director=False, programmer=False, catalogue=False, decay=False, research=False)
    return night.run(DAY, **{**flags, **kw})


def test_the_stage_runs_after_the_twins_and_before_the_anomalies_and_the_overlays(
        tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    report = _run_night(paper=True)
    assert order == ["waiting", "twins", "restamp", "anomalies", "overlays", "paper"]
    text = report.read_text(encoding="utf-8")
    assert f"## {restamp.SECTION_TITLE}" in text and "x-idea" in text


def test_the_flag_switches_the_stage_off(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    _run_night(paper=False, restamp=False)
    assert "restamp" not in order


def test_a_failing_stage_is_a_report_line_and_the_night_goes_on(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)

    def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(night, "restamp_stage", broken)
    report = _run_night(paper=False)
    assert order[-2:] == ["anomalies", "overlays"]
    assert "пересчёт старых результатов" in report.read_text(encoding="utf-8")


def test_nothing_starts_after_the_stop_time_in_the_night(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    _run_night(paper=False, stop_at=datetime.now(UTC) - timedelta(minutes=1))
    assert "restamp" not in order


# ---- the command --------------------------------------------------------------------------------


def test_the_command_is_registered_and_asks_for_something() -> None:
    from typer.testing import CliRunner

    from qlab.cli import app

    out = CliRunner().invoke(app, ["restamp", "run", "--help"])
    assert out.exit_code == 0 and "--idea" in out.output and "--all" in out.output
    assert "--dry-run" in out.output
    night_help = CliRunner().invoke(app, ["night", "run", "--help"])
    assert "--restamp" in night_help.output
    bare = CliRunner().invoke(app, ["restamp", "run"])
    assert bare.exit_code == 2


def test_the_command_dry_run_lists_what_would_be_done_and_changes_nothing(
        world, tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from qlab.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "night").mkdir()
    (tmp_path / "night" / "restamp.yaml").write_text(world.config.read_text(), encoding="utf-8")
    (tmp_path / "night" / "watchlist.yaml").write_text("specs: []\n", encoding="utf-8")
    monkeypatch.setattr(restamp, "SPEC_DIR", world.specs)
    monkeypatch.setattr(restamp, "CONFIG", tmp_path / "night" / "restamp.yaml")
    monkeypatch.setattr(restamp, "WATCHLIST", tmp_path / "night" / "watchlist.yaml")
    monkeypatch.setattr(restamp, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(
        restamp, "restamp_stage",
        lambda day, end, state, **kw: _real_stage(world, day, end, state, **kw))
    out = CliRunner().invoke(app, ["restamp", "run", "--dry-run"])
    assert out.exit_code == 0, out.output
    assert "old-pass" in out.output and "был бы пересчитан" in out.output
    assert not world.calls


def _real_stage(world, day, end, state, **kw):
    kw.pop("noise_trials", None)
    kw.pop("deployable_capital_usd", None)
    return _ORIGINAL(day, end, state, noise_trials=1, config_path=world.config,
                     spec_dir=world.specs, ledger_path=world.ledger, watch_config=world.watch,
                     **kw)


_ORIGINAL = restamp.restamp_stage


def test_a_hand_run_keeps_its_state_apart_from_the_nights(world, tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from qlab.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        restamp, "restamp_stage",
        lambda day, end, state, **kw: _real_stage(world, day, end, state, **kw))
    out = CliRunner().invoke(app, ["restamp", "run", "--idea", "old-pass"])
    assert out.exit_code == 0, out.output
    assert (tmp_path / "data" / "restamp" / "runs").is_dir()
    assert "old-pass" in out.output
