"""Tests for `qlab.overlay_stage`: every overlay and pair on the strategies that passed, bounded,
idempotent, counted, compared with the base (docs/OVERLAYS.md, docs/NIGHT.md).

The stand, the second stage, the comparison and the description of the base are replaced by
recorders; the two bases are real spec files copied into a temporary folder, so nothing under
`specs/`, in `data/overlays` or in the real registry is touched."""

from __future__ import annotations

import shutil
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import qlab.registry.db as db_module
from qlab import night, overlay_compare, overlay_diagnostics, overlay_stage, twin_stage
from qlab.night import ItemResult
from qlab.pipeline import stage2
from qlab.pipeline.spec import load_spec
from qlab.registry import repo
from qlab.registry.deflation import family_trials as deflation_family
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

ROOT = Path(__file__).resolve().parents[2]
XSMOM = "xsmom-research-nobull-causal"
TREND = "trend-binance-cap-gated-causal"
BV2 = "bv2-research-headroom-causal"
SOURCES = {XSMOM: ROOT / "specs" / "research" / "xsmom" / f"{XSMOM}.yaml",
           TREND: ROOT / "specs" / f"{TREND}.yaml",
           BV2: ROOT / "specs" / "research" / "bv2" / f"{BV2}.yaml"}
DAY = date(2026, 10, 27)
END = date(2026, 10, 26)
VARIANTS_PER_BASE = 15          # five overlays with a family: five singles and ten pairs
STAGE2_CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))


_REAL_ANOMALY_REFUSAL = overlay_stage.anomaly_refusal


@pytest.fixture(autouse=True)
def _bases_scanned_clean(monkeypatch):
    """The worlds here have no anomaly scans; a base counts as scanned with no flag fired. The
    tests of the anomaly gate set their own over this."""
    monkeypatch.setattr(overlay_stage, "anomaly_refusal", lambda session, idea_id: None)


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


def _idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.VALIDATED,
          driver: str | None = None) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     driver_id=driver)


def _trial(session, idea_id: str, route: str, *, metrics: dict | None = None):
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={}, data_requirements={"end": str(END)},
        rebalance="1d", costs_model={}, code_ref="a:b")
    if session.get(DataSnapshot, "snap") is None:
        repo.add_data_snapshot(session, id="snap", source="binance", instruments={},
                               range_start=date(2019, 9, 8), range_end=END, path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                          status=TrialStatus.OK, snapshot_id="snap", metrics=metrics or {},
                          route=TrialRoute(route), route_reason="r")


def _series(n: int, seed: int, drift: float = 0.0005) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(drift, 0.02, n),
                     index=pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC"))


@pytest.fixture()
def world(tmp_path, registry, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    for name in (XSMOM, TREND, BV2):
        shutil.copy(SOURCES[name], specs / f"{name}.yaml")
    config = tmp_path / "overlays.yaml"
    config.write_text(yaml.safe_dump({
        "enabled": True, "max_bases_per_night": 2, "max_variants_per_night": 6,
        "slow_variant_minutes": 20, "priority": ["trend", "xsmom", "bv2"],
        "bases": [XSMOM, TREND], "auto": False, "second_stage": True,
        "meta_decisions": {}}), encoding="utf-8")
    with registry.session_scope() as s:
        for name in SOURCES:
            _idea(s, name)
            _trial(s, name, "paper")

    calls: list[tuple] = []
    results: dict[str, str] = {}
    failing: set[str] = set()

    def fake_eval(kind, spec, *, end, noise_trials):
        calls.append(("stand", spec.idea_id, end))
        if spec.idea_id in failing:
            return ItemResult(kind=kind, name=spec.idea_id, route="error", reason="boom",
                              error="RuntimeError: boom")
        route = results.get(spec.idea_id, "needs-forward")
        with registry.session_scope() as s:
            trial = _trial(s, spec.idea_id, route, metrics={
                "ann_return_net": 0.2, "sharpe_net": 0.8, "max_dd": -0.4, "forward_days": 2000.0})
            trial_id = trial.id
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=trial_id, route=route,
                          reason="r")

    def fake_compare(base, variant, session, **kw):
        calls.append(("compare", variant.idea_id, base.idea_id))
        record = overlay_compare.compare_series(
            _series(800, 1), _series(800, 2, drift=0.0015), n_boot=100)
        record.update(base_id=base.idea_id, variant_id=variant.idea_id,
                      computed_at="2026-10-27T00:00:00+00:00")
        return record

    def fake_describe(spec, session, **kw):
        calls.append(("describe", spec.idea_id))
        return overlay_diagnostics.build(spec.idea_id, _series(400, 3), _series(400, 4), None)

    def fake_stage2(spec, *, stage1_route, stage1_trial_id, noise_trials,
                    config=None, **kw):
        calls.append(("stage2", spec.idea_id, stage1_route))
        return stage2.Stage2Result(parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl",
                                   status="evaluated", route="paper", trial_id=777,
                                   stage1_route=stage1_route, combined_route="paper",
                                   parent_status="validated")

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    monkeypatch.setattr(overlay_compare, "compare_specs", fake_compare)
    monkeypatch.setattr(overlay_diagnostics, "compute_for_spec", fake_describe)
    monkeypatch.setattr(stage2, "run_stage2", fake_stage2)
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    return types.SimpleNamespace(
        tmp=tmp_path, specs=specs, config=config, calls=calls, results=results,
        failing=failing, ledger=tmp_path / "ledger.json", store=tmp_path / "store")


def run_stage(w, state=None, **kw) -> dict:
    state = state if state is not None else {"done": {}}
    overlay_stage.overlays_stage(
        DAY, END, state, noise_trials=1, config_path=w.config, spec_dir=w.specs,
        ledger_path=w.ledger, store=w.store, **kw)
    return state


def stands(w) -> list[str]:
    return [c[1] for c in w.calls if c[0] == "stand"]


# ---- which bases ----------


def test_bases_are_the_listed_ones_then_those_that_passed_in_the_order_of_priority(
        world, registry) -> None:
    from qlab.waiting import spec_files

    config = overlay_stage.load_config(world.config)
    auto = types.SimpleNamespace(**{**config.__dict__, "auto": True, "bases": (XSMOM,)})
    with registry.session_scope() as s:
        found, refused = overlay_stage.find_bases(s, spec_files(world.specs), auto)
    # trend before xsmom before bv2 (priority), the listed one ahead of the discovered ones
    assert [b.idea_id for b in found] == [TREND, XSMOM, BV2]
    assert found[1].listed and not found[0].listed and found[2].strategy == "bv2"


def test_a_base_must_be_scanned_with_no_anomaly_flag_fired(world, registry, monkeypatch) -> None:
    """Owner 2026-10-28: overlays only on bases without anomaly flags."""
    import qlab.anomalies.store as anomaly_store
    import qlab.registry.stages as stages
    from qlab.waiting import spec_files

    trial = types.SimpleNamespace(id=7)
    scans = {XSMOM: types.SimpleNamespace(fired=["tail_move_coin_days", "frozen_price"]),
             TREND: types.SimpleNamespace(fired=[]), BV2: None}
    current = {}
    monkeypatch.setattr(stages, "latest_routed",
                        lambda session, idea_id: current.update(id=idea_id) or (trial, None))
    monkeypatch.setattr(anomaly_store, "latest_scan",
                        lambda session, trial_id: scans[current["id"]])
    monkeypatch.setattr(overlay_stage, "anomaly_refusal", _REAL_ANOMALY_REFUSAL)
    config = overlay_stage.load_config(world.config)
    auto = types.SimpleNamespace(**{**config.__dict__, "auto": True, "bases": (XSMOM,)})
    with registry.session_scope() as s:
        found, refused = overlay_stage.find_bases(s, spec_files(world.specs), auto)
    assert [b.idea_id for b in found] == [TREND]
    assert "tail_move_coin_days" in refused[XSMOM] and "no anomaly scan" in refused[BV2]


def test_a_variant_a_second_stage_and_an_idea_without_one_spec_are_no_base(
        world, registry) -> None:
    from qlab.waiting import spec_files

    config = overlay_stage.load_config(world.config)
    (world.specs / "dup.yaml").write_text((world.specs / f"{XSMOM}.yaml").read_text(),
                                          encoding="utf-8")
    with registry.session_scope() as s:
        _idea(s, f"{XSMOM}-ov-own")
        _idea(s, f"{XSMOM}-hl")
        s.get(Idea, f"{XSMOM}-hl").parent_id = XSMOM
        s.get(Idea, f"{XSMOM}-hl").notes = f"{stage2.NOTES_PREFIX}{XSMOM}"
        _idea(s, "no-file")
        found, refused = overlay_stage.find_bases(
            s, spec_files(world.specs), config,
            only=[f"{XSMOM}-ov-own", f"{XSMOM}-hl", "no-file", "nope", XSMOM, TREND])
    assert [b.idea_id for b in found] == [TREND]
    assert refused[f"{XSMOM}-ov-own"].startswith("a variant")
    assert refused[f"{XSMOM}-hl"].startswith("a variant or a second stage")
    assert refused["no-file"] == "no spec file" and refused["nope"].endswith("registry")
    assert refused[XSMOM] == "several spec files"


def test_ideas_that_passed_stage_1_are_found_when_auto_is_on(world, registry, tmp_path) -> None:
    from qlab.waiting import spec_files

    with registry.session_scope() as s:
        _idea(s, "just-bench", IdeaStatus.BENCH)
        _trial(s, "just-bench", "needs-forward")
        config = overlay_stage.load_config(world.config)
        config = types.SimpleNamespace(**{**config.__dict__, "auto": True, "bases": ()})
        found, _ = overlay_stage.find_bases(s, spec_files(world.specs), config)
    assert {b.idea_id for b in found} == set(SOURCES)       # passed: validated or route paper


# ---- the variants: every single, every pair, no triple ----------


def test_a_base_gets_every_single_overlay_and_every_pair_and_no_triple(world, registry) -> None:
    state = run_stage(world, only=[XSMOM], max_variants=100)
    entries = [e for e in state["overlays"]["entries"] if e.get("variant")]
    assert len(entries) == VARIANTS_PER_BASE
    chains = [tuple(e["chain"]) for e in entries]
    assert sorted({len(c) for c in chains}) == [1, 2] and len(set(chains)) == len(chains)
    assert not [c for c in chains if len(c) == 3]
    assert {e["variant"] for e in entries} >= {f"{XSMOM}-ov-own", f"{XSMOM}-ov-own-enter",
                                               f"{XSMOM}-ov-gate-exit"}
    # each variant is a spec file, an idea under the base, a stand run and a comparison
    assert len(list((world.specs / "overlays" / XSMOM).glob("*.yaml"))) == VARIANTS_PER_BASE
    assert len(stands(world)) == VARIANTS_PER_BASE
    with registry.session_scope() as s:
        assert s.get(Idea, f"{XSMOM}-ov-own").parent_id == XSMOM
        assert len(overlay_compare.variants_of(s, XSMOM)) == VARIANTS_PER_BASE
    assert overlay_compare.load_record(f"{XSMOM}-ov-own", world.store)["days"] == 800
    assert all(load_spec(p).selects_causally
               for p in (world.specs / "overlays" / XSMOM).glob("*.yaml"))


def test_the_chains_option_limits_the_variants(world) -> None:
    run_stage(world, only=[XSMOM], chains_only=["own_results", "own_results+gradual_entry"])
    assert stands(world) == [f"{XSMOM}-ov-own", f"{XSMOM}-ov-own-enter"]


# ---- bounded ----------


def test_at_most_the_nightly_cap_of_new_variants_the_rest_wait_for_the_next_night(world) -> None:
    first = run_stage(world)
    assert len(stands(world)) == 6 and first["overlays"]["new"] == 6
    assert len(first["overlays"]["deferred"]) > 0
    deferred = set(first["overlays"]["deferred"])
    second = run_stage(world)                                # the next night: the ledger resumes
    assert len(stands(world)) == 12
    again = [e["variant"] for e in second["overlays"]["entries"] if e.get("variant")]
    assert not set(again) & {e["variant"] for e in first["overlays"]["entries"]
                             if e.get("variant")}
    assert set(again) <= deferred
    for _ in range(3):
        run_stage(world)
    assert len(stands(world)) == 2 * VARIANTS_PER_BASE       # both bases, every variant, once
    quiet = run_stage(world)
    assert len(stands(world)) == 2 * VARIANTS_PER_BASE
    assert quiet["overlays"]["new"] == 0 and overlay_stage.report_lines(quiet["overlays"]) == []


def test_at_most_the_nightly_cap_of_bases_and_a_finished_base_does_not_take_a_place(
        world, tmp_path) -> None:
    config = yaml.safe_load(world.config.read_text())
    config.update(max_bases_per_night=1, max_variants_per_night=100, bases=[TREND, XSMOM, BV2])
    world.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = run_stage(world)
    assert {e["base"] for e in first["overlays"]["entries"]} == {TREND}
    assert any(v.startswith(XSMOM) for v in first["overlays"]["deferred"])
    second = run_stage(world)                                # trend is finished: xsmom is next
    assert {e["base"] for e in second["overlays"]["entries"]} == {XSMOM}
    third = run_stage(world)
    assert {e["base"] for e in third["overlays"]["entries"]} == {BV2}


def test_fixed_variants_are_one_idea_per_point_and_only_for_named_chains(world, registry) -> None:
    state = run_stage(world, only=[XSMOM], kinds=("fixed",))
    assert stands(world) == [] and any("--chain" in n for n in state["overlays"]["notes"])
    run_stage(world, only=[XSMOM], chains_only=["gradual_exit"], kinds=("fixed",))
    expected = [f"{XSMOM}-ov-exit-k{k}-d{d}" for k in (2, 3) for d in (3, 7)]
    assert sorted(stands(world)) == sorted(expected)
    spec = load_spec(world.specs / "overlays" / XSMOM / f"{expected[0]}.yaml")
    base = load_spec(world.specs / f"{XSMOM}.yaml")
    assert spec.code_ref == "qlab.strategies.overlays:Overlaid"
    assert spec.params_fixed_at == base.params_fixed_at      # the base's own dates, not a choice
    with registry.session_scope() as s:                  # every point is a trial of the family
        assert overlay_compare.family_trials(s, XSMOM) == 1 + 4


def test_an_explicit_request_is_not_capped(world) -> None:
    run_stage(world, only=[TREND, XSMOM])
    assert len(stands(world)) == 2 * VARIANTS_PER_BASE


def test_nothing_starts_after_the_stop_time_and_the_report_names_the_stage(world) -> None:
    state = run_stage(world, stop_at=datetime.now(UTC) - timedelta(minutes=1))
    assert stands(world) == [] and state["stopped_at"]["before"].startswith("overlays:")
    assert state["overlays"]["new"] == 0


def test_a_variant_that_started_always_finishes(world) -> None:
    stop = datetime.now(UTC) + timedelta(hours=1)
    state = run_stage(world, stop_at=stop, only=[XSMOM], chains_only=["own_results"])
    assert len(stands(world)) == 1 and "stopped_at" not in state


# ---- idempotent ----------


def test_a_variant_is_run_once_until_its_spec_changes(world) -> None:
    config = yaml.safe_load(world.config.read_text())
    config["bases"] = [XSMOM]
    world.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_stage(world, chains_only=["own_results"])
    assert stands(world) == [f"{XSMOM}-ov-own"]
    again = run_stage(world, chains_only=["own_results"])      # not forced: the ledger holds it
    assert stands(world) == [f"{XSMOM}-ov-own"]
    assert again["overlays"]["new"] == 0 and again["overlays"]["entries"] == []
    run_stage(world, only=[XSMOM], chains_only=["own_results"])  # asked for by name: run again
    assert len(stands(world)) == 2
    text = (world.specs / f"{XSMOM}.yaml").read_text(encoding="utf-8")
    (world.specs / f"{XSMOM}.yaml").write_text(
        text.replace("rebalance_days: 7", "rebalance_days: 14"), encoding="utf-8")
    run_stage(world, chains_only=["own_results"])
    assert len(stands(world)) == 3     # the base's numbers changed: the variant is run again


def test_the_data_end_alone_does_not_make_a_variant_new(world) -> None:
    run_stage(world, chains_only=["own_results"], max_variants=1)
    later = {"done": {}}
    overlay_stage.overlays_stage(
        DAY + timedelta(days=3), END + timedelta(days=3), later, noise_trials=1,
        config_path=world.config, spec_dir=world.specs, ledger_path=world.ledger,
        store=world.store, chains_only=["own_results"])
    assert len(stands(world)) == 2     # the other base's variant of the night; the first is done
    ledger = overlay_stage.load_ledger(world.ledger)
    assert set(ledger) == {f"{XSMOM}-ov-own", f"{TREND}-ov-own"}


def test_a_failed_run_is_an_error_row_and_is_tried_again(world) -> None:
    world.failing.add(f"{XSMOM}-ov-own")
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    entry = state["overlays"]["entries"][0]
    assert entry["action"] == "error" and "boom" in entry["error"]
    assert overlay_stage.load_ledger(world.ledger) == {}
    assert state["overlays"]["new"] == 0 and f"overlays:{XSMOM}-ov-own" not in state["done"]
    assert not [c for c in world.calls if c[0] in ("compare", "stage2")]
    world.failing.clear()
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    assert state["overlays"]["entries"][0]["action"] == "run"


def test_one_bases_failure_does_not_stop_the_others(world, monkeypatch) -> None:
    real = overlay_stage._Run.handle

    def flaky(self, base):
        if base.idea_id == TREND:
            raise RuntimeError("the base broke")
        return real(self, base)

    monkeypatch.setattr(overlay_stage._Run, "handle", flaky)
    state = run_stage(world, only=[TREND, XSMOM], chains_only=["own_results"])
    assert any(e["base"] == TREND and e["action"] == "error" for e in state["overlays"]["entries"])
    assert stands(world) == [f"{XSMOM}-ov-own"]


def test_numbers_without_a_source_are_not_run(world, monkeypatch) -> None:
    monkeypatch.setattr("qlab.pipeline.sources.source_problems",
                        lambda spec: ["no source for params.x"])
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    assert state["overlays"]["entries"][0]["action"] == "error"
    assert "without a source" in state["overlays"]["entries"][0]["error"]
    assert stands(world) == []


def test_a_comparison_that_failed_is_made_up_the_next_night_without_a_new_run(
        world, monkeypatch) -> None:
    real = overlay_compare.compare_specs
    monkeypatch.setattr(overlay_compare, "compare_specs",
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("no snapshot")))
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    assert "no snapshot" in state["overlays"]["entries"][0]["comparison_error"]
    assert overlay_compare.load_record(f"{XSMOM}-ov-own", world.store) is None
    monkeypatch.setattr(overlay_compare, "compare_specs", real)
    again = run_stage(world, chains_only=["own_results"], max_variants=1)
    assert overlay_compare.load_record(f"{XSMOM}-ov-own", world.store) is not None
    assert len([c for c in world.calls if c[:2] == ("stand", f"{XSMOM}-ov-own")]) == 1
    assert any("досчитано позже" in n for n in again["overlays"]["notes"])


def test_a_dry_run_changes_nothing(world) -> None:
    state = run_stage(world, only=[XSMOM], dry_run=True)
    actions = {e["action"] for e in state["overlays"]["entries"]}
    assert actions == {"would-run"} and world.calls == []
    assert not world.ledger.exists() and not (world.specs / "overlays").exists()
    assert state["done"] == {}


def test_the_stage_is_off_without_its_config_and_when_switched_off(world) -> None:
    state = {"done": {}}
    overlay_stage.overlays_stage(DAY, END, state, config_path=world.tmp / "none.yaml",
                                 spec_dir=world.specs, ledger_path=world.ledger)
    assert "overlays" not in state
    config = yaml.safe_load(world.config.read_text())
    config["enabled"] = False
    world.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert "overlays" not in run_stage(world)


# ---- counted, compared, sent on ----------


def test_every_run_is_a_trial_and_counts_in_the_bases_family(world, registry) -> None:
    with registry.session_scope() as s:
        repo.upsert_driver(s, id="momentum", title="m", description="d", kill_condition="k",
                           observable="o")
        s.get(Idea, XSMOM).driver_id = "momentum"
    run_stage(world, only=[XSMOM], chains_only=["own_results", "gradual_entry", "vol_target"])
    run_stage(world, only=[XSMOM], chains_only=["own_results"])      # a forced re-run: a trial
    with registry.session_scope() as s:
        assert overlay_compare.family_trials(s, XSMOM) == 1 + 4      # base + three + one re-run
        count = deflation_family(s, XSMOM, scope="driver")
        assert count.n_trials == 1 + 4
        assert s.get(Idea, f"{XSMOM}-ov-enter").driver_id == "momentum"


@pytest.mark.usefixtures("before_independent_venues")
def test_a_survivor_goes_to_the_second_stage_and_a_rejected_one_does_not(world) -> None:
    world.results[f"{XSMOM}-ov-own"] = "reject"
    world.results[f"{XSMOM}-ov-enter"] = "paper"
    state = run_stage(world, only=[XSMOM], chains_only=["own_results", "gradual_entry"])
    assert [c[1] for c in world.calls if c[0] == "stage2"] == [f"{XSMOM}-ov-enter"]
    by = {e["variant"]: e for e in state["overlays"]["entries"]}
    assert by[f"{XSMOM}-ov-enter"]["stage2"]["route"] == "paper"
    assert "stage2" not in by[f"{XSMOM}-ov-own"]
    run_stage(world, only=[TREND], chains_only=["gradual_exit"], second_stage=False)
    assert [c[1] for c in world.calls if c[0] == "stage2"] == [f"{XSMOM}-ov-enter"]


@pytest.mark.usefixtures("before_independent_venues")
def test_a_survivor_run_without_its_second_stage_owes_it_and_a_later_night_does_it(
        world) -> None:
    world.results[f"{XSMOM}-ov-enter"] = "paper"
    world.results[f"{XSMOM}-ov-own"] = "reject"
    config = yaml.safe_load(world.config.read_text())
    config["bases"] = [XSMOM]
    world.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_stage(world, only=[XSMOM], chains_only=["own_results", "gradual_entry"],
              second_stage=False)
    assert [c for c in world.calls if c[0] == "stage2"] == []
    ledger = overlay_stage.load_ledger(world.ledger)
    assert ledger[f"{XSMOM}-ov-enter"]["stage2_pending"] is True      # a survivor: owed
    assert ledger[f"{XSMOM}-ov-own"]["stage2_pending"] is False       # rejected: nothing owed
    state = run_stage(world, chains_only=["own_results", "gradual_entry"])  # a later night
    assert [c[1] for c in world.calls if c[0] == "stage2"] == [f"{XSMOM}-ov-enter"]
    assert len(stands(world)) == 2                                    # no first-stage re-run
    late = [e for e in state["overlays"]["entries"] if e["action"] == "stage2-late"]
    assert len(late) == 1 and late[0]["stage2"]["route"] == "paper"
    assert overlay_stage.load_ledger(world.ledger)[f"{XSMOM}-ov-enter"]["stage2_pending"] is False
    run_stage(world, chains_only=["own_results", "gradual_entry"])    # and only once
    assert len([c for c in world.calls if c[0] == "stage2"]) == 1
    body = "\n".join(overlay_stage.report_lines(state["overlays"], {}))
    assert "вторых этапов, долгов прошлых ночей: 1" in body


def test_the_variant_is_compared_on_the_bases_data_and_the_base_is_described(world) -> None:
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    compare = [c for c in world.calls if c[0] == "compare"]
    assert compare == [("compare", f"{XSMOM}-ov-own", XSMOM)]
    entry = state["overlays"]["entries"][0]
    assert entry["comparison"]["days"] == 800 and entry["action"] == "run"
    assert entry["seconds"] >= 0 and entry["route"] == "needs-forward"
    base_entry = state["overlays"]["bases"][0]
    assert base_entry["base"] == XSMOM and base_entry["diagnostics"]
    assert overlay_diagnostics.load(XSMOM, world.store)["idea_id"] == XSMOM


def test_the_stage_runs_on_the_bases_last_end_when_asked(world, registry) -> None:
    with registry.session_scope() as s:
        spec = s.query(Spec).filter_by(idea_id=XSMOM).first()
        spec.data_requirements = {"end": "2026-10-20"}
    run_stage(world, only=[XSMOM], chains_only=["own_results"], data_end="base")
    assert [c for c in world.calls if c[0] == "stand"][0][2] == date(2026, 10, 20)
    assert load_spec(world.specs / "overlays" / XSMOM / f"{XSMOM}-ov-own.yaml").data.end \
        == date(2026, 10, 20)


def test_meta_gate_is_offered_only_to_a_base_that_declares_decisions(world) -> None:
    config = yaml.safe_load(world.config.read_text())
    config["meta_decisions"] = {XSMOM: ["trend-binance-cap"]}
    world.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_stage(world, only=[XSMOM, TREND], max_variants=1000)
    xs = [v for v in stands(world) if v.startswith(XSMOM)]
    assert len(xs) == 6 + 15 and f"{XSMOM}-ov-meta" in xs
    assert not [v for v in stands(world) if v.startswith(TREND) and "-meta" in v]


# ---- the report ----------


def test_the_report_gives_the_table_the_counts_and_the_diagnostics(world, registry) -> None:
    state = run_stage(world, only=[XSMOM], chains_only=["own_results", "gradual_entry"])
    body = "\n".join(overlay_stage.report_lines(
        state["overlays"], {XSMOM: _table(registry, world)}))
    assert "Вариантов прогнано за ночь: 2" in body
    assert f"**{XSMOM}**" in body and "Войти в любой день и держать 30 дн." in body
    assert "| `own` |" in body and "| `enter` |" in body
    assert "→" in body and "помогла" in body                 # a verdict in words
    assert "Испытаний в семействе базы: 3" in body           # the base's and two variants'
    assert "Токены не тратятся" in body


def _table(registry, w) -> dict:
    with registry.session_scope() as s:
        return overlay_compare.table(s, XSMOM, w.store)


def test_the_report_names_what_waits_and_what_failed(world) -> None:
    world.failing.add(f"{TREND}-ov-own")
    state = run_stage(world, chains_only=["own_results", "gradual_entry", "vol_target"],
                      max_variants=2)
    body = "\n".join(overlay_stage.report_lines(state["overlays"], {}))
    assert "сбоев: 1" in body and "**сбой**" in body
    assert "ждут следующей ночи" in body


def test_the_night_report_has_the_section_after_the_twins_one(world, registry) -> None:
    state = run_stage(world, only=[XSMOM], chains_only=["own_results"])
    state["results"] = []
    state["twins"] = {"entries": [{"idea": "t", "action": "none", "reason": "r"}], "new": 0}
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    assert f"## {overlay_stage.SECTION_TITLE}" in text
    assert text.index(f"## {twin_stage.SECTION_TITLE}") < text.index(
        f"## {overlay_stage.SECTION_TITLE}")


def test_the_stage_spends_no_tokens() -> None:
    import inspect

    for module in (overlay_stage, overlay_compare, overlay_diagnostics):
        imports = [ln for ln in inspect.getsource(module).splitlines()
                   if ln.lstrip().startswith(("import ", "from "))]
        assert not [ln for ln in imports if "budget" in ln or "agents" in ln]


def test_the_shipped_config_is_read_and_puts_trend_and_xsmom_first_and_bv2_last() -> None:
    config = overlay_stage.load_config(ROOT / "night" / "overlays.yaml")
    assert config.enabled and config.priority == ("trend", "xsmom", "bv2")
    assert config.max_variants_per_night >= 1 and config.max_bases_per_night >= 1
    # Owner 2026-10-28: XSMOM's base is a tail lottery, its overlay variants were retired.
    assert "xsmom-research-nobull-causal" not in config.bases
    assert "trend-binance-cap-gated-causal" in config.bases
    assert config.bases[-1] == "bv2-research-headroom-causal"       # the control comes last


# ---- the night ----------


def _night_env(tmp_path, monkeypatch, order: list[str]):
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "unwatched_bench", lambda w: [])
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")

    def fake_twins(day, end, state, **kw):
        order.append(kw.get("label", "twins"))

    def fake_overlays(day, end, state, **kw):
        order.append("overlays")
        state["overlays"] = {"entries": [], "new": 0, "bases": [], "deferred": [],
                             "notes": ["a note of the stage"]}

    monkeypatch.setattr(twin_stage, "twins_stage", fake_twins)
    monkeypatch.setattr(night, "overlays_stage", fake_overlays)
    monkeypatch.setattr("qlab.waiting.waiting_stage", lambda *a, **kw: order.append("waiting"))
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: order.append("research"))
    monkeypatch.setattr(night, "paper_reconciliation", lambda day: order.append("paper") or [])


def _run_night(**kw):
    flags = dict(graveyard=False, implement=False, search=False, detectors=False, assess=False,
                 director=False, programmer=False, catalogue=False, decay=False)
    return night.run(DAY, **{**flags, **kw})


def test_the_anomaly_stage_runs_before_the_overlays_so_a_base_is_scanned_first(
        tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)

    def fake_anomalies(day, end, state, **kw):
        # two steps of the night call it (docs/PIPELINE.md): the code part, then the agent part
        order.append("anomalies:code" if kw.get("agent") is False else "anomalies:agent")
        state["anomalies"] = {"scanned": []}

    monkeypatch.setattr(night, "anomalies_stage", fake_anomalies)
    _run_night(paper=False, research=False)
    assert order == ["waiting", "twins", "anomalies:code", "anomalies:agent", "overlays"]
    order.clear()
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state2")
    _run_night(paper=False, research=False, anomalies=False)
    assert order == ["waiting", "twins", "overlays"]


def test_the_stage_runs_right_after_the_twins_and_follows_them_unless_given(
        tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    report = _run_night(paper=True, research=False)
    assert order == ["waiting", "twins", "overlays", "paper"]
    assert "a note of the stage" in report.read_text(encoding="utf-8")
    order.clear()
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state2")
    _run_night(paper=False, research=False, overlays=False)
    assert order == ["waiting", "twins"]
    order.clear()
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state3")
    _run_night(paper=False, research=False, waiting=False)
    assert order == []
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state4")
    _run_night(paper=False, research=False, waiting=False, overlays=True)
    assert order == ["overlays"]


def test_a_failing_stage_is_a_report_line_and_the_night_goes_on(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)

    def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(night, "overlays_stage", broken)
    report = _run_night(paper=True, research=False)
    assert order == ["waiting", "twins", "paper"]
    assert "RuntimeError: boom" in report.read_text(encoding="utf-8")


def test_the_whole_night_never_reaches_the_real_stage_in_tests() -> None:
    """conftest keeps `night.overlays_stage` off the owner's registry and files."""
    assert night.overlays_stage(None, None, {}) is None


def test_the_commands_are_registered() -> None:
    from typer.testing import CliRunner

    from qlab.cli import app

    out = CliRunner().invoke(app, ["overlays", "run", "--help"])
    assert out.exit_code == 0 and "--only" in out.output and "--chain" in out.output
    assert "--data-end" in out.output
    assert "--overlays" in CliRunner().invoke(app, ["night", "run", "--help"]).output
    assert CliRunner().invoke(app, ["overlays", "diagnostics", "--help"]).exit_code == 0
    assert CliRunner().invoke(app, ["overlays", "table", "--help"]).exit_code == 0


def test_the_twins_stage_leaves_overlay_variants_alone(world, registry) -> None:
    from qlab.waiting import spec_files

    run_stage(world, only=[XSMOM], chains_only=["own_results"])
    with registry.session_scope() as s:
        found, refused = twin_stage.find_candidates(
            s, spec_files(world.specs), only=[f"{XSMOM}-ov-own"])
        assert found == [] and "overlay variant" in refused[f"{XSMOM}-ov-own"]
        s.get(Idea, f"{XSMOM}-ov-own").status = IdeaStatus.BENCH
        found, _ = twin_stage.find_candidates(s, spec_files(world.specs))
        assert f"{XSMOM}-ov-own" not in [c.idea_id for c in found]
