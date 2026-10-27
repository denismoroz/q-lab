"""Tests for `qlab.twin_stage`: an idea that waits for forward data gets its causal twin on
Binance's history the same night, and a survivor goes on to the second stage (docs/NIGHT.md).

The stand and the second stage are replaced by recorders; the spec files that matter (an XSMOM
experiment and its base) are real files copied into a temporary folder, so nothing under
`specs/` or in the real registry is touched."""

from __future__ import annotations

import shutil
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import qlab.registry.db as db_module
from qlab import night, twin_stage
from qlab.night import ItemResult
from qlab.pipeline import stage2
from qlab.pipeline.spec import StrategySpec, load_spec
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
    TrialRoute,
    TrialStatus,
)

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"
END = date(2026, 10, 26)
DAY = date(2026, 10, 27)
STAGE2_CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))
EXPERIMENT = "xsmom-research-nobull"
BASE = "xsmom-honest-20legs"
TWIN = f"{EXPERIMENT}-causal"


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


def _idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.BENCH) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status)


def _trial(session, idea_id: str, route: str, *, metrics: dict | None = None):
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={}, data_requirements={}, rebalance="1d",
        costs_model={}, code_ref="a:b")
    snapshot = f"snap-{END}"
    if session.get(DataSnapshot, snapshot) is None:
        repo.add_data_snapshot(session, id=snapshot, source="hyperliquid", instruments={},
                               range_start=date(2025, 1, 1), range_end=END, path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                          status=TrialStatus.OK, snapshot_id=snapshot, metrics=metrics or {},
                          route=TrialRoute(route), route_reason="r")


def _plain_spec(idea_id: str, **data) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20", **data},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def _write_plain(path: Path, idea_id: str, **data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain_spec(idea_id, **data).model_dump(
        mode="json", exclude_none=True)), encoding="utf-8")
    return path


@pytest.fixture()
def world(tmp_path, registry, monkeypatch):
    """The XSMOM experiment and its base as files in a temporary `specs/`, both in the registry;
    the stand and the second stage recorded."""
    specs = tmp_path / "specs"
    (specs / "research" / "xsmom").mkdir(parents=True)
    shutil.copy(SPECS / f"{BASE}.yaml", specs / f"{BASE}.yaml")
    shutil.copy(SPECS / "research" / "xsmom" / f"{EXPERIMENT}.yaml",
                specs / "research" / "xsmom" / f"{EXPERIMENT}.yaml")
    plan = tmp_path / "research.yaml"
    plan.write_text(yaml.safe_dump({"strategies": {"xsmom": {
        "ideas": [BASE, EXPERIMENT], "specs": [str(specs / f"{BASE}.yaml")]}}}), encoding="utf-8")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n", encoding="utf-8")
    config = tmp_path / "twins.yaml"
    config.write_text("enabled: true\nmax_new_twins_per_night: 4\nslow_twin_minutes: 20\n",
                      encoding="utf-8")
    with registry.session_scope() as s:
        _idea(s, BASE, IdeaStatus.VALIDATED)
        _trial(s, BASE, "paper")
        _idea(s, EXPERIMENT)
        _trial(s, EXPERIMENT, "needs-forward")

    calls: list[tuple] = []
    results: dict[str, str] = {}

    def fake_eval(kind, spec, *, end, noise_trials, deployable_capital_usd):
        calls.append(("stand", spec.idea_id, end, noise_trials))
        route = results.get(spec.idea_id, "needs-forward")
        with registry.session_scope() as s:
            trial = _trial(s, spec.idea_id, route, metrics={
                "ann_return_net": 0.362, "sharpe_net": 0.9, "max_dd": -0.536,
                "ann_return_net_ex_best_1pct": -0.048, "noise_return_percentile": 1.0,
                "forward_days": 2491.0})
            trial_id = trial.id
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=trial_id, route=route,
                          reason="r")

    def fake_stage2(spec, *, stage1_route, stage1_trial_id, noise_trials, deployable_capital_usd,
                    config=None, **kw):
        calls.append(("stage2", spec.idea_id, stage1_route, stage1_trial_id, kw))
        return stage2.Stage2Result(parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl",
                                   status="evaluated", route="paper", trial_id=777,
                                   stage1_route=stage1_route, combined_route="paper",
                                   parent_status="validated")

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    monkeypatch.setattr(stage2, "run_stage2", fake_stage2)
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    monkeypatch.setattr("qlab.pipeline.sources.source_problems", lambda spec: [])
    return types.SimpleNamespace(
        tmp=tmp_path, specs=specs, plan=plan, watch=watch, config=config, calls=calls,
        results=results, ledger=tmp_path / "ledger.json",
        experiment=specs / "research" / "xsmom" / f"{EXPERIMENT}.yaml")


def _stage(w, state=None, **kw) -> dict:
    state = state if state is not None else {"done": {}}
    twin_stage.twins_stage(
        DAY, END, state, noise_trials=1, config_path=w.config, research_config=w.plan,
        spec_dir=w.specs, ledger_path=w.ledger, watch_config=w.watch, **kw)
    return state


# ---- which ideas get a twin ------------------------------------------------------------


def test_only_an_idea_in_work_that_waits_for_forward_data_is_a_candidate(world, registry) -> None:
    _write_plain(world.specs / "paper-idea.yaml", "paper-idea")
    _write_plain(world.specs / "waiting-idea.yaml", "waiting-idea")
    _write_plain(world.specs / "done-idea.yaml", "done-idea")
    shutil.copy(SPECS / "research" / "xsmom" / "xsmom-research-nobull-retune.yaml",
                world.specs / "research" / "xsmom" / "xsmom-research-nobull-retune.yaml")
    with registry.session_scope() as s:
        for idea, status, route in (("paper-idea", IdeaStatus.BENCH, "paper"),
                                    ("waiting-idea", IdeaStatus.CANDIDATE, "needs-forward"),
                                    ("done-idea", IdeaStatus.REJECTED, "needs-forward"),
                                    ("xsmom-research-nobull-retune", IdeaStatus.BENCH,
                                     "needs-forward")):
            _idea(s, idea, status)
            _trial(s, idea, route)
        # a twin and a second stage are never candidates, whatever their route
        _idea(s, TWIN)
        _trial(s, TWIN, "needs-forward")
        _idea(s, f"{EXPERIMENT}-hl")
        s.get(Idea, f"{EXPERIMENT}-hl").parent_id = EXPERIMENT
        s.get(Idea, f"{EXPERIMENT}-hl").notes = f"{stage2.NOTES_PREFIX}{EXPERIMENT}"
        _trial(s, f"{EXPERIMENT}-hl", "needs-forward")
        from qlab.waiting import spec_files

        found, refused = twin_stage.find_candidates(s, spec_files(world.specs))
        assert [c.idea_id for c in found] == [EXPERIMENT, "waiting-idea"]  # experiments first
        found, refused = twin_stage.find_candidates(
            s, spec_files(world.specs),
            only=["done-idea", "paper-idea", TWIN, "xsmom-research-nobull-retune", "nope"])
    assert [c.idea_id for c in found] == ["done-idea"]  # explicitly asked: status does not matter
    assert refused["paper-idea"].startswith("its latest route is paper")
    assert refused[TWIN] == "a twin or a second stage"
    assert "from the past already" in refused["xsmom-research-nobull-retune"]
    assert refused["nope"] == "the idea is not in the registry"


def test_the_base_pool_is_the_strategys_specs_its_folder_and_the_watch_list(world) -> None:
    _write_plain(world.specs / "research" / "xsmom" / "sibling.yaml", "sibling")
    _write_plain(world.specs / "research" / "xsmom" / "sibling-causal.yaml", "sibling-causal")
    _write_plain(world.specs / "watched.yaml", "watched")
    pool = twin_stage.base_candidates(world.experiment, yaml.safe_load(world.plan.read_text()),
                                      watch=[str(world.specs / "watched.yaml")])
    assert sorted(s.idea_id for _, s in pool) == sorted([BASE, "sibling", "watched"])
    # an idea outside a research folder has the watch list only
    outside = twin_stage.base_candidates(world.specs / f"{BASE}.yaml", {}, watch=[])
    assert outside == []


# ---- the twin is made, run, and goes on to the second stage -------------------------------


def test_the_twin_is_written_next_to_the_spec_run_on_the_stand_and_sent_to_stage_2(
        world, registry) -> None:
    state = _stage(world)
    twin_file = world.experiment.with_name(f"{TWIN}.yaml")
    assert twin_file.is_file()
    twin = load_spec(twin_file)
    assert twin.selects_causally and twin.data.source == "binance"
    assert [c[:2] for c in world.calls] == [("stand", TWIN), ("stage2", TWIN)]
    assert world.calls[0][2] == END  # the stand sees the data up to the last closed day
    assert world.calls[1][2] == "needs-forward"  # stage 2 is told the twin's stage 1 route
    assert world.calls[1][4] == {}  # not a measurement: the call the generator makes
    with registry.session_scope() as s:
        idea = s.get(Idea, TWIN)
        assert idea.parent_id == EXPERIMENT  # one branch of the strategy, not one more place
    entry = state["twins"]["entries"][0]
    assert entry["action"] == "made" and entry["base"] == BASE and entry["kind"] == "wrapper"
    assert entry["metrics"]["ann_return_net_ex_best_1pct"] == -0.048
    assert entry["stage2"]["route"] == "paper"
    assert state["twins"]["new"] == 1 and f"twins:{EXPERIMENT}" in state["done"]


def test_a_twin_is_not_rebuilt_or_rerun_by_the_stage_the_next_night(world) -> None:
    _stage(world)
    text = world.experiment.with_name(f"{TWIN}.yaml").read_text(encoding="utf-8")
    world.calls.clear()
    again = _stage(world)  # the same night, resumed: the done marker
    assert world.calls == [] and again["twins"]["entries"] == []
    tomorrow = _stage(world, {"done": {}})  # next night: the twin is an ordinary idea now
    assert world.calls == [] and tomorrow["twins"]["entries"] == []
    assert world.experiment.with_name(f"{TWIN}.yaml").read_text(encoding="utf-8") == text


def test_an_existing_twin_file_is_never_rebuilt_but_is_run_once_if_it_never_ran(world) -> None:
    from qlab.twin import causal_twin, write_twin

    made = causal_twin(load_spec(world.experiment), base=load_spec(
        world.specs / f"{BASE}.yaml"), spec_path=world.experiment).spec
    path = write_twin(world.experiment, made, EXPERIMENT)
    path.write_text(path.read_text(encoding="utf-8") + "# edited by hand\n", encoding="utf-8")
    edited = path.read_text(encoding="utf-8")
    state = _stage(world)
    assert path.read_text(encoding="utf-8") == edited  # left as it is
    assert state["twins"]["entries"][0]["action"] == "existing"
    assert [c[:2] for c in world.calls] == [("stand", TWIN), ("stage2", TWIN)]


def test_a_twin_that_fails_stage_1_does_not_reach_stage_2(world) -> None:
    world.results[TWIN] = "reject"
    state = _stage(world)
    assert [c[0] for c in world.calls] == ["stand"]
    entry = state["twins"]["entries"][0]
    assert entry["route"] == "reject" and "stage2" not in entry
    lines = "\n".join(twin_stage.entry_lines(entry))
    assert "второй этап не проводился" in lines


def test_a_crashed_twin_run_is_an_error_row_and_is_tried_again(world, monkeypatch) -> None:
    def crash(kind, spec, **kw):
        return ItemResult(kind=kind, name=spec.idea_id, error="ValueError: boom")

    monkeypatch.setattr(night, "evaluate_item", crash)
    state = _stage(world)
    entry = state["twins"]["entries"][0]
    assert entry["action"] == "error" and "boom" in entry["error"]
    assert f"twins:{EXPERIMENT}" not in state["done"]


def test_one_ideas_failure_is_its_row_and_the_others_still_run(world, registry, monkeypatch):
    second = world.specs / "research" / "xsmom" / "a-first.yaml"
    raw = yaml.safe_load(world.experiment.read_text(encoding="utf-8"))
    raw["idea_id"] = "a-first"
    second.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with registry.session_scope() as s:
        _idea(s, "a-first")
        _trial(s, "a-first", "needs-forward")
    real = twin_stage._run_twin

    def flaky(twin, origin, entry, **kw):
        if origin == "a-first":
            raise RuntimeError("disk full")
        return real(twin, origin, entry, **kw)

    monkeypatch.setattr(twin_stage, "_run_twin", flaky)
    state = _stage(world)
    rows = {e["idea"]: e for e in state["twins"]["entries"]}
    assert rows["a-first"]["action"] == "error" and "disk full" in rows["a-first"]["error"]
    assert rows[EXPERIMENT]["action"] == "made"


def test_numbers_without_a_source_are_not_run(world, monkeypatch) -> None:
    monkeypatch.setattr("qlab.pipeline.sources.source_problems", lambda spec: ["x.y"])
    state = _stage(world)
    assert world.calls == [] and state["twins"]["entries"][0]["action"] == "error"
    assert "x.y" in state["twins"]["entries"][0]["error"]


# ---- bounds ---------------------------------------------------------------------------------


def test_at_most_the_nightly_cap_of_new_twins_the_rest_wait_for_the_next_night(world, registry):
    second = world.specs / "research" / "xsmom" / "xsmom-research-nobull-two.yaml"
    raw = yaml.safe_load(world.experiment.read_text(encoding="utf-8"))
    raw["idea_id"] = "xsmom-research-nobull-two"
    second.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with registry.session_scope() as s:
        _idea(s, "xsmom-research-nobull-two")
        _trial(s, "xsmom-research-nobull-two", "needs-forward")
    world.config.write_text("enabled: true\nmax_new_twins_per_night: 1\nslow_twin_minutes: 20\n",
                            encoding="utf-8")
    state = _stage(world)
    assert [e["idea"] for e in state["twins"]["entries"]] == [EXPERIMENT]
    assert state["twins"]["deferred"] == ["xsmom-research-nobull-two"]
    assert "ждут следующей ночи: xsmom-research-nobull-two" in "\n".join(
        twin_stage.report_lines(state["twins"]))
    world.calls.clear()
    tomorrow = _stage(world, {"done": {}})
    assert [e["idea"] for e in tomorrow["twins"]["entries"]] == ["xsmom-research-nobull-two"]
    assert tomorrow["twins"]["deferred"] == []


def test_an_explicit_request_is_not_capped(world) -> None:
    world.config.write_text("enabled: true\nmax_new_twins_per_night: 0\nslow_twin_minutes: 20\n",
                            encoding="utf-8")
    assert _stage(world)["twins"]["entries"] == []  # the night's cap is 0: nothing, and said so
    state = _stage(world, only=[EXPERIMENT])
    assert state["twins"]["entries"][0]["action"] == "made"


def test_nothing_starts_after_the_stop_time_and_the_report_names_the_stage(world) -> None:
    state = {"done": {}}
    _stage(world, state, stop_at=datetime.now(UTC) - timedelta(minutes=1), label="twins-research")
    assert world.calls == [] and state["stopped_at"]["before"].startswith("twins-research:")
    assert not world.experiment.with_name(f"{TWIN}.yaml").exists()
    assert "twins-research" in night.STOP_STAGES and "twins" in night.STOP_STAGES


def test_a_slow_twin_is_reported_as_slow(world, monkeypatch) -> None:
    world.config.write_text("enabled: true\nmax_new_twins_per_night: 4\nslow_twin_minutes: 0\n",
                            encoding="utf-8")
    state = _stage(world)
    entry = state["twins"]["entries"][0]
    assert entry["slow"] is True
    assert "дольше нормы" in "\n".join(twin_stage.entry_lines(entry))


def test_hourly_candles_of_every_binance_perpetual_are_too_heavy_for_a_night() -> None:
    hourly = _plain_spec("hourly", interval="1h", source="binance")
    assert "hours" in twin_stage.heavy_data(hourly, 20)
    named = hourly.model_copy(update={"data": hourly.data.model_copy(
        update={"instruments": ["BTC", "ETH"]})})
    assert twin_stage.heavy_data(named, 20) is None  # a short list is cheap
    assert twin_stage.heavy_data(_plain_spec("daily", source="binance"), 20) is None
    assert twin_stage.heavy_data(_plain_spec("hl", interval="1h"), 20) is None


def test_an_hourly_list_is_heavy_past_the_columns_the_night_allows_and_costs_the_fetch() -> None:
    hourly = _plain_spec("hourly", interval="1h", source="binance",
                         instruments=[f"C{i}" for i in range(41)])
    assert twin_stage.heavy_data(hourly, 20) is None  # no bound given: only the missing list
    reason = twin_stage.heavy_data(hourly, 20, 40, 25.0)
    assert "41 instruments" in reason and "17 minutes" in reason and "40 instruments" in reason
    fits = hourly.model_copy(update={"data": hourly.data.model_copy(
        update={"instruments": [f"C{i}" for i in range(40)]})})
    assert twin_stage.heavy_data(fits, 20, 40, 25.0) is None  # 40 x 25 s is under the 20 minutes
    # the daily panel of the whole universe is cached: a long list costs nothing there
    daily = _plain_spec("daily", source="binance", instruments=[f"C{i}" for i in range(99)])
    assert twin_stage.heavy_data(daily, 20, 40, 25.0) is None


def test_the_real_config_carries_the_threshold_the_seconds_and_the_coin_parameters() -> None:
    config = twin_stage.load_config(ROOT / "night" / "twins.yaml")
    assert config.max_hourly_instruments == 40 and config.seconds_per_instrument == 25.0
    assert config.coin_list_params == ("universe", "coins")
    # 40 columns at the measured rate fit one twin's bound (the reason for the number 40)
    assert 40 * config.seconds_per_instrument < config.slow_twin_minutes * 60


def test_a_ledger_entry_of_the_old_rules_is_not_kept(world) -> None:
    assert twin_stage.fingerprint(world.experiment, None).startswith(
        f"v{twin_stage.LEDGER_VERSION}:")


def _bv2_result():
    from qlab.twin import causal_twin

    names = ["BTC", "ETH", "SOL", "AVAX", "BTC-SPOT", "ETH-SPOT", "SOL-SPOT", "AVAX-SPOT"]
    return causal_twin(load_spec(SPECS / "research" / "bv2" / "bv2-research-shared-pool-hl.yaml"),
                       base=load_spec(SPECS / "bv2-live.yaml"), venue_names=names)


def test_a_twin_with_an_instrument_list_is_run_and_the_report_names_the_list(
        world, monkeypatch) -> None:
    monkeypatch.setattr(twin_stage, "_causal_twin", lambda *a, **k: _bv2_result())
    state = _stage(world)
    entry = state["twins"]["entries"][0]
    assert entry["action"] == "made" and len(entry["instruments"]) == 8
    assert entry["list_source"] == "required_instruments of bv2-research-shared-pool-hl"
    assert entry["fetch_minutes"] == 3.3
    assert [c[0] for c in world.calls] == ["stand", "stage2"]
    text = "\n".join(twin_stage.entry_lines(entry))
    assert "только 8 столбцами" in text and "со дня, когда список уже был известен" in text


def test_a_list_past_the_threshold_is_recorded_as_heavy_with_its_size(world, monkeypatch) -> None:
    monkeypatch.setattr(twin_stage, "_causal_twin", lambda *a, **k: _bv2_result())
    world.config.write_text("enabled: true\nmax_new_twins_per_night: 4\nslow_twin_minutes: 20\n"
                            "max_hourly_instruments: 5\n", encoding="utf-8")
    state = _stage(world)
    entry = state["twins"]["entries"][0]
    assert entry["action"] == "heavy" and world.calls == []
    assert "8 instruments" in entry["reason"] and "5 instruments" in entry["reason"]
    text = "\n".join(twin_stage.entry_lines(entry))
    assert "8 столбцов" in text and "max_hourly_instruments" in text


def test_a_heavy_twin_is_not_written_or_run_but_recorded_and_a_hand_run_can_allow_it(
        world, monkeypatch) -> None:
    monkeypatch.setattr(twin_stage, "heavy_data", lambda spec, minutes, *rest: "too heavy")
    state = _stage(world)
    entry = state["twins"]["entries"][0]
    assert entry["action"] == "heavy" and world.calls == []
    assert not world.experiment.with_name(f"{TWIN}.yaml").exists()
    assert "--allow-heavy" in "\n".join(twin_stage.entry_lines(entry))
    assert EXPERIMENT in twin_stage.load_ledger(world.ledger)
    allowed = _stage(world, {"done": {}}, only=[EXPERIMENT], allow_heavy=True)
    assert allowed["twins"]["entries"][0]["action"] == "made"
    assert [c[0] for c in world.calls] == ["stand", "stage2"]


# ---- no twin: the reason is recorded once ----------------------------------------------------


def test_no_twin_is_recorded_once_and_tried_again_only_when_the_spec_file_changes(
        world, registry) -> None:
    lonely = _write_plain(world.specs / "lonely.yaml", "lonely")
    with registry.session_scope() as s:
        _idea(s, "lonely", IdeaStatus.CANDIDATE)
        _trial(s, "lonely", "needs-forward")
    first = _stage(world, only=None)
    none = [e for e in first["twins"]["entries"] if e["idea"] == "lonely"]
    assert none[0]["action"] == "none" and "no declared alternative" in none[0]["reason"]
    assert "lonely" in twin_stage.load_ledger(world.ledger)
    assert not lonely.with_name("lonely-causal.yaml").exists()
    text = "\n".join(twin_stage.entry_lines(none[0]))
    assert "двойника нет" in text and "Больше не пробуем" in text

    second = _stage(world, {"done": {}})  # the next night: nothing about it, but counted
    assert [e["idea"] for e in second["twins"]["entries"]] == []
    assert second["twins"]["recorded"] == ["lonely"]
    assert twin_stage.report_lines(second["twins"]) == []  # it is not the report's news again

    lonely.write_text(lonely.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    third = _stage(world, {"done": {}})
    assert [e["idea"] for e in third["twins"]["entries"]] == ["lonely"]


def test_a_dry_run_changes_nothing(world) -> None:
    state = _stage(world, only=[EXPERIMENT], dry_run=True)
    assert state["twins"]["entries"][0]["action"] == "would-make"
    assert not world.experiment.with_name(f"{TWIN}.yaml").exists()
    assert world.calls == [] and not world.ledger.exists() and state["done"] == {}
    assert not (world.tmp / "state").exists()  # not even the night's state is saved


def test_the_stage_is_off_without_its_config_and_when_switched_off(world, tmp_path) -> None:
    missing = {"done": {}}
    twin_stage.twins_stage(DAY, END, missing, config_path=tmp_path / "none.yaml",
                           spec_dir=world.specs)
    assert "twins" not in missing
    world.config.write_text("enabled: false\nmax_new_twins_per_night: 4\nslow_twin_minutes: 20\n",
                            encoding="utf-8")
    assert "twins" not in _stage(world)


def test_the_shipped_config_is_read() -> None:
    config = twin_stage.load_config(ROOT / "night" / "twins.yaml")
    assert config.enabled and config.max_new_twins_per_night == 4
    assert config.slow_twin_minutes == 20


def test_the_stage_spends_no_tokens() -> None:
    import inspect

    imports = [ln for ln in inspect.getsource(twin_stage).splitlines()
               if ln.lstrip().startswith(("import ", "from "))]
    assert not [ln for ln in imports if "budget" in ln or "agents" in ln]


# ---- the second stage of an experiment on Binance's history ----------------------------------


def test_only_an_experiment_on_binance_goes_to_the_second_stage() -> None:
    assert twin_stage.goes_to_stage2(_plain_spec("a", source="binance"))
    assert not twin_stage.goes_to_stage2(_plain_spec("b"))


# ---- the morning report -----------------------------------------------------------------------


def test_the_report_gives_the_plain_numbers_the_stage_2_lines_and_the_deferred(world) -> None:
    state = _stage(world)
    state["twins"]["deferred"] = ["later"]
    body = "\n".join(twin_stage.report_lines(state["twins"]))
    assert "в год +36.2%" in body and "Шарп 0.90" in body and "просадка -53.6%" in body
    assert "без лучшего 1% дней -4.8%" in body
    assert "второй этап, Hyperliquid" in body and "ждут следующей ночи: later" in body
    assert f"{TWIN}.yaml" in body  # the twin's file is named


def test_the_night_report_has_the_section_after_the_waiting_one(world, monkeypatch) -> None:
    state = _stage(world)
    state["results"] = []
    path = night.write_report(DAY, state)
    text = path.read_text(encoding="utf-8")
    assert f"## {twin_stage.SECTION_TITLE}" in text and TWIN in text


# ---- the night -----------------------------------------------------------------------------------


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
        state.setdefault("twins", {"entries": [], "new": 0})["entries"].append(
            {"idea": f"idea-{kw.get('label', 'twins')}", "action": "none",
             "reason": "no alternative"})

    monkeypatch.setattr(twin_stage, "twins_stage", fake_twins)
    monkeypatch.setattr("qlab.waiting.waiting_stage",
                        lambda *a, **kw: order.append("waiting"))
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: order.append("research"))
    monkeypatch.setattr(night, "paper_reconciliation", lambda day: order.append("paper") or [])


def _run_night(**kw):
    flags = dict(graveyard=False, implement=False, search=False, detectors=False, assess=False,
                 director=False, programmer=False, catalogue=False, decay=False)
    return night.run(DAY, **{**flags, **kw})


def test_the_stage_runs_right_after_waiting_and_again_after_the_researcher(
        tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    report = _run_night(paper=True, research=True)
    assert order == ["waiting", "twins", "paper", "research", "twins-research"]
    text = report.read_text(encoding="utf-8")
    assert f"## {twin_stage.SECTION_TITLE}" in text and "idea-twins-research" in text


def test_twins_follow_the_waiting_flag_unless_given(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)
    _run_night(paper=False, research=False, waiting=False)
    assert order == []
    order.clear()
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state2")
    _run_night(paper=False, research=False, waiting=False, twins=True)
    assert order == ["twins"]


def test_a_failing_twins_stage_is_a_report_line_and_the_night_goes_on(
        tmp_path, monkeypatch) -> None:
    order: list[str] = []
    _night_env(tmp_path, monkeypatch, order)

    def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(twin_stage, "twins_stage", broken)
    report = _run_night(paper=True, research=False)
    assert order == ["waiting", "paper"]
    text = report.read_text(encoding="utf-8")
    assert "двойники и вторая ступень" in text and "RuntimeError: boom" in text


def test_the_command_is_registered() -> None:
    from typer.testing import CliRunner

    from qlab.cli import app

    out = CliRunner().invoke(app, ["twins", "run", "--help"])
    assert out.exit_code == 0 and "--only" in out.output and "--allow-heavy" in out.output
    night_help = CliRunner().invoke(app, ["night", "run", "--help"])
    assert "--twins" in night_help.output


def test_a_twin_entry_shows_its_result_year_by_year() -> None:
    entry = {"idea": "x", "action": "made", "twin_spec": "x-causal.yaml", "trial": 1,
             "route": "reject", "seconds": 60, "candidates": ["a", "b"],
             "metrics": {"ann_return_net": 0.1, "sharpe_net": 1.0, "max_dd": -0.1,
                         "year_2024_return": 0.2, "year_2024_days": 366.0,
                         "year_2025_return": -0.05, "year_2025_days": 365.0}}
    text = "\n".join(twin_stage.entry_lines(entry))
    assert "по годам: 2024 +20%, 2025 −5%" in text
