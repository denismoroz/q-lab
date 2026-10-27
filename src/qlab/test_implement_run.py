"""The implement stage end to end with no tokens (qlab.implement_run, docs/IMPLEMENTER.md): a card
from the inbox goes through a stubbed implementer, the guards' bookkeeping, the written spec, its
causal version over the card's grid, the baseline on the same panel and the second stage of every
survivor, and the morning report shows the comparison.

The agent, the reviewer, the budget, the stand and the second stage are replaced by stubs; the
registry is a temporary database and the working directory a temporary project, so nothing of the
owner's is touched (the conftest keeps the ledger and the block register away too)."""

from __future__ import annotations

import types
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

import qlab.agents.implementer as implementer
import qlab.registry.db as db_module
from qlab import agent_blocks, implement_run, night, twin_stage
from qlab.agent_reports import agent_reports
from qlab.agents import reviewer
from qlab.agents.implementer import ImplementOutcome
from qlab.agents.reviewer import Finding, ReviewOutcome
from qlab.card_fields import read_card_fields
from qlab.night import ItemResult
from qlab.pipeline import stage2
from qlab.pipeline.spec import load_spec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Idea,
    Profile,
    SourceType,
    Spec,
    TrialRoute,
    TrialStatus,
)

DAY = date(2026, 10, 28)
END = night.last_closed_day(datetime(2026, 10, 28, 3, tzinfo=UTC))
STAGE2_CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))
METRICS = {
    "ann_return_net": 0.123, "sharpe_net": 0.9, "max_dd": -0.31,
    "ann_return_net_ex_best_1pct": 0.045, "noise_return_percentile": 0.97, "forward_days": 800.0,
    "year_2021_return": 0.40, "year_2021_days": 365, "year_2022_return": -0.10,
    "year_2022_days": 365, "years_counted": 2, "years_positive": 1,
}


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


def _trial(session, idea_id: str, route: str, metrics: dict):
    if session.get(Idea, idea_id) is None:
        repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
        session.flush()
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={}, data_requirements={}, rebalance="1d",
        costs_model={}, code_ref="a:b")
    snapshot = f"snap-{END}"
    if session.get(DataSnapshot, snapshot) is None:
        repo.add_data_snapshot(session, id=snapshot, source="binance", instruments={},
                               range_start=date(2019, 9, 8), range_end=END, path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                          status=TrialStatus.OK, snapshot_id=snapshot, metrics=metrics,
                          route=TrialRoute(route), route_reason="r")


def _write(path: Path, raw: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _plain_raw(idea_id: str, **over) -> dict:
    raw = {
        "idea_id": idea_id, "title": idea_id, "code_ref": "qlab.strategies.agent.toy_block:Toy",
        "params": {"signal": {"window_days": 7}, "top_n": 3},
        "data": {"source": "binance", "interval": "1d", "start": "2022-01-01",
                 "end": "2026-10-26"},
        "costs": {"taker_fee_bps": 1.5, "slippage_bps": 0.5}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "params_fixed_at": "2026-10-27",
        "params_fixed_evidence": "written by the implementer agent from the card on 2026-10-27",
        "sources": {"signal.window_days": "card", "top_n": "card", "costs": "card",
                    "min_leg_notional": "card"}}
    raw.update(over)
    return raw


def _card_raw(world, **over) -> dict:
    raw = {
        "idea_id": "toy-idea", "title": "Игрушка", "found_at": "2026-10-27",
        "plain_summary": "Держим монеты, которые выросли.",
        "idea_text": "купить то, что выросло за неделю",
        "baseline_spec": str(world.baseline_file),
        "grid": {"params": {"signal.window_days": [7, 14, 30]}},
        "new_block": {
            "name": "weekly-momentum",
            "params": [{"name": "window_days", "meaning": "сколько суток считать рост"}],
            "rule": "держим монеты, которые сильнее всего выросли за окно"},
        "data": {"venue": "binance", "interval": "1d", "coins": ["BTC", "ETH"],
                 "honest_from": "2022-01-01"}}
    raw.update(over)
    return raw


@pytest.fixture()
def world(tmp_path, registry, monkeypatch):
    """A temporary project: a baseline spec, a queue, stubs for the agent, the reviewer, the
    budget, the stand and the second stage. `world.calls` records, in order, what ran."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(implementer, "QLAB_ROOT", tmp_path)
    baseline_file = _write(tmp_path / "specs" / "base.yaml", _plain_raw(
        "old-strategy", code_ref="qlab.strategies.trend:TrendTSMOMEnsemble",
        params={"lookbacks_days": [30]}, params_fixed_at="2025-03-01",
        costs={"taker_fee_bps": 9.0, "slippage_bps": 9.0}, min_leg_notional=99.0,
        data={"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
              "end": "2026-10-26"},
        sources={"lookbacks_days": "card", "costs": "card", "min_leg_notional": "card"}))
    w = types.SimpleNamespace(
        tmp=tmp_path, baseline_file=baseline_file,
        queue=tmp_path / "night" / "implement_queue.yaml",
        calls=[], agent_calls=[], routes={}, fail={}, review_findings=[], stops=[],
        venue="binance", interval="1d", coins=True, problems=[], not_expressible=None,
        ledger=tmp_path / "ledger.json",
        listing=["BTC", "ETH", "BTC-SPOT", "ETH-SPOT"])

    def add_card(name="toy", **over) -> str:
        path = _write(tmp_path / "cards" / f"{name}.yaml", _card_raw(w, **over))
        w.queue.parent.mkdir(parents=True, exist_ok=True)
        queued = (yaml.safe_load(w.queue.read_text()) or {}).get("cards", []) \
            if w.queue.is_file() else []
        w.queue.write_text(yaml.safe_dump({"cards": [*queued, str(path)]}), encoding="utf-8")
        return str(path)

    def fake_implement(card, *, budget, session, **kw):
        w.agent_calls.append(str(card))
        fields = read_card_fields(card, tmp_path)
        module = fields.new_block.module if fields.new_block else f"plain_{fields.idea_id}"
        code = tmp_path / "src/qlab/strategies/agent" / f"{module}.py"
        reused = code.is_file()
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("window = params['signal']['window_days']\n", encoding="utf-8")
        data = {"source": w.venue, "interval": w.interval, "start": "2022-01-01",
                "end": "2026-10-26"}
        if w.venue == "binance" and w.coins:
            data["instruments"] = ["BTC", "ETH"]
        _write(tmp_path / "specs/agent" / f"{fields.idea_id}.yaml", _plain_raw(
            fields.idea_id, code_ref=f"qlab.strategies.agent.{module}:Toy", data=data))
        return ImplementOutcome(
            idea_id=fields.idea_id, code_path=Path("src/qlab/strategies/agent") / f"{module}.py",
            spec_path=Path("specs/agent") / f"{fields.idea_id}.yaml", agent_reply="Toy",
            problems=list(w.problems), not_expressible=w.not_expressible,
            fields=fields if fields.inbox else None, block_reused=reused)

    def fake_review(**kw):
        return ReviewOutcome(accepted=list(w.review_findings), rejected=[], raw_text=None)

    def fake_eval(kind, spec, *, end, noise_trials, deployable_capital_usd):
        w.calls.append(("stand", spec.idea_id, kind, spec.data.source))
        w.evaluated[spec.idea_id] = spec
        if spec.idea_id in w.fail:
            raise RuntimeError(w.fail[spec.idea_id])
        route = w.routes.get(spec.idea_id, "paper")
        with registry.session_scope() as s:
            trial = _trial(s, spec.idea_id, route, METRICS)
            trial_id = trial.id
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=trial_id, route=route,
                          reason="r")

    def fake_stage2(spec, *, stage1_route, stage1_trial_id, noise_trials, deployable_capital_usd,
                    config=None, **kw):
        w.calls.append(("stage2", spec.idea_id, stage1_route, kw))
        return stage2.Stage2Result(parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl",
                                   status="evaluated", route="paper", trial_id=777,
                                   stage1_route=stage1_route, combined_route="paper",
                                   parent_status="validated", metrics=METRICS,
                                   data_start="2025-01-01", data_end=str(END))

    class _Budget:
        spent_tokens = 1234

    class _Stage:
        def candidate(self, _name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, _stage, candidates):
            return _Stage()

    import qlab.budget as budget_module

    w.evaluated = {}
    w.add_card = add_card
    monkeypatch.setattr(implementer, "implement", fake_implement)
    monkeypatch.setattr(reviewer, "run_review", fake_review)
    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    monkeypatch.setattr(stage2, "run_stage2", fake_stage2)
    monkeypatch.setattr(stage2, "load_config", lambda: STAGE2_CONFIG)
    monkeypatch.setattr("qlab.pipeline.sources.source_problems", lambda spec: [])
    monkeypatch.setattr(twin_stage, "_binance_listing", lambda: w.listing)
    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe", lambda _session: types.SimpleNamespace(
        report=types.SimpleNamespace(reading=None)))
    monkeypatch.setattr(implement_run, "LEDGER", w.ledger)
    monkeypatch.setattr(implement_run, "BASELINE_DIR", tmp_path / "baselines")
    monkeypatch.setattr(agent_blocks, "REGISTRY", tmp_path / "agent_blocks.yaml")

    real_past_stop = night.past_stop

    def past_stop(stop_at, state, stage):
        if any(stage.endswith(s) for s in w.stops):
            state.setdefault("stopped_at", {"time": "t", "stop_at": datetime.now(UTC).isoformat(),
                                            "before": stage})
            return True
        return real_past_stop(stop_at, state, stage)

    monkeypatch.setattr(night, "past_stop", past_stop)
    return w


def _stage(w, state=None, **kw) -> dict:
    state = state if state is not None else {"done": {}}
    night.implement_stage(DAY, state, noise_trials=1, deployable_capital_usd=3000.0,
                          queue=w.queue, **kw)
    return state


def _order(w) -> list[tuple]:
    return [c[:2] for c in w.calls]


# ---- the whole path ------------------------------------------------------------------------


def test_a_card_goes_from_idea_to_verdicts_in_the_order_the_stage_promises(world) -> None:
    world.add_card()
    state = _stage(world)
    assert len(world.agent_calls) == 1
    # a. the spec as written, b. the causal version, c. the baseline, d. stage 2 of survivors
    assert _order(world) == [
        ("stand", "toy-idea"), ("stand", "toy-idea-causal"), ("stand", "toy-idea-base"),
        ("stage2", "toy-idea"), ("stage2", "toy-idea-causal")]
    assert all(c[2] == "implemented" for c in world.calls if c[0] == "stand")
    (entry,) = state["implemented"]
    assert entry["idea_id"] == "toy-idea" and entry["tokens"] == 1234
    assert entry["idea_text"] == "купить то, что выросло за неделю"
    assert set(entry["runs"]) == set(implement_run.STEPS)
    assert all(r["action"] == "ran" for r in entry["runs"].values())
    assert entry["stand"] == "paper: r"  # the line the report printed before the extra runs

    causal, base = world.evaluated["toy-idea-causal"], world.evaluated["toy-idea-base"]
    spec = world.evaluated["toy-idea"]
    assert causal.selects_causally and causal.code_ref.endswith("retune:Retune")
    assert causal.params["inner_code_ref"] == spec.code_ref
    assert list(causal.params["grid"]) == ["signal.window_days"]  # exactly the card's grid
    assert entry["runs"]["causal"]["candidates"] == ["window_days=7", "window_days=14",
                                                     "window_days=30"]
    assert causal.params["refit_every"] == "M" and causal.params["window"] == "expanding"
    assert causal.params["objective"] == "sharpe_net"
    assert causal.data.source == "binance" and causal.data.interval == "1d"
    # the hand-picked coins are judged from the day the card says they were known
    assert causal.judged_from == causal.instruments_chosen_at == date(2022, 1, 1)
    assert causal.data.start == date(2019, 9, 8) and causal.data.instruments == ["BTC", "ETH"]
    # the baseline stands on exactly the causal version's panel, costs and judged part
    assert base.data == causal.data and base.costs == causal.costs
    assert base.judged_from == causal.judged_from and base.params == {"lookbacks_days": [30]}
    assert base.params_fixed_at == date(2025, 3, 1)  # the baseline's own numbers are untouched
    # the files: the twin next to the spec, the baseline in the ledger's folder, one block
    assert (world.tmp / "specs/agent/toy-idea-causal.yaml").is_file()
    assert (world.tmp / "baselines/toy-idea-base.yaml").is_file()
    assert load_spec(world.tmp / "specs/agent/toy-idea-causal.yaml").selects_causally
    (block,) = agent_blocks.load()
    assert block.name == "weekly-momentum" and block.specs == ["specs/agent/toy-idea.yaml"]
    assert entry["block"]["file"] == "src/qlab/strategies/agent/weekly_momentum.py"
    assert entry["block"]["reused"] is False
    ledger = twin_stage.load_ledger(world.ledger)
    assert ledger[world.agent_calls[0]]["status"] == "complete"
    with world_session() as s:
        assert s.get(Idea, "toy-idea-causal").parent_id == "toy-idea"
        assert s.get(Idea, "toy-idea-base").parent_id == "toy-idea"


def world_session():
    from qlab.registry.db import session_scope

    return session_scope()


def test_the_second_stage_is_the_generators_call_not_a_measurement(world) -> None:
    world.add_card()
    _stage(world)
    stage2_calls = [c for c in world.calls if c[0] == "stage2"]
    assert [c[3] for c in stage2_calls] == [{}, {}]  # `measure` is never passed
    assert [c[2] for c in stage2_calls] == ["paper", "paper"]  # told stage 1's route


def test_data_from_the_execution_venue_has_no_second_stage_to_pass(world) -> None:
    world.venue = "hyperliquid"
    world.add_card(data={"venue": "hyperliquid", "interval": "1d", "coins": ["BTC", "ETH"],
                         "honest_from": "2022-01-01"})
    state = _stage(world)
    assert _order(world)[-1] == ("stage2", "toy-idea-causal")
    assert ("stage2", "toy-idea") not in _order(world)
    runs = state["implemented"][0]["runs"]
    assert runs["stage2-spec"]["action"] == "skipped"
    assert "hyperliquid" in runs["stage2-spec"]["reason"]
    # the causal version runs on Binance's history as a proxy for the card's venue
    assert world.evaluated["toy-idea-causal"].execution_venue == "hyperliquid"


def test_only_a_survivor_goes_to_the_second_stage(world) -> None:
    world.routes = {"toy-idea": "reject", "toy-idea-causal": "paper"}
    world.add_card()
    state = _stage(world)
    assert [c[:2] for c in world.calls if c[0] == "stage2"] == [("stage2", "toy-idea-causal")]
    skipped = state["implemented"][0]["runs"]["stage2-spec"]
    assert skipped["action"] == "skipped" and "отвергнута" in skipped["reason"]


def test_a_hand_written_card_behaves_as_before(world) -> None:
    path = _write(world.tmp / "cards" / "old.yaml", {
        "idea_id": "old-idea", "title": "Старая", "plain_summary": "Просто так."})
    world.queue.parent.mkdir(parents=True, exist_ok=True)
    world.queue.write_text(yaml.safe_dump({"cards": [str(path)]}), encoding="utf-8")
    state = _stage(world)
    (entry,) = state["implemented"]
    assert "block" not in entry and "idea_text" not in entry
    assert _order(world)[0] == ("stand", "old-idea")
    runs = entry["runs"]
    assert runs["spec"]["action"] == "ran" and entry["stand"] == "paper: r"
    assert runs["baseline"]["action"] == "skipped"  # the card names no baseline
    assert runs["causal"]["action"] == "none"  # no grid declared on the card, none in the lists
    assert agent_blocks.load() == []


# ---- the night's stop, errors, heavy data, the reviewer -----------------------------------


def test_a_run_cut_by_the_stop_time_is_done_on_the_next_night_without_the_agent(world) -> None:
    world.add_card()
    world.stops = [":causal"]
    first = _stage(world)
    assert _order(world) == [("stand", "toy-idea")]  # the written spec ran, then the stop
    assert "время ночи вышло" in first["implemented"][0]["stopped"]
    ledger = twin_stage.load_ledger(world.ledger)
    assert ledger[world.agent_calls[0]]["status"] == "pending-run"
    assert set(ledger[world.agent_calls[0]]["runs"]) == {"spec"}

    world.stops = []
    world.calls.clear()
    second = _stage(world, {"done": {}})  # the next night: a state of its own
    assert len(world.agent_calls) == 1  # the agent is not called again, no tokens
    assert _order(world) == [("stand", "toy-idea-causal"), ("stand", "toy-idea-base"),
                             ("stage2", "toy-idea"), ("stage2", "toy-idea-causal")]
    (entry,) = second["implemented"]
    assert entry["resumed"] and entry["idea_text"] == "купить то, что выросло за неделю"
    assert set(entry["runs"]) == set(implement_run.STEPS)
    assert twin_stage.load_ledger(world.ledger)[world.agent_calls[0]]["status"] == "complete"
    third = _stage(world, {"done": {}})  # settled: nothing is taken again
    assert third["implemented"] == [] and len(world.agent_calls) == 1


def test_a_step_that_failed_is_tried_once_more_on_the_next_night(world) -> None:
    world.add_card()
    world.fail = {"toy-idea-causal": "the stand fell over"}
    first = _stage(world)
    runs = first["implemented"][0]["runs"]
    assert runs["causal"]["action"] == "error" and runs["causal"]["tries"] == 1
    assert "the stand fell over" in runs["causal"]["error"]
    assert runs["baseline"]["action"] == "ran"  # one step's failure does not stop the others
    assert runs["stage2-causal"]["action"] == "waiting"  # nothing to send on yet
    assert twin_stage.load_ledger(world.ledger)[world.agent_calls[0]]["status"] == "pending-run"
    world.fail = {}
    world.calls.clear()
    second = _stage(world, {"done": {}})
    assert _order(world) == [("stand", "toy-idea-causal"), ("stage2", "toy-idea-causal")]
    (entry,) = second["implemented"]
    assert entry["runs"]["causal"]["action"] == "ran" and entry["runs"]["causal"]["tries"] == 2
    assert len(world.agent_calls) == 1
    assert twin_stage.load_ledger(world.ledger)[world.agent_calls[0]]["status"] == "complete"


def test_a_step_that_fails_twice_is_left_as_it_is(world) -> None:
    world.add_card()
    world.fail = {"toy-idea-causal": "the stand fell over"}
    _stage(world)
    _stage(world, {"done": {}})
    third = _stage(world, {"done": {}})
    assert third["implemented"] == []  # two tries are all a step gets
    rec = twin_stage.load_ledger(world.ledger)[world.agent_calls[0]]
    assert rec["status"] == "complete" and rec["runs"]["causal"]["tries"] == 2


def test_data_too_heavy_for_a_night_is_named_not_fetched(world) -> None:
    world.interval, world.coins = "1h", False
    world.add_card(data={"venue": "binance", "interval": "1h",
                         "universe_rule": "all perpetuals", "honest_from": "2022-01-01"})
    state = _stage(world)
    runs = state["implemented"][0]["runs"]
    for step in ("spec", "causal", "baseline"):
        assert runs[step]["action"] == "heavy" and "hourly" in runs[step]["reason"], step
    assert [c for c in world.calls if c[0] == "stand"] == []  # nothing was fetched or run
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    assert "данные слишком тяжёлые для одной ночи" in text


def test_a_reviewers_verified_gap_runs_the_written_spec_only(world) -> None:
    world.review_findings = [Finding(kind="unexpressed_mechanism", summary="the exit is missing",
                                     source_quote="q", code_ref="c:1",
                                     spec_path="p")]
    world.add_card()
    state = _stage(world)
    assert _order(world) == [("stand", "toy-idea")]  # a run about another strategy is not made
    runs = state["implemented"][0]["runs"]
    assert runs["causal"]["action"] == runs["baseline"]["action"] == "skipped"
    assert "block" not in state["implemented"][0] and agent_blocks.load() == []  # not registered
    assert twin_stage.load_ledger(world.ledger)[world.agent_calls[0]]["status"] == "blocked"
    assert _stage(world, {"done": {}})["implemented"] == []  # waits for the owner


def test_a_failed_attempt_is_retried_once_and_then_waits_for_the_owner(world) -> None:
    world.problems = ["source: a number without a source"]
    card = world.add_card()
    first = _stage(world)
    assert first["implemented"][0]["problems"] and first["implemented"][0]["attempts"] == 1
    assert "runs" not in first["implemented"][0] and _order(world) == []
    second = _stage(world, {"done": {}})
    assert second["implemented"][0]["attempts"] == 2
    third = _stage(world, {"done": {}})
    assert third["implemented"] == [] and world.agent_calls == [card, card]
    text = night.write_report(DAY, second).read_text(encoding="utf-8")
    assert "попыток 2 из 2: больше не пробуем, карточка ждёт владельца" in text


def test_an_honest_not_expressible_is_final(world) -> None:
    world.not_expressible = "NOT EXPRESSIBLE: no order book in the data"
    world.add_card()
    _stage(world)
    assert _stage(world, {"done": {}})["implemented"] == []
    assert len(world.agent_calls) == 1 and _order(world) == []


def test_the_second_use_of_a_block_reuses_it(world) -> None:
    world.add_card("one")
    world.add_card("two", idea_id="toy-idea-two")
    state = _stage(world)
    blocks = [e["block"] for e in state["implemented"]]
    assert [b["reused"] for b in blocks] == [False, True]
    (entry,) = agent_blocks.load()
    assert len(entry.cards) == 2 and len(entry.specs) == 2


# ---- the morning report ------------------------------------------------------------------


def test_the_report_shows_the_idea_the_block_and_the_comparison(world) -> None:
    world.add_card()
    state = _stage(world)
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    section = text.split("## Реализация карточек агентом")[1]
    assert "идея: купить то, что выросло за неделю" in section
    assert "блок `weekly-momentum`" in section
    assert "window_days — сколько суток считать рост" in section
    assert "| версия | данные | в год | Шарп | просадка | без 1% лучших дней | исход |" in section
    assert "| база `old-strategy` |" in section
    assert "| написанная `toy-idea` |" in section
    assert "| с выбором кода `toy-idea-causal` |" in section
    assert "+12.3%" in section and "0.90" in section and "-31.0%" in section
    assert "+4.5%" in section  # the return without the best 1% of days
    assert "по годам: 2021 +40%, 2022 −10%" in section  # the per-year line, once per version
    assert section.count("по годам:") >= 3
    assert "второй этап, Hyperliquid" in section  # stage 2's own lines under the survivors
    assert "торгует с" in section and "window_days=7, window_days=14, window_days=30" in section
    assert "судится только со дня" in section  # the hand-picked list is said to be judged late


def test_a_version_that_did_not_run_is_a_dash_and_a_reason(world) -> None:
    # a grid whose first variant the written spec does not carry: no causal version can be made
    world.add_card(grid={"params": {"signal.window_days": [14, 30]}}, baseline_spec=None)
    state = _stage(world)
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    section = text.split("## Реализация карточек агентом")[1]
    assert "с выбором кода: не сделана — the spec carries signal.window_days = 7" in section
    assert "| написанная `toy-idea` |" in section and "| база" not in section


def test_the_agents_page_gets_the_three_routes(world) -> None:
    world.add_card()
    state = _stage(world)
    night.save_state(DAY, state)
    page = agent_reports(days=1, state_dir=night.STATE_DIR)
    (agent,) = [a for a in page["agents"] if a["key"] == "implementer"]
    (run,) = agent["runs"]
    assert "написанная" in run["result"] and "с выбором кода" in run["result"]
    assert "база" in run["result"]
