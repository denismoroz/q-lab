"""A researcher's experiment that survives stage 1 on Binance's history goes on to the second
stage on Hyperliquid right after its stand run, the call the generator makes (docs/NIGHT.md,
docs/TWO_STAGE.md); its lines stay in the memo, the night's state and the «Агенты» page."""

from __future__ import annotations

import json
import types
from datetime import date
from pathlib import Path

import pytest
import yaml

import qlab.agent_reports as reports
import qlab.agents.researcher as researcher
import qlab.registry.db as db_module
from qlab import night, own_numbers
from qlab.pipeline import stage2
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.registry.models import Base

CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={}, stage1_survives=("paper", "shelf", "needs-infrastructure",
                                                         "needs-forward"))


def _spec(idea_id: str = "exp", source: str = "binance") -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": source, "interval": "1d", "start": "2025-01-01", "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def _result(route: str = "needs-forward", trial_id: int = 41):
    return types.SimpleNamespace(
        trial_id=trial_id, routing=types.SimpleNamespace(route=route, reason="r"),
        metrics={"ann_return_net": 0.1, "sharpe_net": 1.0, "max_dd": -0.2})


@pytest.fixture()
def second(monkeypatch):
    """The second stage replaced by a recorder."""
    calls: list[dict] = []

    def fake(spec, *, stage1_route, stage1_trial_id, noise_trials,
             config=None, **kw):
        calls.append({"spec": spec, "route": stage1_route, "trial": stage1_trial_id,
                      "noise": noise_trials, "kw": kw})
        return stage2.Stage2Result(
            parent_id=spec.idea_id, idea_id=f"{spec.idea_id}-hl", status="evaluated",
            route="paper", trial_id=77, stage1_route=stage1_route, data_start="2025-01-01",
            data_end="2026-10-26", combined_route="paper", parent_status="validated",
            metrics={"ann_return_net": 0.2, "sharpe_net": 1.1, "max_dd": -0.1})

    monkeypatch.setattr(stage2, "run_stage2", fake)
    monkeypatch.setattr(stage2, "load_config", lambda: CONFIG)
    return calls


def _lines(spec, result, opened=None, judged=None):
    return researcher.second_stage_lines(spec, judged or spec, opened or {}, result,
                                         noise_trials=1)


def test_a_survivor_on_binance_goes_to_stage_2_as_the_generator_does(second) -> None:
    lines = _lines(_spec(), _result("needs-forward", 41))
    assert len(second) == 1 and second[0]["route"] == "needs-forward"
    assert second[0]["trial"] == 41 and second[0]["noise"] == 1
    assert second[0]["kw"] == {}  # not a measurement: the parent's status moves with it
    assert lines and all(line.startswith("  - ") for line in lines)
    text = "\n".join(lines)
    assert "второй этап, Hyperliquid" in text and "итог по двум этапам" in text


@pytest.mark.parametrize("route", ["reject", "needs-more-data", "error", "not-evaluable"])
@pytest.mark.usefixtures("before_independent_venues")
def test_a_stage_1_that_did_not_survive_goes_nowhere(second, route) -> None:
    assert _lines(_spec(), _result(route)) == [] and second == []


def test_an_experiment_already_on_hyperliquid_has_nothing_new_to_look_at(second) -> None:
    assert _lines(_spec(source="hyperliquid"), _result()) == [] and second == []


def test_stage_2_switched_off_or_unreadable_never_fails_the_experiment(second, monkeypatch):
    monkeypatch.setattr(stage2, "load_config", lambda: types.SimpleNamespace(
        enabled=False, stage1_survives=("needs-forward",)))
    assert _lines(_spec(), _result()) == []

    def broken():
        raise ValueError("night/stage2.yaml: missing key 'venue'")

    monkeypatch.setattr(stage2, "load_config", broken)
    assert "второй этап: сбой ValueError" in _lines(_spec(), _result())[0]


def test_open_numbers_carry_the_value_chosen_at_the_cut(second, stand, monkeypatch) -> None:
    chosen = {"lookback": 7}
    monkeypatch.setattr(own_numbers, "chosen_params", lambda spec, session: chosen)
    judged = _spec("exp-judged")
    _lines(_spec(), _result(), opened={"cut": "2026-03-01", "candidate_trials": [1, 2]},
           judged=judged)
    sent = second[0]["spec"]
    assert sent.params == chosen and sent.params_fixed_at == date(2026, 3, 1)
    assert sent.idea_id == "exp"  # the idea the experiment is, not its one-choice form


def test_open_numbers_with_no_choice_at_the_cut_have_nothing_to_carry(second, stand, monkeypatch):
    monkeypatch.setattr(own_numbers, "chosen_params", lambda spec, session: None)
    lines = _lines(_spec(), _result(), opened={"cut": "2026-03-01"})
    assert second == [] and "не выбрано ни одного значения" in lines[0]


def test_a_failing_second_stage_is_a_line_not_a_failed_experiment(second, monkeypatch) -> None:
    def boom(*a, **kw):
        raise RuntimeError("no listing")

    monkeypatch.setattr(stage2, "run_stage2", boom)
    assert "второй этап: сбой RuntimeError: no listing" in _lines(_spec(), _result())[0]


# ---- the experiments on the stand, as `research_stage` runs them ---------------------------


@pytest.fixture()
def stand(tmp_path, monkeypatch):
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    monkeypatch.setattr("qlab.pipeline.sources.source_problems", lambda spec: [])
    ran: list[str] = []

    def evaluate(spec, **kw):
        ran.append(spec.idea_id)
        return types.SimpleNamespace(candidate=_result("needs-forward", 41))

    monkeypatch.setattr("qlab.calibration.shape_aware.evaluate_spec_with_shape_aware_bar",
                        evaluate)
    yield ran
    db_module._engine = None
    db_module._SessionLocal = None


def _write(tmp_path: Path, spec: StrategySpec) -> Path:
    path = tmp_path / f"{spec.idea_id}.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")
    return path


def test_the_stage_2_lines_follow_the_experiments_line_in_the_memo_and_the_state(
        tmp_path, stand, second) -> None:
    memo = tmp_path / "memo.md"
    memo.write_text("The memo.\n", encoding="utf-8")
    lines = researcher.run_experiments(
        memo, [_write(tmp_path, _spec("exp-b")), _write(tmp_path, _spec("exp-h", "hyperliquid"))],
        name="s", noise_trials=1)
    top = [i for i, line in enumerate(lines) if line.startswith("- `")]
    assert len(top) == 2 and "exp-b" in lines[top[0]] and "exp-h" in lines[top[1]]
    nested = [line for line in lines if line.startswith("  - ")]
    assert nested and all(top[0] < lines.index(line) < top[1] for line in nested)
    assert [c["spec"].idea_id for c in second] == ["exp-b"]  # the Hyperliquid one does not go
    assert "  - второй этап, Hyperliquid" in memo.read_text(encoding="utf-8")


def test_the_morning_report_and_the_agents_page_show_the_second_stage(
        tmp_path, monkeypatch) -> None:
    stand_lines = ["", "- `exp` (прогон 41): needs-forward — в год +10.0%",
                   "  - второй этап, Hyperliquid, прогон 77: прошла",
                   "  - итог по двум этапам: прошла оба этапа"]
    state = {"done": {}, "results": [], "research": [
        {"strategy": "xsmom", "memo": "m.md", "specs": ["a.yaml"], "tokens": 5,
         "stand": stand_lines}]}
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(reports, "_by_hand", lambda key, seen: [])
    text = night.write_report(date(2026, 10, 27), state).read_text(encoding="utf-8")
    assert "    - `exp` (прогон 41): needs-forward" in text
    assert "      - второй этап, Hyperliquid, прогон 77: прошла" in text
    (tmp_path / "2026-10-27.json").write_text(json.dumps(state, ensure_ascii=False))
    agent = {a["key"]: a for a in reports.agent_reports(state_dir=tmp_path)["agents"]}
    run = agent["researcher"]["runs"][0]
    assert run["result"] == "исходы экспериментов: ждёт проверки вперёд — 1"  # nested: not counted
    assert "xsmom: второй этап, Hyperliquid, прогон 77: прошла" in run["details"]
    assert any(d.startswith("xsmom: exp (прогон 41)") for d in run["details"])


def test_an_experiments_fixation_day_is_named_by_code_before_it_runs(
        tmp_path, stand, second) -> None:
    """Owner 2026-10-29 (В9): the researcher's day is corrected to the start of the last third of
    the window, and the memo says so."""
    from qlab import fixation

    spec = _spec("exp-b")
    path = _write(tmp_path, spec.model_copy(update={
        "params_fixed_at": date(2019, 1, 1), "params_fixed_evidence": "the agent's own day"}))
    memo = tmp_path / "memo.md"
    memo.write_text("The memo.\n", encoding="utf-8")
    lines = researcher.run_experiments(memo, [path], name="s", noise_trials=1)
    named = fixation.named_date(spec.data.start, spec.data.end, fixation.holdout_share())
    assert any("день фиксации назвал код" in line and str(named) in line for line in lines)
    assert load_spec(path).params_fixed_at == named
