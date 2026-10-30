"""The night's queue of closed ideas whose Hyperliquid side was never run (`qlab.second_venue`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import qlab.registry.db as db_module
from qlab import night, second_venue, waiting
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    StageTransition,
    TrialRoute,
    TrialSource,
    TrialStatus,
    Verdict,
    VerdictStage,
)

TODAY = date(2026, 10, 31)
CONFIG = second_venue.Config(enabled=True, max_per_night=2, retry_after_days=14)


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
          rejected_on: date | None = None) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if parent else None)
    session.get(Idea, idea_id).parent_id = parent
    if rejected_on is not None:
        session.add(StageTransition(
            idea_id=idea_id, from_status=IdeaStatus.BENCH, to_status=IdeaStatus.REJECTED,
            at=datetime(rejected_on.year, rejected_on.month, rejected_on.day, 3, tzinfo=UTC),
            reason="trial 1: reject"))


def _trial(session, idea_id: str, route: str, source: str, *, rules: str | None = None) -> int:
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref="a:b")
    trial = repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                           status=TrialStatus.OK, metrics={}, route=TrialRoute(route),
                           route_reason="r", source=TrialSource.IMPORTED)
    if rules:
        repo.add_verdicts(session, [dict(
            idea_id=idea_id, spec_id=spec.id, trial_id=trial.id, stage=VerdictStage.EDGE,
            rule_id="net_edge_positive", rules_version=rules, metric="ann_return_net", value=0.1,
            comparator=">", threshold=0.0, passed=True, data_range_start=date(2025, 1, 1),
            data_range_end=date(2026, 10, 27))])
    return trial.id


def _spec(tmp_path: Path, idea_id: str, source: str = "binance") -> None:
    spec = StrategySpec.model_validate({
        "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {"lookback": 14},
        "data": {"source": source, "interval": "1d", "start": "2020-01-01", "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})
    path = tmp_path / "specs" / f"{idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
                    encoding="utf-8")


def _world(tmp_path, registry) -> None:
    with registry.session_scope() as s:
        _idea(s, "young", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 28))
        _idea(s, "old", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 2))
        _idea(s, "middle", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 20))
        for name in ("young", "old", "middle"):
            _trial(s, name, "reject", "binance")
        # not queued: rejected on Hyperliquid's own data, still open, no spec file, a second stage
        _idea(s, "hl-reject", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 1))
        _trial(s, "hl-reject", "reject", "hyperliquid")
        _idea(s, "still-open", IdeaStatus.BENCH)
        _trial(s, "still-open", "reject", "binance")
        _idea(s, "no-spec", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 3))
        _trial(s, "no-spec", "reject", "binance")
        _idea(s, "old-hl", IdeaStatus.REJECTED, parent="old", rejected_on=date(2026, 10, 2))
        _trial(s, "old-hl", "reject", "hyperliquid", rules="2026-10-29.3")
        # judged on Hyperliquid by independent venues already: nothing owed
        _idea(s, "judged", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 4))
        _trial(s, "judged", "reject", "binance")
        _idea(s, "judged-hl", IdeaStatus.BENCH, parent="judged")
        _trial(s, "judged-hl", "needs-forward", "hyperliquid", rules=second_venue.FIRST_RULES)
        # judged on Hyperliquid under the older rules: owed again
        _idea(s, "stale", IdeaStatus.REJECTED, rejected_on=date(2026, 10, 5))
        _trial(s, "stale", "reject", "binance")
        _idea(s, "stale-hl", IdeaStatus.VALIDATED, parent="stale")
        _trial(s, "stale-hl", "paper", "hyperliquid", rules="2026-10-29.3")
    for name in ("young", "old", "middle", "hl-reject", "still-open", "judged", "stale"):
        _spec(tmp_path, name)


def _find(tmp_path, registry, ledger=None, config=CONFIG):
    from qlab.pipeline.stage2 import Stage2Config

    stage2_config = Stage2Config(
        enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
        slippage_bps=0.9, cost_sources={}, stage1_survives=("paper",))
    with registry.session_scope() as s:
        return second_venue.find_candidates(
            s, waiting.spec_files(tmp_path / "specs"), ledger=ledger or {}, today=TODAY,
            config=config, stage2_config=stage2_config, names=[])


def test_the_queue_is_the_rejected_binance_ideas_with_no_hyperliquid_run_oldest_first(
        tmp_path, registry) -> None:
    _world(tmp_path, registry)

    found, refused = _find(tmp_path, registry)

    assert [c.idea_id for c in found] == ["old", "stale", "middle", "young"]
    assert [c.rejected_on for c in found] == [date(2026, 10, 2), date(2026, 10, 5),
                                              date(2026, 10, 20), date(2026, 10, 28)]
    assert refused == {"no-spec": "нет файла спека"}
    numbers = second_venue.counts(found, CONFIG)
    assert numbers == {"owed": 4, "max_per_night": 2, "nights": 2}  # 4 ideas, 2 a night
    assert second_venue.counts([], CONFIG)["nights"] == 0


def test_a_finished_idea_is_not_queued_again_and_an_undecided_one_waits_its_days(
        tmp_path, registry) -> None:
    _world(tmp_path, registry)
    ledger = {
        "old": {"day": "2026-10-30", "outcome": "revived"},       # done for good
        "stale": {"day": "2026-10-30", "outcome": "rejected"},    # rejected there too: done
        "middle": {"day": "2026-10-20", "outcome": "undecided"},  # 11 days ago: not yet (14)
        "young": {"day": "2026-10-10", "outcome": "undecided"},   # 21 days ago: looked at again
    }

    found, _ = _find(tmp_path, registry, ledger)

    assert [c.idea_id for c in found] == ["young"]
    assert found[0].last_outcome == "undecided" and found[0].last_day == "2026-10-10"


def test_the_stage_runs_the_oldest_up_to_the_cap_and_a_pass_is_cited(
        tmp_path, registry, monkeypatch) -> None:
    _world(tmp_path, registry)
    seen = []

    def fake(plan, end, *, stage1, noise_trials, config, **kw):
        seen.append((plan.idea_id, plan.spec_path, kw))
        won = plan.idea_id == "old"
        return {"idea": plan.idea_id, "step": "stage2", "action": "rerun",
                "route_after": "paper" if won else "needs-forward", "trial": 77,
                "status": "validated" if won else "rejected", "lines": ["a line"]}

    monkeypatch.setattr(waiting, "rerun_stage2", fake)
    cfg = tmp_path / "second_venue.yaml"
    cfg.write_text("enabled: true\nmax_per_night: 2\nretry_after_days: 14\n", encoding="utf-8")
    state = {"done": {}}

    second_venue.second_venue_stage(
        TODAY, TODAY - timedelta(days=1), state, config_path=cfg, spec_dir=tmp_path / "specs",
        ledger_path=tmp_path / "ledger.json")

    assert [s[0] for s in seen] == ["old", "stale"]  # the two oldest rejections, no more
    note = seen[0][2]["settle_note"]
    assert "rejected on Binance on 2026-10-02" in note and "Hyperliquid had not been run" in note
    assert all(s[2]["only_a_pass_moves"] is True for s in seen)  # only a pass revives
    block = state["second_venue"]
    assert (block["owed"], block["nights"], block["owed_after"]) == (4, 2, 2)
    assert [e["outcome"] for e in block["entries"]] == ["revived", "undecided"]
    ledger = second_venue.load_ledger(tmp_path / "ledger.json")
    assert ledger["old"]["outcome"] == "revived" and ledger["stale"]["outcome"] == "undecided"
    assert set(state["done"]) == {"second_venue:old", "second_venue:stale"}
    text = "\n".join(second_venue.report_lines(block))
    assert "`old` (отвергнута 2026-10-02): оживлена: прошла на Hyperliquid" in text
    assert "В очереди: 4, ночей 2 при 2 в ночь." in text

    # the same night again does nothing twice
    seen.clear()
    second_venue.second_venue_stage(
        TODAY, TODAY - timedelta(days=1), state, config_path=cfg, spec_dir=tmp_path / "specs",
        ledger_path=tmp_path / "ledger.json")
    assert [s[0] for s in seen] == ["middle", "young"]  # the first two are done, not repeated


def test_a_crashed_run_is_tried_again_and_a_switched_off_queue_does_nothing(
        tmp_path, registry, monkeypatch) -> None:
    _world(tmp_path, registry)

    def boom(*a, **kw):
        raise RuntimeError("data source down")

    monkeypatch.setattr(waiting, "rerun_stage2", boom)
    cfg = tmp_path / "second_venue.yaml"
    cfg.write_text("enabled: true\nmax_per_night: 1\nretry_after_days: 14\n", encoding="utf-8")
    state = {"done": {}}
    second_venue.second_venue_stage(
        TODAY, TODAY, state, config_path=cfg, spec_dir=tmp_path / "specs",
        ledger_path=tmp_path / "ledger.json")
    entry = state["second_venue"]["entries"][0]
    assert entry["outcome"] == "error" and "data source down" in entry["error"]
    assert "second_venue:old" not in state["done"]  # a failed one is tried again

    off = tmp_path / "off.yaml"
    off.write_text("enabled: false\nmax_per_night: 4\nretry_after_days: 14\n", encoding="utf-8")
    quiet = {"done": {}}
    second_venue.second_venue_stage(TODAY, TODAY, quiet, config_path=off)
    assert "second_venue" not in quiet


def test_the_config_has_no_defaults(tmp_path) -> None:
    assert second_venue.load_config().max_per_night == 4  # the shipped file: four a night
    bad = tmp_path / "bad.yaml"
    bad.write_text("enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        second_venue.load_config(bad)


def test_the_ledger_word_for_a_result() -> None:
    word = second_venue.outcome_of
    assert word({"route": "paper", "parent_status": "validated"}) == "revived"
    assert word({"route": "paper", "parent_status": "rejected"}) == "passed"
    assert word({"route": "reject"}) == "rejected"
    assert word({"route": "needs-forward", "parent_status": "rejected"}) == "undecided"
    assert word({"route": "not-evaluable"}) == "not-evaluable"
    assert word({"error": "x"}) == "error" and word({"skipped": "off"}) == "skipped"


def test_the_shipped_night_does_not_run_the_queue_in_tests(monkeypatch) -> None:
    """conftest.py keeps whole-night tests off the owner's registry: the stage is stubbed there."""
    assert night.second_venue_stage(1) is None
    assert night.venue_gap_stage(1) is None
    assert Verdict.__tablename__ == "verdict"
