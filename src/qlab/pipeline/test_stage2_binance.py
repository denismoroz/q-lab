"""The second stage the other way round (owner, 2026-11-05: «почему вот эта стратегия не
тестировалась на бинансе?»; «да, делай и прогоняй все»): an idea born on Hyperliquid gets a Binance
stage `<idea>-bn` -- the translation back (names, costs, minimum leg, dropped coins, lists of
values), the child's id, the status over both venues, the ladder's Binance step, the night's plan
for an open idea and the queue for a closed one."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import ladder as lad
from qlab import second_venue, waiting
from qlab.forward_series import stage_runs
from qlab.pipeline import stage2, venue_status
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
)
from qlab.registry.stages import (
    binance_child,
    child_id,
    child_id_taken,
    child_venue,
    other_venue_child,
    stage2_child,
    stage2_children,
)

OWNER = "owner 2026-01-01: test fixture"
TWO_LEG = "derived: (5.0 + 10.0) / 2; test fixture"
BINANCE_START = date(2019, 9, 8)
END = date(2026, 10, 25)

BN = stage2.Stage2Config(
    enabled=True, venue="binance", data_start=BINANCE_START, taker_fee_bps=5.0, slippage_bps=0.9,
    cost_sources={"costs.taker_fee_bps": OWNER, "costs.slippage_bps": OWNER,
                  "costs.two_leg_taker_fee_bps": TWO_LEG},
    stage1_survives=("paper", "shelf", "needs-infrastructure", "needs-forward"),
    two_leg_taker_fee_bps=7.5)
HL = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=date(2025, 1, 1), taker_fee_bps=4.5,
    slippage_bps=0.9, cost_sources={"costs.taker_fee_bps": OWNER, "costs.slippage_bps": OWNER},
    stage1_survives=("paper", "shelf", "needs-infrastructure", "needs-forward"))
BINANCE_NAMES = ["BTC", "ETH", "1000BONK", "1000PEPE", "BTC-SPOT", "1000PEPE-SPOT"]


@pytest.fixture(autouse=True)
def tokens(monkeypatch):
    """Hyperliquid's bridge token of BTC, as night/stage2.yaml names it."""
    monkeypatch.setattr(stage2, "hyperliquid_token_coins", lambda *a, **k: {"UBTC": "BTC"})


def _hl_spec(**overrides: object) -> StrategySpec:
    base: dict[str, object] = {
        "idea_id": "trend-x-hl",
        "title": "Тренд на Hyperliquid",
        "code_ref": "qlab.strategies.blocks:Composed",
        "params": {"universe": {"rule": "all"}, "signal": {"block": "momentum",
                                                            "lookback_days": 14},
                   "score_costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9},
                   "grid": {"gate": {"always": {"trade_in": ["bull", "flat", "bear"]},
                                     "skip_flat": {"trade_in": ["bull", "bear"]}}}},
        "data": {"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1),
                 "end": END},
        "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9},
        "min_leg_notional": 10.0,
        "execution_venue": "hyperliquid",
        "unexpressed_mechanisms": [],
        "params_fixed_at": date(2025, 8, 1),
        "params_fixed_evidence": "Warm-up arithmetic.",
        "sources": {k: OWNER for k in ("universe", "signal", "score_costs", "grid", "costs",
                                        "min_leg_notional")},
    }
    base.update(overrides)
    return StrategySpec.model_validate(base)


# ---- the translation back -------------------------------------------------------------------


def test_a_hyperliquid_spec_is_translated_to_binance_with_its_names_costs_and_minimum_leg() -> None:
    spec = _hl_spec(data={"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1),
                          "end": END,
                          "instruments": ["BTC", "kBONK", "kPEPE", "HYPE", "UBTC-SPOT"]})

    t = stage2.translate(spec, BN, venue_names=BINANCE_NAMES)
    child = t.spec

    assert child is not None and child.idea_id == "trend-x-hl-bn"
    assert child.title.endswith("(второй этап: binance)")
    assert child.data.source == "binance" and child.data.start == BINANCE_START
    assert child.data.end == END
    # kBONK -> 1000BONK, kPEPE -> 1000PEPE, the bridge token UBTC -> Binance's BTC spot
    assert child.data.instruments == ["BTC", "1000BONK", "1000PEPE", "BTC-SPOT"]
    assert t.dropped == ["HYPE"]  # Binance has no HYPE: dropped and counted
    assert any("HYPE" in n and "not traded" in n for n in child.differs_from_source)
    assert (child.costs.taker_fee_bps, child.costs.slippage_bps) == (5.0, 0.9)
    assert child.sources["costs.taker_fee_bps"] == OWNER
    assert child.min_leg_notional == 5.0  # venues/binance.yaml, not Hyperliquid's $10
    assert "venues/binance.yaml" in child.sources["min_leg_notional"]
    assert child.execution_venue is None  # Binance's own facts: no adapter, needs-infrastructure
    assert any("born on Hyperliquid" in n for n in child.differs_from_source)


def test_the_numbers_and_the_declared_lists_travel_unchanged() -> None:
    spec = _hl_spec()
    child = stage2.translate(spec, BN).spec
    assert child.params == spec.params  # the grid code chooses from, the scoring costs: stage 1's
    assert child.params_fixed_at == spec.params_fixed_at
    assert child.params_fixed_evidence.startswith("Inherited from stage 1 (trend-x-hl)")


def test_a_two_legged_book_takes_binances_two_leg_fee() -> None:
    child = stage2.translate(_hl_spec(simultaneous_legs=2), BN).spec
    assert child.costs.taker_fee_bps == 7.5
    assert child.sources["costs.taker_fee_bps"] == TWO_LEG


def test_the_candidates_of_a_coin_list_rule_are_mapped_back_and_the_missing_ones_dropped() -> None:
    spec = _hl_spec(params={"universe": {"rule": "top_liquid",
                                         "candidates": ["UBTC", "kPEPE", "HYPE"]}})
    t = stage2.translate(spec, BN, venue_names=BINANCE_NAMES)
    assert t.spec.params["universe"]["candidates"] == ["BTC", "1000PEPE"]
    assert t.dropped_candidates == ["HYPE"]


def test_a_spec_none_of_whose_coins_binance_lists_cannot_be_made() -> None:
    spec = _hl_spec(data={"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1),
                          "end": END, "instruments": ["HYPE", "PURR"]})
    t = stage2.translate(spec, BN, venue_names=BINANCE_NAMES)
    assert t.spec is None and "binance" in t.problems[0]


def test_the_forward_direction_is_unchanged_and_each_spec_knows_its_second_venue() -> None:
    binance = _hl_spec(idea_id="trend-x", execution_venue=None,
                       data={"source": "binance", "interval": "1d", "start": BINANCE_START,
                             "end": END})
    child = stage2.translate(binance, HL).spec
    assert child.idea_id == "trend-x-hl" and child.data.source == "hyperliquid"
    assert child.data.start == date(2025, 1, 1)
    assert stage2.second_venue_of(binance) == "hyperliquid"
    assert stage2.second_venue_of(_hl_spec()) == "binance"


def test_an_idea_that_declares_lists_runs_its_check_on_binance_already() -> None:
    spec = _hl_spec(choice={"axes": {"signal.lookback_days": [7, 14]}, "source": OWNER})
    assert stage2.second_venue_of(spec) == "hyperliquid"


def test_the_binance_config_file_reads_the_binance_row_of_the_cost_table() -> None:
    config = stage2.config_for("binance")
    assert config.venue == "binance" and config.data_start == BINANCE_START
    assert (config.taker_fee_bps, config.slippage_bps, config.two_leg_taker_fee_bps) == (
        5.0, 0.9, 7.5)
    assert config.min_daily_volume_usd is None and config.spot_legs == {}


def test_the_crash_ledger_keeps_the_two_stages_of_a_hyperliquid_idea_apart() -> None:
    assert stage2.failure_key("x") == "x"
    assert stage2.failure_key("x", "binance") == "x-bn"


# ---- a registry ----------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.CANDIDATE,
          parent: str | None = None, stage2_child_: bool = True) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=(f"Second stage of {parent} (docs/TWO_STAGE.md)"
                            if parent and stage2_child_ else None)))
    session.flush()


_STEP = [0]


def _run(session, idea_id: str, route: str, source: str, code_ref: str = "x:Y") -> int:
    _STEP[0] += 1
    spec = repo.add_spec(session, idea_id=idea_id, version=_STEP[0], params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref=code_ref)
    trial = repo.add_trial(
        session, spec_id=spec.id, config_hash=f"h{_STEP[0]}", code_sha="x", params={},
        status=TrialStatus.OK, route=TrialRoute(route), route_reason="r", metrics={},
        source=TrialSource.IMPORTED, stand_version="2026-11-04.2",
        started_at=datetime(2026, 11, 1, 2, tzinfo=UTC))
    return trial.id


def _settle(session, idea_id: str, trial_id: int) -> venue_status.Settled:
    return venue_status.settle(session, idea_id, trial_id=trial_id, rules_version="2026-11-04.2",
                               run_data_end=date(2026, 11, 2))


def _status(session, idea_id: str) -> IdeaStatus:
    session.expire_all()
    return session.get(Idea, idea_id).status


def hl_born(session, hl: str, bn: str | None, status: IdeaStatus = IdeaStatus.CANDIDATE) -> int:
    _idea(session, "h", status)
    t = _run(session, "h", hl, "hyperliquid")
    if bn is not None:
        _idea(session, "h-bn", IdeaStatus.CANDIDATE, parent="h")
        t = _run(session, "h-bn", bn, "binance")
    return t


# ---- the child's id ------------------------------------------------------------------------


def test_the_binance_child_is_idea_bn_and_is_told_apart_from_the_hyperliquid_one(session) -> None:
    assert child_id("h", "binance") == "h-bn" and child_id("h") == "h-hl"
    _idea(session, "h")
    _idea(session, "h-hl", parent="h")
    _idea(session, "h-bn", parent="h")
    assert binance_child(session, "h").id == "h-bn" and stage2_child(session, "h").id == "h-hl"
    assert [c.id for c in stage2_children(session, "h")] == ["h-hl", "h-bn"]
    assert other_venue_child(session, "h").id == "h-bn"
    assert child_venue(session.get(Idea, "h-bn")) == "binance"
    assert child_venue(session.get(Idea, "h-hl")) == "hyperliquid"
    assert venue_status.family_root(session, "h-bn") == "h"


def test_an_id_taken_by_another_idea_is_never_written_to(session) -> None:
    _idea(session, "g")
    _idea(session, "g-bn", parent=None)  # another experiment that happens to end in -bn
    assert child_id_taken(session, "g", "binance") and not child_id_taken(session, "g")
    assert binance_child(session, "g") is None


# ---- the status over both venues -----------------------------------------------------------


def test_a_hyperliquid_idea_that_passes_on_binance_stands_on_binance(session) -> None:
    t = hl_born(session, "reject", "paper")
    settled = _settle(session, "h", t)
    assert (settled.view.binance, settled.view.hyperliquid) == ("paper", "reject")
    assert settled.view.combined.route == "needs-infrastructure"  # no Binance adapter
    assert settled.view.pass_ideas == ("h-bn",) and settled.view.pass_other_venue
    assert _status(session, "h") == IdeaStatus.BENCH


def test_a_closed_hyperliquid_idea_is_revived_by_a_binance_pass(session) -> None:
    t = hl_born(session, "reject", "paper", status=IdeaStatus.REJECTED)
    _settle(session, "h", t)
    assert _status(session, "h") != IdeaStatus.REJECTED


def test_a_hyperliquid_idea_rejected_on_both_venues_is_closed_with_its_binance_stage(
        session) -> None:
    t = hl_born(session, "reject", "reject")
    _settle(session, "h", t)
    assert _status(session, "h") == IdeaStatus.REJECTED
    assert _status(session, "h-bn") == IdeaStatus.REJECTED  # no open child under a closed parent


def test_a_hyperliquid_pass_stands_whatever_binance_said(session) -> None:
    t = hl_born(session, "paper", "reject")
    _settle(session, "h", t)
    assert _status(session, "h") == IdeaStatus.VALIDATED


def test_a_binance_stage_given_up_on_lets_the_hyperliquid_reject_close_the_idea(
        session, monkeypatch) -> None:
    monkeypatch.setattr(stage2, "load_failures",
                        lambda *a, **k: {"h-bn": {"given_up": True, "permanent": True,
                                                  "error": "ValueError: x"}})
    t = hl_born(session, "reject", None)
    settled = _settle(session, "h", t)
    assert settled.view.binance is None and not settled.view.bn_owed
    assert "Binance не проводился" in settled.reason
    assert _status(session, "h") == IdeaStatus.REJECTED


def test_a_production_book_owes_no_binance_stage(session) -> None:
    hl_born(session, "needs-forward", None, status=IdeaStatus.PAPER)
    view = venue_status.venue_view(session, "h")
    assert view.binance is None and not view.bn_owed


# ---- the ladder ----------------------------------------------------------------------------


def test_the_ladders_binance_step_of_a_hyperliquid_idea_is_its_binance_stage(session) -> None:
    hl_born(session, "needs-forward", "reject", status=IdeaStatus.BENCH)
    found = lad.build_ladder(session, "h", lad.Ledgers(independent=True))
    step2 = found.steps[1]
    assert step2.venue == "binance" and step2.other_idea == "h-bn"
    assert step2.outcome == lad.Outcome.FAILED and "версия на Binance «h-bn»" in step2.detail


def test_the_ladder_says_the_binance_stage_is_owed_until_it_runs(session) -> None:
    _idea(session, "o", IdeaStatus.BENCH)
    _run(session, "o", "needs-forward", "hyperliquid")
    _idea(session, "c", IdeaStatus.REJECTED)
    _run(session, "c", "reject", "hyperliquid")

    open_step = lad.build_ladder(session, "o", lad.Ledgers(independent=True)).steps[1]
    closed_step = lad.build_ladder(session, "c", lad.Ledgers(independent=True)).steps[1]

    assert open_step.reason == lad.Reason.WAITS_FOR_NIGHT and "o-bn" in open_step.detail
    assert closed_step.reason == lad.Reason.SECOND_VENUE_QUEUED


def test_the_binance_stage_page_shows_itself_as_the_binance_step(session) -> None:
    hl_born(session, "paper", "reject", status=IdeaStatus.VALIDATED)
    steps = lad.build_ladder(session, "h-bn", lad.Ledgers(independent=True)).steps
    assert steps[1].venue == "binance" and steps[1].outcome == lad.Outcome.FAILED
    assert steps[2].venue == "hyperliquid" and steps[2].other_idea == "h"


def test_the_charts_list_the_binance_stage_after_the_first(session) -> None:
    hl_born(session, "paper", "reject", status=IdeaStatus.VALIDATED)
    two, stages = stage_runs(session, "h")
    assert two and [(s.number, s.idea_id, s.source) for s in stages] == [
        (1, "h", "hyperliquid"), (2, "h-bn", "binance")]


# ---- the night: the open idea's plan, the closed idea's queue ------------------------------


def _write_spec(tmp_path: Path, spec: StrategySpec) -> Path:
    import yaml

    path = tmp_path / "specs" / "research" / f"{spec.idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True),
                                   allow_unicode=True), encoding="utf-8")
    return path


def test_an_open_hyperliquid_idea_gets_a_binance_stage_in_the_nights_plan(
        session, tmp_path) -> None:
    _idea(session, "trend-x-hl", IdeaStatus.BENCH)
    _run(session, "trend-x-hl", "needs-forward", "hyperliquid")
    path = _write_spec(tmp_path, _hl_spec())

    plans = waiting.plan_waiting(session, end=END, watched=set(),
                                 specs={"trend-x-hl": [path]}, stage2_config=HL,
                                 binance_config=BN)

    bn = [p for p in plans if p.step == waiting.BINANCE_STEP]
    assert len(bn) == 1 and bn[0].due and bn[0].idea_id == "trend-x-hl"
    assert "Binance" in bn[0].why
    # without the Binance config there is no such plan
    assert not [p for p in waiting.plan_waiting(
        session, end=END, watched=set(), specs={"trend-x-hl": [path]}, stage2_config=HL)
        if p.step == waiting.BINANCE_STEP]


def test_the_queue_of_closed_ideas_takes_a_closed_hyperliquid_idea_for_binance(
        session, tmp_path) -> None:
    _idea(session, "trend-x-hl", IdeaStatus.REJECTED)
    _run(session, "trend-x-hl", "paper", "hyperliquid")  # closed by its stricter look there
    path = _write_spec(tmp_path, _hl_spec())
    config = second_venue.Config(True, 4, 14, 30.0)

    found, refused = second_venue.find_candidates(
        session, {"trend-x-hl": [path]}, ledger={}, today=END, config=config,
        stage2_config=HL, names=[], binance_config=BN, binance_names=BINANCE_NAMES)

    assert [(c.idea_id, c.venue) for c in found] == [("trend-x-hl", "binance")]
    assert refused == {}
    assert "born on Hyperliquid" in second_venue.settle_note(found[0])


def test_the_queue_leaves_a_hyperliquid_idea_whose_binance_stage_has_run(
        session, tmp_path) -> None:
    _idea(session, "trend-x-hl", IdeaStatus.REJECTED)
    _run(session, "trend-x-hl", "reject", "hyperliquid")
    _idea(session, "trend-x-hl-bn", IdeaStatus.REJECTED, parent="trend-x-hl")
    _run(session, "trend-x-hl-bn", "reject", "binance")
    path = _write_spec(tmp_path, _hl_spec())

    found, _ = second_venue.find_candidates(
        session, {"trend-x-hl": [path]}, ledger={}, today=END,
        config=second_venue.Config(True, 4, 14, 30.0), stage2_config=HL, names=[],
        binance_config=BN, binance_names=BINANCE_NAMES)

    assert found == []


def test_the_report_names_the_venue_a_closed_idea_was_revived_on() -> None:
    lines = second_venue.report_lines({"entries": [
        {"idea": "trend-x-hl", "rejected_on": "2026-11-03", "outcome": "revived",
         "venue": "binance", "status": "bench", "at": "2026-11-05T01:00:00+00:00"}]})
    text = "\n".join(lines)
    assert "оживлена: прошла на Binance" in text and "отвергнута на Hyperliquid" in text


def test_the_report_of_a_binance_stage_says_where_the_idea_was_born() -> None:
    text = "\n".join(stage2.report_lines({"route": "reject"}, {
        "status": "evaluated", "venue": "binance", "route": "paper", "trial_id": 9,
        "data_start": "2019-09-08", "data_end": "2026-10-25", "metrics": {},
        "combined_route": "needs-infrastructure", "independent": True, "parent_status": "bench",
        "stage1_route": "paper",
        "venue_routes": {"binance": "paper", "hyperliquid": "reject"}}))
    assert "второй этап, Binance (идея родилась на Hyperliquid" in text
    # the Hyperliquid verdict is the venue's (its stricter look rejected), not the first look's
    assert "прошла на Binance" in text and "на Hyperliquid отвергнута" in text


def test_a_closed_idea_that_binance_cannot_decide_yet_is_said_to_stay_closed() -> None:
    text = "\n".join(stage2.report_lines({"route": "paper"}, {
        "status": "evaluated", "venue": "binance", "route": "needs-forward", "trial_id": 9,
        "data_start": "2019-09-08", "data_end": "2026-11-04", "metrics": {},
        "combined_route": "needs-forward", "independent": True, "parent_status": "rejected",
        "stage1_route": "paper",
        "venue_routes": {"binance": "needs-forward", "hyperliquid": "reject"}}))
    assert "идея остаётся закрытой" in text and "открытой" not in text
