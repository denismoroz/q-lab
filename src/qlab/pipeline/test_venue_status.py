"""One function decides an idea's status from both venues' verdicts
(`qlab.pipeline.venue_status`, docs/VENUE_SPLIT.md, rules 2026-10-30.2): an idea passes if it
passes anywhere, is rejected only when no venue passed and none is pending, and one venue rejecting
while the other has not spoken leaves it open. Every pairing of data source and second stage, and
the cases that used to leave orphans."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.pipeline import stage2, venue_status
from qlab.registry import repo
from qlab.registry.lifecycle import CLOSED_STATUSES
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
          parent: str | None = None) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if parent else None))
    session.flush()


_STEP = [0]


def _run(session, idea_id: str, route: str, source: str) -> int:
    _STEP[0] += 1
    spec = repo.add_spec(session, idea_id=idea_id, version=_STEP[0], params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref="x:Y")
    trial = repo.add_trial(
        session, spec_id=spec.id, config_hash=f"h{_STEP[0]}", code_sha="x", params={},
        status=TrialStatus.OK, route=TrialRoute(route), route_reason="r", metrics={},
        source=TrialSource.IMPORTED, stand_version="2026-10-30.2",
        started_at=datetime(2026, 11, 1, 2, tzinfo=UTC))
    return trial.id


def _settle(session, idea_id: str, trial_id: int) -> venue_status.Settled:
    return venue_status.settle(session, idea_id, trial_id=trial_id, rules_version="2026-10-30.2",
                               run_data_end=date(2026, 11, 2))


def status(session, idea_id: str) -> IdeaStatus:
    session.expire_all()
    return session.get(Idea, idea_id).status


def no_orphan(session) -> None:
    """No second stage is open under a closed first stage."""
    for child in session.query(Idea).filter(Idea.parent_id.isnot(None)):
        if child.notes and child.notes.startswith("Second stage of "):
            parent = session.get(Idea, child.parent_id)
            if parent.status in (IdeaStatus.REJECTED, IdeaStatus.RETIRED):
                assert child.status in CLOSED_STATUSES, (child.id, child.status)


# ---- Binance data (whatever the spec's execution venue says), second stage on Hyperliquid -----


def binance_family(session, b: str, hl: str | None, status_: IdeaStatus = IdeaStatus.CANDIDATE):
    _idea(session, "i", status_)
    t = _run(session, "i", b, "binance")
    if hl is not None:
        _idea(session, "i-hl", IdeaStatus.CANDIDATE, parent="i")
        t = _run(session, "i-hl", hl, "hyperliquid")
    return t


def test_a_binance_reject_with_hyperliquid_not_yet_run_leaves_the_idea_open(session) -> None:
    t = binance_family(session, "reject", None)
    settled = _settle(session, "i", t)

    assert settled.view.hl_owed and settled.view.combined.route == "needs-forward"
    assert status(session, "i") == IdeaStatus.BENCH
    assert "second venue" in settled.reason


def test_a_binance_reject_with_hyperliquid_pending_leaves_the_idea_open(session) -> None:
    t = binance_family(session, "reject", "needs-forward")
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.BENCH


def test_a_hyperliquid_pass_makes_the_idea_pass_whatever_binance_said(session) -> None:
    t = binance_family(session, "reject", "paper")
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.VALIDATED


def test_a_binance_pass_with_a_hyperliquid_reject_stands_on_binance_alone(session) -> None:
    t = binance_family(session, "needs-infrastructure", "reject")
    settled = _settle(session, "i", t)
    assert settled.view.combined.route == "needs-infrastructure"
    assert status(session, "i") == IdeaStatus.BENCH


def test_both_venues_rejecting_closes_the_idea_and_its_second_stage(session) -> None:
    t = binance_family(session, "reject", "reject")
    # the second stage was opened by its own run; closing the parent closes it with it
    _idea_hl = session.get(Idea, "i-hl")
    _idea_hl.status = IdeaStatus.BENCH
    session.flush()
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.REJECTED
    assert status(session, "i-hl") == IdeaStatus.REJECTED
    no_orphan(session)


def test_a_binance_reject_closes_when_hyperliquid_cannot_judge_the_idea(
        session, monkeypatch) -> None:
    """Hyperliquid's side is not owed when the second stage cannot be made (here: its crashes were
    given up on): nothing is pending, the reject stands."""
    monkeypatch.setattr(stage2, "load_failures", lambda *a, **k: {"i": {"given_up": True}})
    t = binance_family(session, "reject", None)
    settled = _settle(session, "i", t)
    assert not settled.view.hl_owed
    assert status(session, "i") == IdeaStatus.REJECTED


def test_a_binance_reject_closes_when_the_strategy_has_no_hyperliquid_panel(
        session, monkeypatch) -> None:
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("binance",))
    t = binance_family(session, "reject", None)
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.REJECTED


def test_a_crashed_second_stage_is_pending_until_given_up_and_then_not_run(
        session, monkeypatch) -> None:
    t = binance_family(session, "reject", "error")
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.BENCH  # tried again: the venue has not spoken

    monkeypatch.setattr(stage2, "load_failures", lambda *a, **k: {"i": {"given_up": True}})
    settled = _settle(session, "i", t)
    assert settled.view.hyperliquid is None  # «не проводился: сбой»
    assert status(session, "i") == IdeaStatus.REJECTED
    no_orphan(session)


def test_a_crash_never_opens_a_closed_idea(session) -> None:
    t = binance_family(session, "reject", "error", IdeaStatus.REJECTED)
    settled = _settle(session, "i", t)
    assert settled.stayed_closed
    assert status(session, "i") == IdeaStatus.REJECTED


def test_a_pending_second_venue_does_not_reopen_a_closed_idea(session) -> None:
    t = binance_family(session, "reject", "needs-forward", IdeaStatus.REJECTED)
    settled = _settle(session, "i", t)
    assert settled.stayed_closed and status(session, "i") == IdeaStatus.REJECTED


def test_a_pass_revives_a_closed_idea_on_new_data(session) -> None:
    t = binance_family(session, "reject", "paper", IdeaStatus.REJECTED)
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.VALIDATED


def test_a_run_that_decides_nothing_moves_nothing(session) -> None:
    t = binance_family(session, "not-evaluable", None, IdeaStatus.BENCH)
    _settle(session, "i", t)
    assert status(session, "i") == IdeaStatus.BENCH


# ---- Hyperliquid data only, with and without a same-venue second stage -----------------------


def hl_family(session, first: str, second: str | None):
    _idea(session, "h", IdeaStatus.CANDIDATE)
    t = _run(session, "h", first, "hyperliquid")
    if second is not None:
        _idea(session, "h-hl", IdeaStatus.CANDIDATE, parent="h")
        t = _run(session, "h-hl", second, "hyperliquid")
    return t


def test_a_hyperliquid_only_idea_is_judged_by_its_one_venue(session, monkeypatch) -> None:
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("hyperliquid",))
    t = hl_family(session, "paper", None)
    settled = _settle(session, "h", t)
    assert settled.view.binance is None and not settled.view.hl_owed
    assert status(session, "h") == IdeaStatus.VALIDATED


def test_a_hyperliquid_only_reject_closes_it_when_binance_cannot_judge_it(
        session, monkeypatch) -> None:
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("hyperliquid",))
    t = hl_family(session, "reject", None)
    _settle(session, "h", t)
    assert status(session, "h") == IdeaStatus.REJECTED


def test_a_hyperliquid_reject_with_binance_not_yet_run_leaves_the_idea_open(session) -> None:
    """Owner, 2026-11-05: «почему вот эта стратегия не тестировалась на бинансе?». An idea born
    on Hyperliquid owes a Binance verdict as a Binance idea owes a Hyperliquid one."""
    t = hl_family(session, "reject", None)
    settled = _settle(session, "h", t)
    assert settled.view.bn_owed and settled.view.binance == "needs-forward"
    assert settled.view.combined.route == "needs-forward"
    assert status(session, "h") == IdeaStatus.BENCH
    assert "Binance has not been run yet" in settled.reason


def test_a_hyperliquid_only_idea_whose_stricter_second_look_rejects_is_rejected(
        session, monkeypatch) -> None:
    """The case of trend-research-flat-choice-hl-ov-gate-enter (2026-11-03): the first look on
    Hyperliquid said paper, the same venue under the live engine's liquidity floor rejected. It
    was read as «passed on Binance» and parked on the bench; both are Hyperliquid, both must
    pass. (Binance cannot judge it here; with a Binance stage owed it would wait for it.)"""
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("hyperliquid",))
    t = hl_family(session, "paper", "reject")
    settled = _settle(session, "h", t)
    assert settled.view.binance is None and settled.view.hyperliquid == "reject"
    assert status(session, "h") == IdeaStatus.REJECTED
    no_orphan(session)


def test_a_hyperliquid_only_idea_passes_when_both_looks_pass(session) -> None:
    t = hl_family(session, "paper", "paper")
    _settle(session, "h", t)
    assert status(session, "h") == IdeaStatus.VALIDATED


def test_a_hyperliquid_first_look_that_waits_keeps_the_idea_open_whatever_the_second_says(
        session) -> None:
    t = hl_family(session, "needs-forward", "reject")
    _settle(session, "h", t)
    assert status(session, "h") == IdeaStatus.BENCH


def test_a_second_stage_run_settles_its_parent(session, monkeypatch) -> None:
    """The same function from the other side: the id of the second stage names the family."""
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("hyperliquid",))
    t = hl_family(session, "paper", "reject")
    assert venue_status.family_root(session, "h-hl") == "h"
    _settle(session, "h-hl", t)
    assert status(session, "h") == IdeaStatus.REJECTED


# ---- Binance only ---------------------------------------------------------------------------


def test_a_binance_only_pass_is_needs_infrastructure(session, monkeypatch) -> None:
    monkeypatch.setattr(stage2, "strategy_sources", lambda code_ref: ("binance",))
    t = binance_family(session, "paper", None)
    settled = _settle(session, "i", t)
    assert settled.view.combined.route == "needs-infrastructure"
    assert status(session, "i") == IdeaStatus.BENCH


# ---- no orphan whatever the order of the runs ------------------------------------------------


@pytest.mark.parametrize("b", ["reject", "paper", "needs-infrastructure", "needs-forward",
                               "error", "not-evaluable"])
@pytest.mark.parametrize("hl", [None, "reject", "paper", "needs-forward", "error",
                                "not-evaluable"])
def test_no_pairing_leaves_a_second_stage_open_under_a_rejected_parent(session, b, hl) -> None:
    t = binance_family(session, b, hl)
    if hl is not None:
        session.get(Idea, "i-hl").status = IdeaStatus.BENCH  # its own run left it open
        session.flush()
    _settle(session, "i", t)
    no_orphan(session)
    if status(session, "i") == IdeaStatus.REJECTED:  # closed only when no venue passed or waits
        assert "reject" in (stage2.venue_state(b), stage2.venue_state(hl))
        assert "pass" not in (stage2.venue_state(b), stage2.venue_state(hl))
        assert "pending" not in (stage2.venue_state(b), stage2.venue_state(hl))


@pytest.mark.parametrize("current", [IdeaStatus.BENCH, IdeaStatus.VALIDATED,
                                     IdeaStatus.REJECTED])
def test_settling_twice_changes_nothing_the_second_time(session, current) -> None:
    t = binance_family(session, "reject", "needs-forward", current)
    _settle(session, "i", t)
    first = status(session, "i")
    session.query(Idea)  # no new run
    _settle(session, "i", t)
    assert status(session, "i") == first
