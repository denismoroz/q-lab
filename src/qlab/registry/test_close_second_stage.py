"""A second stage closes with its first stage (owner, 2026-10-29, docs/EVALUATION_PATHS.md Р10)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.registry import repo
from qlab.registry.lifecycle import apply_route, close_second_stage
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


def _idea(session, idea_id: str, status: IdeaStatus, parent: str | None = None,
          stage2: bool = False) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent}" if stage2 else None))
    session.flush()


def _trial(session, idea_id: str) -> int:
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": "binance"}, rebalance="1d",
                         costs_model={}, code_ref="x:Y")
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="x", params={},
                          status=TrialStatus.OK, route=TrialRoute.REJECT, route_reason="r",
                          source=TrialSource.IMPORTED,
                          started_at=datetime(2026, 10, 28, tzinfo=UTC)).id


@pytest.mark.parametrize("child_status", [IdeaStatus.VALIDATED, IdeaStatus.BENCH,
                                          IdeaStatus.CANDIDATE])
def test_a_rejected_first_stage_rejects_its_second_stage(session, child_status) -> None:
    _idea(session, "a", IdeaStatus.VALIDATED)
    _idea(session, "a-hl", child_status, parent="a", stage2=True)
    trial = _trial(session, "a")

    apply_route(session, idea_id="a", route=TrialRoute.REJECT, route_reason="failed",
                trial_id=trial, rules_version="v1", run_data_end=None)

    assert session.get(Idea, "a").status == IdeaStatus.REJECTED
    assert session.get(Idea, "a-hl").status == IdeaStatus.REJECTED
    move = session.query(StageTransition).filter_by(idea_id="a-hl").one()
    assert "closes with its first stage" in move.reason and "a was rejected" in move.reason


def test_a_child_in_a_human_decision_is_left_alone(session) -> None:
    _idea(session, "a", IdeaStatus.VALIDATED)
    _idea(session, "a-hl", IdeaStatus.PAPER, parent="a", stage2=True)

    assert close_second_stage(session, parent_id="a", closed_as=IdeaStatus.REJECTED,
                              reason="x") is None
    assert session.get(Idea, "a-hl").status == IdeaStatus.PAPER


def test_a_child_already_closed_is_not_written_twice(session) -> None:
    _idea(session, "a", IdeaStatus.RETIRED)
    _idea(session, "a-hl", IdeaStatus.REJECTED, parent="a", stage2=True)

    assert close_second_stage(session, parent_id="a", closed_as=IdeaStatus.RETIRED,
                              reason="x") is None
    assert session.query(StageTransition).count() == 0


def test_an_idea_with_no_second_stage_has_nothing_to_close(session) -> None:
    _idea(session, "solo", IdeaStatus.RETIRED)
    # an experiment whose id happens to be `<id>-hl` is NOT a second stage
    _idea(session, "solo-hl", IdeaStatus.BENCH, parent="solo", stage2=False)

    assert close_second_stage(session, parent_id="solo", closed_as=IdeaStatus.RETIRED,
                              reason="x") is None
    assert session.get(Idea, "solo-hl").status == IdeaStatus.BENCH


def test_only_a_rejection_or_a_retirement_closes_a_child(session) -> None:
    with pytest.raises(ValueError, match="rejected or retired"):
        close_second_stage(session, parent_id="a", closed_as=IdeaStatus.DECAYED, reason="x")
