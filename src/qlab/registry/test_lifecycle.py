"""Tests for `qlab.registry.lifecycle.next_status` -- the policy by which a
run's route moves its idea (docs/TASKS.md T31). Pure function: no database."""

from __future__ import annotations

from datetime import date

import pytest

from qlab.registry.lifecycle import PRODUCTION_STATUSES, next_status
from qlab.registry.models import IdeaStatus, TrialRoute

DATA_END = date(2026, 9, 20)


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        (TrialRoute.REJECT, IdeaStatus.REJECTED),
        (TrialRoute.SHELF, IdeaStatus.BENCH),
        (TrialRoute.PAPER, IdeaStatus.VALIDATED),
    ],
)
def test_deciding_routes_move_a_candidate(route: TrialRoute, expected: IdeaStatus) -> None:
    decision = next_status(
        IdeaStatus.CANDIDATE, route, run_data_end=DATA_END, rejected_on_data_end=None
    )
    assert decision.new_status == expected


@pytest.mark.parametrize(
    "route", [TrialRoute.NEEDS_MORE_DATA, TrialRoute.NOT_EVALUABLE, TrialRoute.ERROR]
)
@pytest.mark.parametrize("current", [IdeaStatus.CANDIDATE, IdeaStatus.VALIDATED])
def test_routes_that_decide_nothing_move_nothing(
    route: TrialRoute, current: IdeaStatus
) -> None:
    decision = next_status(current, route, run_data_end=DATA_END, rejected_on_data_end=None)
    assert decision.new_status is None
    assert route.value in decision.reason


def test_paper_route_never_produces_paper_status() -> None:
    """Research -> paper is the owner's gate, for every starting status."""
    for current in IdeaStatus:
        decision = next_status(
            current, TrialRoute.PAPER, run_data_end=DATA_END, rejected_on_data_end=None
        )
        assert decision.new_status in (None, IdeaStatus.VALIDATED)


@pytest.mark.parametrize("current", sorted(PRODUCTION_STATUSES))
@pytest.mark.parametrize("route", [TrialRoute.REJECT, TrialRoute.SHELF, TrialRoute.PAPER])
def test_production_statuses_are_never_moved_by_a_backtest(
    current: IdeaStatus, route: TrialRoute
) -> None:
    decision = next_status(current, route, run_data_end=DATA_END, rejected_on_data_end=None)
    assert decision.new_status is None


def test_a_later_reject_demotes_a_validated_idea() -> None:
    decision = next_status(
        IdeaStatus.VALIDATED, TrialRoute.REJECT, run_data_end=DATA_END, rejected_on_data_end=None
    )
    assert decision.new_status == IdeaStatus.REJECTED


def test_same_status_is_not_a_transition() -> None:
    decision = next_status(
        IdeaStatus.REJECTED, TrialRoute.REJECT, run_data_end=DATA_END, rejected_on_data_end=DATA_END
    )
    assert decision.new_status is None


def test_revival_on_new_data_is_allowed() -> None:
    decision = next_status(
        IdeaStatus.REJECTED,
        TrialRoute.PAPER,
        run_data_end=date(2026, 12, 1),
        rejected_on_data_end=DATA_END,
    )
    assert decision.new_status == IdeaStatus.VALIDATED


@pytest.mark.parametrize("run_end", [DATA_END, date(2026, 1, 1), None])
def test_revival_on_old_data_is_refused(run_end: date | None) -> None:
    decision = next_status(
        IdeaStatus.REJECTED,
        TrialRoute.PAPER,
        run_data_end=run_end,
        rejected_on_data_end=DATA_END,
    )
    assert decision.new_status is None
    assert "nothing new" in decision.reason


def test_revival_from_an_undated_rejection_is_refused() -> None:
    """Imported dossiers often carry no data range; newness cannot be shown."""
    decision = next_status(
        IdeaStatus.REJECTED,
        TrialRoute.SHELF,
        run_data_end=date(2026, 12, 1),
        rejected_on_data_end=None,
    )
    assert decision.new_status is None
    assert "undated" in decision.reason
