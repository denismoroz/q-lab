"""How one run's route moves its idea along the lifecycle (docs/TASKS.md T31).

Before this module existed, `qlab.pipeline.evaluate` computed a route for
every run and nothing ever acted on it: `stage_transition` stayed empty, and
the funnel's status counts were the imported history of the previous
framework, not q-lab's own work. Seven ideas sat at `candidate` after their
latest run had rejected them.

The policy is one pure function, `next_status`, plus `apply_route`, which
reads the two facts the policy needs from the registry and writes the
transition. The policy answers three questions the backlog entry raised
before any code was written:

1. **A `paper` route does not put an idea into `paper`.** Research -> paper
   is the owner's gate (decision of 2026-09-19: "человек — гейт на
   переходе research → paper → live"), for the same reason
   `decide_route` never returns `live`. A run that passes every rule and is
   affordable moves the idea to `validated`: checked, waiting for a human.
   A run that passes but is not affordable now moves it to `bench`.

2. **The latest run decides.** "The best run so far" would be selection on
   the outcome -- the very thing deflation exists to correct. Every run of
   an idea is a recorded trial and counts in deflation whether or not it
   moved the status.

3. **A run that decided nothing moves nothing.** `needs-more-data`,
   `not-evaluable` and `error` leave the status where it is. Above all they
   never move an idea to `rejected`: a rejection claims "tested, does not
   work", and these runs did not test it (T24, T26).

Two guards sit on top of the mapping:

- **Production states belong to production.** An idea in `paper`, `live`,
  `decayed` or `retired` got there through a human decision or a real
  shutdown; a backtest never overrides that. The trial and its verdicts are
  still recorded -- only the status is left alone.

- **Revival needs new data** (CLAUDE.md: "Воскрешение с кладбища — только
  на данных, которых не было в момент отказа"). A `rejected` idea moves to
  `validated`/`bench` only if the new run's data ends after the latest data
  any failing verdict of this idea was rendered on. If no failing verdict
  of this idea carries a date (imported dossiers often do not), newness
  cannot be shown and the idea stays rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.registry import repo
from qlab.registry.models import Idea, IdeaStatus, TrialRoute, Verdict

ROUTE_TARGET: dict[TrialRoute, IdeaStatus] = {
    TrialRoute.REJECT: IdeaStatus.REJECTED,
    TrialRoute.SHELF: IdeaStatus.BENCH,
    # Passed every strategy rule, waiting for infrastructure -- idle on the
    # bench like a shelf idea waiting for capital, never the graveyard (T35).
    TrialRoute.NEEDS_INFRASTRUCTURE: IdeaStatus.BENCH,
    TrialRoute.PAPER: IdeaStatus.VALIDATED,
}
"""Routes that decide something, and the status each one moves an idea to.
A route absent from this mapping decided nothing (see module docstring)."""

PRODUCTION_STATUSES: frozenset[IdeaStatus] = frozenset(
    {IdeaStatus.PAPER, IdeaStatus.LIVE, IdeaStatus.DECAYED, IdeaStatus.RETIRED}
)
"""Statuses set by people or by a real shutdown, never by a backtest."""


@dataclass(frozen=True, slots=True)
class StatusDecision:
    """What `next_status` decided. `new_status is None` means "leave it",
    and `reason` says why -- every outcome carries a reason, including the
    ones that change nothing, so a caller can always print it."""

    new_status: IdeaStatus | None
    reason: str


def next_status(
    current: IdeaStatus,
    route: TrialRoute,
    *,
    run_data_end: date | None,
    rejected_on_data_end: date | None,
) -> StatusDecision:
    """The status an idea in `current` moves to after a run routed `route`.

    `run_data_end` is the last date of the data this run's verdict was
    rendered on. `rejected_on_data_end` is the latest data end of any
    failing verdict this idea already has (None when none is dated); it
    only matters for an idea that is currently `rejected`.
    """
    target = ROUTE_TARGET.get(route)
    if target is None:
        return StatusDecision(None, f"route {route.value} decides nothing; status unchanged")

    if current in PRODUCTION_STATUSES:
        return StatusDecision(
            None,
            f"idea is in {current.value}, which only a human or a real shutdown changes; "
            f"route {route.value} recorded, status unchanged",
        )

    if current == target:
        return StatusDecision(None, f"already {current.value}")

    if current == IdeaStatus.REJECTED:
        if rejected_on_data_end is None:
            return StatusDecision(
                None,
                "rejected on undated data: cannot show this run used data the rejection "
                "did not have; stays rejected",
            )
        if run_data_end is None or run_data_end <= rejected_on_data_end:
            return StatusDecision(
                None,
                f"rejected on data up to {rejected_on_data_end}; this run's data ends "
                f"{run_data_end} and holds nothing new; stays rejected",
            )

    return StatusDecision(target, f"route {route.value}")


def apply_route(
    session: Session,
    *,
    idea_id: str,
    route: TrialRoute,
    route_reason: str,
    trial_id: int,
    rules_version: str | None,
    run_data_end: date | None,
) -> StatusDecision:
    """Apply `next_status` to `idea_id` and write the transition if it moves.

    Must run in the same session as the trial it describes, so the trial,
    its verdicts and the status change commit or roll back together.

    The failing-verdict date is read EXCLUDING `trial_id`'s own verdicts:
    the question is what data the idea was rejected on before this run.
    """
    idea = session.get(Idea, idea_id)
    if idea is None:
        raise ValueError(f"idea {idea_id!r} does not exist")

    rejected_on_data_end = (
        session.query(func.max(Verdict.data_range_end))
        .filter(
            Verdict.idea_id == idea_id,
            Verdict.passed.is_(False),
            Verdict.trial_id.is_distinct_from(trial_id),
        )
        .scalar()
    )

    decision = next_status(
        idea.status,
        route,
        run_data_end=run_data_end,
        rejected_on_data_end=rejected_on_data_end,
    )
    if decision.new_status is not None:
        repo.set_status(
            session,
            idea_id=idea_id,
            new_status=decision.new_status,
            reason=f"trial {trial_id}: {route.value} -- {route_reason}",
            rules_version=rules_version,
            trial_id=trial_id,
        )
    return decision


__all__ = [
    "PRODUCTION_STATUSES",
    "ROUTE_TARGET",
    "StatusDecision",
    "apply_route",
    "next_status",
]
