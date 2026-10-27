"""The link between the two stages of an idea's evaluation (docs/TWO_STAGE.md).

An idea the generator wrote is judged twice: on Binance's long history (stage 1)
and, if it survived, as the same mechanism on Hyperliquid (stage 2). Stage 2 is
judged under an idea of its own -- the CHILD, id `<parent>-hl`, `parent_id`
pointing at the idea it checks, notes starting with `NOTES_PREFIX` -- so its
trials, verdicts and data ranges are the child's, exactly like any other run.
The idea people see and the cap on strategies counts is the PARENT (a child is a
variant: it takes no place of its own, `qlab.registry.strategies`); the parent's
status is the verdict of both stages together (`qlab.pipeline.stage2.combine`).

This module only reads that link; nothing here writes a status or a verdict.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from qlab.registry.models import Idea, Spec, Trial, Verdict

NOTES_PREFIX = "Second stage of "
"""Starts the `notes` of a child idea. `qlab.funnel.idea_source` reads it: a child
is not a new idea the generator made, it is the second look at one."""

CHILD_SUFFIX = "-hl"


def child_id(parent_id: str) -> str:
    return f"{parent_id}{CHILD_SUFFIX}"


def is_stage2_child(idea: Idea) -> bool:
    return bool(idea.parent_id) and (idea.notes or "").startswith(NOTES_PREFIX)


def stage2_child(session: Session, parent_id: str) -> Idea | None:
    """The idea that holds `parent_id`'s second stage, if it has run."""
    idea = session.get(Idea, child_id(parent_id))
    if idea is not None and idea.parent_id == parent_id and is_stage2_child(idea):
        return idea
    return None


def child_id_taken(session: Session, parent_id: str) -> bool:
    """Whether the id a second stage of `parent_id` would take (`<parent>-hl`) already belongs to
    an idea that is not that second stage -- a researcher's own `bv2-research-shared-pool-hl`
    next to `bv2-research-shared-pool`. Writing the second stage there would run a trial under,
    and move the status of, an idea that is another experiment (found 2026-10-27)."""
    idea = session.get(Idea, child_id(parent_id))
    return idea is not None and not (idea.parent_id == parent_id and is_stage2_child(idea))


def latest_routed(session: Session, idea_id: str) -> tuple[Trial, Spec] | None:
    """The idea's latest trial that carries a route, with the spec it ran."""
    row = (session.query(Trial, Spec).join(Spec, Trial.spec_id == Spec.id)
           .filter(Spec.idea_id == idea_id, Trial.route.isnot(None))
           .order_by(Trial.id.desc()).first())
    return None if row is None else (row[0], row[1])


def data_range(session: Session, trial_id: int) -> tuple[date, date] | None:
    """The data a trial's verdicts were rendered on: from the earliest start to the
    latest end among its dated verdicts (a split run has a selection and a forward
    range). None when it wrote no dated verdict."""
    rows = (session.query(Verdict.data_range_start, Verdict.data_range_end)
            .filter(Verdict.trial_id == trial_id, Verdict.data_range_end.isnot(None)).all())
    if not rows:
        return None
    starts = [r[0] for r in rows if r[0] is not None]
    return (min(starts) if starts else min(r[1] for r in rows), max(r[1] for r in rows))


__all__ = ["CHILD_SUFFIX", "NOTES_PREFIX", "child_id", "child_id_taken", "data_range",
           "is_stage2_child", "latest_routed", "stage2_child"]
