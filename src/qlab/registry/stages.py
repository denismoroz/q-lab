"""The link between the two stages of an idea's evaluation (docs/TWO_STAGE.md).

An idea the generator wrote is judged twice: on Binance's long history (stage 1)
and, if it survived, as the same mechanism on Hyperliquid (stage 2). Stage 2 is
judged under an idea of its own -- the CHILD, id `<parent>-hl`, `parent_id`
pointing at the idea it checks, notes starting with `NOTES_PREFIX` -- so its
trials, verdicts and data ranges are the child's, exactly like any other run.
The idea people see and the cap on strategies counts is the PARENT (a child is a
variant: it takes no place of its own, `qlab.registry.strategies`); the parent's
status is the verdict of both stages together (`qlab.pipeline.stage2.combine`).

The second stage goes both ways (owner, 2026-11-05, on an idea born on Hyperliquid: «почему вот
эта стратегия не тестировалась на бинансе?»; «да, делай и прогоняй все»). An idea judged on
Binance gets a Hyperliquid child `<parent>-hl`; an idea born on Hyperliquid (its own spec's data
is Hyperliquid's) gets a Binance child `<parent>-bn` -- next to the `-hl` child it may already
have, which is the same venue under the live engine's liquidity floor. Both children carry the same
`NOTES_PREFIX`, so every reader that folds a second stage under its parent folds both.

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
BINANCE_CHILD_SUFFIX = "-bn"
"""The suffix of the Binance second stage of an idea born on Hyperliquid. No idea of the registry
ended in it when it was chosen (2026-11-05); the researcher's own Binance versions are
`<idea>-binance` and the causal twins `<idea>-causal`, ideas of their own."""
HYPERLIQUID = "hyperliquid"
BINANCE = "binance"
SUFFIX_OF_VENUE = {HYPERLIQUID: CHILD_SUFFIX, BINANCE: BINANCE_CHILD_SUFFIX}


def child_id(parent_id: str, venue: str = HYPERLIQUID) -> str:
    """The id of `parent_id`'s second stage on `venue` (`<parent>-hl`, `<parent>-bn`)."""
    return f"{parent_id}{SUFFIX_OF_VENUE[venue]}"


def is_stage2_child(idea: Idea) -> bool:
    return bool(idea.parent_id) and (idea.notes or "").startswith(NOTES_PREFIX)


def child_venue(idea: Idea) -> str | None:
    """The venue of a second stage (`binance` for `<parent>-bn`, else `hyperliquid`); None for an
    idea that is not a second stage."""
    if not is_stage2_child(idea):
        return None
    return BINANCE if idea.id == child_id(str(idea.parent_id), BINANCE) else HYPERLIQUID


def stage2_child(session: Session, parent_id: str, venue: str = HYPERLIQUID) -> Idea | None:
    """The idea that holds `parent_id`'s second stage on `venue`, if it has run."""
    idea = session.get(Idea, child_id(parent_id, venue))
    if idea is not None and idea.parent_id == parent_id and is_stage2_child(idea):
        return idea
    return None


def binance_child(session: Session, parent_id: str) -> Idea | None:
    """The Binance second stage (`<parent>-bn`) of an idea born on Hyperliquid, if it has run."""
    return stage2_child(session, parent_id, BINANCE)


def stage2_children(session: Session, parent_id: str) -> list[Idea]:
    """Every second stage of `parent_id`: the Hyperliquid one first, then the Binance one."""
    return [c for v in (HYPERLIQUID, BINANCE) if (c := stage2_child(session, parent_id, v))]


def other_venue_child(session: Session, parent_id: str) -> Idea | None:
    """The second stage that is the idea's OTHER venue: the Binance child of an idea born on
    Hyperliquid when it has one, else the Hyperliquid child (a Binance idea's other venue, or a
    Hyperliquid idea's stricter look on the same venue)."""
    return binance_child(session, parent_id) or stage2_child(session, parent_id)


def child_id_taken(session: Session, parent_id: str, venue: str = HYPERLIQUID) -> bool:
    """Whether the id a second stage of `parent_id` would take (`<parent>-hl`, `<parent>-bn`)
    already belongs to an idea that is not that second stage -- a researcher's own
    `bv2-research-shared-pool-hl` next to `bv2-research-shared-pool`. Writing the second stage
    there would run a trial under, and move the status of, an idea that is another experiment
    (found 2026-10-27)."""
    idea = session.get(Idea, child_id(parent_id, venue))
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


__all__ = ["BINANCE", "BINANCE_CHILD_SUFFIX", "CHILD_SUFFIX", "HYPERLIQUID", "NOTES_PREFIX",
           "SUFFIX_OF_VENUE", "binance_child", "child_id", "child_id_taken", "child_venue",
           "data_range", "is_stage2_child", "latest_routed", "other_venue_child", "stage2_child",
           "stage2_children"]
