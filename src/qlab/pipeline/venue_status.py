"""The one place where an idea's status is decided from the verdicts of both venues.

The owner's rule (2026-10-30, docs/VENUE_SPLIT.md, rules 2026-10-30.2): Binance and Hyperliquid are
judged independently; an idea passes if it passes anywhere; it is rejected only when no venue
passed and none is still pending; one venue rejecting while the other has not spoken leaves the
idea open.

Before this module two code paths applied the rule, each by its own guess of which venue a run
belonged to: `evaluate._record` (a first-stage run, and only when
`spec.execution_venue or spec.data.source` was "binance") and `stage2._settle` (the second stage).
A spec with Binance data and `execution_venue: hyperliquid` took neither the independent path, so a
Binance reject closed its idea at once and a later Hyperliquid pass reopened it; a spec on
Hyperliquid data whose second stage is the same venue with the live engine's liquidity floor was
read as "Binance passed" and parked on the bench for ever (found 2026-11-03).

Now both call `settle`. The venue of a run is the venue of its DATA (`data_requirements.source` of
the spec the trial ran), whatever `execution_venue` says; the verdicts come from the registry (the
latest routed run of the idea and of its second stage), not from the run in hand, so every path
reaches the same status from the same facts.

* A root idea on Binance data: Binance = its latest run; Hyperliquid = its second stage's latest
  run. No second-stage run yet: Hyperliquid is OWED (pending) while the second stage can still be
  made, and none (cannot judge) once it cannot (switched off, the strategy has no Hyperliquid
  panel, its crashes were given up on, the id is taken).
* A root idea on Hyperliquid data: the Hyperliquid verdict is its run together with its same-venue
  second stage (the live engine's liquidity floor): both must pass, a reject of the stricter look
  rejects once the first look has decided, a first look that still waits keeps the venue pending.
  Binance = its Binance second stage's latest run (`<idea>-bn`, owner 2026-11-05: «почему вот эта
  стратегия не тестировалась на бинансе?»; «да, делай и прогоняй все»). No Binance run yet:
  Binance is OWED (pending) while that stage can be made, exactly as Hyperliquid is for a Binance
  idea, and none once it cannot (switched off, the strategy has no Binance panel, the id is taken,
  its crashes were given up on, or the idea is a book on paper or live, whose check is real
  trading).
* The crash rules of 2026-10-31 hold: a crash decides nothing, never opens a closed idea (only a
  pass changes a closed idea), and after `MAX_FAILED_ATTEMPTS` crashes the venue is «не проводился:
  сбой», read from the same ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from qlab.pipeline import stage2
from qlab.registry.lifecycle import (
    CLOSED_STATUSES,
    ROUTE_TARGET,
    StatusDecision,
    apply_route,
    close_second_stage,
)
from qlab.registry.models import Idea, IdeaStatus, TrialRoute
from qlab.registry.stages import (
    binance_child,
    child_id_taken,
    is_stage2_child,
    latest_routed,
    stage2_child,
)

BINANCE = "binance"
HYPERLIQUID = "hyperliquid"

OWED_WHY = ("Hyperliquid has not been run yet and its second stage can still be made: the idea "
            "waits for the second venue")
OWED_BINANCE_WHY = ("Binance has not been run yet and the idea's Binance stage can still be made "
                    "(it was born on Hyperliquid): the idea waits for the second venue")


@dataclass(frozen=True)
class VenueView:
    """What the registry says about the two venues of one idea."""

    root_id: str
    child_id: str | None
    root_venue: str | None
    """The venue of the root idea's latest run (its data source); None when it has no run."""
    binance: str | None
    """The route standing for Binance (None: no run, or no Binance version of this idea)."""
    hyperliquid: str | None
    """The route standing for Hyperliquid (None: no run and none owed, or given up on)."""
    hl_owed: bool
    """Hyperliquid has no run yet but its second stage can still be made."""
    combined: stage2.Combined
    pass_ideas: tuple[str, ...]
    """The ideas whose failing verdicts date the data a revival must beat: those of the venue
    that gives the pass."""
    pass_other_venue: bool
    bn_child_id: str | None = None
    """The Binance second stage (`<idea>-bn`) of an idea born on Hyperliquid, when it exists."""
    bn_owed: bool = False
    """Binance has no run yet but the idea's Binance stage can still be made."""


@dataclass(frozen=True)
class Settled:
    view: VenueView
    decision: StatusDecision
    stayed_closed: bool = False
    note: str = ""

    @property
    def reason(self) -> str:
        if self.stayed_closed:
            return (f"{self.view.combined.reason}; the idea stays closed: only a pass revives a "
                    "closed idea, a crash or a verdict that cannot decide yet is not one")
        return self.view.combined.reason


def family_root(session: Session, idea_id: str) -> str:
    """The idea whose status is the two venues' verdict: a second stage's parent, else itself."""
    idea = session.get(Idea, idea_id)
    if idea is not None and is_stage2_child(idea) and idea.parent_id:
        return idea.parent_id
    return idea_id


def _route(trial) -> str | None:
    return None if trial is None else str(getattr(trial.route, "value", trial.route))


def _source(spec_row) -> str | None:
    return None if spec_row is None else (spec_row.data_requirements or {}).get("source")


def _given_up(root_id: str, venue: str = HYPERLIQUID) -> bool:
    key = stage2.failure_key(root_id, venue)
    return bool((stage2.load_failures().get(key) or {}).get("given_up"))


def _given_up_why(root_id: str, venue: str) -> str:
    failure = stage2.load_failures().get(stage2.failure_key(root_id, venue)) or {}
    how = ("постоянная причина" if failure.get("permanent")
           else f"{failure.get('attempts', '?')} попытки подряд")
    word = stage2.VENUE_TITLE.get(venue, venue)
    return f"{word} не проводился: сбой ({how}): {failure.get('error', '')}"


def second_stage_possible(session: Session, root_id: str, code_ref: str) -> bool:
    """Whether Hyperliquid's side of a Binance-data idea can still be made: the stage is on, the
    strategy has a Hyperliquid panel, its id is free, and its crashes were not given up on."""
    try:
        config = stage2.load_config()
    except (OSError, ValueError):
        return False
    if not config.enabled or _given_up(root_id) or child_id_taken(session, root_id):
        return False
    sources = stage2.strategy_sources(code_ref)
    return sources is None or config.venue in sources


def binance_stage_possible(session: Session, root_id: str, code_ref: str) -> bool:
    """Whether Binance's side of an idea born on Hyperliquid can still be made: the Binance stage
    is on (`night/stage2_binance.yaml`), the strategy has a Binance panel, `<idea>-bn` is free, its
    crashes were not given up on, and the idea is not a book on paper or live (their numbers are
    fixed on production and their check is real trading, docs/ONE_CHECK.md)."""
    from qlab.registry.lifecycle import PRODUCTION_STATUSES

    try:
        config = stage2.config_for(BINANCE)
    except (OSError, ValueError):
        return False
    idea = session.get(Idea, root_id)
    if (not config.enabled or idea is None or idea.status in PRODUCTION_STATUSES
            or _given_up(root_id, BINANCE) or child_id_taken(session, root_id, BINANCE)):
        return False
    sources = stage2.strategy_sources(code_ref)
    return sources is None or BINANCE in sources


def _same_venue(first: str | None, second: str | None) -> str | None:
    """Hyperliquid's route for an idea whose own run is on Hyperliquid and whose second stage is
    the same venue under the live engine's floor: both must pass."""
    if second is None:
        return first
    if first is None:
        return second
    state = stage2.venue_state(first)
    if state == "pending":
        return first  # the first look has not spoken
    if state == "reject":
        return first
    if state == "pass":
        sec = stage2.venue_state(second)
        if sec == "pass":
            return second
        if sec == "reject":
            return "reject"
        return "needs-forward"  # the stricter look decided nothing yet: it never passes the idea
    return first


def venue_view(session: Session, idea_id: str) -> VenueView:
    """The two venues' standing routes for the idea (or its parent, when `idea_id` is a second
    stage) and what they make together (`stage2.combine_venues`)."""
    root_id = family_root(session, idea_id)
    child = stage2_child(session, root_id)
    root_run = latest_routed(session, root_id)
    child_run = None if child is None else latest_routed(session, child.id)
    root_route = None if root_run is None else _route(root_run[0])
    child_route = None if child_run is None else _route(child_run[0])
    root_venue = None if root_run is None else _source(root_run[1])
    given_up = _given_up(root_id)
    if child_route == "error" and given_up:
        child_route = None  # «не проводился: сбой»: the venue did not judge
    why = ""
    if given_up and child_route is None:
        why = _given_up_why(root_id, HYPERLIQUID)
    owed = False
    bn_owed = False
    bn = None if root_venue == BINANCE else binance_child(session, root_id)
    binance: str | None = None
    hyperliquid: str | None
    pass_ideas: tuple[str, ...] = (root_id,)
    other_venue = False
    if root_venue == BINANCE:
        binance = root_route
        hyperliquid = child_route
        if child_run is None and root_run is not None and root_route is not None:
            owed = second_stage_possible(session, root_id, root_run[1].code_ref)
            if owed:
                hyperliquid, why = "needs-forward", OWED_WHY
        hl_ideas = () if child is None else (child.id,)
    else:
        hyperliquid = _same_venue(root_route, child_route)
        hl_ideas = (root_id, *(() if child is None else (child.id,)))
        # Born on Hyperliquid: Binance is its `<idea>-bn` stage (owner, 2026-11-05).
        bn_run = None if bn is None else latest_routed(session, bn.id)
        binance = None if bn_run is None else _route(bn_run[0])
        bn_given_up = _given_up(root_id, BINANCE)
        if binance == "error" and bn_given_up:
            binance = None
        if bn_given_up and binance is None:
            why = "; ".join(w for w in (why, _given_up_why(root_id, BINANCE)) if w)
        if (bn_run is None and root_venue == HYPERLIQUID and root_run is not None
                and root_route is not None):
            bn_owed = binance_stage_possible(session, root_id, root_run[1].code_ref)
            if bn_owed:
                binance = "needs-forward"
                why = "; ".join(w for w in (why, OWED_BINANCE_WHY) if w)
    combined = stage2.combine_venues(binance, hyperliquid, why=why)
    if stage2.venue_state(hyperliquid) == "pass":
        pass_ideas, other_venue = hl_ideas, True
    elif bn is not None and stage2.venue_state(binance) == "pass":
        pass_ideas, other_venue = (bn.id,), True
    return VenueView(root_id, None if child is None else child.id, root_venue, binance,
                     hyperliquid, owed, combined, pass_ideas, other_venue,
                     None if bn is None else bn.id, bn_owed)


def settle(session: Session, idea_id: str, *, trial_id: int | None, rules_version: str | None,
           run_data_end: date | None, note: str = "") -> Settled:
    """Move the idea's status by the two venues' verdicts, whichever path made the run.

    A closed idea (rejected, retired, decayed) changes only on a pass; the verdict of a run that
    decided nothing (`needs-more-data`, `not-evaluable`, `error`) changes nothing. An idea left
    closed closes its second stage with it, so no open second stage sits under a closed parent."""
    view = venue_view(session, idea_id)
    root = session.get(Idea, view.root_id)
    if root is None:
        raise ValueError(f"idea {view.root_id!r} does not exist")
    combined = view.combined
    reason = f"{note}; {combined.reason}" if note else combined.reason
    if root.status in CLOSED_STATUSES and combined.route not in stage2.PASSED:
        settled = Settled(view, StatusDecision(None, "closed idea; only a pass revives it"),
                          stayed_closed=True, note=note)
    else:
        decision = apply_route(
            session, idea_id=view.root_id, route=TrialRoute(combined.route), route_reason=reason,
            trial_id=trial_id, rules_version=rules_version, run_data_end=run_data_end,
            venue_ideas=view.pass_ideas if view.pass_other_venue else None)
        settled = Settled(view, decision, note=note)
    session.refresh(root)
    if root.status in (IdeaStatus.REJECTED, IdeaStatus.RETIRED):
        for venue in (HYPERLIQUID, BINANCE):
            close_second_stage(session, parent_id=view.root_id, closed_as=root.status,
                               reason=f"the first stage {view.root_id} is {root.status.value}",
                               rules_version=rules_version, venue=venue)
    return settled


def target_status(view: VenueView) -> IdeaStatus | None:
    """The status the venues together put an idea in (None: they decide nothing)."""
    return ROUTE_TARGET.get(TrialRoute(view.combined.route))


__all__ = ["OWED_BINANCE_WHY", "OWED_WHY", "Settled", "VenueView", "binance_stage_possible",
           "family_root", "second_stage_possible", "settle", "target_status", "venue_view"]
