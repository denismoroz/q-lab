"""Recompute every idea's status from its two venues' verdicts (docs/VENUE_SPLIT.md, section on
the single status function).

The owner's rule (2026-10-30): an idea passes if it passes on any venue, is rejected only when no
venue passed and none is pending, and is open while one venue has rejected and the other has not
spoken. Before 2026-11-04 two code paths applied it by their own guess of a run's venue, so some
ideas were closed by one venue's reject (and sometimes reopened later), and one sat on the bench
whose status matched neither venue. `qlab.pipeline.venue_status` is now the only place that decides;
this script runs it over the registry once:

* every open idea (candidate .. bench, validated),
* every idea rejected since `--since` (default 2026-10-30, the day the venues became independent),
* every open second stage under a closed first stage (an orphan).

A closed idea changes only to a pass or back to open (the closure was the defect); an open one to
whatever the two venues make together. Each move is a `stage_transition` row that says why. Trials
and verdicts are never touched.

    uv run python scripts/repair_venue_statuses.py            # dry run: lists, writes nothing
    uv run python scripts/repair_venue_statuses.py --apply    # backup first, then writes

    --db PATH         the registry (default: QLAB_DB or data/qlab.db)
    --failures PATH   the second stages' crash ledger (default data/stage2/failures.json)
    --since YYYY-MM-DD

A dry run opens the registry read-only through a copy, so a night running at the same time is not
disturbed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

DEFAULT_SINCE = date(2026, 10, 30)
WHY = ("venue status repair 2026-11-04 (docs/VENUE_SPLIT.md): the status is what both venues' "
       "latest verdicts make together, by the owner's rule of 2026-10-30; {detail}")


@dataclass
class Change:
    idea_id: str
    before: str
    after: str
    kind: str
    detail: str
    binance: str | None = None
    hyperliquid: str | None = None


def _closed_since(session, idea_id: str, since: date) -> bool:
    from qlab.registry.models import IdeaStatus, StageTransition

    row = (session.query(StageTransition)
           .filter(StageTransition.idea_id == idea_id,
                   StageTransition.to_status == IdeaStatus.REJECTED)
           .order_by(StageTransition.id.desc()).first())
    return row is not None and row.at is not None and row.at.date() >= since


def _reopen_child(session, child_id: str, parent_id: str) -> list[Change]:
    """A second stage that was closed only because its first stage was (`close_second_stage`)
    follows the parent back: it takes the status its own latest route gives."""
    from qlab.registry.lifecycle import ROUTE_TARGET
    from qlab.registry.models import Idea, IdeaStatus, StageTransition, TrialRoute
    from qlab.registry.stages import latest_routed

    child = session.get(Idea, child_id)
    if child is None or child.status != IdeaStatus.REJECTED:
        return []
    closing = (session.query(StageTransition).filter(StageTransition.idea_id == child_id)
               .order_by(StageTransition.id.desc()).first())
    if closing is None or not (closing.reason or "").startswith("second stage closes with"):
        return []  # closed by its own verdict
    routed = latest_routed(session, child_id)
    target = None if routed is None else ROUTE_TARGET.get(TrialRoute(routed[0].route))
    if target is None or target == IdeaStatus.REJECTED:
        return []
    return [Change(child_id, child.status.value, target.value, "child",
                   f"it was closed only with its first stage {parent_id}, which is open again")]


def plan(session, since: date) -> list[Change]:
    """The moves the single status function asks for. Reads only."""
    from qlab.pipeline import venue_status
    from qlab.registry.lifecycle import CLOSED_STATUSES, PRODUCTION_STATUSES
    from qlab.registry.models import Idea, IdeaStatus, TrialSource
    from qlab.registry.stages import is_stage2_child, latest_routed

    found: list[Change] = []
    judged: dict[str, IdeaStatus] = {}
    for idea in session.query(Idea).order_by(Idea.id):
        if idea.id.startswith("noise-") or is_stage2_child(idea):
            continue
        if idea.status in PRODUCTION_STATUSES:
            continue
        if idea.status in CLOSED_STATUSES and not (
                idea.status == IdeaStatus.REJECTED and _closed_since(session, idea.id, since)):
            continue
        routed = latest_routed(session, idea.id)
        if routed is None or routed[0].source != TrialSource.QLAB:
            continue  # no q-lab run: an imported dossier or a card, nothing to recompute from
        view = venue_status.venue_view(session, idea.id)
        target = venue_status.target_status(view)
        if target is None:
            continue  # the venues together decide nothing: the status is left
        judged[idea.id] = target
        closed = idea.status in CLOSED_STATUSES
        if target == idea.status:
            continue
        if closed and target == IdeaStatus.REJECTED:
            continue
        kind = ("reopen" if closed and target != IdeaStatus.VALIDATED else
                "revive" if closed else
                "close" if target == IdeaStatus.REJECTED else "move")
        found.append(Change(idea.id, idea.status.value, target.value, kind, view.combined.reason,
                            view.binance, view.hyperliquid))
        if closed and view.child_id is not None:
            found.extend(_reopen_child(session, view.child_id, idea.id))
    # A second stage open under a closed first stage, counting the moves above.
    for child in session.query(Idea).filter(Idea.parent_id.isnot(None)).order_by(Idea.id):
        if not is_stage2_child(child) or child.status in CLOSED_STATUSES | PRODUCTION_STATUSES:
            continue
        parent = session.get(Idea, child.parent_id)
        parent_status = judged.get(parent.id, parent.status)
        if parent_status in (IdeaStatus.REJECTED, IdeaStatus.RETIRED):
            found.append(Change(child.id, child.status.value, parent_status.value, "orphan",
                                f"its first stage {parent.id} is {parent_status.value}"))
    return found


def apply(session, changes: list[Change]) -> None:
    from qlab.registry import repo
    from qlab.registry.lifecycle import close_second_stage
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.rules.loader import load_latest

    version = load_latest().version
    for change in changes:
        if session.get(Idea, change.idea_id).status.value == change.after:
            continue  # already moved by closing its first stage
        repo.set_status(session, idea_id=change.idea_id, new_status=IdeaStatus(change.after),
                        reason=WHY.format(detail=change.detail), rules_version=version)
        if change.after == IdeaStatus.REJECTED.value and change.kind != "orphan":
            close_second_stage(session, parent_id=change.idea_id, closed_as=IdeaStatus.REJECTED,
                               reason=f"the first stage {change.idea_id} is rejected",
                               rules_version=version)


def backup(db: Path) -> Path:
    """A consistent copy of the registry next to it (sqlite's own backup, safe while it is used)."""
    target_dir = db.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"qlab-before-venue-repair-{datetime.now(UTC):%Y%m%d-%H%M%S}.db"
    source = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    return target


def _read_only_copy(db: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="qlab-venue-repair-")) / "qlab.db"
    source = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(tmp)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    return tmp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write (a backup is made first)")
    parser.add_argument("--db", default=os.environ.get("QLAB_DB", "data/qlab.db"))
    parser.add_argument("--failures", default=None)
    parser.add_argument("--since", type=date.fromisoformat, default=DEFAULT_SINCE)
    args = parser.parse_args(argv)

    db = Path(args.db).resolve()
    if not db.exists():
        print(f"no registry at {db}", file=sys.stderr)
        return 2
    working = db
    if args.apply:
        print(f"backup: {backup(db)}")
    else:
        working = _read_only_copy(db)  # the dry run never opens the real file
    os.environ["QLAB_DB"] = str(working)

    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope

    if args.failures:
        stage2.FAILURES = Path(args.failures)
    elif not stage2.FAILURES.exists() and (db.parent / "stage2" / "failures.json").exists():
        stage2.FAILURES = db.parent / "stage2" / "failures.json"

    with session_scope() as session:
        changes = plan(session, args.since)
        for c in changes:
            print(f"{c.kind:7} {c.idea_id}: {c.before} -> {c.after}"
                  f"   [Binance {c.binance or '-'}; Hyperliquid {c.hyperliquid or '-'}]")
            print(f"        {c.detail}")
        print(f"{len(changes)} change(s)" + ("" if args.apply else " (dry run: nothing written)"))
        if args.apply:
            apply(session, changes)
    if not args.apply:
        shutil.rmtree(working.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
