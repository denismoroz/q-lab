"""One-off repair of the night of 2026-10-31 (owner: «да, чини»).

Three closed ideas of the queue of closed ideas (frab-binance-spot-allbinance, -300, -400) were
moved rejected -> bench because their Hyperliquid run crashed ("database is locked"), and a crash
is not a pass: only a pass revives a closed idea. They are put back to `rejected`, each move a
`stage_transition` row that says why, and their second stages (`<idea>-hl`) are closed the way the
closing code does (`qlab.registry.lifecycle.close_second_stage`), unless they are closed already.
Trials are never erased: the partial noise-book trials of the crashed runs stay.

    QLAB_DB=... uv run python scripts/repair_2026_10_31_queue.py [--apply]

Without --apply it only says what it would do.
"""

from __future__ import annotations

import sys

from qlab.registry import repo
from qlab.registry.db import session_scope
from qlab.registry.lifecycle import close_second_stage
from qlab.registry.models import Idea, IdeaStatus, StageTransition

IDEAS = ("frab-binance-spot-allbinance", "frab-binance-spot-allbinance-300",
         "frab-binance-spot-allbinance-400")
WHY = ("a crash reopened this closed idea by mistake (the second-venue queue's Hyperliquid run "
       "failed with 'database is locked' on the night of 2026-10-31 and the idea moved rejected -> "
       "bench); only a pass revives a closed idea; put back to rejected, owner 2026-10-31 "
       "«да, чини»")


def main(apply: bool) -> None:
    with session_scope() as session:
        for idea_id in IDEAS:
            idea = session.get(Idea, idea_id)
            last = (session.query(StageTransition).filter(StageTransition.idea_id == idea_id)
                    .order_by(StageTransition.id.desc()).first())
            print(f"{idea_id}: {idea.status.value}; last move {last.from_status.value} -> "
                  f"{last.to_status.value}: {(last.reason or '')[:80]}")
            if idea.status != IdeaStatus.BENCH or last.from_status != IdeaStatus.REJECTED:
                print("  not in the state the crash left: skipped")
                continue
            if not apply:
                print("  would: bench -> rejected, and close its second stage")
                continue
            repo.set_status(session, idea_id=idea_id, new_status=IdeaStatus.REJECTED, reason=WHY)
            closed = close_second_stage(
                session, parent_id=idea_id, closed_as=IdeaStatus.REJECTED,
                reason=f"the first stage {idea_id} is rejected again ({WHY})")
            print(f"  rejected again; second stage closed: {closed or 'already closed or none'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
