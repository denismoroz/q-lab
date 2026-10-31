"""What a run will do with each step, decided before it starts (a pure function of the table).

The same function answers the console's preview («что запустится») and the runner's first pass, so
the preview cannot promise what the run then does differently. A failure that happens DURING the
run adds blocked steps the plan could not know; the runner applies the same rule to them
(`blocking_parent`).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from qlab.nightflow.table import Table

RUN, OFF, BLOCKED, OUTSIDE = "run", "off", "blocked", "outside"


@dataclass(frozen=True)
class Planned:
    step: str
    action: str  # run / off / blocked / outside (not part of this partial run)
    blocked_by: str | None = None
    reason: str = ""

    def to_json(self) -> dict:
        return {"step": self.step, "action": self.action, "blocked_by": self.blocked_by,
                "reason": self.reason}


def blocking_parent(table: Table, step_id: str, outcome: dict[str, str], blocked_by: dict[str, str],
                    done_today: Collection[str]) -> tuple[str, str] | None:
    """The `data` parent that stops `step_id`, with the reason, or None.

    `outcome` is what each step above it has come to: ok / failed / blocked / off / skipped / cut /
    outside / interrupted. A failed or blocked parent blocks (the root cause is named, not the
    nearest blocked step); a switched-off parent blocks unless it was done earlier the same night
    (its output of tonight exists). A parent that was skipped (not due tonight) or is outside a
    partial run does not block: nothing is wrong with it."""
    for parent in table.by_id[step_id].hard_parents:
        got = outcome.get(parent)
        if got in ("failed", "interrupted"):
            return parent, "failed"
        if got == "blocked":
            return blocked_by.get(parent, parent), "blocked"
        if got == "off" and parent not in done_today:
            return parent, "off"
    return None


def make_plan(table: Table, *, off: Collection[str] = (), only: Collection[str] | None = None,
              done_today: Collection[str] = ()) -> list[Planned]:
    unknown = (set(off) | set(only or ())) - set(table.ids)
    if unknown:
        raise ValueError(f"unknown steps: {', '.join(sorted(unknown))}")
    outcome: dict[str, str] = {}
    blocked_by: dict[str, str] = {}
    plan: list[Planned] = []
    for step in table.steps:
        if only is not None and step.id not in only:
            outcome[step.id] = OUTSIDE
            plan.append(Planned(step.id, OUTSIDE, reason="не входит в этот запуск"))
            continue
        if step.id in off:
            outcome[step.id] = OFF
            done = step.id in done_today
            plan.append(Planned(step.id, OFF, reason=(
                "выключен; сегодня уже выполнен, зависимые шаги идут" if done else "выключен")))
            continue
        found = blocking_parent(table, step.id, outcome, blocked_by, done_today)
        if found is not None:
            root, why = found
            outcome[step.id], blocked_by[step.id] = BLOCKED, root
            title = table.by_id[root].title
            plan.append(Planned(step.id, BLOCKED, root, (
                f"зависит от «{title}», который выключен и сегодня не выполнен" if why == "off"
                else f"зависит от «{title}»")))
            continue
        outcome[step.id] = RUN
        plan.append(Planned(step.id, RUN))
    return plan
