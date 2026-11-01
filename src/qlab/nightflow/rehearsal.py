"""Stand-ins for the agent steps, for rehearsing the lanes without spending a token
(docs/PIPELINE.md, «Дорожки»; `scripts/rehearse_lanes.py`).

A stub step sleeps for a given time, saving a key of its own every few seconds (the night's state
is then written from two lanes at once), and may run the stand on given specs the way the researcher
and the generator do (trials written from the agents' lane while the calculations lane writes its
own). It calls no agent and touches no token guard. `stubbed_steps` swaps the stubs in for the real
agent steps; `table_with_stub_keys` lets the table's ownership check know about the stubs' keys."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from pathlib import Path

from qlab.nightflow.runner import StepImpl
from qlab.nightflow.table import GATING_KINDS, Table, validate


def stub_key(step_id: str) -> str:
    return f"rehearsal_{step_id.replace('-', '_')}"


def make_stub(step_id: str, seconds: float, specs: list[str], *, every: float = 2.0) -> StepImpl:
    """A step that sleeps `seconds` and runs the stand on `specs`."""
    key = stub_key(step_id)

    def run(ctx) -> None:
        from qlab import night
        from qlab.pipeline.spec import load_spec

        block = ctx.state.setdefault(key, {"slept": 0.0, "evaluated": []})
        for path in specs:
            spec = load_spec(Path(path))
            result = night.evaluate_item(step_id, spec, end=ctx.end, noise_trials=ctx.noise_trials)
            block["evaluated"].append({"idea": result.name, "trial": result.trial_id,
                                       "route": result.route, "error": result.error})
            ctx.save()
        started = time.monotonic()
        while (left := seconds - (time.monotonic() - started)) > 0:
            time.sleep(min(every, left))
            block["slept"] = round(time.monotonic() - started, 1)
            ctx.save()

    return StepImpl(run, step_id)


def stubbed_steps(real: dict[str, StepImpl], plan: Mapping[str, tuple[float, list[str]]]
                  ) -> dict[str, StepImpl]:
    """`real` with the steps named in `plan` ({step: (seconds, specs for the stand)}) replaced by
    stubs."""
    out = dict(real)
    for step_id, (seconds, specs) in plan.items():
        out[step_id] = make_stub(step_id, seconds, specs)
    return out


def table_with_stub_keys(table: Table, step_ids: list[str]) -> Table:
    """The table with each stub step also owning its stub key (validated again)."""
    steps = [dataclasses.replace(s, state_keys=(*s.state_keys, stub_key(s.id)))
             if s.id in step_ids else s for s in table.steps]
    new = Table(steps)
    validate(new)
    return new


def idle_while_ready(table: Table, spans: Mapping[str, tuple[float, float]], lane: str,
                     grace: float = 0.0) -> list[tuple[str, float, float]]:
    """The stretches in which `lane` did nothing although one of its steps could have started:
    (the step that was ready, from when the lane stood idle, until when) for every idle gap longer
    than `grace` seconds that a step started after, while it was ready. `spans` is {step: (start,
    end)} in seconds of the steps that ran. A step is ready when every parent it really waits for
    (`GATING_KINDS`) that ran has ended. An empty answer is the proof that the lane was never idle
    while one of its steps was ready (run 31, 2026-11-01: the agents' lane stood idle for 100
    minutes behind a budget edge)."""
    mine = sorted((s for s in table.ids if s in spans and table.by_id[s].lane == lane),
                  key=lambda s: spans[s][0])
    ready = {s: max((spans[p][1] for p in table.parents(s, GATING_KINDS) if p in spans),
                    default=0.0) for s in mine}
    gaps: list[tuple[float, float]] = []
    free = 0.0
    for step in mine:
        start, end = spans[step]
        if start > free:
            gaps.append((free, start))
        free = max(free, end)
    waits: list[tuple[str, float, float]] = []
    for gap_from, gap_to in gaps:
        for step in mine:
            if spans[step][0] < gap_to:
                continue  # started before the gap ended: it was not waiting through it
            idle_from = max(gap_from, ready[step])
            if gap_to - idle_from > grace:
                waits.append((step, idle_from, gap_to))
    return waits
