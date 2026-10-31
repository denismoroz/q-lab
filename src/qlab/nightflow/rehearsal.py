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
from qlab.nightflow.table import Table, validate


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
