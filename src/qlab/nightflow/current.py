"""Which step of the night the current thread is running, and in which lane.

The token guard is not safe for two agent calls at once (it reads the windows, divides what is
left, and records the spend, call by call), so every agent call of the night comes from the agents'
lane, which runs one step at a time. `run_agent` asks this module and refuses a call made by a step
of another lane: «вызов агента вне бюджетного гарда невозможен по конструкции» holds for the lanes
too. A thread that runs no step (the console, a hand command, a test) is not restricted."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

AGENT_LANE = "agents"
_local = threading.local()


class AgentOutsideItsLane(RuntimeError):
    """A step that is not in the agents' lane tried to call an agent."""


@contextmanager
def running(step_id: str, lane: str) -> Iterator[None]:
    before = getattr(_local, "step", None)
    _local.step = (step_id, lane)
    try:
        yield
    finally:
        _local.step = before


def current() -> tuple[str, str] | None:
    return getattr(_local, "step", None)


def require_agents_lane() -> None:
    found = current()
    if found is not None and found[1] and found[1] != AGENT_LANE:
        raise AgentOutsideItsLane(
            f"step {found[0]!r} is in the {found[1]!r} lane and may not call an agent: the token "
            f"guard serves one agent step at a time, and those are the {AGENT_LANE!r} lane's")
