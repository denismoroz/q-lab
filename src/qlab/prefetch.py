"""Fetch the data of the night's later items while an earlier one computes (docs/NIGHT_SPEED.md).

Most of a night's wall time is the venues' pacing: a request every half second, a few thousand
requests to bring the raw cache and the snapshots of the watch list up to the last closed day
(night of 2026-10-29: about an hour before the watch list was done, with the machine idle). The
fetch of the next item does not depend on the result of the one before it, so a background thread
builds the snapshots of the items still to run, in the order they will be asked for, while the
main thread computes.

The thread only calls `qlab.pipeline.evaluate.resolve_panel` on the data block the item will ask
for, so it fetches exactly what the item would have fetched, and writes the same snapshot rows.
Two builds of one request never run at once (`resolve_panel` takes a lock per request: the item
waits for the thread's build instead of starting its own), and two fetches from one venue never
overlap (`qlab.data.snapshot` takes a lock per source), so the request rate to a venue stays what
a single process makes. A fetch that fails here is only noted: the item asks again when its turn
comes and fails or succeeds as it would have without the thread.

`QLAB_PREFETCH=0` turns it off (the test suite does).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import structlog

from qlab.pipeline.spec import StrategySpec

PREFETCH_ENV = "QLAB_PREFETCH"

logger = structlog.get_logger()


def enabled() -> bool:
    """On unless `QLAB_PREFETCH` is 0/false/off/no."""
    return os.environ.get(PREFETCH_ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


class Prefetcher:
    """Builds the snapshots of `specs` one after another in a daemon thread.

    `fetch(spec)` is what builds one (default: resolve the spec's panel in a registry session of
    its own); a test passes its own. `report()` says what has been done, for the night's state.
    """

    def __init__(self, specs: Iterable[StrategySpec],
                 fetch: Callable[[StrategySpec], object] | None = None) -> None:
        self._specs = list(specs)
        self._fetch = fetch or _resolve
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qlab-prefetch", daemon=True)
        self._lock = threading.Lock()
        self._done: list[str] = []
        self._failed: dict[str, str] = {}
        self._started_at: str | None = None
        self._finished_at: str | None = None

    def start(self) -> Prefetcher:
        self._started_at = datetime.now(UTC).isoformat()
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        """Ask the thread to stop after the fetch in hand and wait for it. A fetch is not
        interrupted: the raw cache keeps what it has finished, and the next run goes on."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        for spec in self._specs:
            if self._stop.is_set():
                break
            try:
                self._fetch(spec)
            except Exception as exc:  # noqa: BLE001 - the item asks again and fails on its own
                with self._lock:
                    self._failed[spec.idea_id] = f"{type(exc).__name__}: {exc}"
                logger.warning("prefetch_failed", idea=spec.idea_id, error=str(exc))
            else:
                with self._lock:
                    self._done.append(spec.idea_id)
        self._finished_at = datetime.now(UTC).isoformat()

    def report(self) -> dict:
        with self._lock:
            return {"planned": len(self._specs), "done": list(self._done),
                    "failed": dict(self._failed), "started_at": self._started_at,
                    "finished_at": self._finished_at}


def _resolve(spec: StrategySpec) -> None:
    from qlab.pipeline.evaluate import resolve_panel
    from qlab.registry.db import session_scope

    with session_scope() as session:
        resolve_panel(session, spec)


def start(specs: Iterable[StrategySpec]) -> Prefetcher | None:
    """Start fetching `specs` in the background, or None when prefetch is off or there is
    nothing to fetch."""
    specs = list(specs)
    if not enabled() or not specs:
        return None
    return Prefetcher(specs).start()
