"""The night's state, one slice per step (docs/PIPELINE.md, «Дорожки»).

The stages of the night used to share one dict and save it whole (`night.save_state`). With steps
running together that breaks twice: a save that walks the dict while another step adds a key to it
fails ("dictionary changed size during iteration"), and two steps that write one key lose one
write. Here every step works on a private copy and the hub merges back only what the step owns:

* `view(step_id)` is a deep copy of the whole state, as it stands when the step starts. The stage
  reads other steps' results from it (they were written before, because of the table's edges) and
  writes its own keys into it, exactly as it wrote into the shared dict;
* `save(view)` (what `night.save_state` does for a view) merges the step's own keys (the table's
  `state_keys`) and its done-markers into the master state, under one lock, and writes the file
  atomically. Nothing walks a dict another thread is changing: the master is touched only by the
  hub, and a view only by its step;
* a key that a step does not own but changed in its view is a conflict: the write would be lost.
  It is counted (`violations`), printed and, with `strict`, raised. The proof run and the tests are
  strict. The table already forbids two steps owning one key unless one is the other's ancestor
  (`qlab.nightflow.table.validate`), so the two can never run at once.

The keys no step owns (`account`, `limits_off`, `prefetch`, `stage_errors`, `stopped_at`,
`deadline`) belong to the hub; they are changed only through its methods.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

from qlab.nightflow.table import Table

#: Written by the run itself or through the hub's methods, never by one step's stage.
HUB_KEYS = frozenset({"account", "limits_off", "prefetch", "stage_errors", "stopped_at",
                      "deadline", "done"})
_MISSING = object()


class ForeignStateWrite(RuntimeError):
    """A step changed a key of the night's state that it does not own (strict mode)."""


class StepView(dict):
    """The night's state as one step sees it: a private copy (see the module docstring). It is a
    plain dict for the stage; `hub` and `owner` are how `night.save_state` finds where to merge."""

    hub: NightStateHub
    owner: str
    seed: dict


class NightStateHub:
    def __init__(self, master: dict, table: Table | None = None, *,
                 write: Callable[[str], None] | None = None,
                 observers: Callable[[], Iterable[Callable[[dict], None]]] | None = None,
                 strict: bool = False, log: Callable[[str], None] | None = None) -> None:
        self.master = master
        self.table = table
        self.lock = threading.RLock()
        self._write = write
        self._observers = observers or (lambda: ())
        self.strict = strict
        self.violations: list[str] = []
        self.saves = 0
        self._log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self._open: dict[int, StepView] = {}

    # ------------------------------------------------------------------ views

    def owned_keys(self, step_id: str) -> frozenset[str]:
        if self.table is None or step_id not in self.table.by_id:
            return frozenset()
        return frozenset(self.table.by_id[step_id].state_keys)

    def done_prefixes(self, step_id: str) -> tuple[str, ...]:
        if self.table is None or step_id not in self.table.by_id:
            return ()
        return self.table.by_id[step_id].done_prefixes

    def view(self, step_id: str) -> StepView:
        """A private copy of the state for `step_id`, taken now."""
        with self.lock:
            data = copy.deepcopy(self.master)
            seed = copy.deepcopy(data)
        view = StepView(data)
        view.hub, view.owner, view.seed = self, step_id, seed
        with self.lock:
            self._open[id(view)] = view
        return view

    def close(self, view: StepView) -> None:
        with self.lock:
            self._open.pop(id(view), None)

    # ------------------------------------------------------------------ merging and saving

    def save(self, view: StepView) -> None:
        """Merge the step's own keys and done-markers into the master and write the file."""
        with self.lock:
            self._merge(view)
            self._flush_locked()
        self._notify()

    def save_master(self) -> None:
        """Write the master as it stands (the keys the run itself changed)."""
        with self.lock:
            self._flush_locked()
        self._notify()

    def snapshot(self) -> dict:
        """A deep copy of the master, for readers outside a step (the journal's item tracker)."""
        with self.lock:
            return copy.deepcopy(self.master)

    def save_open_views(self) -> None:
        """A stop of the whole night: merge what the steps that are still running have in their
        views (they may still be changing them, so a copy that fails is tried again, then left)."""
        with self.lock:
            views = list(self._open.values())
        for view in views:
            for _ in range(5):
                try:
                    self.save(view)
                    break
                except RuntimeError:  # the dict changed under the copy
                    continue

    def _merge(self, view: StepView) -> None:
        owned = self.owned_keys(view.owner)
        for key in owned:
            if key in view:
                self.master[key] = copy.deepcopy(view[key])
        self._merge_done(view)
        self._check_foreign(view, owned)

    def _merge_done(self, view: StepView) -> None:
        seed_done = view.seed.get("done") or {}
        mine = view.get("done") or {}
        master_done = self.master.setdefault("done", {})
        prefixes = self.done_prefixes(view.owner)
        for key, stamp in list(mine.items()):
            if seed_done.get(key) == stamp:
                continue
            if prefixes and not key.startswith(prefixes):
                self._violation(view, f"done-marker {key!r} is outside its prefixes {prefixes}")
            master_done[key] = stamp
        for key in seed_done:
            if key not in mine:
                if prefixes and not key.startswith(prefixes):
                    self._violation(view, f"removed the done-marker {key!r} of another step")
                master_done.pop(key, None)

    def _check_foreign(self, view: StepView, owned: frozenset[str]) -> None:
        for key in set(view) | set(view.seed):
            if key in owned or key in HUB_KEYS:
                continue
            try:
                same = view.get(key, _MISSING) == view.seed.get(key, _MISSING)
            except Exception:  # noqa: BLE001 - values that cannot be compared are not state
                same = True
            if not same:
                self._violation(view, f"changed the key {key!r}, which it does not own")

    def _violation(self, view: StepView, what: str) -> None:
        text = f"step {view.owner!r} {what}: the write is lost"
        self.violations.append(text)
        self._log(f"[state] {text}")
        if self.strict:
            raise ForeignStateWrite(text)

    def _flush_locked(self) -> None:
        self.saves += 1
        if self._write is not None:
            self._write(json.dumps(self.master, indent=1, default=str))

    def _notify(self) -> None:
        observers = list(self._observers())
        if not observers:
            return
        snapshot = self.snapshot()
        for observer in observers:
            try:
                observer(snapshot)
            except Exception:  # noqa: BLE001 - the journal must never fail a save of the night
                pass

    # ------------------------------------------------------------------ the keys the hub owns

    def set_key(self, key: str, value: Any) -> None:
        with self.lock:
            self.master[key] = value

    def record_error(self, name: str, error: str) -> None:
        """A failed step's line in `stage_errors` (the report's «Сбои стадий»)."""
        with self.lock:
            self.master.setdefault("stage_errors", {})[name] = error
            self._flush_locked()
        self._notify()

    def update(self, fn: Callable[[dict], Any]) -> Any:
        """Run `fn(master)` under the lock: the one way for code outside a step to change the
        master after the steps have started."""
        with self.lock:
            return fn(self.master)
