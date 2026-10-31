"""Runs the night's steps from the table (docs/PIPELINE.md).

A step that raises is marked `failed` and nothing else stops: only its `data` descendants are
marked `blocked` (with `blocked_by` naming the root cause). Every other step still runs. Each step's
row is written to the journal when it starts and again when it ends.

`concurrency` is the number of steps that may run together. With 1 the night goes in the table's
order, as it always did. With more, the steps run in lanes (docs/PIPELINE.md, «Дорожки»): a step
starts when every parent has ended and its lane (`Step.lane`) is free, and at most one step runs
in a lane. Each step works on its own view of the night's state (`qlab.nightflow.statehub`), so
the steps share no dict; a failed step blocks only its `data` descendants, and a long step of
one lane delays nothing in another.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qlab.nightflow.plan import BLOCKED, OFF, OUTSIDE, blocking_parent
from qlab.nightflow.table import Step, Table

#: Walked in order to name a failed item found in a step's state block.
NAME_KEYS = ("name", "idea", "idea_id", "strategy", "card", "path", "id", "trial")


@dataclass
class StepImpl:
    """How the night runs one step of the table.

    `run(ctx)` does the work and raises if the step failed. `skip_reason(ctx)` returns why the
    step has nothing to do tonight (a flag, a cadence), or None. `is_cut(ctx)` says whether the
    night's stop time has passed (it records the cut in the state for the report). `inspect(ctx)`
    returns an error text for a step that swallowed its own failure into the state, or None.
    `finals(ctx)` lists the items the done-markers cannot show, as (name, outcome, text); without
    it the items found in the step's state with an error text are listed as failed. `note(ctx)`
    is a line for the step's row, shown on the page (what was left undone and why), or None."""

    run: Callable[[Any], None]
    error_name: str
    skip_reason: Callable[[Any], str | None] | None = None
    is_cut: Callable[[Any], bool] | None = None
    inspect: Callable[[Any], str | None] | None = None
    finals: Callable[[Any], list[tuple[str, str, str]]] | None = None
    note: Callable[[Any], str | None] | None = None


def describe_error(exc: BaseException) -> str:
    where = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    tail = f" ({Path(where.filename).name}:{where.lineno})" if where else ""
    return f"{type(exc).__name__}: {exc}{tail}"


def failed_items(step: Step, state: dict) -> list[tuple[str, str]]:
    """Items that ended in an error, found generically: every dict under the step's state keys
    that carries an `error` text (or an `action` of `error`).

    A failure is named after its own dict, or after the nearest dict around it that has a name
    (a stage's brief inside an idea's entry), or after the state key it was found under; a row
    with no name is never made. A dict that is reported is not searched further: the failure
    inside it is the same one."""
    found: list[tuple[str, str]] = []

    def walk(node: Any, inherited: str) -> None:
        if len(found) >= 50:
            return
        if isinstance(node, dict):
            name = next((str(node[k]) for k in NAME_KEYS if node.get(k) is not None), inherited)
            text = node.get("error") or (node.get("why") if node.get("action") == "error" else None)
            if isinstance(text, str) and text:
                found.append((name, text[:300]))
                return
            for value in node.values():
                if isinstance(value, dict | list):
                    walk(value, name)
        elif isinstance(node, list):
            for value in node:
                walk(value, inherited)

    for key in step.state_keys:
        walk(state.get(key), key)
    return found


def new_failed_items(step: Step, state: dict, before: Counter) -> list[tuple[str, str]]:
    """The failures in the step's state that were not there when this run of the step began.

    The night's state file keeps the whole day: a stage that is started again the same day (the
    button after an interrupted night) finds the earlier attempts' entries in its block. Those
    belong to the earlier attempts' step rows; counting them again would show a crash of the
    morning in the evening's run. The comparison is by (name, text) with multiplicity, so the
    same failure happening again in this run is still new."""
    left = Counter(before)
    fresh: list[tuple[str, str]] = []
    for item in failed_items(step, state):
        if left[item] > 0:
            left[item] -= 1
        else:
            fresh.append(item)
    return fresh


def item_seconds(step: Step, state: dict, name: str) -> float | None:
    """The seconds a stage recorded for the item `name` in its own entry (`seconds`), the last
    entry winning. The time between two done-markers includes whatever the stage did in between;
    the stage's own figure is what the item took."""
    found: float | None = None

    def walk(node: Any) -> None:
        nonlocal found
        if isinstance(node, dict):
            if any(str(node.get(k)) == name for k in NAME_KEYS if node.get(k) is not None):
                seconds = node.get("seconds")
                if isinstance(seconds, int | float) and not isinstance(seconds, bool):
                    found = float(seconds)
            for value in node.values():
                if isinstance(value, dict | list):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for key in step.state_keys:
        walk(state.get(key))
    return found


class ItemTracker:
    """Turns the night's done-markers into journal items, as the stages write them.

    Every stage marks a finished piece of work in `state["done"]` with the time, and saves the
    state after it. `observe` (called by `save_state`) sees a new marker under a running step's
    prefix and records an item whose duration is the time since the step's previous marker, or
    since the step began. Nothing in the stages had to change to be seen.

    The rows are written by a thread of their own, never by the stage's: `save_state` is called
    from inside stages, some of which hold an unfinished registry transaction, and a journal write
    from that same thread would wait for a lock that thread itself holds. The writer simply waits
    until the stage has committed. `end` drains the queue, so a step's row counts all its items."""

    def __init__(self, table: Table, journal) -> None:
        self.table = table
        self.journal = journal
        self._active: dict[str, dict[str, Any]] = {}
        self._failed_before: dict[str, Counter] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None

    def _write_all(self) -> None:
        while True:
            args, kwargs = self._queue.get()
            try:
                self.journal.add_item(*args, **kwargs)
            except Exception:  # noqa: BLE001 - an item lost is a gap in the journal, not a failure
                pass
            finally:
                self._queue.task_done()

    def _put(self, *args, **kwargs) -> None:
        if self._writer is None:
            self._writer = threading.Thread(target=self._write_all, name="journal-items",
                                            daemon=True)
            self._writer.start()
        self._queue.put((args, kwargs))

    def drain(self, timeout: float = 120.0) -> None:
        """Wait until every queued item is written (or the timeout: a stuck registry must not
        hold the night)."""
        if self._writer is None:
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)

    def begin(self, step: Step, step_run_id: int, state: dict) -> None:
        with self._lock:
            self._active[step.id] = {
                "step": step, "run": step_run_id, "seen": set(state.get("done") or {}),
                "last": datetime.now(UTC)}
            self._failed_before[step.id] = Counter(failed_items(step, state))

    def failed_before(self, step_id: str) -> Counter:
        """The failures the step's state already held when its current run began."""
        with self._lock:
            return Counter(self._failed_before.get(step_id, ()))

    def end(self, step_id: str, state: dict) -> None:
        self.observe(state)
        with self._lock:
            self._active.pop(step_id, None)
        self.drain()

    def observe(self, state: dict) -> None:
        with self._lock:
            active = list(self._active.values())
        for slot in active:
            step: Step = slot["step"]
            if not step.done_prefixes:
                continue
            try:
                done = dict(state.get("done") or {})
            except RuntimeError:  # the dict changed under the copy; the next save sees it
                return
            fresh = sorted((ts, key) for key, ts in done.items()
                           if key not in slot["seen"] and key.startswith(step.done_prefixes))
            for ts, key in fresh:
                slot["seen"].add(key)
                try:
                    moment = datetime.fromisoformat(ts)
                except (TypeError, ValueError):
                    moment = datetime.now(UTC)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                name = key.split(":", 1)[1] if ":" in key else key
                spent = max(0.0, (moment - slot["last"]).total_seconds())
                slot["last"] = max(moment, slot["last"])
                exact = item_seconds(step, state, name)
                if exact is not None:
                    spent = exact
                self._put(slot["run"], name, "ok", duration_s=round(spent, 2), finished_at=moment)


class Runner:
    def __init__(self, table: Table, impls: dict[str, StepImpl], ctx: Any, *, journal, run_id: int,
                 off: Collection[str] = (), only: Collection[str] | None = None,
                 done_today: Collection[str] = (), concurrency: int = 1,
                 tracker: ItemTracker | None = None, log: Callable[[str], None] | None = None,
                 on_failure: Callable[[str, str], None] | None = None) -> None:
        missing = set(table.ids) - set(impls)
        if missing:
            raise ValueError(f"steps with no implementation: {', '.join(sorted(missing))}")
        self.table, self.impls, self.ctx = table, impls, ctx
        self.journal, self.run_id = journal, run_id
        self.off, self.only, self.done_today = set(off), set(only) if only is not None else None, \
            set(done_today)
        self.concurrency = max(1, int(concurrency))
        self.tracker = tracker
        self.log = log or (lambda text: print(text, flush=True))
        self.on_failure = on_failure
        self.outcome: dict[str, str] = {}
        self.blocked_by: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.RLock()
        self._ctx_of: dict[str, Any] = {}
        self._rows: dict[str, int] = {}  # step id -> journal row, for the steps running now
        self._stopping = threading.Event()

    # ------------------------------------------------------------------ one step

    def _step_ctx(self, step: Step) -> Any:
        """The context the step runs with: its own view of the night's state when the night's
        context makes one (`NightContext.for_step`), else the shared one (a test's stub)."""
        with self._lock:
            found = self._ctx_of.get(step.id)
        if found is None:
            make = getattr(self.ctx, "for_step", None)
            found = make(step) if make is not None else self.ctx
            with self._lock:
                self._ctx_of[step.id] = found
        return found

    def _finish_without_running(self, step: Step, state: str, **fields) -> None:
        with self._lock:
            self.outcome[step.id] = state
            if fields.get("blocked_by"):
                self.blocked_by[step.id] = fields["blocked_by"]
        self.journal.record_step(self.run_id, step.id, state, **fields)
        self.log(f"=== step {step.id}: {state}"
                 + (f" ({fields['note']})" if fields.get("note") else ""))

    def _precheck(self, step: Step) -> bool:
        """Records the step as not run and returns False when it is outside the run, switched off
        or blocked by a parent. Needs no lane: it only reads what the parents came to."""
        with self._lock:
            if self.only is not None and step.id not in self.only:
                self.outcome[step.id] = OUTSIDE
                return False
            if step.id in self.off:
                off_note = "выключен владельцем" + (
                    "; сегодня уже выполнен" if step.id in self.done_today else "")
                found = None
            else:
                off_note = None
                found = blocking_parent(self.table, step.id, self.outcome, self.blocked_by,
                                        self.done_today)
        if off_note is not None:
            self._finish_without_running(step, OFF, note=off_note)
            return False
        if found is not None:
            root, why = found
            title = self.table.by_id[root].title
            self._finish_without_running(
                step, BLOCKED, blocked_by=root,
                note=(f"«{title}» выключен и сегодня не выполнен" if why == "off"
                      else f"зависит от упавшего шага «{title}»"))
            return False
        return True

    def _start_checks(self, step: Step) -> bool:
        """The step's own reasons not to run now (a flag, a cadence, the night's stop time),
        asked when its turn has come. Records the step and returns False when it does not run."""
        impl = self.impls[step.id]
        sctx = self._step_ctx(step)
        if impl.skip_reason is not None and (reason := impl.skip_reason(sctx)):
            self._finish_without_running(step, "skipped", note=reason)
            return False
        if impl.is_cut is not None and impl.is_cut(sctx):
            self._finish_without_running(step, "cut", note="вышел срок ночи, новое не начато")
            return False
        return True

    def _decide_before_start(self, step: Step) -> bool:
        return self._precheck(step) and self._start_checks(step)

    def _run_step(self, step: Step) -> None:
        impl = self.impls[step.id]
        sctx = self._step_ctx(step)
        step_run_id = self.journal.start_step(self.run_id, step.id)
        with self._lock:
            self._rows[step.id] = step_run_id
        if self.tracker is not None:
            self.tracker.begin(step, step_run_id, sctx.state)
        self.log(f"=== step {step.id}: start")
        started = time.monotonic()
        result, error = "ok", None
        try:
            impl.run(sctx)
            if (save := getattr(sctx, "save", None)) is not None:
                save()  # the saved state is what the item tracker reads
            if impl.inspect is not None and (swallowed := impl.inspect(sctx)):
                result, error = "failed", swallowed
        except Exception as exc:  # noqa: BLE001 - a failed step is a row, not the end of the night
            result, error = "failed", describe_error(exc)
            traceback.print_exc(file=sys.stderr)
            try:
                if (save := getattr(sctx, "save", None)) is not None:
                    save()  # what the step had done before it failed stays in the night's state
            except Exception:  # noqa: BLE001
                pass
        except BaseException as exc:
            # A stop of the whole night (Ctrl-C, SIGTERM): the row says so, the stop goes on.
            try:
                self._finish_row(step, step_run_id, "interrupted", describe_error(exc), sctx)
            except Exception:  # noqa: BLE001 - the stop matters more than the row
                self._interrupted_row(step.id, describe_error(exc))
            raise
        if self._stopping.is_set():
            return  # the night was stopped meanwhile; the row already says so
        self._finish_row(step, step_run_id, result, error, sctx)
        self.log(f"=== step {step.id}: {result} in {time.monotonic() - started:.1f}s"
                 + (f" -- {error[:200]}" if error else ""))

    def _interrupted_row(self, step_id: str, error: str) -> None:
        with self._lock:
            row = self._rows.pop(step_id, None)
            self.outcome.setdefault(step_id, "interrupted")
        if row is not None:
            self.journal.finish_step(row, "interrupted", error=error,
                                     tokens=self.table.by_id[step_id].tokens)

    def _finish_row(self, step: Step, step_run_id: int, result: str, error: str | None,
                    sctx: Any) -> None:
        impl = self.impls[step.id]
        state = sctx.state
        if self.tracker is not None:
            snapshot = getattr(sctx, "snapshot", None)
            self.tracker.end(step.id, snapshot() if snapshot is not None else state)
        if impl.finals is not None:
            extra = impl.finals(sctx)
        else:
            found = (new_failed_items(step, state, self.tracker.failed_before(step.id))
                     if self.tracker is not None else failed_items(step, state))
            extra = [(name, "failed", text) for name, text in found]
        for name, outcome, text in extra:
            self.journal.add_item(step_run_id, name, outcome, detail=text,
                                  duration_s=item_seconds(step, state, name))
        note = impl.note(sctx) if impl.note is not None else None
        self.journal.finish_step(step_run_id, result, error=error, note=note, tokens=step.tokens)
        with self._lock:
            self.outcome[step.id] = result
            self._rows.pop(step.id, None)
            if error:
                self.errors[step.id] = error
        if (release := getattr(sctx, "release", None)) is not None:
            release()
        if result == "failed" and error and self.on_failure is not None:
            self.on_failure(step.id, error)

    # ------------------------------------------------------------------ the whole table

    def execute(self) -> dict[str, str]:
        if self.concurrency == 1:
            for step in self.table.steps:
                if self._decide_before_start(step):
                    self._run_step(step)
            return self.outcome
        return self._execute_in_lanes()

    def _execute_in_lanes(self) -> dict[str, str]:
        """A step starts when every parent (of every kind) has ended and its lane is free; at most
        `concurrency` steps at a time and one per lane. Among the steps ready for a lane the one
        first in the table's order goes, so a step waiting for another lane does not hold up the
        ones behind it. The coordinator is this thread; the steps run in threads of their own,
        which are daemons: a stop of the night (SIGTERM, Ctrl-C) must end the process, not wait
        for a step that may run for hours."""
        pending = list(self.table.steps)
        busy: dict[str, str] = {}  # lane -> the step running in it
        ended: queue.Queue = queue.Queue()
        threads: dict[str, threading.Thread] = {}

        def work(step: Step) -> None:
            try:
                self._run_step(step)
                ended.put((step.id, None))
            except BaseException as exc:  # noqa: BLE001 - handed to the coordinator
                ended.put((step.id, exc))

        try:
            while pending or busy:
                started_something = False
                for step in list(pending):
                    if any(p not in self.outcome for p in self.table.parents(step.id)):
                        continue
                    if not self._precheck(step):
                        pending.remove(step)
                        started_something = True
                        continue
                    lane = step.lane_key
                    if lane in busy or len(busy) >= self.concurrency:
                        continue
                    pending.remove(step)
                    started_something = True
                    if not self._start_checks(step):
                        continue
                    busy[lane] = step.id
                    threads[step.id] = threading.Thread(
                        target=work, args=(step,), name=f"step-{step.id}", daemon=True)
                    threads[step.id].start()
                if busy:
                    try:
                        step_id, exc = ended.get(timeout=1.0)
                    except queue.Empty:
                        continue  # wakes the thread now and then so that a signal is taken at once
                    busy.pop(self.table.by_id[step_id].lane_key, None)
                    threads.pop(step_id).join()
                    if exc is not None:
                        raise exc  # a stop of the night (BaseException) comes out here
                elif pending and not started_something:
                    raise RuntimeError("the step table has a cycle")
        except BaseException as exc:
            self._stopping.set()
            for step_id in list(busy.values()):
                self._interrupted_row(step_id, describe_error(exc))
            raise
        return self.outcome
