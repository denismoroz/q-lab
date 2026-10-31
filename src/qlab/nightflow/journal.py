"""The journal of the night's steps (tables `pipeline_run`, `pipeline_step_run`, `pipeline_item`).

Written live by the runner: a step's row exists the moment the step starts, and its end, its
duration and its items are added as they happen, so a night that is stopped half way still shows
which step it was in. The journal describes; it decides nothing. A journal failure (a locked file,
a missing table) is a warning on stderr and never stops the night (`SafeJournal`).

Reading is in the second half: the states the page paints, a step's history, its statistics.
"""

from __future__ import annotations

import os
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.registry.models import PipelineItem, PipelineRun, PipelineStepRun, TokenSpend

#: The states a step can be in. `pending` is only ever shown, never stored.
STEP_STATES = ("running", "ok", "failed", "blocked", "off", "skipped", "cut", "interrupted")
DONE_STATES = ("ok",)
TERMINAL_STATES = ("ok", "failed", "blocked", "off", "skipped", "cut", "interrupted")


def _now() -> datetime:
    return datetime.now(UTC)


def pid_alive(pid: int | None) -> bool:
    """Whether a process with this id exists (any user's: we may not signal it)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _aware(value: datetime | None) -> datetime | None:
    """SQLite gives back naive datetimes; everything stored here is UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def tokens_between(session: Session, start: datetime, end: datetime) -> int:
    """Tokens the agents spent between two moments (`token_spend`, all four kinds)."""
    total = session.query(func.coalesce(func.sum(
        func.coalesce(TokenSpend.tokens_in, 0) + func.coalesce(TokenSpend.tokens_out, 0)
        + func.coalesce(TokenSpend.tokens_cache_write, 0)
        + func.coalesce(TokenSpend.tokens_cache_read, 0)), 0)).filter(
        TokenSpend.at >= start, TokenSpend.at <= end).scalar()
    return int(total or 0)


class Journal:
    """Writes the journal; every call is its own short transaction."""

    def __init__(self, scope: Callable | None = None) -> None:
        if scope is None:
            from qlab.registry.db import session_scope as scope
        self._scope = scope

    # ------------------------------------------------------------------ runs

    def start_run(self, night_date: date, *, trigger: str, steps_off: list[str],
                  only: list[str] | None, restart: str | None, concurrency: int, pid: int,
                  command: str | None = None, note: str | None = None) -> int:
        with self._scope() as session:
            run = PipelineRun(night_date=night_date, trigger=trigger, state="running",
                              started_at=_now(), pid=pid, concurrency=concurrency,
                              steps_off=sorted(steps_off), only=sorted(only) if only else None,
                              restart=restart, command=command, note=note, approximate=False)
            session.add(run)
            session.flush()
            return run.id

    def finish_run(self, run_id: int, state: str, note: str | None = None) -> None:
        with self._scope() as session:
            run = session.get(PipelineRun, run_id)
            run.state, run.finished_at = state, _now()
            if note:
                run.note = note

    def sweep_interrupted(self) -> int:
        """Rows left `running` by a process that died. Called when a runner holds the lock, so a
        running row belongs to a process that is gone; a process that still lives (a second
        journal on another lock, as in a test) is left alone."""
        with self._scope() as session:
            old = _now() - timedelta(days=1)  # no run lasts a day: a reused process id is no excuse
            runs = [r for r in session.query(PipelineRun).filter(PipelineRun.state == "running")
                    if not pid_alive(r.pid) or _aware(r.started_at) < old]
            for run in runs:
                run.state, run.finished_at = "interrupted", run.finished_at or _now()
                for row in session.query(PipelineStepRun).filter(
                        PipelineStepRun.run_id == run.id, PipelineStepRun.state == "running"):
                    row.state, row.finished_at = "interrupted", _now()
                    row.error = row.error or "процесс остановлен до конца шага"
            return len(runs)

    # ------------------------------------------------------------------ steps

    def start_step(self, run_id: int, step_id: str) -> int:
        with self._scope() as session:
            row = PipelineStepRun(run_id=run_id, step_id=step_id, state="running",
                                  started_at=_now(), items_done=0, items_failed=0,
                                  approximate=False)
            session.add(row)
            session.flush()
            return row.id

    def finish_step(self, step_run_id: int, state: str, *, error: str | None = None,
                    note: str | None = None, tokens: bool = False) -> None:
        with self._scope() as session:
            row = session.get(PipelineStepRun, step_run_id)
            row.state, row.finished_at = state, _now()
            started = _aware(row.started_at)
            row.duration_s = round((row.finished_at - started).total_seconds(), 3)
            row.error, row.note = error, note or row.note
            counts = dict(session.query(PipelineItem.outcome, func.count()).filter(
                PipelineItem.step_run_id == step_run_id).group_by(PipelineItem.outcome).all())
            row.items_done, row.items_failed = counts.get("ok", 0), counts.get("failed", 0)
            if tokens:
                row.tokens = tokens_between(session, started, row.finished_at)

    def record_step(self, run_id: int, step_id: str, state: str, *, blocked_by: str | None = None,
                    note: str | None = None) -> None:
        """A step that did not run (off, blocked, skipped, cut): one finished row."""
        with self._scope() as session:
            moment = _now()
            session.add(PipelineStepRun(
                run_id=run_id, step_id=step_id, state=state, started_at=moment,
                finished_at=moment, duration_s=0.0, items_done=0, items_failed=0,
                blocked_by=blocked_by, note=note, approximate=False))

    def add_item(self, step_run_id: int, name: str, outcome: str, *,
                 duration_s: float | None = None, detail: str | None = None,
                 finished_at: datetime | None = None) -> None:
        with self._scope() as session:
            session.add(PipelineItem(step_run_id=step_run_id, name=name, outcome=outcome,
                                     duration_s=duration_s, detail=detail,
                                     finished_at=finished_at or _now()))
            row = session.get(PipelineStepRun, step_run_id)
            if outcome == "ok":
                row.items_done += 1
            elif outcome == "failed":
                row.items_failed += 1

    # ------------------------------------------------------------------ reading for the runner

    def done_today(self, night_date: date) -> set[str]:
        """Steps that ended `ok` in some run of this night: a switched-off parent among them does
        not block its children (its output of tonight exists)."""
        with self._scope() as session:
            rows = session.query(PipelineStepRun.step_id).join(
                PipelineRun, PipelineRun.id == PipelineStepRun.run_id).filter(
                PipelineRun.night_date == night_date, PipelineStepRun.state == "ok").distinct()
            return {r[0] for r in rows}


class NullJournal:
    """The journal of a run that keeps none (the suite's whole-night tests, which must not touch
    the owner's registry)."""

    def start_run(self, *a, **k) -> int:
        return 0

    def finish_run(self, *a, **k) -> None:
        return None

    def sweep_interrupted(self) -> int:
        return 0

    def start_step(self, *a, **k) -> int:
        return 0

    def finish_step(self, *a, **k) -> None:
        return None

    def record_step(self, *a, **k) -> None:
        return None

    def add_item(self, *a, **k) -> None:
        return None

    def done_today(self, night_date: date) -> set[str]:
        return set()


class SafeJournal:
    """Wraps a journal so that no failure of it reaches the night."""

    def __init__(self, inner: Journal | NullJournal) -> None:
        self._inner = inner
        self.failures: list[str] = []

    def __getattr__(self, name: str):
        method = getattr(self._inner, name)

        def call(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - see the class docstring
                text = f"{name}: {type(exc).__name__}: {exc}"
                self.failures.append(text)
                print(f"[journal] {text}", file=sys.stderr)
                return {"done_today": set(), "start_run": 0, "start_step": 0,
                        "sweep_interrupted": 0}.get(name)
        return call


# ---------------------------------------------------------------------------- reading for the page


@dataclass
class StepState:
    step_id: str
    state: str
    run_id: int | None = None
    step_run_id: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    items_done: int = 0
    items_failed: int = 0
    tokens: int | None = None
    error: str | None = None
    blocked_by: str | None = None
    note: str | None = None
    approximate: bool = False

    def to_json(self) -> dict:
        return {
            "step": self.step_id, "state": self.state, "run_id": self.run_id,
            "step_run_id": self.step_run_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": self.duration_s, "items_done": self.items_done,
            "items_failed": self.items_failed, "tokens": self.tokens, "error": self.error,
            "blocked_by": self.blocked_by, "note": self.note, "approximate": self.approximate,
        }


def step_state(row: PipelineStepRun) -> StepState:
    return StepState(
        step_id=row.step_id, state=row.state, run_id=row.run_id, step_run_id=row.id,
        started_at=_aware(row.started_at), finished_at=_aware(row.finished_at),
        duration_s=row.duration_s, items_done=row.items_done, items_failed=row.items_failed,
        tokens=row.tokens, error=row.error, blocked_by=row.blocked_by, note=row.note,
        approximate=row.approximate)


def latest_run(session: Session) -> PipelineRun | None:
    return session.query(PipelineRun).filter(PipelineRun.approximate.is_(False)).order_by(
        PipelineRun.started_at.desc(), PipelineRun.id.desc()).first() or session.query(
        PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc()).first()


def night_states(session: Session, night_date: date) -> dict[str, StepState]:
    """Each step's latest row among all runs of the night: a restart of one step shows its new
    row and leaves the rest as they were."""
    rows = session.query(PipelineStepRun).join(
        PipelineRun, PipelineRun.id == PipelineStepRun.run_id).filter(
        PipelineRun.night_date == night_date).order_by(PipelineStepRun.id).all()
    return {row.step_id: step_state(row) for row in rows}


def run_json(run: PipelineRun, *, alive: bool | None = None) -> dict:
    """A run for the API. `alive` is whether a process holds the run lock: a row that says
    `running` while nobody does is an interrupted run, whatever its row last said."""
    state = run.state
    if state == "running" and alive is False:
        state = "interrupted"
    return {
        "id": run.id, "night_date": run.night_date.isoformat(), "trigger": run.trigger,
        "state": state, "started_at": _aware(run.started_at).isoformat(),
        "finished_at": _aware(run.finished_at).isoformat() if run.finished_at else None,
        "pid": run.pid, "concurrency": run.concurrency, "steps_off": run.steps_off or [],
        "only": run.only, "restart": run.restart, "note": run.note,
        "approximate": run.approximate,
    }


def step_history(session: Session, step_id: str, limit: int = 30) -> list[dict]:
    rows = (session.query(PipelineStepRun, PipelineRun)
            .join(PipelineRun, PipelineRun.id == PipelineStepRun.run_id)
            .filter(PipelineStepRun.step_id == step_id)
            .order_by(PipelineStepRun.id.desc()).limit(limit).all())
    out = []
    for row, run in rows:
        item = step_state(row).to_json()
        item.update(night_date=run.night_date.isoformat(), trigger=run.trigger)
        out.append(item)
    return out


def step_items(session: Session, step_run_id: int, limit: int = 500) -> list[dict]:
    rows = (session.query(PipelineItem).filter(PipelineItem.step_run_id == step_run_id)
            .order_by(PipelineItem.id).limit(limit).all())
    return [{"name": r.name, "outcome": r.outcome, "duration_s": r.duration_s, "detail": r.detail,
             "finished_at": _aware(r.finished_at).isoformat() if r.finished_at else None}
            for r in rows]


def step_stats(session: Session, step_id: str, *, approximate: bool = False) -> dict:
    """Runs, success share, durations and tokens of a step. Only measured runs by default; the
    rows rebuilt from old nights (`approximate`) are a separate count, never mixed in."""
    rows = session.query(PipelineStepRun).filter(
        PipelineStepRun.step_id == step_id, PipelineStepRun.state.in_(("ok", "failed")),
        PipelineStepRun.approximate.is_(approximate)).order_by(PipelineStepRun.id).all()
    ok = [r for r in rows if r.state == "ok"]
    durations = [r.duration_s for r in rows if r.duration_s is not None]
    ok_durations = [r.duration_s for r in ok if r.duration_s is not None]
    tokens = [r.tokens for r in rows if r.tokens]
    last_error = next((r for r in reversed(rows) if r.state == "failed" and r.error), None)
    blocked = session.query(func.count()).select_from(PipelineStepRun).filter(
        PipelineStepRun.step_id == step_id, PipelineStepRun.state == "blocked",
        PipelineStepRun.approximate.is_(approximate)).scalar()
    return {
        "runs": len(rows), "ok": len(ok), "failed": len(rows) - len(ok), "blocked": int(blocked),
        "success_share": round(len(ok) / len(rows), 4) if rows else None,
        "avg_s": round(statistics.fmean(ok_durations), 1) if ok_durations else None,
        "median_s": round(statistics.median(ok_durations), 1) if ok_durations else None,
        "max_s": round(max(durations), 1) if durations else None,
        "min_s": round(min(ok_durations), 1) if ok_durations else None,
        "tokens_avg": round(statistics.fmean(tokens)) if tokens else None,
        "tokens_total": sum(tokens) if tokens else 0,
        "last_error": last_error.error if last_error else None,
        "last_error_at": (_aware(last_error.finished_at).isoformat()
                          if last_error and last_error.finished_at else None),
        "approximate": approximate,
    }


def last_item_seconds(session: Session, step_id: str, name: str) -> float | None:
    """How long the latest finished item `name` of step `step_id` took, from the journal: what a
    stage reads to estimate the same piece of work the next time. None when it was never done."""
    row = (session.query(PipelineItem.duration_s)
           .join(PipelineStepRun, PipelineStepRun.id == PipelineItem.step_run_id)
           .filter(PipelineStepRun.step_id == step_id, PipelineItem.name == name,
                   PipelineItem.duration_s.isnot(None), PipelineItem.duration_s > 0)
           .order_by(PipelineItem.id.desc()).first())
    return float(row[0]) if row else None


def last_finished(session: Session, step_id: str) -> StepState | None:
    row = (session.query(PipelineStepRun).filter(
        PipelineStepRun.step_id == step_id, PipelineStepRun.state.in_(("ok", "failed")))
        .order_by(PipelineStepRun.id.desc()).first())
    return step_state(row) if row else None
