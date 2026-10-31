"""The journal's history before the journal: rebuilt from the nights' saved state.

`data/night/<date>.json` lists what a night did and WHEN each item of work was marked done, and
`data/night/logs/<date>.log` has the start and end of the process. That is enough for an honest
approximation: a step that left done-markers ran from its first marker to its last (a lower
bound: the first item's own time is before the first marker), a step that left its block in the
state ran, a step named in `stage_errors` failed. Everything written here is marked
`approximate`, kept apart from measured rows in every statistic, and never says a step took
longer or shorter than the markers show. Nothing is guessed for a step that left no trace.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.nightflow.table import Table, load_table
from qlab.registry.models import PipelineItem, PipelineRun, PipelineStepRun

STATE_DIR = Path("data/night")
NOTE = "восстановлено из состояния ночи: времена приблизительные"
_LOG_LINE = re.compile(r"^=== (\d{4}-\d\d-\d\dT[\d:]+)Z (start|end \(exit (\d+)\))")


def _parse(ts: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def log_segments(log: Path) -> list[tuple[datetime, datetime | None, int | None]]:
    """The process runs the night's log records: (start, end, exit code); the end is None for a
    run that never wrote one (killed). A night resumed after a stop has several."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[tuple[datetime, datetime | None, int | None]] = []
    for line in lines:
        found = _LOG_LINE.match(line)
        if not found:
            continue
        moment = datetime.fromisoformat(found.group(1)).replace(tzinfo=UTC)
        if found.group(2) == "start":
            out.append((moment, None, None))
        elif out and out[-1][1] is None:
            out[-1] = (out[-1][0], moment, int(found.group(3)))
    return out


def log_bounds(log: Path) -> tuple[datetime | None, datetime | None, int | None]:
    """The first start and the last end the night's log records, with the last exit code."""
    segments = log_segments(log)
    if not segments:
        return None, None, None
    last = segments[-1]
    return segments[0][0], last[1], last[2]


def _span(moments: list[datetime], segments: list[tuple[datetime, datetime | None, int | None]],
          ) -> float:
    """Seconds from the first to the last of `moments`, counted inside each process run so that a
    pause between two runs of the same night is not a step's time."""
    if not segments:
        return (max(moments) - min(moments)).total_seconds()
    total = 0.0
    starts = [s for s, _, _ in segments]
    for at, begin in enumerate(starts):
        until = starts[at + 1] if at + 1 < len(starts) else None
        inside = [m for m in moments if m >= begin and (until is None or m < until)]
        if inside:
            total += (max(inside) - min(inside)).total_seconds()
    return total


def rebuild_night(table: Table, day: date, state: dict, log: Path) -> dict | None:
    """The run and step rows one night's state implies, or None when it implies nothing."""
    markers: dict[str, list[tuple[datetime, str]]] = {}
    done = state.get("done") or {}
    claimed: set[str] = set()  # a marker belongs to the first step that names its prefix
    for step in table.steps:
        if not step.done_prefixes:
            continue
        found = []
        for key, ts in done.items():
            moment = _parse(ts)
            if moment and key not in claimed and key.startswith(step.done_prefixes):
                claimed.add(key)
                found.append((moment, key.split(":", 1)[1] if ":" in key else key))
        if found:
            markers[step.id] = sorted(found)
    segments = log_segments(log)
    errors = state.get("stage_errors") or {}
    rows = []
    for step in table.steps:
        found = markers.get(step.id)
        failed = errors.get(step.id)
        present = any(key in state for key in step.state_keys)
        if not (found or failed or present):
            continue
        row = {"step": step.id, "state": "failed" if failed else "ok", "error": failed,
               "started_at": found[0][0] if found else None,
               "finished_at": found[-1][0] if found else None, "items": []}
        if found:
            row["duration_s"] = round(_span([m for m, _ in found], segments), 1)
            previous = None
            for moment, name in found:
                gap = round((moment - previous).total_seconds(), 1) if previous else None
                row["items"].append({"name": name, "duration_s": gap, "finished_at": moment})
                previous = moment
        rows.append(row)
    start, end, code = log_bounds(log)
    stamps = [m for found in markers.values() for m, _ in found]
    if not rows and not stamps:
        return None
    return {"night_date": day, "started_at": start or (min(stamps) if stamps else None),
            "finished_at": end, "exit_code": code, "steps": rows}


def backfill(session: Session, state_dir: Path | None = None,
             table: Table | None = None) -> list[str]:
    """Insert one approximate run per night file that has none yet; returns the nights added."""
    folder = state_dir or STATE_DIR
    table = table or load_table()
    have = {d for (d,) in session.query(PipelineRun.night_date).filter(
        PipelineRun.approximate.is_(True))}
    made: list[str] = []
    for path in sorted(folder.glob("????-??-??.json")):
        day = date.fromisoformat(path.stem)
        if day in have:
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        built = rebuild_night(table, day, state, folder / "logs" / f"{path.stem}.log")
        if built is None or built["started_at"] is None:
            continue
        ended = built["finished_at"]
        run = PipelineRun(
            night_date=day, trigger="schedule", started_at=built["started_at"],
            finished_at=ended, state=("failed" if state.get("stage_errors") else "ok")
            if ended else "interrupted", concurrency=1, steps_off=[], only=None, restart=None,
            command=None, note=NOTE, approximate=True)
        session.add(run)
        session.flush()
        for row in built["steps"]:
            step_run = PipelineStepRun(
                run_id=run.id, step_id=row["step"], state=row["state"],
                started_at=row["started_at"], finished_at=row["finished_at"],
                duration_s=row.get("duration_s"), items_done=len(row["items"]), items_failed=0,
                error=row["error"], note=NOTE, approximate=True)
            session.add(step_run)
            session.flush()
            for item in row["items"]:
                session.add(PipelineItem(step_run_id=step_run.id, name=item["name"],
                                         outcome="ok", duration_s=item["duration_s"],
                                         finished_at=item["finished_at"]))
        made.append(path.stem)
    return made
