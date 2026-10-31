"""The pipeline page's routes (docs/PIPELINE.md): the steps, their states and history, and the
console's one way to change something -- start a run.

The reads copy stored rows (`qlab.nightflow.journal`) and the declared table; they decide
nothing. The three POST routes only START a night on its own process (`qlab.nightflow.control`);
the API itself writes nothing to the registry. They answer only to the local console: the request
must come from the loopback, carry the console's own header and name a local origin and host, so
a page on another site cannot press the button.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from qlab.api.deps import get_session
from qlab.nightflow import control, journal
from qlab.nightflow.table import Table, load_table

CONSOLE_HEADER = "x-qlab-console"
_LOCAL_HOST = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
_LOOPBACK = ("127.0.0.1", "::1", "localhost")

SessionDep = Annotated[Session, Depends(get_session)]


class RunBody(BaseModel):
    off: list[str] = Field(default_factory=list)


class RestartBody(BaseModel):
    step: str
    off: list[str] = Field(default_factory=list)


class PreviewBody(BaseModel):
    off: list[str] = Field(default_factory=list)
    restart: str | None = None


def local_only(request: Request) -> None:
    """Refuse a request that did not come from the local console."""
    client = request.client.host if request.client else None
    if client not in _LOOPBACK:
        raise HTTPException(403, "Запуск доступен только с этой машины.")
    if not _LOCAL_HOST.match(request.headers.get("host", "")):
        raise HTTPException(403, "Запуск доступен только по локальному адресу.")
    origin = request.headers.get("origin")
    if origin is not None and not _LOCAL_ORIGIN.match(origin):
        raise HTTPException(403, "Запрос с чужой страницы отклонён.")
    if request.headers.get(CONSOLE_HEADER) != "1":
        raise HTTPException(403, "Запрос не из консоли q-lab.")


LocalOnly = Depends(local_only)


def _table() -> Table:
    try:
        return load_table()
    except Exception as exc:  # noqa: BLE001 - a broken table is shown, not a 500 page
        raise HTTPException(500, f"night/pipeline.yaml не читается: {exc}") from exc


def _step_json(table: Table, step_id: str) -> dict[str, Any]:
    step = table.by_id[step_id]
    waves, waves_free = table.layers(), table.layers(("data", "fresh", "exclusive"))
    children = [{"step": s.id, "kind": e.kind, "why": e.why}
                for s in table.steps for e in s.depends_on if e.step == step_id]
    return {
        "id": step.id, "title": step.title, "description": step.description, "kind": step.kind,
        "tokens": step.tokens, "item_kind": step.items, "refresh": step.refresh,
        "depends_on": [{"step": e.step, "kind": e.kind, "why": e.why} for e in step.depends_on],
        "children": children, "wave": waves[step.id], "wave_without_budget": waves_free[step.id],
        "can_run_with": table.can_run_with(step.id),
        "can_run_with_without_budget": table.can_run_with(
            step.id, ("data", "fresh", "exclusive")),
        "restart_set": table.restart_set(step.id),
    }


def _alive() -> bool:
    return control.lock_holder() is not None


def _states(session: Session, run, alive: bool, table: Table) -> dict[str, dict]:
    """Each step's latest row in the run's night. While the run is going, the steps it covers
    that it has not reached yet are `pending`, not what an earlier run of the night said: a page
    that showed yesterday's green under a running night could not tell what had run."""
    out = {}
    for step_id, state in journal.night_states(session, run.night_date).items():
        item = state.to_json()
        if item["state"] == "running" and not alive:
            item["state"] = "interrupted"
        out[step_id] = item
    if alive and run.state == "running":
        covered = set(run.only) if run.only else set(table.ids)
        for step_id in covered:
            if step_id in out and out[step_id]["run_id"] != run.id:
                del out[step_id]
    return out


def register(app: FastAPI) -> None:
    @app.get("/api/pipeline")
    def pipeline(session: SessionDep) -> dict[str, Any]:
        """The step table with each step's state in the latest night, the latest run, whether a
        run is going now, and the statistics the page shows next to each step."""
        table = _table()
        holder = control.lock_holder()
        alive = holder is not None
        body: dict[str, Any] = {
            "steps": [_step_json(table, s) for s in table.ids], "run": None, "states": {},
            "running": alive, "holder": holder, "journal_ready": True,
            "refusal": control.refusal(),
        }
        try:
            run = journal.latest_run(session)
            if run is not None:
                body["run"] = journal.run_json(run, alive=alive)
                body["states"] = _states(session, run, alive, table)
            for step in body["steps"]:
                stats = journal.step_stats(session, step["id"])
                step["stats"] = {k: stats[k] for k in (
                    "runs", "success_share", "avg_s", "median_s", "max_s", "tokens_avg")}
                if not stats["runs"]:
                    approx = journal.step_stats(session, step["id"], approximate=True)
                    step["stats_approximate"] = {k: approx[k] for k in (
                        "runs", "success_share", "avg_s", "median_s", "max_s")}
                last = journal.last_finished(session, step["id"])
                step["last"] = last.to_json() if last else None
        except OperationalError:
            body["journal_ready"] = False
        return body

    @app.get("/api/pipeline/steps/{step_id}")
    def pipeline_step(step_id: str, session: SessionDep) -> dict[str, Any]:
        table = _table()
        if step_id not in table.by_id:
            raise HTTPException(404, f"Нет шага {step_id}")
        body = _step_json(table, step_id)
        body.update(history=[], items=[], stats=None, stats_approximate=None, state=None)
        try:
            alive = _alive()
            run = journal.latest_run(session)
            if run is not None:
                state = _states(session, run, alive, table).get(step_id)
                body["state"] = state
                if state and state["step_run_id"]:
                    body["items"] = journal.step_items(session, state["step_run_id"])
            body["history"] = journal.step_history(session, step_id)
            body["stats"] = journal.step_stats(session, step_id)
            body["stats_approximate"] = journal.step_stats(session, step_id, approximate=True)
            neighbours = {n["step"] for n in body["depends_on"] + body["children"]}
            if run is not None:
                every = _states(session, run, alive, table)
                body["neighbour_states"] = {n: every.get(n, {}).get("state") for n in neighbours}
        except OperationalError:
            body["journal_ready"] = False
        return body

    @app.get("/api/pipeline/runs")
    def pipeline_runs(session: SessionDep, limit: int = 20) -> dict[str, Any]:
        from qlab.registry.models import PipelineRun

        alive = _alive()
        try:
            rows = session.query(PipelineRun).order_by(
                PipelineRun.started_at.desc(), PipelineRun.id.desc()).limit(
                max(1, min(limit, 100))).all()
        except OperationalError:
            return {"runs": [], "journal_ready": False}
        newest = rows[0].id if rows else None
        return {"runs": [journal.run_json(r, alive=alive if r.id == newest else False)
                         for r in rows], "journal_ready": True}

    def _done_today(session: Session) -> set[str]:
        from qlab.registry.models import PipelineRun, PipelineStepRun

        try:
            rows = session.query(PipelineStepRun.step_id).join(
                PipelineRun, PipelineRun.id == PipelineStepRun.run_id).filter(
                PipelineRun.night_date == control.today(), PipelineStepRun.state == "ok")
            return {r[0] for r in rows.distinct()}
        except OperationalError:
            return set()

    def _checked(table: Table, off: list[str], restart: str | None) -> None:
        unknown = (set(off) | ({restart} if restart else set())) - set(table.ids)
        if unknown:
            raise HTTPException(422, f"Нет таких шагов: {', '.join(sorted(unknown))}")

    def _preview(session: Session, table: Table, off: list[str], restart: str | None) -> dict:
        from qlab import night

        skips: dict[str, str] = {}
        try:
            skips = night.schedule_skips(control.today())
        except Exception:  # noqa: BLE001 - the preview without the schedule is still true
            skips = {}
        return control.preview(table, done_today=_done_today(session), off=off, restart=restart,
                               skips=skips)

    @app.post("/api/pipeline/preview", dependencies=[LocalOnly])
    def pipeline_preview(body: PreviewBody, session: SessionDep) -> dict[str, Any]:
        """What a run would do with these steps switched off, or with this step restarted."""
        table = _table()
        _checked(table, body.off, body.restart)
        return {**_preview(session, table, body.off, body.restart), "refusal": control.refusal()}

    @app.post("/api/pipeline/run", dependencies=[LocalOnly])
    def pipeline_run(body: RunBody, session: SessionDep) -> dict[str, Any]:
        """Start the whole night now, with these steps switched off."""
        table = _table()
        _checked(table, body.off, None)
        preview = _preview(session, table, body.off, None)
        started = control.start_detached(control.run_arguments(off=body.off))
        if not started["started"]:
            raise HTTPException(409, started["reason"])
        return {**started, "preview": preview}

    @app.post("/api/pipeline/restart", dependencies=[LocalOnly])
    def pipeline_restart(body: RestartBody, session: SessionDep) -> dict[str, Any]:
        """Restart a step and every step that depends on its result."""
        table = _table()
        _checked(table, body.off, body.step)
        preview = _preview(session, table, body.off, body.step)
        started = control.start_detached(control.run_arguments(off=body.off, restart=body.step))
        if not started["started"]:
            raise HTTPException(409, started["reason"])
        return {**started, "preview": preview}
