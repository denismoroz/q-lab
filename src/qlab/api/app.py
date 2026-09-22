"""Read-only HTTP API over the registry, for the owner's console.

**Read-only by construction**: every route is a GET, and there is no
repository write call anywhere in this package. The registry's integrity
rules (docs/REGISTRY.md) are enforced on the write path — a UI that could
write would be a second, unpoliced way into the tables.

Nothing here computes a verdict, a threshold or a metric. Values are
copied from stored rows; the aggregate pass/fail is fail-closed and lives
in the rules engine, never in a view.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.api.cards import card_summary, load_cards
from qlab.api.deps import get_session
from qlab.api.serializers import (
    idea_detail_json,
    idea_row_json,
    spec_json,
    trial_json,
    verdict_json,
)
from qlab.registry.models import Driver, Idea, Spec, Trial, Verdict
from qlab.registry.queries import funnel_stats

# The dev frontend runs on Vite's default port and proxies /api, so CORS is
# normally never exercised; it is allowed here only so opening the API
# directly from a dev page fails loudly on its own terms rather than as an
# opaque CORS error.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

MAX_PAGE = 200

# Declared as Annotated aliases rather than call-valued defaults: FastAPI
# reads them identically, and the signatures stay readable when every
# paginated route repeats the same three parameters.
SessionDep = Annotated[Session, Depends(get_session)]
LimitParam = Annotated[int, Query(ge=1, le=MAX_PAGE)]
OffsetParam = Annotated[int, Query(ge=0)]


def _page(session: Session, query: Any, limit: int, offset: int, to_json: Any) -> dict[str, Any]:
    """Run a paginated query and return `{total, limit, offset, items}`.

    Pagination is not optional on this API: `trial` alone holds thousands
    of rows, and a single idea can carry thousands of verdicts. Nothing is
    ever returned unbounded.
    """
    # `Query.count()` wraps the query in a subquery, which keeps the FROM and
    # any joins/filters intact; selecting a bare `func.count()` off the query
    # drops the FROM clause and silently answers 1.
    total = query.order_by(None).count()
    rows = query.limit(limit).offset(offset).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [to_json(row) for row in rows],
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="q-lab registry console",
        description="Read-only views over the q-lab registry (docs/REGISTRY.md).",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # 1. Funnel
    # ------------------------------------------------------------------

    @app.get("/api/funnel")
    def funnel(session: SessionDep) -> dict[str, Any]:
        """The counts `queries.funnel_stats` computes — no extra aggregation."""
        stats = funnel_stats(session)
        return {
            "ideas_by_status": stats.ideas_by_status,
            "verdicts_by_stage": stats.verdicts_by_stage,
            "verdicts_by_outcome": stats.verdicts_by_outcome,
            "decayed_by_shutdown_cause": stats.decayed_by_shutdown_cause,
        }

    # ------------------------------------------------------------------
    # 2. Registry and graveyard
    # ------------------------------------------------------------------

    @app.get("/api/ideas")
    def list_ideas(session: SessionDep) -> dict[str, Any]:
        rows = (
            session.query(Idea, Driver)
            .outerjoin(Driver, Idea.driver_id == Driver.id)
            .order_by(Idea.status, Idea.id)
            .all()
        )
        return {"items": [idea_row_json(idea, driver) for idea, driver in rows]}

    def _load_idea(session: Session, idea_id: str) -> Idea:
        idea = session.get(Idea, idea_id)
        if idea is None:
            raise HTTPException(status_code=404, detail=f"idea not found: {idea_id}")
        return idea

    @app.get("/api/ideas/{idea_id}")
    def get_idea(idea_id: str, session: SessionDep) -> dict[str, Any]:
        idea = _load_idea(session, idea_id)
        driver = session.get(Driver, idea.driver_id) if idea.driver_id else None

        spec_ids = [row[0] for row in session.query(Spec.id).filter(Spec.idea_id == idea_id)]
        trial_count = (
            session.query(func.count(Trial.id)).filter(Trial.spec_id.in_(spec_ids)).scalar()
            if spec_ids
            else 0
        )
        # Counted by the same three-kind split docs/REGISTRY.md defines, so
        # the header can never imply an idea has more decisions than it does.
        verdicts = session.query(Verdict).filter(Verdict.idea_id == idea_id)
        counts = {
            "specs": len(spec_ids),
            "trials": trial_count,
            "verdicts": verdicts.count(),
            "verdicts_decision": verdicts.filter(Verdict.passed.isnot(None)).count(),
            "verdicts_unknown": verdicts.filter(
                Verdict.passed.is_(None), Verdict.value.is_(None)
            ).count(),
            "verdicts_measurement": verdicts.filter(
                Verdict.passed.is_(None), Verdict.value.isnot(None)
            ).count(),
        }
        detail = idea_detail_json(idea, driver)
        detail["counts"] = counts
        return detail

    @app.get("/api/ideas/{idea_id}/specs")
    def idea_specs(
        idea_id: str,
        session: SessionDep,
        limit: LimitParam = 50,
        offset: OffsetParam = 0,
    ) -> dict[str, Any]:
        _load_idea(session, idea_id)
        query = (
            session.query(Spec)
            .filter(Spec.idea_id == idea_id)
            .order_by(Spec.version.desc(), Spec.id.desc())
        )
        return _page(session, query, limit, offset, spec_json)

    @app.get("/api/ideas/{idea_id}/verdicts")
    def idea_verdicts(
        idea_id: str,
        session: SessionDep,
        limit: LimitParam = 50,
        offset: OffsetParam = 0,
    ) -> dict[str, Any]:
        _load_idea(session, idea_id)
        query = (
            session.query(Verdict)
            .filter(Verdict.idea_id == idea_id)
            .order_by(Verdict.decided_at.desc(), Verdict.id.desc())
        )
        return _page(session, query, limit, offset, verdict_json)

    @app.get("/api/ideas/{idea_id}/trials")
    def idea_trials(
        idea_id: str,
        session: SessionDep,
        limit: LimitParam = 50,
        offset: OffsetParam = 0,
    ) -> dict[str, Any]:
        _load_idea(session, idea_id)
        query = (
            session.query(Trial)
            .join(Spec, Trial.spec_id == Spec.id)
            .filter(Spec.idea_id == idea_id)
            .order_by(Trial.started_at.desc(), Trial.id.desc())
        )
        page = _page(session, query, limit, offset, lambda trial: trial)
        specs = {
            spec.id: spec
            for spec in session.query(Spec).filter(Spec.idea_id == idea_id).all()
        }
        page["items"] = [
            trial_json(
                trial,
                idea_id=idea_id,
                code_ref=specs[trial.spec_id].code_ref if trial.spec_id in specs else None,
                spec_version=specs[trial.spec_id].version if trial.spec_id in specs else None,
            )
            for trial in page["items"]
        ]
        return page

    # ------------------------------------------------------------------
    # 3. Candidate cards
    # ------------------------------------------------------------------

    @app.get("/api/cards")
    def list_cards() -> dict[str, Any]:
        return {"items": [card_summary(card) for card in load_cards()]}

    @app.get("/api/cards/{idea_id}")
    def get_card(idea_id: str) -> dict[str, Any]:
        for card in load_cards():
            if card["idea_id"] == idea_id:
                return card
        raise HTTPException(status_code=404, detail=f"card not found: {idea_id}")

    # ------------------------------------------------------------------
    # 4. Trials ledger
    # ------------------------------------------------------------------

    @app.get("/api/trials")
    def list_trials(
        session: SessionDep,
        limit: LimitParam = 50,
        offset: OffsetParam = 0,
    ) -> dict[str, Any]:
        """The ledger, newest first. Always paginated on the server."""
        query = session.query(Trial).order_by(Trial.started_at.desc(), Trial.id.desc())
        page = _page(session, query, limit, offset, lambda trial: trial)

        trials = page["items"]
        spec_ids = {trial.spec_id for trial in trials}
        rows = (
            session.query(Spec, Idea)
            .join(Idea, Spec.idea_id == Idea.id)
            .filter(Spec.id.in_(spec_ids))
            .all()
            if spec_ids
            else []
        )
        routing = {spec.id: (spec, idea) for spec, idea in rows}
        page["items"] = [
            trial_json(
                trial,
                idea_id=routing[trial.spec_id][1].id if trial.spec_id in routing else None,
                idea_title=routing[trial.spec_id][1].title if trial.spec_id in routing else None,
                code_ref=routing[trial.spec_id][0].code_ref if trial.spec_id in routing else None,
                spec_version=(
                    routing[trial.spec_id][0].version if trial.spec_id in routing else None
                ),
            )
            for trial in trials
        ]
        return page

    return app


app = create_app()

__all__ = ["app", "create_app"]
