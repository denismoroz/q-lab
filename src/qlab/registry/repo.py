"""Thin data-access layer over the registry models.

No business logic and no verdict computation lives here: verdicts arrive
fully formed from the rules engine (src/qlab/rules) and are just persisted.

Every function takes a live ``Session`` and leaves committing to the caller
(typically via ``db.session_scope()``), so several repo calls can be composed
into one atomic transaction — which is exactly what ``set_status`` relies on
internally to write the idea's new status and its ``stage_transition`` row
together.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from qlab.registry.models import (
    AssetClass,
    DataSnapshot,
    Driver,
    Idea,
    IdeaStatus,
    Profile,
    ShutdownCause,
    SourceType,
    Spec,
    StageTransition,
    Trial,
    TrialSource,
    TrialStatus,
    Verdict,
)


def _now() -> datetime:
    return datetime.now(UTC)


def upsert_driver(
    session: Session,
    *,
    id: str,
    title: str,
    description: str,
    kill_condition: str,
    observable: str,
) -> Driver:
    """Insert or update a `driver` row by its slug id."""
    driver = session.get(Driver, id)
    if driver is None:
        driver = Driver(
            id=id,
            title=title,
            description=description,
            kill_condition=kill_condition,
            observable=observable,
        )
        session.add(driver)
    else:
        driver.title = title
        driver.description = description
        driver.kill_condition = kill_condition
        driver.observable = observable
    session.flush()
    return driver


def upsert_idea(
    session: Session,
    *,
    id: str,
    title: str,
    source_type: SourceType,
    asset_class: AssetClass,
    profile: Profile,
    source_url: str | None = None,
    claimed_edge: str | None = None,
    driver_id: str | None = None,
    status: IdeaStatus = IdeaStatus.CANDIDATE,
    notes: str | None = None,
    shutdown_cause: ShutdownCause | None = None,
) -> Idea:
    """Insert or update an `idea` row by its slug id.

    This does not touch `stage_transition` — use `set_status` for status
    changes on an idea that already exists, so the funnel ledger stays
    authoritative. This function is for creating an idea or editing its
    non-status metadata. `shutdown_cause` is only meaningful for
    `status=DECAYED` (see `ck_idea_decayed_requires_shutdown_cause`); the
    caller is responsible for supplying one whenever it sets that status.
    """
    idea = session.get(Idea, id)
    now = _now()
    if idea is None:
        idea = Idea(
            id=id,
            title=title,
            source_type=source_type,
            source_url=source_url,
            claimed_edge=claimed_edge,
            asset_class=asset_class,
            driver_id=driver_id,
            profile=profile,
            status=status,
            shutdown_cause=shutdown_cause,
            created_at=now,
            updated_at=now,
            notes=notes,
        )
        session.add(idea)
    else:
        idea.title = title
        idea.source_type = source_type
        idea.source_url = source_url
        idea.claimed_edge = claimed_edge
        idea.asset_class = asset_class
        idea.driver_id = driver_id
        idea.profile = profile
        idea.notes = notes
        idea.shutdown_cause = shutdown_cause
        idea.updated_at = now
    session.flush()
    return idea


def add_spec(
    session: Session,
    *,
    idea_id: str,
    version: int,
    params: Mapping,
    data_requirements: Mapping,
    rebalance: str,
    costs_model: Mapping,
    code_ref: str,
) -> Spec:
    """Append a new `spec` version for an idea. (idea_id, version) is unique."""
    spec = Spec(
        idea_id=idea_id,
        version=version,
        params=dict(params),
        data_requirements=dict(data_requirements),
        rebalance=rebalance,
        costs_model=dict(costs_model),
        code_ref=code_ref,
        created_at=_now(),
    )
    session.add(spec)
    session.flush()
    return spec


def add_data_snapshot(
    session: Session,
    *,
    id: str,
    source: str,
    instruments: Mapping,
    range_start: date,
    range_end: date,
    path: str,
    rows: int,
    fetched_at: datetime | None = None,
) -> DataSnapshot:
    """Record a `data_snapshot`. Not required by T2's test matrix but needed
    by `add_trial` callers that want a snapshot to reference."""
    snapshot = DataSnapshot(
        id=id,
        source=source,
        instruments=dict(instruments),
        range_start=range_start,
        range_end=range_end,
        fetched_at=fetched_at or _now(),
        path=path,
        rows=rows,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def add_trial(
    session: Session,
    *,
    spec_id: int,
    config_hash: str,
    code_sha: str,
    params: Mapping,
    status: TrialStatus,
    snapshot_id: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    metrics: Mapping | None = None,
    kept: bool = True,
    token_cost: int | None = None,
    cpu_seconds: float | None = None,
    source: TrialSource = TrialSource.QLAB,
) -> Trial:
    """Append a `trial` row. Trial is append-only — this never updates an
    existing row; every run, including discarded ones, gets its own row."""
    trial = Trial(
        spec_id=spec_id,
        config_hash=config_hash,
        snapshot_id=snapshot_id,
        code_sha=code_sha,
        params=dict(params),
        started_at=started_at or _now(),
        finished_at=finished_at,
        metrics=dict(metrics) if metrics is not None else None,
        status=status,
        kept=kept,
        token_cost=token_cost,
        cpu_seconds=cpu_seconds,
        source=source,
    )
    session.add(trial)
    session.flush()
    return trial


def add_verdicts(session: Session, verdicts: Iterable[Mapping]) -> list[Verdict]:
    """Batch-insert already-decided `verdict` rows.

    Each item is a mapping of Verdict column names to values (idea_id,
    stage, rule_id, rules_version, metric, value, comparator, threshold,
    passed, data_range_start, data_range_end, ...). The rules engine is the
    only producer of these rows — this function just persists them.
    """
    rows = [Verdict(**dict(v)) for v in verdicts]
    for row in rows:
        if row.decided_at is None:
            row.decided_at = _now()
    session.add_all(rows)
    session.flush()
    return rows


def set_status(
    session: Session,
    *,
    idea_id: str,
    new_status: IdeaStatus,
    reason: str | None = None,
    rules_version: str | None = None,
) -> Idea:
    """Move an idea to `new_status`, atomically logging a `stage_transition`.

    Both writes happen against the same Session/transaction: if the caller
    rolls back (e.g. an exception before commit), neither the idea's status
    nor the transition row are persisted. Raises ValueError if the idea does
    not exist.
    """
    idea = session.get(Idea, idea_id)
    if idea is None:
        raise ValueError(f"idea {idea_id!r} does not exist")

    from_status = idea.status
    now = _now()

    idea.status = new_status
    idea.updated_at = now

    transition = StageTransition(
        idea_id=idea_id,
        from_status=from_status,
        to_status=new_status,
        at=now,
        reason=reason,
        rules_version=rules_version,
    )
    session.add(transition)
    session.flush()
    return idea


__all__ = [
    "upsert_driver",
    "upsert_idea",
    "add_spec",
    "add_data_snapshot",
    "add_trial",
    "add_verdicts",
    "set_status",
]
