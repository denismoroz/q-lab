"""SQLAlchemy 2.0 (synchronous) ORM models for the q-lab registry.

Schema contract: docs/REGISTRY.md. This module only defines the tables — no
business logic, no verdict computation. See registry/repo.py for the thin
data-access layer and registry/db.py for engine/session setup.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for every registry table."""


def _enum_column(enum_cls: type[enum.Enum], **kwargs: object) -> SqlEnum:
    """Build a sqlalchemy.Enum that stores the Python Enum's *value*.

    Several enums here use hyphenated values (e.g. ``"venue-event"``,
    ``"crypto-perp"``) that cannot be Python identifiers, so the member
    name and value differ. SQLAlchemy's Enum type stores ``.name`` by
    default, which would put the wrong string in the DB — this forces it
    to store ``.value`` instead, matching docs/REGISTRY.md verbatim.
    """
    return SqlEnum(
        enum_cls,
        values_callable=lambda cls: [member.value for member in cls],
        **kwargs,
    )


# --------------------------------------------------------------------------
# Enums (values match docs/REGISTRY.md exactly)
# --------------------------------------------------------------------------


class SourceType(enum.StrEnum):
    PAPER = "paper"
    GITHUB = "github"
    FORUM = "forum"
    VENUE_EVENT = "venue-event"
    GRAVEYARD = "graveyard"
    INTERNAL = "internal"


class AssetClass(enum.StrEnum):
    CRYPTO_PERP = "crypto-perp"
    CRYPTO_SPOT = "crypto-spot"
    DEFI = "defi"
    FX = "fx"


class Profile(enum.StrEnum):
    CARRY = "carry"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean-reversion"
    ARB = "arb"
    OTHER = "other"


class IdeaStatus(enum.StrEnum):
    CANDIDATE = "candidate"
    SPECCING = "speccing"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    BENCH = "bench"
    LIVE = "live"
    REJECTED = "rejected"
    RETIRED = "retired"


class TrialStatus(enum.StrEnum):
    OK = "ok"
    ERROR = "error"


class TrialSource(enum.StrEnum):
    QLAB = "qlab"
    IMPORTED = "imported"


class VerdictStage(enum.StrEnum):
    PREFLIGHT = "preflight"
    EDGE = "edge"
    TAIL = "tail"
    PROFILE = "profile"
    CAPACITY = "capacity"
    CORRELATION = "correlation"


class RevivalOutcome(enum.StrEnum):
    REVIVED = "revived"
    STILL_DEAD = "still-dead"
    PENDING = "pending"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class Driver(Base):
    """`driver` — what the edge feeds on."""

    __tablename__ = "driver"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    kill_condition: Mapped[str] = mapped_column(String, nullable=False)
    observable: Mapped[str] = mapped_column(String, nullable=False)


class Idea(Base):
    """`idea` — a candidate strategy."""

    __tablename__ = "idea"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[SourceType] = mapped_column(_enum_column(SourceType), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    claimed_edge: Mapped[str | None] = mapped_column(String, nullable=True)
    asset_class: Mapped[AssetClass] = mapped_column(_enum_column(AssetClass), nullable=False)
    driver_id: Mapped[str | None] = mapped_column(
        ForeignKey("driver.id"), nullable=True, index=True
    )
    profile: Mapped[Profile] = mapped_column(_enum_column(Profile), nullable=False)
    status: Mapped[IdeaStatus] = mapped_column(
        _enum_column(IdeaStatus), nullable=False, default=IdeaStatus.CANDIDATE, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


class Spec(Base):
    """`spec` — formalization of an idea. Unique on (idea_id, version)."""

    __tablename__ = "spec"
    __table_args__ = (UniqueConstraint("idea_id", "version", name="uq_spec_idea_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("idea.id"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    data_requirements: Mapped[dict] = mapped_column(JSON, nullable=False)
    rebalance: Mapped[str] = mapped_column(String, nullable=False)
    costs_model: Mapped[dict] = mapped_column(JSON, nullable=False)
    code_ref: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DataSnapshot(Base):
    """`data_snapshot` — reproducibility anchor for a trial's input data."""

    __tablename__ = "data_snapshot"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # sha256 of contents
    source: Mapped[str] = mapped_column(String, nullable=False)
    instruments: Mapped[dict] = mapped_column(JSON, nullable=False)
    range_start: Mapped[date] = mapped_column(Date, nullable=False)
    range_end: Mapped[date] = mapped_column(Date, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)


class Trial(Base):
    """`trial` — a single run. Append-only: every run is written, kept or not."""

    __tablename__ = "trial"
    __table_args__ = (
        CheckConstraint(
            "snapshot_id IS NOT NULL OR source = 'imported'",
            name="ck_trial_snapshot_required_unless_imported",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spec_id: Mapped[int] = mapped_column(ForeignKey("spec.id"), nullable=False, index=True)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_snapshot.id"), nullable=True, index=True
    )
    code_sha: Mapped[str] = mapped_column(String, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[TrialStatus] = mapped_column(
        _enum_column(TrialStatus), nullable=False, index=True
    )
    kept: Mapped[bool] = mapped_column(nullable=False, default=True)
    token_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cpu_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[TrialSource] = mapped_column(
        _enum_column(TrialSource), nullable=False, default=TrialSource.QLAB
    )


class Verdict(Base):
    """`verdict` — a stage decision, rendered by the rules engine (never by a model)."""

    __tablename__ = "verdict"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("idea.id"), nullable=False, index=True)
    spec_id: Mapped[int | None] = mapped_column(ForeignKey("spec.id"), nullable=True, index=True)
    trial_id: Mapped[int | None] = mapped_column(
        ForeignKey("trial.id"), nullable=True, index=True
    )
    stage: Mapped[VerdictStage] = mapped_column(_enum_column(VerdictStage), nullable=False)
    rule_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    rules_version: Mapped[str] = mapped_column(String, nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    comparator: Mapped[str] = mapped_column(String, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(nullable=False, index=True)
    data_range_start: Mapped[date] = mapped_column(Date, nullable=False)
    data_range_end: Mapped[date] = mapped_column(Date, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)


class RevivalCheck(Base):
    """`revival_check` — graveyard re-maturation tracking."""

    __tablename__ = "revival_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("idea.id"), nullable=False, index=True)
    verdict_id: Mapped[int] = mapped_column(ForeignKey("verdict.id"), nullable=False, index=True)
    rejected_data_end: Mapped[date] = mapped_column(Date, nullable=False)
    eligible_at: Mapped[date] = mapped_column(Date, nullable=False)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[RevivalOutcome] = mapped_column(
        _enum_column(RevivalOutcome), nullable=False, default=RevivalOutcome.PENDING, index=True
    )


class StageTransition(Base):
    """`stage_transition` — funnel ledger; the source of truth for funnel counts."""

    __tablename__ = "stage_transition"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("idea.id"), nullable=False, index=True)
    from_status: Mapped[IdeaStatus | None] = mapped_column(_enum_column(IdeaStatus), nullable=True)
    to_status: Mapped[IdeaStatus] = mapped_column(
        _enum_column(IdeaStatus), nullable=False, index=True
    )
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    rules_version: Mapped[str | None] = mapped_column(String, nullable=True, index=True)


class TokenSpend(Base):
    """`token_spend` — budget ledger for the "price of one survivor" metric."""

    __tablename__ = "token_spend"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    idea_id: Mapped[str | None] = mapped_column(ForeignKey("idea.id"), nullable=True, index=True)
    agent: Mapped[str] = mapped_column(String, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False)
    usd_est: Mapped[float] = mapped_column(Float, nullable=False)


class Dossier(Base):
    """`dossier` — rendered artifact pointer for an idea."""

    __tablename__ = "dossier"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("idea.id"), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    rendered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rules_version: Mapped[str] = mapped_column(String, nullable=False, index=True)
