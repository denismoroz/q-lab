"""SQLAlchemy 2.0 (synchronous) ORM models for the q-lab registry.

Schema contract: docs/REGISTRY.md. This module only defines the tables — no
business logic, no verdict computation. See registry/repo.py for the thin
data-access layer and registry/db.py for engine/session setup.
"""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime

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


def _utcnow() -> datetime:
    """Timestamp default: rows must never be writable without one."""
    return datetime.now(UTC)


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
    """Idea lifecycle: candidate -> speccing -> implemented -> validated ->
    bench -> paper -> live, with terminal states rejected / decayed / retired.

    BENCH vs PAPER is a distinction worth a dedicated state, not a note in
    `notes`: BENCH means validated and idle — it cleared the offline checks
    but nothing is currently running. PAPER means running forward, on live
    data, with no capital at risk — it is the only stage that produces
    out-of-sample evidence that cannot be back-fitted, because the data
    arriving in paper did not exist when the strategy was built. Collapsing
    the two into one status (as the original graveyard seed did, calling
    paper-trading strategies "bench") erases exactly that distinction: a
    strategy sitting idle and a strategy accumulating forward evidence are
    in fundamentally different places on the road to LIVE.
    """

    CANDIDATE = "candidate"
    SPECCING = "speccing"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    BENCH = "bench"
    PAPER = "paper"
    LIVE = "live"
    # Terminal states, and the difference between them is the point: REJECTED
    # never traded, DECAYED did and stopped working, RETIRED was withdrawn for
    # a reason other than decay (superseded, venue gone, owner's choice).
    # Only DECAYED carries a survival time, which is what answers "how long
    # does a working strategy last" — the other half of the ceiling question.
    REJECTED = "rejected"
    DECAYED = "decayed"
    RETIRED = "retired"


class ShutdownCause(enum.StrEnum):
    """Why a DECAYED idea was actually shut off — distinct from the fact
    that it decayed at all.

    This distinction carries the project's weight (docs/PLAN.md: "how long
    does a working strategy last, and does that price rise" is the whole
    point of q-lab). Only `EDGE_DECAYED` — a real edge that worked and then
    stopped — contributes to that survival-time statistic. `FALSE_DISCOVERY`
    means the screening itself was wrong: the idea was never actually
    profitable, and whatever validation let it through production was a
    measurement or selection error (e.g. XSMOM, selected on decorrelation
    with FRAB alone, called profitable, and only shown never to have been
    profitable once a real loss forced a re-evaluation). A false discovery
    that reached production says nothing about the market changing — mixing
    it into "how long strategies survive" corrupts the one number the owner
    actually needs to plan replacement capacity around. `EXECUTION` covers
    a shutdown caused by infrastructure/venue/implementation problems, not
    the edge itself. `OWNER_CHOICE` covers a deliberate shutdown for reasons
    unrelated to performance (capital reallocation, venue exit, etc.).
    """

    EDGE_DECAYED = "edge-decayed"
    FALSE_DISCOVERY = "false-discovery"
    EXECUTION = "execution"
    OWNER_CHOICE = "owner-choice"


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
    """`idea` — a candidate strategy.

    `shutdown_cause` is required whenever `status = 'decayed'` (see the
    CheckConstraint below): a death cannot be recorded without a cause. The
    cause matters more than it looks — see `ShutdownCause`'s docstring —
    because only `edge-decayed` deaths contribute to the "how long does a
    working strategy last" statistic that the whole project exists to
    produce (docs/PLAN.md). A `false-discovery` idea that reached
    production says the screening was wrong, not that the market changed;
    counting it as a decayed edge would corrupt that one number.
    """

    __tablename__ = "idea"
    __table_args__ = (
        CheckConstraint(
            "status != 'decayed' OR shutdown_cause IS NOT NULL",
            name="ck_idea_decayed_requires_shutdown_cause",
        ),
    )

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
    shutdown_cause: Mapped[ShutdownCause | None] = mapped_column(
        _enum_column(ShutdownCause), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


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
    """`verdict` — a stage decision, rendered by the rules engine (never by a model).

    There are exactly three legitimate kinds of row, and mixing them up
    means fabricated thresholds silently entering the registry:

    | kind        | `value` | comparator/threshold | `passed` | when                     |
    |-------------|---------|-----------------------|----------|--------------------------|
    | decision    | set     | set                   | set      | rule applied to a metric |
    | unknown     | null    | set                   | null     | rule exists, no metric   |
    | measurement | set     | **null**              | null     | eyeballed number, no rule |

    **Measurement** rows exist for old dossiers full of numbers ("Sharpe
    0.77") that a human judged by eye, with no formalized rule behind them.
    Recording such a row as a decision would require inventing a threshold —
    and that invented threshold would later surface in `near_threshold()` as
    a real near-miss, and could steer real capital. So a measurement is
    stored with no comparator/threshold, conventionally `rule_id =
    "unidentified"`, and never participates in screening (`passed` is
    always null for it). q-lab itself always applies a real rule, so a null
    comparator/threshold is only legal for `source == "imported"`. The four
    CheckConstraints below encode exactly the three rows above and nothing
    else — see each constraint's name for which invariant it is.

    `data_range_start`/`data_range_end` may be null only for `source ==
    "imported"` rows (see `ck_verdict_data_range_required_unless_imported`),
    for historical verdicts pulled from a graveyard document that did not
    record a data range. **A verdict with an unknown data range cannot take
    part in automatic graveyard revival**: `ripe_for_revival` needs the
    rejection's data end date to tell which data is actually new, so such a
    row can only ever be read, never used to schedule a `revival_check`.

    Historical verdicts also arrive with `rules_version = "frab-legacy"` —
    a sentinel meaning "whatever criteria frab used pre-q-lab", not a real
    `YYYY-MM-DD.N` rules version. Don't parse `rules_version` assuming that
    format without checking for this sentinel first.
    """

    __tablename__ = "verdict"
    __table_args__ = (
        CheckConstraint(
            "passed IS NULL OR "
            "(value IS NOT NULL AND comparator IS NOT NULL AND threshold IS NOT NULL)",
            name="ck_verdict_decision_requires_value_comparator_threshold",
        ),
        CheckConstraint(
            "(comparator IS NULL AND threshold IS NULL) "
            "OR (comparator IS NOT NULL AND threshold IS NOT NULL)",
            name="ck_verdict_comparator_threshold_together",
        ),
        CheckConstraint(
            "comparator IS NOT NULL OR source = 'imported'",
            name="ck_verdict_measurement_requires_imported",
        ),
        CheckConstraint(
            "value IS NOT NULL OR passed IS NULL",
            name="ck_verdict_unevaluated_metric_has_no_decision",
        ),
        CheckConstraint(
            "(data_range_start IS NOT NULL AND data_range_end IS NOT NULL) "
            "OR source = 'imported'",
            name="ck_verdict_data_range_required_unless_imported",
        ),
    )

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
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    comparator: Mapped[str | None] = mapped_column(String, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(nullable=True, index=True)
    data_range_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    data_range_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[TrialSource] = mapped_column(
        _enum_column(TrialSource), nullable=False, default=TrialSource.QLAB
    )


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
