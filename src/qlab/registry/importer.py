"""Importer: load an extracted graveyard YAML file into the registry.

The input shape is documented in ``docs/REGISTRY.md`` and the T5 task spec
(``seed/graveyard.yaml``). This module never reads or writes anything under
``seed/`` itself — the caller passes a path (typically that default) and
this module treats its contents as arbitrary, untrusted-but-structured
input to validate and persist.

Two very different kinds of "bad data" are handled differently:

- **Structural** problems (wrong types, unknown enum values, unexpected
  fields, missing required fields) fail the *whole* import loudly, via
  :class:`GraveyardImportError`, before a single row is written. Writing
  half a graveyard would be worse than writing none of it.
- A single verdict that doesn't fit one of the three legitimate row kinds
  from docs/REGISTRY.md ("decision" / "unknown" / "measurement" — see
  :func:`_classify_verdict`) is not a structural problem — pydantic parses
  it fine — but it violates the ``verdict`` table's invariants. That is a
  data-quality issue in the *source* document, so it is reported in
  :class:`ImportReport`.data_errors and only that one verdict is skipped;
  the rest of the file still imports.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from sqlalchemy.orm import Session

from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Driver,
    Idea,
    IdeaStatus,
    Profile,
    ShutdownCause,
    SourceType,
    TrialSource,
    Verdict,
    VerdictStage,
)
from qlab.rules import LEGACY_SENTINEL_VERSION

# --------------------------------------------------------------------------
# Input schema (pydantic) — mirrors docs/REGISTRY.md / the T5 task spec.
#
# `extra="forbid"` is deliberate: an unexpected field is far more likely to
# be a typo or drift in the extractor that produced the file than
# something safe to silently ignore, and this importer must fail loudly on
# malformed input rather than write partial data.
# --------------------------------------------------------------------------


class DriverIn(BaseModel):
    """One `drivers[]` entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    kill_condition: str
    observable: str


class VerdictIn(BaseModel):
    """One `ideas[].verdicts[]` entry.

    `value`/`comparator`/`threshold`/`passed` are independently optional
    here — pydantic does not enforce docs/REGISTRY.md's "three legitimate
    kinds of verdict row" invariant (decision / unknown / measurement).
    That check is a data-quality concern handled in
    :func:`import_graveyard` via `_classify_verdict`, not a
    structural-schema concern (see module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    stage: VerdictStage
    metric: str
    value: float | None = None
    comparator: str | None = None
    threshold: float | None = None
    passed: bool | None = None
    data_range_start: date | None = None
    data_range_end: date | None = None
    decided_at: date
    rule_id: str | None = None
    source_file: str
    source_quote: str


class IdeaIn(BaseModel):
    """One `ideas[]` entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    source_type: SourceType
    source_url: str | None = None
    claimed_edge: str | None = None
    asset_class: AssetClass
    driver_id: str | None = None
    profile: Profile
    status: IdeaStatus
    shutdown_cause: ShutdownCause | None = None
    notes: str | None = None
    verdicts: list[VerdictIn] = []

    @model_validator(mode="after")
    def _decayed_requires_shutdown_cause(self) -> IdeaIn:
        """Mirrors `ck_idea_decayed_requires_shutdown_cause`: a death cannot
        be recorded without a cause, and this must fail loudly and by name
        here rather than surface later as an opaque IntegrityError from the
        database."""
        if self.status == IdeaStatus.DECAYED and self.shutdown_cause is None:
            raise ValueError(
                f"idea {self.id!r} has status 'decayed' but no shutdown_cause — "
                "a death cannot be recorded without a cause"
            )
        return self


class GraveyardFile(BaseModel):
    """Top-level shape of `seed/graveyard.yaml`."""

    model_config = ConfigDict(extra="forbid")

    drivers: list[DriverIn] = []
    ideas: list[IdeaIn] = []


UNIDENTIFIED_RULE_ID = "unidentified"
"""Stored in `verdict.rule_id` when the source document's `rule_id` is
null, i.e. "the rule behind this rejection could not be identified". The
column is NOT NULL, so a literal sentinel string is used instead of null;
`ImportReport.verdicts_unidentified_rule` counts how often this happens —
a large count is itself a finding about how undocumented the old
screening process was.

Every **measurement** row (see `_classify_verdict`) has `rule_id ==
UNIDENTIFIED_RULE_ID` by definition, per docs/REGISTRY.md: a number with
no `comparator`/`threshold` behind it is, by construction, a number no
rule was ever identified for.
"""

VerdictKind = str  # "decision" | "unknown" | "measurement" — see _classify_verdict


def _classify_verdict(
    *,
    value: float | None,
    passed: bool | None,
    comparator: str | None,
    threshold: float | None,
) -> tuple[VerdictKind | None, str | None]:
    """Classify one verdict row into the three legitimate kinds from
    docs/REGISTRY.md ("Три законных вида строки в verdict"), or explain why
    it's none of them.

    - **decision**: `value`, `comparator`, `threshold`, `passed` all set —
      a rule was applied to a computed metric.
    - **unknown**: `value`/`passed` both null, `comparator`/`threshold`
      set — a rule exists but the metric wasn't computed.
    - **measurement**: `value` set, `comparator`/`threshold`/`passed` all
      null — a number the old graveyard recorded with no formal criterion
      behind it (a human decided, not code). Never fabricate a threshold
      to force this into a "decision": a made-up threshold would later
      surface in `near_threshold` as a fake near-miss.

    Returns `(kind, None)` when the row is one of the three, or
    `(None, reason)` when it is none of them (a genuine data error in the
    source document).
    """
    if (comparator is None) != (threshold is None):
        return None, "comparator and threshold must be null together or set together"

    has_criterion = comparator is not None
    if has_criterion:
        if (value is None) != (passed is None):
            return None, "value and passed must be null together or set together"
        return ("unknown" if value is None else "decision"), None

    # No criterion at all -> this can only be a measurement.
    if passed is not None:
        return None, "passed cannot be set without a comparator/threshold to decide against"
    if value is None:
        return None, "row has neither a value nor a comparator/threshold — nothing to record"
    return "measurement", None


class GraveyardImportError(ValueError):
    """The graveyard file failed structural validation.

    Raised before any database write. The message lists every offending
    idea/driver, index and field so a human can fix the source file
    without having to read a raw pydantic traceback.
    """


def _safe_get(raw: Any, path: list[Any]) -> Any:
    cur = raw
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def _describe_loc(loc: tuple[Any, ...], raw: Mapping[str, Any]) -> str:
    """Turn a pydantic error `loc` tuple into a human-readable location.

    E.g. `("ideas", 2, "verdicts", 1, "value")` becomes
    `"idea 'cross-exchange-spread' verdict[1] field 'value'"` when the raw
    dict has an id at that position, falling back to a positional index
    when it doesn't (e.g. the id field itself is what's wrong).
    """
    if not loc:
        return "<root>"
    section = loc[0]
    if section == "ideas" and len(loc) >= 2:
        idx = loc[1]
        idea_id = _safe_get(raw, ["ideas", idx, "id"]) or f"#{idx}"
        rest = loc[2:]
        if len(rest) >= 2 and rest[0] == "verdicts":
            v_idx = rest[1]
            field_part = f" field {rest[2]!r}" if len(rest) > 2 else ""
            return f"idea {idea_id!r} verdict[{v_idx}]{field_part}"
        field_part = f" field {rest[0]!r}" if rest else ""
        return f"idea {idea_id!r}{field_part}"
    if section == "drivers" and len(loc) >= 2:
        idx = loc[1]
        driver_id = _safe_get(raw, ["drivers", idx, "id"]) or f"#{idx}"
        field_part = f" field {loc[2]!r}" if len(loc) > 2 else ""
        return f"driver {driver_id!r}{field_part}"
    return ".".join(str(part) for part in loc)


def parse_graveyard(raw: Mapping[str, Any]) -> GraveyardFile:
    """Validate an already-YAML-parsed mapping against the graveyard schema.

    Raises `GraveyardImportError` with one line per problem, naming the
    offending idea/driver id where possible, instead of surfacing a raw
    pydantic `ValidationError`.
    """
    try:
        return GraveyardFile.model_validate(raw)
    except ValidationError as exc:
        lines = ["graveyard file failed validation:"]
        for err in exc.errors():
            lines.append(f"  - {_describe_loc(err['loc'], raw)}: {err['msg']}")
        raise GraveyardImportError("\n".join(lines)) from exc


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DataError:
    """A single verdict that failed a semantic (not structural) check."""

    idea_id: str
    verdict_index: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"idea {self.idea_id!r} verdict[{self.verdict_index}]: {self.message}"


@dataclass(slots=True)
class ImportReport:
    """Summary of one `import_graveyard` call."""

    drivers_inserted: int = 0
    drivers_updated: int = 0
    ideas_inserted: int = 0
    ideas_updated: int = 0
    verdicts_inserted: int = 0
    verdicts_skipped_duplicate: int = 0
    verdicts_null_value: int = 0
    verdicts_unidentified_rule: int = 0
    verdicts_measurement: int = 0
    data_errors: list[DataError] = field(default_factory=list)

    @property
    def verdicts_skipped_data_error(self) -> int:
        return len(self.data_errors)

    @property
    def verdicts_skipped_total(self) -> int:
        return self.verdicts_skipped_duplicate + self.verdicts_skipped_data_error


# --------------------------------------------------------------------------
# Dedup key
#
# The `verdict` table has no dedicated dedup column, so the hash is
# embedded as `import_key=<hex>` at the start of `verdict.note` — the same
# field already carries the source-file/source-quote provenance, so this
# keeps everything about "where did this row come from" in one greppable
# place instead of adding a schema column this task must not touch.
# --------------------------------------------------------------------------

_IMPORT_KEY_RE = re.compile(r"^import_key=([0-9a-f]{64})")


def _verdict_dedup_key(
    *,
    idea_id: str,
    stage: VerdictStage,
    rule_id: str,
    metric: str,
    value: float | None,
    threshold: float | None,
    decided_at: date,
    source_quote: str,
) -> str:
    """Deterministic hash identifying "the same historical verdict".

    Used to make `import_graveyard` idempotent: running it twice against
    an unchanged graveyard file must not duplicate `verdict` rows.
    `rule_id` here is the value *after* the null -> "unidentified"
    substitution, since that's what actually ends up in the row.
    `threshold` is null for measurement rows (no criterion) — never
    fabricated, so the hash must tolerate that too.
    """
    parts = [
        idea_id,
        stage.value,
        rule_id,
        metric,
        "" if value is None else repr(float(value)),
        "" if threshold is None else repr(float(threshold)),
        decided_at.isoformat(),
        source_quote,
    ]
    raw = "\x1f".join(parts)  # unit separator: avoids field-boundary ambiguity
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_note(*, source_file: str, source_quote: str, import_key: str) -> str:
    """Stable, greppable provenance note.

    `source_file`/`source_quote` have no columns of their own in the
    `verdict` table (docs/REGISTRY.md) — this is what makes every
    imported number auditable back to the document it came from.
    """
    return f"import_key={import_key} source_file={source_file} source_quote={source_quote!r}"


def _existing_import_keys(session: Session) -> set[str]:
    keys: set[str] = set()
    rows = (
        session.query(Verdict.note)
        .filter(Verdict.source == TrialSource.IMPORTED, Verdict.note.isnot(None))
        .all()
    )
    for (note,) in rows:
        match = _IMPORT_KEY_RE.match(note or "")
        if match:
            keys.add(match.group(1))
    return keys


def _to_decided_at_datetime(d: date) -> datetime:
    """`verdict.decided_at` is a timezone-aware DateTime column; the source
    document only ever records a date, so midnight UTC is used."""
    return datetime.combine(d, time.min, tzinfo=UTC)


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


def import_graveyard(session: Session, path: Path | str) -> ImportReport:
    """Load `path` (a graveyard.yaml-shaped file) into the registry.

    Idempotent: drivers and ideas upsert by id (via `registry.repo`);
    verdicts dedup by a hash of (idea_id, stage, rule_id, metric, value,
    threshold, decided_at, source_quote) embedded in `verdict.note` —
    running this twice against the same file leaves row counts unchanged
    on the second run.

    Every written verdict gets `source="imported"` and
    `rules_version="frab-legacy"` (`qlab.rules.LEGACY_SENTINEL_VERSION`),
    regardless of what (if anything) the source file says, per
    docs/REGISTRY.md.

    Raises `GraveyardImportError` (structural validation failure) or
    `FileNotFoundError` before writing anything to `session`. A verdict
    that doesn't fit one of the three legitimate row kinds (see
    `_classify_verdict`) is instead reported in the returned
    `ImportReport.data_errors` and skipped, while the rest of the file
    still imports — see the module docstring for why this is handled
    differently from a structural failure.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no graveyard file at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise GraveyardImportError(f"{path} must contain a YAML mapping at top level")

    graveyard = parse_graveyard(raw)

    report = ImportReport()
    existing_keys = _existing_import_keys(session)

    for driver in graveyard.drivers:
        is_new = session.get(Driver, driver.id) is None
        repo.upsert_driver(
            session,
            id=driver.id,
            title=driver.title,
            description=driver.description,
            kill_condition=driver.kill_condition,
            observable=driver.observable,
        )
        if is_new:
            report.drivers_inserted += 1
        else:
            report.drivers_updated += 1

    to_insert: list[dict[str, Any]] = []

    for idea in graveyard.ideas:
        is_new = session.get(Idea, idea.id) is None
        repo.upsert_idea(
            session,
            id=idea.id,
            title=idea.title,
            source_type=idea.source_type,
            source_url=idea.source_url,
            claimed_edge=idea.claimed_edge,
            asset_class=idea.asset_class,
            driver_id=idea.driver_id,
            profile=idea.profile,
            status=idea.status,
            notes=idea.notes,
            shutdown_cause=idea.shutdown_cause,
        )
        if is_new:
            report.ideas_inserted += 1
        else:
            report.ideas_updated += 1

        for v_idx, verdict in enumerate(idea.verdicts):
            kind, error = _classify_verdict(
                value=verdict.value,
                passed=verdict.passed,
                comparator=verdict.comparator,
                threshold=verdict.threshold,
            )
            if kind is None:
                report.data_errors.append(
                    DataError(idea_id=idea.id, verdict_index=v_idx, message=error or "invalid")
                )
                continue

            if kind == "measurement" and verdict.rule_id is not None:
                # docs/REGISTRY.md: a measurement has no identified rule by
                # definition — a `rule_id` alongside a null comparator/
                # threshold is a self-contradictory source row.
                report.data_errors.append(
                    DataError(
                        idea_id=idea.id,
                        verdict_index=v_idx,
                        message=(
                            "measurement row (no comparator/threshold) must not carry a "
                            f"rule_id, got {verdict.rule_id!r}"
                        ),
                    )
                )
                continue

            rule_id = verdict.rule_id
            if rule_id is None:
                rule_id = UNIDENTIFIED_RULE_ID
                report.verdicts_unidentified_rule += 1

            import_key = _verdict_dedup_key(
                idea_id=idea.id,
                stage=verdict.stage,
                rule_id=rule_id,
                metric=verdict.metric,
                value=verdict.value,
                threshold=verdict.threshold,
                decided_at=verdict.decided_at,
                source_quote=verdict.source_quote,
            )
            if import_key in existing_keys:
                report.verdicts_skipped_duplicate += 1
                continue
            existing_keys.add(import_key)

            note = _build_note(
                source_file=verdict.source_file,
                source_quote=verdict.source_quote,
                import_key=import_key,
            )

            to_insert.append(
                dict(
                    idea_id=idea.id,
                    stage=verdict.stage,
                    rule_id=rule_id,
                    rules_version=LEGACY_SENTINEL_VERSION,
                    metric=verdict.metric,
                    value=verdict.value,
                    comparator=verdict.comparator,
                    threshold=verdict.threshold,
                    passed=verdict.passed,
                    data_range_start=verdict.data_range_start,
                    data_range_end=verdict.data_range_end,
                    decided_at=_to_decided_at_datetime(verdict.decided_at),
                    note=note,
                    source=TrialSource.IMPORTED,
                )
            )
            report.verdicts_inserted += 1
            if kind == "unknown":
                report.verdicts_null_value += 1
            elif kind == "measurement":
                report.verdicts_measurement += 1

    if to_insert:
        repo.add_verdicts(session, to_insert)

    return report


__all__ = [
    "UNIDENTIFIED_RULE_ID",
    "DataError",
    "DriverIn",
    "GraveyardFile",
    "GraveyardImportError",
    "IdeaIn",
    "ImportReport",
    "VerdictIn",
    "import_graveyard",
    "parse_graveyard",
]
