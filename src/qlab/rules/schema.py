"""Pydantic models for the versioned screening-rules config.

Format is defined in ``docs/REGISTRY.md`` under "Правила отбора —
версионируемый конфиг". These models are pure data + validation: no I/O,
no evaluation logic (that lives in :mod:`qlab.rules.engine`).
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_VERSION_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d+)$")


def parse_version(version: str) -> tuple[date, int]:
    """Parse a ``YYYY-MM-DD.N`` rules version into a sortable key.

    Raises ``ValueError`` if the string doesn't match the format. Sorting by
    this key (not lexicographically) is what makes ``2026-09-20.10`` come
    after ``2026-09-20.9``.
    """
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"invalid rules version {version!r}, expected format YYYY-MM-DD.N")
    date_part, seq_part = match.groups()
    try:
        parsed_date = date.fromisoformat(date_part)
    except ValueError as exc:
        raise ValueError(f"invalid rules version {version!r}: bad date part") from exc
    return parsed_date, int(seq_part)


class Stage(StrEnum):
    """Screening stage. Order below is the order rules are evaluated in."""

    PREFLIGHT = "preflight"
    EDGE = "edge"
    TAIL = "tail"
    PROFILE = "profile"
    CAPACITY = "capacity"
    CORRELATION = "correlation"


# Canonical processing order — deliberately NOT declaration order in a
# ruleset file. See docs/PLAN.md M4: preflight (executability) runs before
# any other stage, ahead of the order used in SCREENING.md.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.PREFLIGHT,
    Stage.EDGE,
    Stage.TAIL,
    Stage.PROFILE,
    Stage.CAPACITY,
    Stage.CORRELATION,
)


class Comparator(StrEnum):
    """Strict set of comparison operators a rule may use."""

    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="

    def compare(self, value: float, threshold: float) -> bool:
        if self is Comparator.GT:
            return value > threshold
        if self is Comparator.GE:
            return value >= threshold
        if self is Comparator.LT:
            return value < threshold
        if self is Comparator.LE:
            return value <= threshold
        if self is Comparator.EQ:
            return value == threshold
        if self is Comparator.NE:
            return value != threshold
        raise AssertionError(f"unhandled comparator {self!r}")  # pragma: no cover


class Rule(BaseModel):
    """A single screening rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: Stage
    metric: str
    comparator: Comparator
    threshold: float
    fatal: bool = False
    near_margin: float | None = Field(
        default=None,
        description=(
            "Relative 'almost passed' margin, as a fraction of |threshold|. "
            "Meaningless when threshold == 0 — use near_margin_abs instead."
        ),
    )
    near_margin_abs: float | None = Field(
        default=None,
        description=(
            "Absolute 'almost passed' margin, in the metric's own units. "
            "Required (not derivable) when threshold == 0: see "
            "qlab.rules.nearness.classify_nearness()."
        ),
    )
    rationale: str | None = None

    @field_validator("id", "metric")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("near_margin", "near_margin_abs")
    @classmethod
    def _margin_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("near margin must be >= 0")
        return value


class RetiredRule(BaseModel):
    """A rule that used to exist but has been removed from active screening.

    Kept around so `killed_by_retired_rules()` can find candidates that were
    only killed by a rule nobody believes in anymore.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    retired_in: str
    reason: str

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("retired_in")
    @classmethod
    def _validate_retired_in(cls, value: str) -> str:
        parse_version(value)
        return value


class RuleSet(BaseModel):
    """One version of the screening config: `rules/<version>.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    based_on: str | None = None
    rules: list[Rule] = Field(default_factory=list)
    retired: list[RetiredRule] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        parse_version(value)
        return value

    @field_validator("based_on")
    @classmethod
    def _validate_based_on(cls, value: str | None) -> str | None:
        if value is not None:
            parse_version(value)
        return value

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> RuleSet:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r} in ruleset {self.version}")
            seen.add(rule.id)
        return self

    def version_key(self) -> tuple[date, int]:
        """Sort key for this ruleset's version (date, sequence-number)."""
        return parse_version(self.version)
