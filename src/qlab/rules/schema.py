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


class RuleKind(StrEnum):
    """What a failing rule says about a candidate (docs/TASKS.md T35).

    A STRATEGY rule failing means "tested, and the strategy falls short"
    (edge, tails, capital). An INFRASTRUCTURE rule failing means "we cannot
    run it here today" -- no execution adapter, no forward data, no atomic
    execution. The second is a statement about our setup, not about the
    strategy, so it must neither stop the strategy rules from being
    evaluated nor send the idea to the graveyard
    (`qlab.pipeline.evaluate.decide_route`'s `needs-infrastructure`).
    """

    STRATEGY = "strategy"
    INFRASTRUCTURE = "infrastructure"


class FitPeriodUse(StrEnum):
    """What a rule's failure means on the SELECTION period -- the data the
    strategy's parameters were chosen on (docs/FIT_VS_FORWARD.md).

    That period flatters the strategy: its parameters were picked to look
    good there. So a failure there is CONCLUSIVE for a rule whose failure
    the flattery cannot explain (below the economic floor even on data it
    was fitted to; more capital than we will ever have). For a rule about
    whether the number can be trusted at all -- a universe chosen with
    hindsight, separation from luck -- the selection period is exactly where
    the number cannot be trusted, so the result there is INFORMATIONAL: it
    is recorded and shown, and the forward test decides.
    """

    CONCLUSIVE = "conclusive"
    INFORMATIONAL = "informational"


class ForwardResolution(BaseModel):
    """When a FAILED forward test is long enough to count as a failure
    rather than as "too early to tell" (docs/FIT_VS_FORWARD.md).

    The forward window needs about ((z_conf + z_power) / S)^2 years to tell a
    strategy whose selection-period Sharpe is S from zero (one-sided test at
    `confidence`, with `power`). A shorter forward test that fails routes
    `needs-forward`, not `reject`. A forward test that PASSES is never held
    back by this: the rules themselves already ask for separation from
    noise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: float = Field(gt=0.5, lt=1.0)
    power: float = Field(gt=0.0, lt=1.0)
    rationale: str


class Rule(BaseModel):
    """A single screening rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: Stage
    metric: str
    comparator: Comparator
    threshold: float
    fatal: bool = False
    kind: RuleKind = RuleKind.STRATEGY
    fit_period: FitPeriodUse = FitPeriodUse.CONCLUSIVE
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
    forward_resolution: ForwardResolution | None = None
    """Present from 2026-10-02.1 on. A ruleset without it judges the whole
    window as one, exactly as before the split existed, so older verdicts
    stay reproducible under their own rules."""

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
