"""Pydantic models for the versioned screening-rules config.

Format is defined in ``docs/REGISTRY.md`` under "Правила отбора —
версионируемый конфиг". These models are pure data + validation: no I/O,
no evaluation logic (that lives in :mod:`qlab.rules.engine`).
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import StrEnum
from typing import Literal

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


class ShortForwardUse(StrEnum):
    """What a rule's failure means on a forward test too short to resolve
    the strategy's Sharpe (docs/FIT_VS_FORWARD.md).

    DECIDE: the failure stands whatever the length. WAIT: the rule asks
    whether the result is separable from luck, and a short window cannot
    answer that -- its failure reads "too early to tell" (owner, 2026-10-02:
    "сделай, чтобы тоже читалось как «рано судить»")."""

    DECIDE = "decide"
    WAIT = "wait"


class ForwardResolution(BaseModel):
    """When a FAILED forward test is long enough to count as a failure
    rather than as "too early to tell" (docs/FIT_VS_FORWARD.md).

    The forward window needs about ((z_conf + z_power) / S)^2 years to tell a
    strategy whose selection-period Sharpe is S from zero (one-sided test at
    `confidence`, with `power`). A shorter forward test decides nothing in
    either direction and routes `needs-forward`: a failure is too early to
    call, and so is a pass (18 days of trend read +446% a year and beat all
    200 noise books).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: float = Field(gt=0.5, lt=1.0)
    power: float = Field(gt=0.0, lt=1.0)
    rationale: str


class HoldoutShare(BaseModel):
    """When a forward test is long enough to decide (docs/FIT_VS_FORWARD.md),
    from 2026-10-24.1 in place of `ForwardResolution`.

    Owner, 2026-10-24: «данные которые доступны для стратегии поделить 2/3 и
    1/3. первую часть использовать для тестирования подбора а оставшуюся 1/3
    для прогона вперед». The forward test decides once it is at least `share`
    of the parameters' fitting window plus itself. Telling a Sharpe of 1 from
    zero instead takes about six years of forward data alone; the luck guard
    is the rules judged on that third (`shape_aware_edge`, regime coverage),
    not its length.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    share: float = Field(gt=0.0, lt=1.0)
    rationale: str

    def days_needed(self, fit_days: float) -> float:
        """Whole forward days for the forward test to be at least `share` of
        fit plus forward (rounded to 1e-6 first: a third written as a float
        must not turn 731 / 2 into 365.49999...)."""
        return float(math.ceil(round(fit_days * self.share / (1.0 - self.share), 6)))


class NoisePower(BaseModel):
    """When the noise bar can tell a good strategy from luck on THIS run
    (docs/FIT_VS_FORWARD.md), from 2026-10-24.2.

    Measured 2026-10-24 on strategy-b-v2's shape and 170 days: books with a
    known +20% a year over random passed the 99% noise bar 2% of the time,
    empty books 1.3% -- the bar rejected good and empty alike. So the run's
    own noise books are asked first: a book of their shape whose true return
    is `good_return` a year, how often does it pass? Below `min_power` a
    failure of a `wait` rule is too early to tell, not a rejection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    good_return: float
    min_power: float = Field(gt=0.0, lt=1.0)
    rationale: str
    return_metric: str = "ann_return_net"
    """The metric of the noise books the bar ranks on (and `good_return` is a level of):
    the stand's yearly return up to rules 2026-10-29.1, `cap_ann_return` (a percent of the
    strategy's own money, docs/CAPITAL_BASE.md) from 2026-10-29.2."""
    percentile_metric: str = "noise_return_percentile"
    """The metric of the rule that is the noise bar: its threshold is the bar the power is
    asked about, and a failure of it is the one a weak bar turns into "too early to tell"."""
    power_metric: str = "noise_power"
    """The key the measured power is stored under in a run's metrics."""


class VenueSplit(BaseModel):
    """Binance and Hyperliquid are judged separately, each on its own data, coins, costs and
    minimum leg, from 2026-10-30.1 (docs/VENUE_SPLIT.md; owner, 2026-10-30: «да, делай раздельно
    Binance и HL»).

    The idea's route is then the better of the two venues' routes, not a pass at both: a pass on
    Hyperliquid is `paper` (the lab can trade there), a pass on Binance alone is
    `needs-infrastructure` (the lab has no Binance order adapter: a fact about our code, never a
    reject), and an idea is rejected only when no venue passed and none still waits. Where the
    noise bar on Hyperliquid's short window has no power, Hyperliquid's verdict is the consistency
    check with Binance's history (`Stage2Confirmation`), taken on the coins Hyperliquid also lists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: Literal["best_of_venues"]
    """How the two venues' routes make the idea's: the better one."""
    shared_coins_history: bool
    """Whether the consistency check compares Hyperliquid's window with Binance's history on the
    coins Hyperliquid also lists (True) or with Binance's whole history (False)."""
    rationale: str


class Stage2Confirmation(BaseModel):
    """How stage 2 of the idea generator's evaluation (Hyperliquid) confirms the
    verdict of stage 1 (Binance's long history) where its own noise comparison
    cannot decide (docs/TWO_STAGE.md), from 2026-10-27.1.

    Owner, 2026-10-27, «делай все»: the proof comes from stage 1's years of
    data; stage 2 is a consistency check, not a second independent proof. When
    the noise bar has less than `NoisePower.min_power` on the stage 2 window,
    the comparison is replaced by: the stage 2 result must not fall in the
    bottom `alert_percentile` of 100 same-length segments of stage 1's forward
    history, and its total return after costs must be above zero. The
    percentile is the owner's alarm for paper books and is NOT repeated here:
    it is read from `alert_percentile_from`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alert_percentile_from: str
    """A YAML file with a top-level `alert_percentile` (0-100): the owner's
    threshold, `night/decay.yaml`. While it is null nothing is confirmed."""
    rationale: str
    total_metric: str = "total_return_net"
    """The stage 2 total return the positivity rule reads: the stand's up to rules
    2026-10-29.1, `cap_total_return` (on the strategy's own money) from 2026-10-29.2."""


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
    on_short_forward: ShortForwardUse = ShortForwardUse.DECIDE
    skip_when_undefined: bool = Field(
        default=False,
        description=(
            "From 2026-10-28.1: a metric that is absent because it is NOT DEFINED for "
            "this run (the year summaries with fewer than two counted years) means the "
            "rule does not apply: the row is written as 'not defined', and it neither "
            "passes nor fails, nor makes the result indecisive. Without this flag an "
            "absent metric is 'unknown' and routes needs-more-data."
        ),
    )
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


class InformationalRule(BaseModel):
    """A reading shown next to the verdict that DECIDES nothing (2026-10-27.2).

    It is not a `Rule`: it never becomes a verdict row, never fails a candidate and
    never moves a route. `qlab.rules.engine.read_informational` compares the metric
    with `threshold` only to say "this would hold / would not hold", so the owner
    sees what a threshold would have done before choosing one. A null `threshold`
    means no line is drawn yet: the number is shown on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    metric: str
    comparator: Comparator | None = None
    threshold: float | None = None
    rationale: str

    @field_validator("id", "metric")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _line_is_whole(self) -> InformationalRule:
        if (self.comparator is None) != (self.threshold is None):
            raise ValueError(
                f"informational rule {self.id!r}: comparator and threshold go together")
        return self


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
    holdout_share: HoldoutShare | None = None
    """From 2026-10-24.1, replacing `forward_resolution`: how long a forward
    test must be, as a share of the data, rather than the years it takes to
    resolve a Sharpe."""
    noise_power: NoisePower | None = None
    """From 2026-10-24.2: a `wait` rule's failure counts only where the noise
    bar can tell a good strategy from luck."""
    stage2_confirmation: Stage2Confirmation | None = None
    """From 2026-10-27.1: where the noise bar of stage 2 cannot decide, stage 2
    confirms stage 1's verdict instead (docs/TWO_STAGE.md)."""
    venue_split: VenueSplit | None = None
    """From 2026-10-30.1: Binance and Hyperliquid are judged separately and the idea's route is the
    better of the two (docs/VENUE_SPLIT.md). Absent: both stages must pass, as before."""
    informational: list[InformationalRule] = Field(default_factory=list)
    """From 2026-10-27.2: readings shown beside the verdict that decide nothing
    (the year-by-year view, docs/YEARS.md)."""
    regime_coverage_days: int | None = None
    """From 2026-10-02.3: a forward test decides only once it has seen every
    market regime (`qlab.regimes`) for at least this many days; otherwise it
    is too early to tell (docs/REGIMES.md)."""
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

    @model_validator(mode="after")
    def _unique_informational_ids(self) -> RuleSet:
        ids = [reading.id for reading in self.informational]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate informational id in ruleset {self.version}")
        return self

    @model_validator(mode="after")
    def _one_forward_length_rule(self) -> RuleSet:
        if self.forward_resolution is not None and self.holdout_share is not None:
            raise ValueError("a ruleset sets forward_resolution or holdout_share, not both")
        return self

    @property
    def splits_fit_forward(self) -> bool:
        """The ruleset judges a forward test apart from the selection period."""
        return self.forward_resolution is not None or self.holdout_share is not None

    def version_key(self) -> tuple[date, int]:
        """Sort key for this ruleset's version (date, sequence-number)."""
        return parse_version(self.version)
