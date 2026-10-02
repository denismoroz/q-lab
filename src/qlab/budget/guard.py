"""The token budget guard (CLAUDE.md, "Бюджет токенов"; docs/BUDGET.md).

The nightly pipeline runs on the owner's subscription, sharing its quota with
the owner's own work; the guard is the only protection. The dangerous window
is the seven-day one: the five-hour window recovers by morning on its own,
the weekly one resets once a week, and one greedy night would lock the owner
out for days.

**The night's ceiling is computed, not set** (CLAUDE.md):

    night target = weekly utilization now
                   + (70% - weekly utilization now) / nights until reset

It slows down by itself after a greedy night and speeds up early in the week.
Utilization is READ, not estimated: every call reports the shared windows
(`qlab.budget.usage`), the owner's interactive use included. The token
conversion is only needed to decide in advance whether the next call fits and
to give each candidate a cap in tokens; it is calibrated on the ledger (pairs
of tokens spent and utilization gained) and, before any data, taken from
CLAUDE.md's measurement (~470k tokens -> +4 pp of the weekly window).

**Ceilings are layered**: the night, then the stage, then the candidate.
Stages run in the order CLAUDE.md fixes -- re-checking the matured graveyard
and live strategies' health, then implementing and running candidates that
passed preflight, then searching for new ones -- each using what the earlier
ones left, so the search is what gets cut first. Within a stage each candidate
may use the stage's remaining tokens divided by the candidates still waiting;
a candidate that reaches its cap stops without taking the others' share.

**Exhaustion is a normal stop, not a failure**: `NightExhausted` and
`CandidateExhausted` are raised for the pipeline to save its state and carry
on next night (or with the next candidate).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from qlab.budget.usage import UsageReading
from qlab.registry.models import TokenSpend

WEEKLY_CEILING = 0.70
"""CLAUDE.md: «потолок_на_ночь = (70% недельного окна − потрачено за неделю) /
ночей до сброса»."""

PRIOR_UTIL_PER_TOKEN = 0.04 / 470_000
"""CLAUDE.md, measurement of 2026-09-20 on the Pro plan: ~470k tokens moved
the weekly window by +4 percentage points. Used until the ledger has data."""

READING_RESOLUTION = 0.01
"""Utilization is reported with two decimals (0.42, 0.17 on 2026-10-02): one
reading can be off by this much, which bounds the calibration from above."""

MIN_CALL_TOKENS_PRIOR = 39_906
"""The cheapest call measured: a one-word haiku reply on 2026-10-02 cost
39,857 cache-write + 10 input + 39 output tokens (the CLI's own system
prompt). No call can cost less; used until the ledger has calls."""

USD_PER_TOKEN_PRIOR = 0.079919 / 39_906
"""The same call's reported `total_cost_usd` (list price) per token; used to
turn a token cap into the CLI's own `--max-budget-usd` stop until the ledger
has calls."""


class Stage(enum.IntEnum):
    """Pipeline stages in the order they run; the last is cut first."""

    RECHECK = 1  # matured graveyard, live strategies' health
    IMPLEMENT = 2  # implement and run candidates that passed preflight
    SEARCH = 3  # look for new candidates


class BudgetExhausted(Exception):  # noqa: N818 - a normal stop, not an error
    """A ceiling was reached. Not a failure: save state and stop cleanly."""


class NightExhausted(BudgetExhausted):
    """The night's ceiling (or a window's limit) is reached: stop everything."""


class CandidateExhausted(BudgetExhausted):
    """This candidate's cap is reached: stop it, the others go on."""


@dataclass(frozen=True)
class Calibration:
    util_per_token: float
    usd_per_token: float
    min_call_tokens: int
    source: str  # "prior" or "ledger (N pairs)"


def _calls(session: Session) -> list[TokenSpend]:
    return session.query(TokenSpend).order_by(TokenSpend.at, TokenSpend.id).all()


def _total(row: TokenSpend) -> int:
    return (row.tokens_in or 0) + (row.tokens_out or 0) + (row.tokens_cache_write or 0) + (
        row.tokens_cache_read or 0
    )


def calibrate(session: Session) -> Calibration:
    """Weekly utilization per token, from consecutive calls in the same weekly
    window: (sum of utilization gained + one reading's resolution) / tokens
    spent -- an upper bound, so the estimate starts cautious and tightens as
    tokens accumulate. Utilization gained between two calls also counts the
    owner's own use in between, which can only make it more cautious."""
    rows = [r for r in _calls(session) if _total(r) > 0]
    usd = sum(r.usd_est or 0.0 for r in rows)
    tokens = sum(_total(r) for r in rows)
    usd_per_token = usd / tokens if usd > 0 and tokens > 0 else USD_PER_TOKEN_PRIOR
    min_call = min((_total(r) for r in rows), default=MIN_CALL_TOKENS_PRIOR)

    gained, spent, pairs = 0.0, 0, 0
    for prev, cur in zip(rows, rows[1:], strict=False):
        if (prev.seven_day_util is None or cur.seven_day_util is None
                or prev.seven_day_resets_at != cur.seven_day_resets_at):
            continue
        gained += max(0.0, cur.seven_day_util - prev.seven_day_util)
        spent += _total(cur)
        pairs += 1
    if pairs == 0:
        return Calibration(PRIOR_UTIL_PER_TOKEN, usd_per_token, min_call, "prior")
    return Calibration((gained + READING_RESOLUTION) / spent, usd_per_token, min_call,
                       f"ledger ({pairs} pairs)")


def nights_until(resets_at: datetime, now: datetime) -> int:
    """Nightly runs left before the weekly window resets, this one included."""
    return max(1, math.ceil((resets_at - now) / timedelta(days=1)))


@dataclass
class NightBudget:
    """One night's run. Open it with the latest reading of the shared windows."""

    start_util: float
    target_util: float
    resets_at: datetime
    calibration: Calibration
    current_util: float
    spent_tokens: int = 0
    stopped: str | None = None
    _stage_spent: dict[Stage, int] = field(default_factory=dict)

    @classmethod
    def open(cls, session: Session, reading: UsageReading, now: datetime | None = None
             ) -> NightBudget:
        if reading.seven_day is None:
            raise NightExhausted("no reading of the weekly window: the guard does not guess it")
        now = now or datetime.now(UTC)
        start = reading.seven_day.utilization
        nights = nights_until(reading.seven_day.resets_at, now)
        target = start + max(0.0, WEEKLY_CEILING - start) / nights
        budget = cls(start_util=start, target_util=target, resets_at=reading.seven_day.resets_at,
                     calibration=calibrate(session), current_util=start)
        budget.observe(reading)
        return budget

    def tokens_left(self) -> int:
        return max(0, int((self.target_util - self.current_util) / self.calibration.util_per_token))

    def observe(self, reading: UsageReading | None) -> None:
        """Take a call's reading; stop the night on a window limit or the target."""
        if reading is None:
            return
        if reading.seven_day is not None:
            self.current_util = reading.seven_day.utilization
        if reading.status != "allowed":
            self.stopped = f"the subscription reports status {reading.status!r}"
        elif reading.five_hour is not None and reading.five_hour.utilization >= 1.0:
            self.stopped = "the five-hour window is full"
        elif self.current_util >= self.target_util:
            self.stopped = (f"weekly utilization {self.current_util:.0%} reached tonight's "
                            f"target {self.target_util:.0%}")

    def check(self) -> None:
        if self.stopped:
            raise NightExhausted(self.stopped)
        if self.tokens_left() < self.calibration.min_call_tokens:
            self.stopped = "what is left tonight is smaller than the cheapest call"
            raise NightExhausted(self.stopped)

    def stage(self, stage: Stage, candidates: int) -> StageBudget:
        if candidates < 1:
            raise ValueError("a stage needs at least one candidate")
        return StageBudget(night=self, stage=stage, waiting=candidates)


@dataclass
class StageBudget:
    night: NightBudget
    stage: Stage
    waiting: int

    def candidate(self, idea_id: str | None) -> CandidateBudget:
        """The next candidate's cap: what is left tonight shared among the
        candidates still waiting in this stage (the later stages get what this
        one leaves)."""
        self.night.check()
        if self.waiting < 1:
            raise ValueError("every candidate of this stage already had its budget")
        cap = self.night.tokens_left() // self.waiting
        self.waiting -= 1
        return CandidateBudget(night=self.night, stage=self.stage, idea_id=idea_id, cap_tokens=cap)


@dataclass
class CandidateBudget:
    night: NightBudget
    stage: Stage
    idea_id: str | None
    cap_tokens: int
    spent_tokens: int = 0

    def remaining(self) -> int:
        return max(0, min(self.cap_tokens - self.spent_tokens, self.night.tokens_left()))

    def check(self) -> None:
        self.night.check()
        if self.remaining() < self.night.calibration.min_call_tokens:
            raise CandidateExhausted(
                f"candidate {self.idea_id!r} used {self.spent_tokens:,} of {self.cap_tokens:,} "
                "tokens; what is left is smaller than the cheapest call"
            )

    def record(self, tokens: int, reading: UsageReading | None) -> None:
        self.spent_tokens += tokens
        self.night.spent_tokens += tokens
        self.night._stage_spent[self.stage] = self.night._stage_spent.get(self.stage, 0) + tokens
        self.night.observe(reading)


@dataclass(frozen=True)
class ExplicitNoBudget:
    """The only way to call an agent without a budget: say so, and why.
    CLAUDE.md: «Забыть про бюджет нельзя, можно только явно отказаться от него.»"""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError("refusing the budget needs a stated reason")


__all__ = [
    "BudgetExhausted",
    "Calibration",
    "CandidateBudget",
    "CandidateExhausted",
    "ExplicitNoBudget",
    "NightBudget",
    "NightExhausted",
    "Stage",
    "StageBudget",
    "calibrate",
    "nights_until",
]
