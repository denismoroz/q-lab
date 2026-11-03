"""What «скамейка» (the bench) means for one idea, read off its venue answers.

The registry status `bench` stays as it is (docs/REGISTRY.md). Owner, 2026-11-03: it mixes three
different things, «да, разведи на три надписи». The label the owner reads is derived here, by code,
from what each venue said about the idea (the latest routed run per venue, as the ladder reads it:
docs/LADDER.md, docs/VENUE_SPLIT.md). The model has no say in it, and nothing here writes.

Reading, in this order (the first that fits wins):

1. **passed, no execution** -- Binance passed and routed `needs-infrastructure` (the lab has no
   Binance order adapter), and Hyperliquid has not passed. The idea works where it cannot trade.
2. **waits for the second venue** -- one venue rejected and the other has not spoken (it is
   undecided, or has no run). The idea stays open because half of the answer is missing.
3. **too early to judge** -- a venue answered `needs-forward` / `needs-more-data`, or its noise bar
   had no power on the window (a bar that would pass a truly earning strategy less than
   `NO_POWER_BELOW` of the time says nothing either way).
4. anything else is «скамейка — причина не определена». That is an honest answer, not a gap to be
   hidden: such an idea is listed (`undetermined_ideas`) so the owner can see which they are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from qlab.ladder import Ledgers, Outcome, build_ladder
from qlab.registry.models import Idea, IdeaStatus

if TYPE_CHECKING:
    from qlab.ladder import Ladder

NO_POWER_BELOW = 0.8
"""`noise_power.min_power` of the rules (rules/2026-11-03.1.yaml): below it the noise bar decides
neither way. A run whose pass or failure rests on such a bar is read «рано судить» by the rules
themselves (route `needs-forward`); this number is only used to name the same state when a venue's
answer is undecided for another route."""

WAITING_ROUTES = frozenset({"needs-forward", "needs-more-data"})
"""The routes that mean «not enough data yet» (docs/LADDER.md, `ROUTE_OUTCOME`)."""

NO_EXECUTION_ROUTE = "needs-infrastructure"


class BenchReason(StrEnum):
    TOO_EARLY = "too-early"
    PASSED_NO_EXECUTION = "passed-no-execution"
    AWAITS_SECOND_VENUE = "awaits-second-venue"
    UNDETERMINED = "undetermined"


BENCH_LABEL_RU: dict[BenchReason, str] = {
    BenchReason.TOO_EARLY: "рано судить",
    BenchReason.PASSED_NO_EXECUTION: "прошла, нет исполнения",
    BenchReason.AWAITS_SECOND_VENUE: "ждёт вторую площадку",
    BenchReason.UNDETERMINED: "скамейка — причина не определена",
}

BENCH_HINT_RU: dict[BenchReason, str] = {
    BenchReason.TOO_EARLY: (
        "Площадка ответила «рано судить»: после выбора чисел данных мало, или сравнение со "
        "случайными книгами на таком окне ничего не различает. Нужны новые данные, а не решение."),
    BenchReason.PASSED_NO_EXECUTION: (
        "Идея прошла правила на Binance, но торговать там мы не можем: нашего адаптера заявок "
        "для Binance нет. Это факт о нашем коде, а не отказ стратегии."),
    BenchReason.AWAITS_SECOND_VENUE: (
        "Одна площадка идею отвергла, вторая ещё не ответила. Идея открыта, пока не скажет "
        "вторая."),
    BenchReason.UNDETERMINED: (
        "Статус «скамейка», но по прогонам код не смог назвать причину. Такие идеи собраны в "
        "список, чтобы их не терять."),
}

# Fixed display order of the four, also the order of the filter chips.
BENCH_ORDER: tuple[BenchReason, ...] = (
    BenchReason.TOO_EARLY, BenchReason.PASSED_NO_EXECUTION, BenchReason.AWAITS_SECOND_VENUE,
    BenchReason.UNDETERMINED)


@dataclass(frozen=True)
class VenueAnswer:
    """What one venue said, in the four things the reading needs."""

    venue: str
    outcome: Outcome
    route: str | None = None
    noise_power: float | None = None


def answers_of(ladder: Ladder) -> tuple[VenueAnswer, ...]:
    """The two venues of a ladder as answers (Binance, Hyperliquid)."""
    return tuple(VenueAnswer(v.venue, v.outcome, v.route, v.noise_power) for v in ladder.venues)


def _powerless(answer: VenueAnswer) -> bool:
    return answer.noise_power is not None and answer.noise_power < NO_POWER_BELOW


def _early(answer: VenueAnswer) -> bool:
    return answer.outcome == Outcome.UNDECIDED and (
        answer.route in WAITING_ROUTES or _powerless(answer))


def bench_reason(answers: Iterable[VenueAnswer]) -> BenchReason:
    """Which of the four labels a bench idea with these venue answers gets (module docstring)."""
    by = {a.venue: a for a in answers}
    binance, hl = by.get("binance"), by.get("hyperliquid")
    both = [a for a in (binance, hl) if a is not None]
    if (binance is not None and binance.outcome == Outcome.PASSED
            and binance.route == NO_EXECUTION_ROUTE
            and not (hl is not None and hl.outcome == Outcome.PASSED)):
        return BenchReason.PASSED_NO_EXECUTION
    failed = [a for a in both if a.outcome == Outcome.FAILED]
    silent = [a for a in both if a.outcome in (Outcome.UNDECIDED, Outcome.NOT_RUN)]
    if len(both) == 2 and len(failed) == 1 and len(silent) == 1:
        return BenchReason.AWAITS_SECOND_VENUE
    if any(_early(a) for a in both):
        return BenchReason.TOO_EARLY
    return BenchReason.UNDETERMINED


def bench_reason_of_ladder(ladder: Ladder) -> BenchReason | None:
    """The label of a ladder's idea; None when the idea is not on the bench."""
    if ladder.status != IdeaStatus.BENCH.value:
        return None
    return bench_reason(answers_of(ladder))


def bench_block(reason: BenchReason | None) -> dict[str, str] | None:
    """What the API hands the page: the key (a filter value), the words and the hint."""
    if reason is None:
        return None
    return {"reason": reason.value, "key": status_key("bench", reason),
            "label_ru": BENCH_LABEL_RU[reason], "hint_ru": BENCH_HINT_RU[reason]}


def status_key(status: str, reason: BenchReason | None) -> str:
    """The filter value of an idea: its status, and for the bench `bench-<reason>`."""
    return f"bench-{reason.value}" if status == IdeaStatus.BENCH.value and reason else status


def bench_reasons(session: Session, ledgers: Ledgers | None = None) -> dict[str, BenchReason]:
    """The label of every bench idea of the registry (calibration noise left out)."""
    out: dict[str, BenchReason] = {}
    ledgers = ledgers or Ledgers()
    ids = [i for (i,) in session.query(Idea.id).filter(Idea.status == IdeaStatus.BENCH)
           if not i.startswith("noise-")]
    for idea_id in sorted(ids):
        found = build_ladder(session, idea_id, ledgers)
        if found is not None:
            out[idea_id] = bench_reason(answers_of(found))
    return out


def bench_counts(reasons: Mapping[str, BenchReason]) -> list[dict[str, object]]:
    """Counts per label in the fixed order, a zero kept so the page lists all four."""
    return [{**(bench_block(r) or {}), "count": sum(1 for v in reasons.values() if v == r)}
            for r in BENCH_ORDER]


def undetermined_ideas(reasons: Mapping[str, BenchReason]) -> list[str]:
    """The bench ideas the reading could not name a reason for, sorted."""
    return sorted(i for i, r in reasons.items() if r == BenchReason.UNDETERMINED)


__all__ = [
    "BENCH_HINT_RU", "BENCH_LABEL_RU", "BENCH_ORDER", "BenchReason", "NO_POWER_BELOW",
    "VenueAnswer", "answers_of", "bench_block", "bench_counts", "bench_reason",
    "bench_reason_of_ladder", "bench_reasons", "status_key", "undetermined_ideas",
]
