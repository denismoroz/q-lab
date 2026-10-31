"""The ladder of an idea: one list of seven steps, the same for every idea (docs/LADDER.md).

Owner, 2026-10-29: «поидее все стратегии должны быть оценены по одному принципу - а у нас везде
все по разному» (docs/EVALUATION_PATHS.md). The paths that put an idea on the stand differ by
idea; the ladder does not. Every idea shows the same seven steps, in the same order, each with an
outcome and, where the step was not run, the reason -- a name from a CLOSED list, never silence.

This module is a **reading**: a pure function from the registry (read only) and the ledgers the
night keeps (`Ledgers`) to the steps. It writes nothing, computes no metric and decides no verdict;
the outcome of a step is the stand's own route for that run (`qlab.pipeline.evaluate`), translated:

    paper / shelf / needs-infrastructure   -> passed   (every strategy rule passed; `shelf` is
                                              only in runs made before rules 2026-10-29.3)
    needs-forward / needs-more-data / error -> undecided (the fourth outcome, owner answer В3:
                                              «не решён — рано судить»)
    reject                                 -> failed
    not-evaluable                          -> not run  (nothing was tested; the reason names why)

The steps (docs/EVALUATION_PATHS.md, section 6):

1. description and the checks before the calculation (the idea has a run the stand accepted);
2. the verdict for Binance, on Binance's long history, coins, costs and minimum leg (the idea itself
   if it runs there, its causal twin if it does not);
3. the verdict for Hyperliquid, on Hyperliquid's own data, coins, costs and minimum leg, the same
   numbers (the second stage, or the idea's own run);
4. the strategy's money against the capital each venue's run needed (one line per venue);
5. readings that decide nothing (years, protection, regimes, anomalies, overlays);
6. paper, four to six weeks;
7. the owner's decision.

Venue split (2026-10-30, docs/VENUE_SPLIT.md; owner: «да, делай раздельно Binance и HL»): steps 2
and 3 are the two venues side by side, each with its own outcome, its own check of the strategy's
money (step 4 has a line per venue) and the fact about execution there (Binance has no order
adapter of ours: a pass there is «нужна инфраструктура», never a reject). `Ladder.venues` and
`Ladder.summary_ru` say which venue passed, which did not and which was not run, and why.

Step 4 (phase B, 2026-10-29): the strategy's money is a field of the spec itself
(`StrategySpec.strategy_capital_usd`, cited in its `sources`), and the capital a run needed
(`min_capital_usd`) is compared with THAT, never with a common sum: the route `shelf` is gone
(rules 2026-10-29.3), a run that passed every rule is `paper`, and whether it fits its money is
this step's outcome. A spec that states no money reads «деньги стратегии не заданы»."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from qlab import capital_text
from qlab.registry.lifecycle import CLOSED_STATUSES
from qlab.registry.models import AnomalyScan, Idea, IdeaStatus, Spec, Trial, Verdict
from qlab.registry.stages import (
    child_id,
    child_id_taken,
    data_range,
    is_stage2_child,
    latest_routed,
    stage2_child,
)

TWIN_SUFFIX = "-causal"
BINANCE_SUFFIX = "-binance"
VARIANT_INFIX = "-ov-"
BINANCE = "binance"
DEFAULT_STAGE1_SURVIVES = frozenset({"paper", "shelf", "needs-infrastructure", "needs-forward"})
DEFAULT_ANOMALY_ROUTES = frozenset({"paper", "shelf", "needs-infrastructure", "needs-forward"})
GATE_STATUSES = frozenset({IdeaStatus.VALIDATED, IdeaStatus.PAPER, IdeaStatus.LIVE})
"""Statuses of an idea that passed every rule of the stand and waits for, or has, the owner."""
MIN_POWER = 0.8
"""The planks' strength below which the noise bar cannot tell (rules 2026-10-29.2)."""


# --------------------------------------------------------------------------
# Words


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNDECIDED = "undecided"
    NOT_RUN = "not_run"


OUTCOME_RU: dict[Outcome, str] = {
    Outcome.PASSED: "пройден",
    Outcome.FAILED: "не пройден",
    Outcome.UNDECIDED: "не решён — рано судить",
    Outcome.NOT_RUN: "не проводился",
}


class Reason(StrEnum):
    """Why a step was not run. A CLOSED list: a new reason is a line here and in docs/LADDER.md,
    never a free sentence in a step."""

    NO_BINANCE_VERSION = "no_binance_version"
    NO_CHOICE = "no_choice"
    HEAVY_DATA = "heavy_data"
    FIRST_STEP_UNDECIDED = "first_step_undecided"
    NO_STRATEGY_MONEY = "no_strategy_money"
    CLOSED_BEFORE = "closed_before"
    WAITS_FOR_NIGHT = "waits_for_night"
    CARD_WITHOUT_IMPLEMENTATION = "card_without_implementation"
    NO_DELISTED_PRICES = "no_delisted_prices"
    # Added in phase A, each one a state the code already tells apart:
    NOT_EVALUABLE = "not_evaluable"
    ID_TAKEN = "id_taken"
    NOT_REACHED = "not_reached"
    NOT_IN_PAPER = "not_in_paper"
    IN_PRODUCTION = "in_production"
    TWIN_ONLY_WHILE_WAITING = "twin_only_while_waiting"
    TWIN_REFUSED_OTHER = "twin_refused_other"
    # Added with the venue split (2026-10-30):
    NO_VERSION_ON_VENUE = "no_version_on_venue"
    # Added with independent venues (2026-10-30.2):
    SECOND_VENUE_QUEUED = "second_venue_queued"
    # Added with the crash rules (2026-10-31):
    STAGE2_FAILED = "stage2_failed"


REASON_RU: dict[Reason, str] = {
    Reason.NO_BINANCE_VERSION: "нет версии на Binance",
    Reason.NO_CHOICE: "выбирать числам не из чего",
    Reason.HEAVY_DATA: "данные слишком тяжелы за ночь",
    Reason.FIRST_STEP_UNDECIDED: "первая ступень не решила — сверять не с чем",
    Reason.NO_STRATEGY_MONEY: "деньги стратегии не заданы",
    Reason.CLOSED_BEFORE: "идея закрыта до этой ступени",
    Reason.WAITS_FOR_NIGHT: "ждёт очереди ночи",
    Reason.CARD_WITHOUT_IMPLEMENTATION: "карточка без реализации",
    Reason.NO_DELISTED_PRICES: "нет цен выбывших инструментов",
    Reason.NOT_EVALUABLE: "проверить нечем: стенд не выражает механизм",
    Reason.ID_TAKEN: "идентификатор второго этапа занят другой идеей",
    Reason.NOT_REACHED: "ещё не дошла: предыдущие ступени не пройдены",
    Reason.NOT_IN_PAPER: "владелец не переводил в бумагу",
    Reason.IN_PRODUCTION: "идея уже в бумаге или в деньгах — двойник не строится",
    Reason.TWIN_ONLY_WHILE_WAITING: "двойник строится, только пока идея ждёт проверки вперёд",
    Reason.TWIN_REFUSED_OTHER: "двойник не построен по другой причине (слова кода в подробностях)",
    Reason.NO_VERSION_ON_VENUE: "нет версии на этой площадке",
    Reason.SECOND_VENUE_QUEUED: "вторая площадка не проводилась: стоит в очереди закрытых идей",
    Reason.STAGE2_FAILED: "не проводился: сбой",
}

STEP_KEYS = ("description", "binance", "hyperliquid", "money", "readings", "paper", "owner")
STEP_TITLES = (
    "Описание и проверки до расчёта",
    "Binance: вердикт на данных, монетах и издержках Binance",
    "Hyperliquid: вердикт на данных, монетах и издержках Hyperliquid",
    "Деньги стратегии: по строке на площадку",
    "Показания: ничего не решают",
    "Бумага 4–6 недель",
    "Решение владельца",
)
STEP_OUTCOME_RU: dict[int, dict[Outcome, str]] = {
    5: {Outcome.PASSED: "записаны"},
    7: {Outcome.PASSED: "принято владельцем", Outcome.UNDECIDED: "ждёт владельца",
        Outcome.FAILED: "закрыта"},
}
"""Words that fit a step better than the general four. Step 5 decides nothing, so «пройден» would
mislead there: its good state reads «записаны». Step 7 is the owner's, not the stand's."""

ROUTE_OUTCOME: dict[str, Outcome] = {
    "paper": Outcome.PASSED,
    "shelf": Outcome.PASSED,
    "needs-infrastructure": Outcome.PASSED,
    "needs-forward": Outcome.UNDECIDED,
    "needs-more-data": Outcome.UNDECIDED,
    "error": Outcome.UNDECIDED,
    "reject": Outcome.FAILED,
}

ROUTE_NOTE_RU: dict[str, str] = {
    "paper": "прошла все правила стенда",
    "shelf": "прошла все правила; до правил 2026-10-29.3 стенд сравнивал нужный капитал с общей "
             "суммой, а не с деньгами стратегии (их читает ступень 4); так записан старый прогон",
    "needs-infrastructure": "прошла все правила этой площадки; нужна инфраструктура: нашего "
                            "адаптера заявок на ней нет (факт о нашем коде, не отказ стратегии); "
                            "торговать мы можем там, где адаптер есть, — на Hyperliquid",
    "needs-forward": "правила пока не могут решить: мало данных после выбора чисел",
    "needs-more-data": "не хватило метрики, нужной правилу",
    "error": "прогон упал",
    "reject": "правила стенда отвергли",
}


# --------------------------------------------------------------------------
# The shapes a step is made of


@dataclass(frozen=True)
class StepRun:
    """The run behind a step: which trial, on which data, under which versions."""

    idea_id: str
    trial_id: int
    route: str | None
    venue: str | None
    data_start: str | None
    data_end: str | None
    rules_version: str | None
    stand_version: str | None
    numbers: str
    """The run's leading numbers on the strategy's own money (`qlab.capital_text`)."""
    forward_days: float | None = None
    forward_days_needed: float | None = None
    noise_percentile: float | None = None
    noise_power: float | None = None
    min_capital_usd: float | None = None
    consistency_trial: int | None = None
    """The first-stage trial this run's pass was checked against instead of the noise bar (the
    consistency check, docs/TWO_STAGE.md); None: the noise bar decided, or no pass."""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Reading:
    """One reading of step 5. It decides nothing."""

    name: str
    recorded: bool
    text: str
    reason: Reason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "recorded": self.recorded, "text": self.text,
                "reason": None if self.reason is None else self.reason.value,
                "reason_ru": None if self.reason is None else REASON_RU[self.reason]}


@dataclass(frozen=True)
class VenueMoney:
    """The check of the strategy's money on one venue: the capital that venue's run needed against
    the strategy's own money (step 4 has one line per venue)."""

    venue: str
    outcome: Outcome
    detail: str
    reason: Reason | None = None

    @property
    def outcome_ru(self) -> str:
        return OUTCOME_RU[self.outcome]

    @property
    def reason_ru(self) -> str | None:
        return None if self.reason is None else REASON_RU[self.reason]

    def to_dict(self) -> dict[str, Any]:
        return {"venue": self.venue, "outcome": self.outcome.value, "outcome_ru": self.outcome_ru,
                "reason": None if self.reason is None else self.reason.value,
                "reason_ru": self.reason_ru, "detail": self.detail}


@dataclass(frozen=True)
class Step:
    number: int
    key: str
    title: str
    outcome: Outcome
    detail: str
    reason: Reason | None = None
    run: StepRun | None = None
    readings: tuple[Reading, ...] = ()
    other_idea: str | None = None
    """The idea whose run the step shows when it is not this one's own (the first stage of a
    second stage, the causal twin of an original)."""
    venue: str | None = None
    """The venue of steps 2 and 3 (`binance`, `hyperliquid`)."""
    execution_ru: str | None = None
    """What the lab can do on that venue: the fact about its order adapter, in words."""
    money: tuple[VenueMoney, ...] = ()
    """Step 4 only: the money check of each venue."""

    @property
    def outcome_ru(self) -> str:
        return STEP_OUTCOME_RU.get(self.number, {}).get(self.outcome, OUTCOME_RU[self.outcome])

    @property
    def reason_ru(self) -> str | None:
        return None if self.reason is None else REASON_RU[self.reason]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "key": self.key, "title": self.title,
            "outcome": self.outcome.value, "outcome_ru": self.outcome_ru,
            "reason": None if self.reason is None else self.reason.value,
            "reason_ru": self.reason_ru, "detail": self.detail,
            "run": None if self.run is None else self.run.to_dict(),
            "readings": [r.to_dict() for r in self.readings], "other_idea": self.other_idea,
            "venue": self.venue, "execution_ru": self.execution_ru,
            "money": [m.to_dict() for m in self.money],
        }

    def short_ru(self) -> str:
        """«пройден», «не проводился: нет версии на Binance»: one cell of a table."""
        if self.outcome == Outcome.NOT_RUN:
            return f"не проводился: {self.reason_ru}"
        return self.outcome_ru


VENUE_TITLE = {"binance": "Binance", "hyperliquid": "Hyperliquid"}
EXECUTION_RU = {
    True: "у нас есть адаптер заявок (frab на 10.8.0.5): прошедшая правила идея идёт в бумагу",
    False: "адаптера заявок на этой площадке у нас нет (ни в q-lab, ни во frab): идея, прошедшая "
           "все правила, получает «нужна инфраструктура» — это факт о нашем коде, а не отказ "
           "стратегии",
    None: "есть ли у нас адаптер заявок на этой площадке, не записано",
}


def execution_ru(venue: str) -> str:
    """The venue's fact about execution, in words (`venues/<venue>.yaml`, `execution_adapter`)."""
    from qlab.venues.config import load_venue

    config = load_venue(venue)
    return EXECUTION_RU[None if config is None else config.execution_adapter]


def adapter(venue: str) -> bool | None:
    from qlab.venues.config import load_venue

    config = load_venue(venue)
    return None if config is None else config.execution_adapter


@dataclass(frozen=True)
class VenueView:
    """One venue of an idea, read off its ladder: its verdict, what the lab can do there, its
    money check. The two views are what the console and the morning report show side by side."""

    venue: str
    title: str
    outcome: Outcome
    outcome_ru: str
    detail: str
    reason: Reason | None
    route: str | None
    adapter: bool | None
    execution_ru: str
    money: VenueMoney | None
    trial_id: int | None
    run_idea: str | None
    noise_power: float | None = None
    consistency_trial: int | None = None
    """The first-stage trial the pass was checked against instead of the noise bar; None when
    the noise bar decided (or there is no pass)."""

    @property
    def reason_ru(self) -> str | None:
        return None if self.reason is None else REASON_RU[self.reason]

    @property
    def by_history_only(self) -> bool:
        """A pass that rests on the consistency check with Binance's history, not on the noise bar
        (the bar had no power on this venue's short window)."""
        return self.outcome == Outcome.PASSED and self.consistency_trial is not None

    def history_note_ru(self) -> str:
        """«на Hyperliquid — только сверкой с историей Binance, планка шума там не решает (сила
        10%)»: what a pass of this venue rests on, when it is not the noise bar."""
        power = ("сила не измерена" if self.noise_power is None
                 else f"сила {self.noise_power:.0%}")
        return (f"на {self.title} — только сверкой с историей Binance, планка шума там не решает "
                f"({power})")

    def word_ru(self) -> str:
        """«пройдена», «не пройдена», «рано судить», «не проводилась: причина»."""
        return {Outcome.PASSED: "пройден", Outcome.FAILED: "не пройден",
                Outcome.UNDECIDED: "не решён (рано судить)",
                Outcome.NOT_RUN: f"не проводился: {self.reason_ru}"}[self.outcome]

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue, "title": self.title, "outcome": self.outcome.value,
            "outcome_ru": self.outcome_ru, "word_ru": self.word_ru(), "detail": self.detail,
            "reason": None if self.reason is None else self.reason.value,
            "reason_ru": self.reason_ru, "route": self.route, "adapter": self.adapter,
            "execution_ru": self.execution_ru,
            "money": None if self.money is None else self.money.to_dict(),
            "trial_id": self.trial_id, "run_idea": self.run_idea,
            "noise_power": self.noise_power, "consistency_trial": self.consistency_trial,
            "by_history_only": self.by_history_only,
        }


def summary_ru(views: tuple[VenueView, ...]) -> str:
    """Which venue passed and which did not, in one sentence (docs/VENUE_SPLIT.md). A pass that
    rests only on the check against Binance's history, because the noise bar had no power on the
    venue's short window, says so (owner, 2026-10-30: it must not read as a plain pass)."""
    by = {v.venue: v for v in views}
    binance, hl = by["binance"], by["hyperliquid"]
    history = "".join(f"; {v.history_note_ru()}" for v in (binance, hl) if v.by_history_only)
    if binance.outcome == Outcome.PASSED and hl.outcome == Outcome.PASSED:
        return "прошла на обеих площадках: на Binance и на Hyperliquid" + history
    if binance.outcome == Outcome.PASSED or hl.outcome == Outcome.PASSED:
        yes, no = (binance, hl) if binance.outcome == Outcome.PASSED else (hl, binance)
        tail = {Outcome.FAILED: "не прошла", Outcome.UNDECIDED: "рано судить",
                Outcome.NOT_RUN: f"не проводилась ({no.reason_ru})"}[no.outcome]
        extra = ("; на Binance у нас нет адаптера, идея ждёт инфраструктуры"
                 if yes.venue == "binance" else "")
        return f"прошла на {yes.title}, на {no.title} — {tail}{extra}{history}"
    words = {Outcome.FAILED: "не прошла", Outcome.UNDECIDED: "рано судить",
             Outcome.NOT_RUN: "не проводилась"}
    return (f"ни одна площадка не пройдена: на Binance — {words[binance.outcome]}, "
            f"на Hyperliquid — {words[hl.outcome]}")


@dataclass(frozen=True)
class Ladder:
    idea_id: str
    status: str
    steps: tuple[Step, ...]
    venue_gap: Any = None
    """`qlab.venue_gap.VenueGap` when one venue passed and the other did not: why they differ,
    from measurements (None when the venues agree)."""

    @property
    def venues(self) -> tuple[VenueView, ...]:
        """Binance and Hyperliquid read off steps 2, 3 and 4: the same two lines for every idea."""
        money = {m.venue: m for m in self.steps[3].money}
        out = []
        for step in (self.steps[1], self.steps[2]):
            assert step.venue is not None
            run = step.run
            out.append(VenueView(
                venue=step.venue, title=VENUE_TITLE[step.venue], outcome=step.outcome,
                outcome_ru=step.outcome_ru, detail=step.detail, reason=step.reason,
                route=None if run is None else run.route, adapter=adapter(step.venue),
                execution_ru=step.execution_ru or execution_ru(step.venue),
                money=money.get(step.venue),
                trial_id=None if run is None else run.trial_id,
                run_idea=None if run is None else run.idea_id,
                noise_power=None if run is None else run.noise_power,
                consistency_trial=None if run is None else run.consistency_trial))
        return tuple(out)

    @property
    def summary_ru(self) -> str:
        return summary_ru(self.venues)

    def to_dict(self) -> dict[str, Any]:
        return {"idea_id": self.idea_id, "status": self.status,
                "steps": [s.to_dict() for s in self.steps],
                "venues": [v.to_dict() for v in self.venues], "summary_ru": self.summary_ru,
                "venue_gap": None if self.venue_gap is None else self.venue_gap.to_dict()}


@dataclass(frozen=True)
class Ledgers:
    """Everything outside the registry the ladder reads (read only). All of it optional: a missing
    file is an empty mapping."""

    twins: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """`data/twins/ledger.json`: ideas for which no twin was built, with the reason in the code's
    words."""
    overlays: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """`data/overlays/ledger.json`: variants, whether their second stage is still owed."""
    restamp: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """`data/restamp/ledger.json`: the last recomputation of an old result."""
    implement: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """`data/implement/ledger.json`: cards the implementer works on."""
    queue: frozenset[str] = frozenset()
    """Card ideas named in `night/implement_queue.yaml`."""
    deferred_twins: frozenset[str] = frozenset()
    """Ideas the last night's twins stage did not reach (its nightly cap)."""
    money: Mapping[str, float] = field(default_factory=dict)
    """Idea -> `strategy_capital_usd` of its spec files (`specs/**/*.yaml`); an idea whose files
    state different sums, or none, is absent."""
    paper_books: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Idea -> its paper or live book (`night/paper_books.yaml`)."""
    stage1_survives: frozenset[str] = DEFAULT_STAGE1_SURVIVES
    """The first-stage routes that go on to the second (`night/stage2.yaml`)."""
    independent: bool = False
    """The newest rules judge the two venues independently (rules 2026-10-30.2): no first-stage
    route stops the second venue."""
    anomaly_routes: frozenset[str] = DEFAULT_ANOMALY_ROUTES
    """The routes the anomaly scan reads (`night/anomalies.yaml`)."""
    spec_stems: frozenset[str] = frozenset()
    """File names (without `.yaml`) found under `specs/`: by convention a spec carries its
    idea's id, so an idea with a spec file and no run is queued, not missing."""
    stage2_failures: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """`data/stage2/failures.json`: ideas whose second stage crashed, with the attempts so far and
    whether the idea was given up on (the Hyperliquid side then stays «не проводился: сбой»)."""
    root: Path | None = None
    """Where the stored comparisons and exports are looked for (None: the working directory)."""


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


_IDEA_LINE = re.compile(r"^idea_id\s*:\s*[\"']?([^\s\"'#]+)", re.M)
_MONEY_LINE = re.compile(r"^strategy_capital_usd\s*:\s*([0-9][0-9._]*)\s*(?:#.*)?$", re.M)


def spec_money(spec_dir: Path) -> dict[str, float]:
    """Idea -> the strategy's own money its spec files state (`strategy_capital_usd`). Read from
    the files' text, not through `load_spec`: the page of an idea must not load 160 specs. Two
    files of one idea that state different sums leave the idea without a sum: nobody can say which
    is meant."""
    seen: dict[str, set[float]] = {}
    for path in sorted(spec_dir.rglob("*.yaml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        money = _MONEY_LINE.search(text)
        idea = _IDEA_LINE.search(text)
        if money is None or idea is None:
            continue
        try:
            seen.setdefault(idea.group(1), set()).add(float(money.group(1).replace("_", "")))
        except ValueError:
            continue
    return {idea: next(iter(sums)) for idea, sums in seen.items() if len(sums) == 1}


def _independent() -> bool:
    from qlab.pipeline.stage2 import independent_venues

    return independent_venues()


def load_ledgers(root: Path = Path(".")) -> Ledgers:
    """The ledgers and configs as the night leaves them. Never raises: a missing or broken file
    reads as empty, and the ladder then names the reason it can name."""
    night = _yaml(root / "night" / "implement_queue.yaml")
    queue = frozenset(Path(str(c)).stem for c in night.get("cards") or [])
    books = _yaml(root / "night" / "paper_books.yaml")
    held: dict[str, Mapping[str, Any]] = {}
    for block in ("books", "live_books"):
        held.update({k: v for k, v in (books.get(block) or {}).items() if isinstance(v, dict)})
    money = spec_money(root / "specs")
    stage2 = _yaml(root / "night" / "stage2.yaml")
    anomalies = _yaml(root / "night" / "anomalies.yaml")
    states = sorted((root / "data" / "night").glob("2???-??-??.json"))
    deferred: frozenset[str] = frozenset()
    if states:
        deferred = frozenset((_json(states[-1]).get("twins") or {}).get("deferred") or [])
    stems = frozenset(p.stem for p in (root / "specs").rglob("*.yaml"))
    return Ledgers(
        twins=_json(root / "data" / "twins" / "ledger.json"),
        overlays=_json(root / "data" / "overlays" / "ledger.json"),
        restamp=_json(root / "data" / "restamp" / "ledger.json"),
        implement=_json(root / "data" / "implement" / "ledger.json"),
        stage2_failures=_json(root / "data" / "stage2" / "failures.json"),
        queue=queue, deferred_twins=deferred, money=money, paper_books=held, spec_stems=stems,
        stage1_survives=frozenset(stage2.get("stage1_survives") or DEFAULT_STAGE1_SURVIVES),
        independent=_independent(),
        anomaly_routes=frozenset((anomalies.get("scan") or {}).get("routes")
                                 or DEFAULT_ANOMALY_ROUTES),
        root=root)


# --------------------------------------------------------------------------
# Reading the registry


@dataclass
class _Run:
    idea_id: str
    trial: Trial
    spec: Spec

    @property
    def route(self) -> str | None:
        return None if self.trial.route is None else str(self.trial.route.value)

    @property
    def venue(self) -> str | None:
        return (self.spec.data_requirements or {}).get("source")

    @property
    def metrics(self) -> dict:
        return self.trial.metrics or {}


def _latest_run(session: Session, idea_id: str) -> _Run | None:
    found = latest_routed(session, idea_id)
    return None if found is None else _Run(idea_id, found[0], found[1])


def _latest_spec(session: Session, idea_id: str) -> Spec | None:
    return (session.query(Spec).filter(Spec.idea_id == idea_id)
            .order_by(Spec.id.desc()).first())


def _number(m: Mapping[str, Any], key: str) -> float | None:
    value = m.get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


def _power(m: Mapping[str, Any]) -> float | None:
    """The noise bar's strength, on the strategy's own money when the run has it."""
    cap = _number(m, "noise_cap_power")
    return cap if cap is not None else _number(m, "noise_power")


def _percentile(m: Mapping[str, Any]) -> float | None:
    cap = _number(m, "noise_cap_return_percentile")
    return cap if cap is not None else _number(m, "noise_return_percentile")


def _step_run(session: Session, run: _Run) -> StepRun:
    window = data_range(session, run.trial.id)
    rules = (session.query(Verdict.rules_version)
             .filter(Verdict.trial_id == run.trial.id, Verdict.rules_version.isnot(None))
             .limit(1).scalar())
    m = run.metrics
    return StepRun(
        idea_id=run.idea_id, trial_id=run.trial.id, route=run.route, venue=run.venue,
        data_start=None if window is None else window[0].isoformat(),
        data_end=None if window is None else window[1].isoformat(),
        rules_version=rules, stand_version=run.trial.stand_version,
        numbers=capital_text.numbers_ru(m, noise="present") if m else "",
        forward_days=_number(m, "forward_days"),
        forward_days_needed=_number(m, "forward_days_needed"),
        noise_percentile=_percentile(m), noise_power=_power(m),
        min_capital_usd=_number(m, "min_capital_usd"),
        consistency_trial=(None if _number(m, "stage1_trial_id") is None
                           else int(_number(m, "stage1_trial_id") or 0)))


def _short(text: str | None, n: int = 260) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _days_ru(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ") + " дн."


def _stand_words(run: _Run) -> str:
    return f" Слова стенда: «{_short(run.trial.route_reason)}»." if run.trial.route_reason else ""


def _run_detail(run: _Run, step_run: StepRun) -> str:
    """The plain-words detail of a step that was run."""
    route = run.route or "?"
    m = run.metrics
    parts = [ROUTE_NOTE_RU.get(route, route)]
    if run.venue:
        parts[0] += f" (данные {run.venue})"
    if route == "needs-forward":
        have, need = step_run.forward_days, step_run.forward_days_needed
        if have is not None and need is not None:
            parts.append(f"проверки вперёд {_days_ru(have)} из {_days_ru(need)} нужных")
        elif have is not None:
            parts.append(f"проверки вперёд {_days_ru(have)}; ждёт ещё и охвата режимов рынка")
    power, pct = step_run.noise_power, step_run.noise_percentile
    if power is not None and pct is not None:
        text = f"обгоняет шум {pct:.0%}, сила планки {power:.0%}"
        if power < MIN_POWER:
            text += (" — меньше 80%: планка шума такой длины не различает, и её провал "
                     "значит «рано судить»")
            if route == "paper" and 0.99 <= pct < 0.995:
                text += "; пройдена по самой границе"
        parts.append(text)
    if m.get("stage1_trial_id") is not None:
        parts.append(f"планку заменила сверка с историей первого этапа (прогон "
                     f"{int(m['stage1_trial_id'])})")
    if route == "reject":
        parts.append(_stand_words(run).strip())
    return "; ".join(p for p in parts if p).replace("; Слова", ". Слова")


def _blocked_from_run(run: _Run) -> tuple[Reason, str] | None:
    """A run of a mechanism the stand could not test says why in the reason's own words."""
    if run.route != "not-evaluable":
        return None
    text = run.trial.route_reason or ""
    if re.search(r"delisted|survivors? only|only survivors", text):
        return Reason.NO_DELISTED_PRICES, (
            "источник не отдаёт цен выбывших инструментов, остались бы только выжившие; "
            f"слова стенда: «{_short(text, 200)}»")
    return Reason.NOT_EVALUABLE, f"слова стенда: «{_short(text, 200)}»"


def _from_run(session: Session, number: int, run: _Run, *, other: str | None = None,
              prefix: str = "") -> Step:
    """A step shown from a run the stand has made."""
    blocked = _blocked_from_run(run)
    sr = _step_run(session, run)
    if blocked is not None:
        return Step(number, STEP_KEYS[number - 1], STEP_TITLES[number - 1], Outcome.NOT_RUN,
                    prefix + blocked[1], reason=blocked[0], run=sr, other_idea=other)
    outcome = ROUTE_OUTCOME.get(run.route or "", Outcome.UNDECIDED)
    return Step(number, STEP_KEYS[number - 1], STEP_TITLES[number - 1], outcome,
                prefix + _run_detail(run, sr), run=sr, other_idea=other)


def _not_run(number: int, reason: Reason, detail: str = "") -> Step:
    return Step(number, STEP_KEYS[number - 1], STEP_TITLES[number - 1], Outcome.NOT_RUN,
                detail, reason=reason)


# --------------------------------------------------------------------------
# Why a twin does not exist


def twin_reason(text: str) -> Reason:
    """The closed reason for a refusal recorded in `data/twins/ledger.json` (the code's words)."""
    low = text.lower()
    if "hourly candles" in low or "max_hourly_instruments" in low or "allow-heavy" in low:
        return Reason.HEAVY_DATA
    if ("binance counterpart" in low or "not listed on binance" in low
            or "binance does not list" in low or "spot legs come from" in low
            or "which binance does not" in low):
        return Reason.NO_BINANCE_VERSION
    if ("no declared alternative" in low or "no pair with the base" in low
            or "picked by hand" in low or "differ in" in low or "owner chooses" in low):
        return Reason.NO_CHOICE
    return Reason.TWIN_REFUSED_OTHER


def _selects_by_code(spec: Spec | None) -> bool:
    if spec is None:
        return False
    return str(spec.code_ref or "").endswith(("retune:Retune", "retune.Retune")) or bool(
        (spec.params or {}).get("selects_causally"))


# --------------------------------------------------------------------------
# The seven steps


def _blocker(session: Session, idea: Idea, ledgers: Ledgers) -> tuple[Reason, str]:
    """Why an idea with no run has none."""
    spec = _latest_spec(session, idea.id)
    if spec is not None or idea.id in ledgers.spec_stems:
        return Reason.WAITS_FOR_NIGHT, "спек записан, прогона ещё нет: его сделает ночь"
    rec = ledgers.implement.get(idea.id)
    if idea.id in ledgers.queue or (rec and rec.get("status") not in ("failed", "blocked")):
        return Reason.WAITS_FOR_NIGHT, "карточка стоит в очереди реализации ночи"
    return Reason.CARD_WITHOUT_IMPLEMENTATION, (
        "есть карточка, а кода и спека нет; в очереди реализации её нет")


def _step1(session: Session, idea: Idea, own: _Run | None, ledgers: Ledgers) -> Step:
    if own is None:
        reason, why = _blocker(session, idea, ledgers)
        return _not_run(1, reason, why)
    blocked = _blocked_from_run(own)
    sr = _step_run(session, own)
    if blocked is not None:
        return Step(1, STEP_KEYS[0], STEP_TITLES[0], Outcome.NOT_RUN, blocked[1],
                    reason=blocked[0], run=sr)
    if own.route == "error":
        return Step(1, STEP_KEYS[0], STEP_TITLES[0], Outcome.UNDECIDED,
                    "прогон упал до ответа." + _stand_words(own), run=sr)
    return Step(1, STEP_KEYS[0], STEP_TITLES[0], Outcome.PASSED,
                "стенд принял описание: у каждого числа есть источник, данные собраны, "
                "заглядывания вперёд нет (это проверяется на каждом прогоне)", run=sr)


def _no_second_stage(session: Session, idea: Idea, own: _Run, ledgers: Ledgers) -> Step:
    """Step 3 of a Binance idea whose second stage does not exist."""
    failed = ledgers.stage2_failures.get(idea.id) or {}
    if failed.get("given_up"):
        # The second stage crashed for good (twice in a row, or for a reason that does not pass):
        # a crash is no verdict, so the Hyperliquid step is «не проводился: сбой» and the idea
        # stands on Binance's verdict (night of 2026-10-31).
        return _not_run(3, Reason.STAGE2_FAILED,
                        f"прогон на Hyperliquid упал ({failed.get('attempts')} раз подряд"
                        + ("; причина постоянная" if failed.get("permanent") else "")
                        + f"): {_short(str(failed.get('error')), 220)}")
    if idea.status in CLOSED_STATUSES:
        if ledgers.independent and idea.status == IdeaStatus.REJECTED and own.route == "reject":
            # Rejected on Binance before the venues were judged independently: Hyperliquid is
            # owed, and the night's queue of closed ideas runs it (night/second_venue.yaml).
            return _not_run(3, Reason.SECOND_VENUE_QUEUED,
                            "отвергнута на Binance до того, как площадки стали оцениваться "
                            "независимо; Hyperliquid ещё не смотрел, ночь берёт такие идеи по "
                            "очереди, самые старые первыми")
        return _not_run(3, Reason.CLOSED_BEFORE,
                        f"статус «{idea.status.value}»: до второго этапа дело не дошло")
    if child_id_taken(session, idea.id):
        return _not_run(3, Reason.ID_TAKEN,
                        f"имя «{child_id(idea.id)}» принадлежит другой идее (опыту "
                        "исследователя); писать второй этап под ним нельзя")
    if not ledgers.independent and own.route not in ledgers.stage1_survives:
        return _not_run(3, Reason.FIRST_STEP_UNDECIDED,
                        "первый этап ничего не решил или отвергнут: второму этапу нечего "
                        "подтверждать")
    owed = (ledgers.overlays.get(idea.id) or {}).get("stage2_pending")
    return _not_run(3, Reason.WAITS_FOR_NIGHT,
                    ("первый этап прошёл или ждёт, " if own.route in ledgers.stage1_survives
                     else "Hyperliquid оценивается независимо от Binance, ")
                    + "второй этап ещё не записан"
                    + (" (в журнале надстроек он значится в долгу)" if owed else ""))


def _twin_step(session: Session, idea: Idea, own: _Run | None, spec: Spec | None,
               ledgers: Ledgers) -> Step:
    """Step 2 of an idea whose own run is not on Binance: a Binance version of it, or why not."""
    for suffix in (TWIN_SUFFIX, BINANCE_SUFFIX):
        other = session.get(Idea, idea.id + suffix)
        if other is None or other.parent_id != idea.id:
            continue
        run = _latest_run(session, other.id)
        if run is None:
            return _not_run(2, Reason.WAITS_FOR_NIGHT,
                            f"версия на Binance «{other.id}» записана, прогона ещё нет")
        hl = stage2_child(session, other.id)
        hl_run = None if hl is None else _latest_run(session, hl.id)
        kind = "двойник" if suffix == TWIN_SUFFIX else "версия на Binance"
        tail = ""
        if hl_run is not None:
            tail = f"; её второй этап «{hl.id}»: {_route_ru(hl_run.route)}"
        return _suffixed(
            _from_run(session, 2, run, other=other.id, prefix=f"{kind} «{other.id}»: "), tail)
    if _selects_by_code(spec):
        return _not_run(2, Reason.NO_BINANCE_VERSION,
                        "числа здесь уже выбирает код, но версии этой идеи на данных Binance нет")
    if idea.status in (IdeaStatus.PAPER, IdeaStatus.LIVE):
        return _not_run(2, Reason.IN_PRODUCTION,
                        f"статус «{idea.status.value}»: двойников строят только идеям в работе")
    recorded = ledgers.twins.get(idea.id)
    if recorded:
        reason = twin_reason(str(recorded.get("reason") or ""))
        return _not_run(2, reason, "слова кода (запись " + str(recorded.get("day") or "?") + "): "
                        + _short(str(recorded.get("reason")), 300))
    if own is not None and own.route != "needs-forward":
        return _not_run(2, Reason.TWIN_ONLY_WHILE_WAITING,
                        f"последний исход «{_route_ru(own.route)}»: решён, двойник не строится")
    if idea.id in ledgers.deferred_twins:
        return _not_run(2, Reason.WAITS_FOR_NIGHT,
                        "двойник не уместился в ночной предел (4 новых за ночь)")
    return _not_run(2, Reason.WAITS_FOR_NIGHT,
                    "двойник ещё не рассматривался: идея ждёт проверки вперёд, очередь ночи")


def _suffixed(step: Step, tail: str) -> Step:
    if not tail:
        return step
    return Step(step.number, step.key, step.title, step.outcome, step.detail + tail,
                reason=step.reason, run=step.run, readings=step.readings,
                other_idea=step.other_idea)


def _route_ru(route: str | None) -> str:
    from qlab.funnel import ROUTE_TITLE

    return ROUTE_TITLE.get(route or "", route or "—")


def _steps23(session: Session, idea: Idea, own: _Run | None, spec: Spec | None,
             ledgers: Ledgers) -> tuple[Step, Step]:
    if own is None:
        reason, why = _blocker(session, idea, ledgers)
        return _not_run(2, reason, why), _not_run(3, reason, why)
    if is_stage2_child(idea):
        parent = _latest_run(session, idea.parent_id) if idea.parent_id else None
        first = (_from_run(session, 2, parent, other=idea.parent_id,
                           prefix=f"первый этап «{idea.parent_id}»: ")
                 if parent is not None else
                 _not_run(2, Reason.WAITS_FOR_NIGHT, "у первого этапа нет прогона"))
        return first, _from_run(session, 3, own)
    if own.venue == BINANCE:
        second = stage2_child(session, idea.id)
        run = None if second is None else _latest_run(session, second.id)
        if run is not None:
            return (_from_run(session, 2, own),
                    _from_run(session, 3, run, other=second.id,
                              prefix=f"второй этап «{second.id}»: "))
        return _from_run(session, 2, own), _no_second_stage(session, idea, own, ledgers)
    return _twin_step(session, idea, own, spec, ledgers), _from_run(session, 3, own)


def strategy_money(idea: Idea | str, ledgers: Ledgers) -> float | None:
    """The money of the idea's strategy: the only place step 4 learns it, the spec's
    `strategy_capital_usd`. A second stage (`<idea>-hl`) is the same strategy on another venue and
    has its first stage's money when its own file states none."""
    idea_id = idea if isinstance(idea, str) else idea.id
    found = ledgers.money.get(idea_id)
    if found is None and not isinstance(idea, str) and idea.parent_id and is_stage2_child(idea):
        found = ledgers.money.get(idea.parent_id)
    return found


def _venue_money(idea: Idea, step: Step, money: float | None) -> VenueMoney:
    """The money check of one venue (step 2 or 3 is the venue's run): the capital that run needed
    against the strategy's own money. A venue with no run has nothing to check."""
    venue = step.venue or ""
    if idea.status in CLOSED_STATUSES and (step.outcome == Outcome.NOT_RUN or money is None):
        return VenueMoney(venue, Outcome.NOT_RUN, f"статус «{idea.status.value}»",
                          Reason.CLOSED_BEFORE)
    if step.outcome == Outcome.NOT_RUN:
        why = f" ({step.reason_ru})" if step.reason_ru else ""
        return VenueMoney(venue, Outcome.NOT_RUN,
                          f"на этой площадке прогона нет{why}: деньги проверять не на чем",
                          Reason.NO_VERSION_ON_VENUE)
    need = None if step.run is None else step.run.min_capital_usd
    if money is None:
        note = "" if need is None else f"прогону нужно ${need:,.0f}; сравнить не с чем"
        return VenueMoney(venue, Outcome.NOT_RUN, note, Reason.NO_STRATEGY_MONEY)
    outcome, detail = capital_fit(need, money)
    return VenueMoney(venue, outcome, detail)


def _step4(idea: Idea, steps: list[Step], ledgers: Ledgers) -> Step:
    """The strategy's money against the capital each venue's run needed: one line per venue. The
    step's own outcome is the venue where the lab can execute (Hyperliquid) when it has a line,
    else Binance's; the lines are what the page and the report show."""
    money = strategy_money(idea, ledgers)
    lines = tuple(_venue_money(idea, s, money) for s in (steps[1], steps[2]))
    binance, hl = lines
    lead = hl if hl.outcome != Outcome.NOT_RUN else binance
    detail = " · ".join(f"{VENUE_TITLE[m.venue]}: {m.outcome_ru}"
                        + (f" ({m.reason_ru})" if m.reason_ru else "")
                        + (f", {m.detail}" if m.detail else "") for m in lines)
    if lead.outcome == Outcome.NOT_RUN:
        reasons = [m.reason for m in lines]
        reason = (Reason.NO_STRATEGY_MONEY if Reason.NO_STRATEGY_MONEY in reasons
                  else Reason.CLOSED_BEFORE if Reason.CLOSED_BEFORE in reasons
                  else Reason.NO_VERSION_ON_VENUE)
        return Step(4, STEP_KEYS[3], STEP_TITLES[3], Outcome.NOT_RUN, detail, reason=reason,
                    money=lines)
    return Step(4, STEP_KEYS[3], STEP_TITLES[3], lead.outcome, detail, money=lines)


def capital_fit(need: float | None, money: float) -> tuple[Outcome, str]:
    """Whether the capital a run needed fits the strategy's own money: the whole of step 4."""
    if need is None:
        return Outcome.UNDECIDED, f"у стратегии ${money:,.0f}; нужный капитал в прогоне не измерен"
    if need <= money:
        return Outcome.PASSED, f"нужно ${need:,.0f} против денег стратегии ${money:,.0f}"
    return Outcome.FAILED, f"нужно ${need:,.0f}, своих ${money:,.0f}: не помещается"


def _readings(session: Session, idea: Idea, main: _Run | None, ledgers: Ledgers) -> list[Reading]:
    out: list[Reading] = []
    m = main.metrics if main is not None else {}
    from qlab.year_view import years_line

    line = years_line(m) if m else None
    out.append(Reading("years", line is not None,
                       line or "годы не определены: полных лет в проверке вперёд меньше двух"))
    if any(k.startswith("prot_") for k in m):
        out.append(Reading("protection", True, "защита Bv2 (P1–P4) записана, ничего не решает"))
    if "regime_bull_days" in m:
        out.append(Reading(
            "regimes", True,
            "режимы рынка в проверке: рост {} дн., боковик {} дн., падение {} дн.".format(
                *(int(_number(m, f"regime_{k}_days") or 0) for k in ("bull", "flat", "bear")))))
    scan = (session.query(AnomalyScan).filter(AnomalyScan.idea_id == idea.id)
            .order_by(AnomalyScan.trial_id.desc(), AnomalyScan.scanned_at.desc()).first())
    if scan is not None:
        fired = list(scan.fired or [])
        out.append(Reading(
            "anomalies", True,
            (f"сработало признаков: {len(fired)} ({', '.join(fired)})" if fired
             else "ни один признак не сработал") + f"; прогон {scan.trial_id}"))
    elif idea.status in CLOSED_STATUSES:
        out.append(Reading("anomalies", False, "аномалии не сканировались: идея закрыта",
                           Reason.CLOSED_BEFORE))
    elif main is not None and main.route in ledgers.anomaly_routes:
        out.append(Reading("anomalies", False, "аномалии ещё не сканировались: очередь ночи",
                           Reason.WAITS_FOR_NIGHT))
    else:
        out.append(Reading("anomalies", False,
                           "аномалии не сканировались: у прогона нет исхода, который сканируют",
                           Reason.NOT_REACHED))
    out += _overlay_reading(session, idea)
    return out


def _overlay_reading(session: Session, idea: Idea) -> list[Reading]:
    ids = [i for (i,) in session.query(Idea.id).filter(
        Idea.parent_id == idea.id, Idea.id.like(f"{idea.id}{VARIANT_INFIX}%"))]
    if not ids:
        return []
    from qlab import overlay_compare

    try:
        counts = overlay_compare.table(session, idea.id)["counts"]
        text = (f"надстроек {len(ids)}: помогли {counts.get('helped', 0)}, не помогли "
                f"{counts.get('not-helped', 0)}, неотличимо "
                f"{counts.get('indistinguishable', 0)} (сравнение с базой, ничего не меняет)")
    except Exception:  # noqa: BLE001 - a reading must not fail the page
        text = f"надстроек {len(ids)}; сравнение с базой не прочитано"
    return [Reading("overlays", True, text)]


def _step5(session: Session, idea: Idea, steps: list[Step], own: _Run | None,
           ledgers: Ledgers) -> Step:
    main = own
    if main is None:
        twin_run = next((s for s in steps[1:3] if s.run is not None and s.other_idea), None)
        main = None if twin_run is None else _latest_run(session, twin_run.other_idea)
    if main is None:
        why = steps[0].reason or Reason.NOT_REACHED
        return _not_run(5, why, "прогона нет, показывать нечего")
    readings = tuple(_readings(session, idea, main, ledgers))
    return Step(5, STEP_KEYS[4], STEP_TITLES[4], Outcome.PASSED,
                "; ".join(r.text for r in readings), run=_step_run(session, main),
                readings=readings)


def _step6(session: Session, idea: Idea, ledgers: Ledgers) -> Step:
    book = ledgers.paper_books.get(idea.id)
    if book is not None and (book.get("equity_from") or not book.get("paper_file")):
        return _not_run(6, Reason.NOT_IN_PAPER,
                        "живая книга: идея сразу в деньгах, бумажной книги у неё не было")
    if book is not None:
        text = f"книга «{book.get('label') or book.get('book') or idea.id}»"
        try:
            from qlab.paper_rule import paper_rule_for_idea

            rule = paper_rule_for_idea(session, idea.id, books=dict(ledgers.paper_books))
        except Exception:  # noqa: BLE001 - the page must not fail over a missing export
            rule = None
        if rule is None:
            return Step(6, STEP_KEYS[5], STEP_TITLES[5], Outcome.UNDECIDED,
                        text + ": выгрузка книги не прочитана, срок не посчитан")
        return Step(6, STEP_KEYS[5], STEP_TITLES[5],
                    Outcome.PASSED if rule.ready else Outcome.UNDECIDED, rule.line())
    if idea.status in CLOSED_STATUSES:
        return _not_run(6, Reason.CLOSED_BEFORE, f"статус «{idea.status.value}»")
    if idea.status in GATE_STATUSES:
        return _not_run(6, Reason.NOT_IN_PAPER, "")
    return _not_run(6, Reason.NOT_REACHED, "")


def _step7(session: Session, idea: Idea, steps: list[Step]) -> Step:
    status = idea.status
    if status in (IdeaStatus.PAPER, IdeaStatus.LIVE):
        return Step(7, STEP_KEYS[6], STEP_TITLES[6], Outcome.PASSED,
                    f"владелец перевёл идею: статус «{status.value}»")
    if status == IdeaStatus.VALIDATED:
        return Step(7, STEP_KEYS[6], STEP_TITLES[6], Outcome.UNDECIDED,
                    "прошла правила стенда, ждёт решения владельца")
    if status in (IdeaStatus.RETIRED, IdeaStatus.DECAYED):
        from qlab.registry.models import StageTransition

        row = (session.query(StageTransition).filter(
            StageTransition.idea_id == idea.id, StageTransition.to_status == status)
            .order_by(StageTransition.id.desc()).first())
        why = f": {_short(row.reason, 300)}" if row is not None and row.reason else ""
        return Step(7, STEP_KEYS[6], STEP_TITLES[6], Outcome.FAILED,
                    f"идея выведена ({status.value}){why}")
    if status == IdeaStatus.REJECTED:
        return _not_run(7, Reason.CLOSED_BEFORE, "отвергнута стендом, до владельца не дошла")
    return _not_run(7, Reason.NOT_REACHED, "")


def _on_venue(step: Step, venue: str) -> Step:
    """A step of steps 2 and 3 named for its venue, with what the lab can do there."""
    return dataclasses.replace(step, venue=venue, execution_ru=execution_ru(venue))


def _failed_rules(session: Session, trial_id: int) -> tuple:
    """The strategy rules a run did not pass, with their numbers (the infrastructure rules are a
    fact about our code, not about the strategy, and are left out)."""
    from qlab import venue_gap
    from qlab.rules.loader import load_latest
    from qlab.rules.schema import RuleKind

    try:
        kinds = {r.id: r.kind for r in load_latest().rules}
    except Exception:  # noqa: BLE001 - a reading must not fail the page
        kinds = {}
    rows = (session.query(Verdict).filter(Verdict.trial_id == trial_id, Verdict.passed.is_(False))
            .order_by(Verdict.id).all())
    seen: set[str] = set()
    out = []
    for row in rows:
        if kinds.get(row.rule_id) == RuleKind.INFRASTRUCTURE or row.rule_id in seen:
            continue
        seen.add(row.rule_id)
        out.append(venue_gap.FailedRule(
            row.rule_id, row.metric, None if row.value is None else float(row.value),
            row.comparator, None if row.threshold is None else float(row.threshold)))
    return tuple(out)


def _venue_facts(session: Session, step: Step, money: VenueMoney | None):
    """What `qlab.venue_gap` needs of one venue, read off its step and its run."""
    from qlab import venue_gap

    run = step.run
    common = dict(venue=step.venue or "", title=VENUE_TITLE.get(step.venue or "", step.venue or ""),
                  outcome=step.outcome.value,
                  money_outcome=None if money is None else money.outcome.value,
                  money_detail=None if money is None else money.detail)
    if run is None or step.outcome == Outcome.NOT_RUN:
        return venue_gap.VenueFacts(**common, not_run_reason=step.reason_ru,
                                    not_run_detail=_short(step.detail, 160) or None,
                                    run_idea=None if run is None else run.idea_id,
                                    trial_id=None if run is None else run.trial_id)
    trial = session.get(Trial, run.trial_id)
    spec = None if trial is None else session.get(Spec, trial.spec_id)
    costs = (spec.costs_model or {}) if spec is not None else {}
    from qlab.year_view import capital_view, year_rows

    metrics = (trial.metrics or {}) if trial is not None else {}
    rows = year_rows(capital_view(metrics) or metrics)
    return venue_gap.VenueFacts(
        years=tuple((r.year, r.ret, r.days) for r in rows),
        **common, route=run.route, run_idea=run.idea_id, trial_id=run.trial_id,
        data_start=run.data_start, data_end=run.data_end, forward_days=run.forward_days,
        forward_days_needed=run.forward_days_needed, noise_percentile=run.noise_percentile,
        noise_power=run.noise_power, consistency_trial=run.consistency_trial,
        taker_fee_bps=costs.get("taker_fee_bps"), slippage_bps=costs.get("slippage_bps"),
        failed_rules=_failed_rules(session, run.trial_id) if step.outcome != Outcome.PASSED
        else ())


def _venue_gap(session: Session, idea_id: str, steps: list[Step], ledgers: Ledgers):
    """Why the venues differ, when exactly one passed (`qlab.venue_gap`); None when they agree.
    The profit share needs the two runs taken apart, which the night measures and stores; a page
    only reads it."""
    from qlab import venue_gap

    money = {m.venue: m for m in steps[3].money}
    facts = [_venue_facts(session, step, money.get(step.venue or "")) for step in steps[1:3]]
    pair = venue_gap.applies(facts[0], facts[1])
    kind = "verdict"
    if pair is None and venue_gap.uneven(facts[0], facts[1]):
        pair, kind = (facts[0], facts[1]), "uneven"
    if pair is None:
        return None
    passed, other = pair
    shares, why = ([], "одна из площадок не проводилась: сравнивать монеты не с чем")
    if passed.trial_id is not None and other.trial_id is not None:
        shares, why = venue_gap.load_stored(
            idea_id, (passed.trial_id, other.trial_id), ledgers.root)
    return venue_gap.compose(passed, other, shares=shares, shares_why=why, kind=kind)


def build_ladder(session: Session, idea_id: str, ledgers: Ledgers | None = None) -> Ladder | None:
    """The seven steps of one idea; None when the idea does not exist. Reads only."""
    ledgers = ledgers or Ledgers()
    idea = session.get(Idea, idea_id)
    if idea is None:
        return None
    own = _latest_run(session, idea_id)
    spec = own.spec if own is not None else _latest_spec(session, idea_id)
    first = _step1(session, idea, own, ledgers)
    if first.outcome == Outcome.NOT_RUN and own is not None:
        # The run was a refusal to test: nothing after it has anything to compare.
        reason = Reason.FIRST_STEP_UNDECIDED
        two = _not_run(2, reason, "первая ступень не дала прогона, которому есть что сравнивать")
        three = _not_run(3, reason, "то же")
    else:
        two, three = _steps23(session, idea, own, spec, ledgers)
    steps = [first, _on_venue(two, "binance"), _on_venue(three, "hyperliquid")]
    steps.append(_step4(idea, steps, ledgers))
    steps.append(_step5(session, idea, steps, own, ledgers))
    steps.append(_step6(session, idea, ledgers))
    steps.append(_step7(session, idea, steps))
    gap = _venue_gap(session, idea_id, steps, ledgers)
    return Ladder(idea_id, idea.status.value, tuple(steps), gap)


def build_ladders(session: Session, idea_ids: list[str], ledgers: Ledgers | None = None
                  ) -> list[Ladder]:
    ledgers = ledgers or Ledgers()
    out = []
    for idea_id in idea_ids:
        found = build_ladder(session, idea_id, ledgers)
        if found is not None:
            out.append(found)
    return out


# --------------------------------------------------------------------------
# The morning report

SECTION_TITLE = "Лестница открытых идей"
SYMBOL = {Outcome.PASSED: "✓", Outcome.FAILED: "✗", Outcome.UNDECIDED: "~", Outcome.NOT_RUN: "·"}
STATUS_ORDER = {"live": 0, "paper": 1, "validated": 2, "bench": 3, "candidate": 4}


def open_idea_ids(session: Session) -> list[str]:
    """The ideas the owner follows: open, calibration noise left out, in the order of their
    standing (live, paper, validated, bench, candidate), then by name."""
    def parent_closed(idea: Idea) -> bool:
        """A second stage that still waits while its first stage is closed (the night's queue of
        closed ideas runs it; only a pass revives the pair) is not an open idea of its own."""
        parent = session.get(Idea, idea.parent_id) if idea.parent_id and is_stage2_child(idea) \
            else None
        return parent is not None and parent.status in CLOSED_STATUSES

    rows = [(i.status.value, i.id) for i in session.query(Idea)
            if i.status not in CLOSED_STATUSES and not i.id.startswith("noise-")
            and not parent_closed(i)]
    return [name for _, name in sorted(rows, key=lambda r: (STATUS_ORDER.get(r[0], 9), r[1]))]


def report_lines(session: Session, ledgers: Ledgers | None = None) -> list[str]:
    """One table of every open idea's ladder for the morning report: a symbol per step, Binance
    and Hyperliquid side by side (steps 2, 3 and the money of step 4 once per venue), which venue
    passed, then the count of reasons steps were not run. A reading; nothing here decides."""
    ladders = build_ladders(session, open_idea_ids(session), ledgers)
    if not ladders:
        return []
    lines = [
        "Одни и те же семь ступеней для каждой идеи (`docs/LADDER.md`), Binance и Hyperliquid "
        "рядом, у каждой площадки свой вердикт (`docs/VENUE_SPLIT.md`): 1 описание и проверки, "
        "2 Binance, 3 Hyperliquid, 4 деньги стратегии (Binance / Hyperliquid), 5 показания, "
        "6 бумага, 7 владелец. ✓ пройден, ✗ не пройден, ~ не решён (рано судить), · не проводился, "
        "причина названа на странице идеи; у пятой ступени ✓ значит «показания записаны». "
        "Прошедшая на Binance идея — «нужна инфраструктура»: адаптера Binance у нас нет.", "",
        "| идея | статус | 1 | 2 Binance | 3 Hyperliquid | 4 деньги B / H | 5 | 6 | 7 | "
        "итог по площадкам | почему площадки расходятся |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    reasons: dict[Reason, int] = {}
    for found in ladders:
        s = found.steps
        money = "/".join(SYMBOL[m.outcome] for m in s[3].money)
        cells = [SYMBOL[s[0].outcome], SYMBOL[s[1].outcome], SYMBOL[s[2].outcome], money,
                 SYMBOL[s[4].outcome], SYMBOL[s[5].outcome], SYMBOL[s[6].outcome]]
        lines.append(f"| {found.idea_id} | {found.status} | " + " | ".join(cells)
                     + f" | {found.summary_ru} | "
                     + ("" if found.venue_gap is None else found.venue_gap.short_ru) + " |")
        for step in found.steps:
            if step.reason is not None:
                reasons[step.reason] = reasons.get(step.reason, 0) + 1
    lines.append("")
    lines.append("Не проводилось, по причинам (ступеней): " + "; ".join(
        f"{REASON_RU[r]} — {n}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])) + ".")
    return lines


__all__ = [
    "SECTION_TITLE", "report_lines", "open_idea_ids",
    "BINANCE_SUFFIX", "DEFAULT_ANOMALY_ROUTES", "DEFAULT_STAGE1_SURVIVES", "GATE_STATUSES",
    "Ladder", "Ledgers", "OUTCOME_RU", "Outcome", "REASON_RU", "ROUTE_OUTCOME", "Reading",
    "Reason", "STEP_KEYS", "STEP_TITLES", "Step", "StepRun", "TWIN_SUFFIX", "VENUE_TITLE",
    "VenueMoney", "VenueView", "build_ladder", "build_ladders", "load_ledgers", "strategy_money",
    "summary_ru", "twin_reason",
]
