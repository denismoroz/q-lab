"""One standard for every strategy page, checked by code (docs/IDEA_STANDARD.md).

Owner, 2026-11-04: «гдето они есть гдето нет - разберись пожалуйста с это кашей. составь список
фич который должен быть у стратегии и прогони все стратегии под одну гребенку».

The fourteen things every strategy page has (`ITEMS`). For one idea `check` returns them in order,
each with a state:

- `есть`      -- the page shows it;
- `нет`       -- the page lacks it (a gap or a defect), with the reason in plain Russian;
- `не нужно`  -- it does not apply to this idea (never run, no second venue by its nature), with
  the reason.

Nothing here computes a backtest, a metric or a verdict, and nothing is written. `gather` reads what
the idea's page already reads (registry rows, the ladder, the stored money series, the spec file,
the raw data's first days) into `Facts`; `check` is a pure function of `Facts`, so a test builds the
facts by hand. Calibration noise books and ideas the owner put away (`retired`) are skipped
entirely (`skipped`).
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

STATE_HAVE = "есть"
STATE_MISSING = "нет"
STATE_NA = "не нужно"
STATES = (STATE_HAVE, STATE_MISSING, STATE_NA)

NEVER_RUN_RU = "ещё не прогонялась"
NO_RUN_CLOSED_RU = "прогона нет: идея закрыта до расчёта"
SKIPPED_STATUSES = frozenset({"retired"})

RULES_WITH_SPLIT = "2026-11-04.1"
"""The first ruleset whose forward test is the last third of the data (2/3 and 1/3)."""

VENUE_START_FALLBACK = {"binance": date(2019, 9, 8), "hyperliquid": date(2025, 1, 1),
                        "hyperliquid-xyz": date(2025, 10, 1)}
"""The first day of the data we hold per venue, when the raw cache is not at hand."""
VENUE_RU = {"binance": "Binance", "hyperliquid": "Hyperliquid",
            "hyperliquid-xyz": "Hyperliquid xyz"}
START_SLACK_DAYS = 7
"""A start this many days after the data's first day is still «the start of the data»."""
WARMUP_SLACK_DAYS = 30
WARMUP_FACTOR = 1.5

# Reasons a venue's step of the ladder may legitimately have no answer (the idea has no version
# there by its nature): the venue then counts as answered by its reason. Any other reason is a gap.
LEGIT_ABSENT = frozenset({
    "no_binance_version", "no_version_on_venue", "heavy_data", "not_evaluable",
    "no_delisted_prices", "in_production", "closed_before", "card_without_implementation"})


@dataclass(frozen=True)
class ItemDef:
    number: int
    key: str
    title: str
    short: str


ITEMS: tuple[ItemDef, ...] = (
    ItemDef(1, "plain", "Простыми словами: что стратегия делает, на чём зарабатывает, в чём риск",
            "Слова"),
    ItemDef(2, "card", "Карточка: откуда идея и кто её предложил", "Карточка"),
    ItemDef(3, "outcome", "Итог и почему: статус и одна фраза причины", "Итог"),
    ItemDef(4, "ladder", "Лестница оценки: семь ступеней, у каждой исход или причина пропуска",
            "Лестница"),
    ItemDef(5, "chart", "График денег стратегии за весь период данных, по каждой площадке",
            "График"),
    ItemDef(6, "years", "По годам: доход и худшее падение каждого года", "Годы"),
    ItemDef(7, "numbers", "Главные числа на деньгах стратегии, отдельно за подбор и за проверку "
                          "вперёд", "Числа"),
    ItemDef(8, "noise", "Планка шума: лучше ли случайных книг и хватает ли данных, чтобы это "
                        "сказать", "Шум"),
    ItemDef(9, "money", "Нужные деньги: сколько нужно на все ордера и как работает на $1k, $3k, "
                        "$10k", "Деньги"),
    ItemDef(10, "venues", "Обе площадки: у каждой свой итог или причина, почему его нет",
            "Площадки"),
    ItemDef(11, "data", "Данные: с какого по какой день и почему не раньше", "Данные"),
    ItemDef(12, "coins", "Монеты и кто их выбрал: правило кода, слово владельца, объявленный "
                         "выбор или боевая книга", "Монеты"),
    ItemDef(13, "costs", "Издержки: комиссии, проскальзывание, плата за удержание контракта",
            "Издержки"),
    ItemDef(14, "double", "Проверка на двойника: нет ли уже такой же идеи", "Двойник"),
)
N_ITEMS = len(ITEMS)


@dataclass(frozen=True)
class Item:
    number: int
    key: str
    title: str
    short: str
    state: str
    reason: str | None = None
    """Why the item is not `есть` (plain Russian); None when it is."""
    detail: str | None = None
    """What the page shows for an item it has (the data range, the coins): one or two lines."""

    @property
    def text(self) -> str | None:
        return self.reason if self.state != STATE_HAVE else self.detail

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "key": self.key, "title": self.title, "short": self.short,
                "state": self.state, "reason": self.reason, "detail": self.detail}


# --------------------------------------------------------------------------
# What the check reads


@dataclass(frozen=True)
class CoinFacts:
    """Where the coin list of one spec comes from. `kind`: `code` (a rule of code picks the
    coins), `knowable` (the list was the obvious pick on every past day, by code), `owner` (the
    owner's word quoted in the spec), `choice` (a declared choice code makes), `production` (the
    production book's own list), `source` (the idea's own source names it), `agent` (nobody gave
    a source: a defect), `unknown` (the spec file could not be read)."""

    kind: str
    text: str
    coins: tuple[str, ...] = ()

    @property
    def defect(self) -> bool:
        return self.kind in ("agent", "unknown")


@dataclass(frozen=True)
class StageFacts:
    """What the check reads of one stage (one venue) of an idea."""

    number: int
    idea_id: str
    venue: str | None
    """The data venue of the stage's spec (`binance`, `hyperliquid`)."""
    trial_id: int | None = None
    series_window: str | None = None
    """`whole`, `forward_only` or None (no series stored)."""
    spec_start: date | None = None
    spec_end: date | None = None
    run_end: date | None = None
    judged_from: date | None = None
    instruments_chosen_at: date | None = None
    params_fixed_at: date | None = None
    data_reason: str | None = None
    """`sources["data.start"]`: the reason the spec gives for its first day."""
    knowable_from: date | None = None
    knowable_plain: str | None = None
    warmup_days: int | None = None
    coverage_day: date | None = None
    coverage_coin: str | None = None
    coins: CoinFacts | None = None
    has_spec: bool = False
    costs: Mapping[str, Any] | None = None
    own_source: str | None = None
    """`own_mechanism.idea_from` of the spec: where a self-made idea comes from."""


@dataclass(frozen=True)
class Facts:
    idea_id: str
    status: str
    has_run: bool
    plain_source: str | None = None
    """`qlab.plain` source of the description (`none` when there is no text)."""
    card: str | None = None
    """The idea whose card the page shows (its own id, or an ancestor's)."""
    outcome: Mapping[str, Any] | None = None
    """The latest routed trial as the page leads with it (`trial_json`)."""
    rules_version: str | None = None
    ladder: Mapping[str, Any] | None = None
    forward: Mapping[str, Any] | None = None
    stages: tuple[StageFacts, ...] = ()
    venue_starts: Mapping[str, date | None] = field(default_factory=dict)
    duplicate: Mapping[str, Any] | None = None
    double_checked: bool = False


# --------------------------------------------------------------------------
# The fourteen checks (pure)


def _item(number: int, state: str, reason: str | None = None, detail: str | None = None) -> Item:
    d = ITEMS[number - 1]
    return Item(d.number, d.key, d.title, d.short, state, reason, detail)


def _have(number: int, detail: str | None = None) -> Item:
    return _item(number, STATE_HAVE, detail=detail)


def _missing(number: int, reason: str) -> Item:
    return _item(number, STATE_MISSING, reason=reason)


def _na(number: int, reason: str) -> Item:
    return _item(number, STATE_NA, reason=reason)


def _venue(name: str | None) -> str:
    return VENUE_RU.get(name or "", name or "площадка")


def _no_run_reason(facts: Facts) -> str:
    return NEVER_RUN_RU if facts.status == "candidate" else NO_RUN_CLOSED_RU


def _plain(facts: Facts) -> Item:
    if facts.plain_source and facts.plain_source != "none":
        return _have(1)
    return _missing(1, "простого описания нет: стратегия не относится ни к одному описанному "
                       "семейству, а карточки у неё нет")


def _card(facts: Facts) -> Item:
    if facts.card == facts.idea_id:
        return _have(2)
    if facts.card:
        return _have(2, f"карточка идеи, от которой произошла эта: {facts.card}")
    if any(s.own_source for s in facts.stages):
        return _have(2, "источник записан в самом спеке")
    return _missing(2, "карточки нет: не записано, откуда идея и кто её предложил")


def _outcome(facts: Facts) -> Item:
    trial = (facts.outcome or {}).get("trial")
    if trial and trial.get("route"):
        return _have(3)
    if not facts.has_run:
        return _na(3, _no_run_reason(facts))
    return _missing(3, "исход прогона не записан: прогоны сделаны до 2026-10-01, когда исход ещё "
                       "не сохраняли")


def _ladder(facts: Facts) -> Item:
    steps = (facts.ladder or {}).get("steps") or []
    if len(steps) == 7:
        return _have(4)
    return _missing(4, "лестница оценки не построена")


def _chart(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(5, _no_run_reason(facts))
    problems = []
    drawn = 0
    for stage in facts.stages:
        if stage.trial_id is None:
            continue
        name = _venue(stage.venue)
        if stage.series_window == "whole":
            drawn += 1
        elif stage.series_window == "forward_only":
            problems.append(f"{name}: сохранён только отрезок проверки вперёд — весь период будет "
                            "после пересчёта")
        else:
            problems.append(f"{name}: ряд денег ещё не построен — будет после пересчёта")
    if problems:
        return _missing(5, "; ".join(problems))
    if drawn == 0:
        return _missing(5, "ряд денег не найден ни у одного этапа")
    return _have(5)


def _trial_view(facts: Facts) -> Mapping[str, Any] | None:
    return (facts.outcome or {}).get("trial")


def _years(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(6, _no_run_reason(facts))
    trial = _trial_view(facts)
    if trial is None:
        return _missing(6, "исход прогона не записан, разбить по годам нечего")
    view = trial.get("year_view")
    if not view or not view.get("years"):
        return _missing(6, "по годам не разбито: прогон сделан до того, как разбивку стали писать")
    if not view.get("capital"):
        return _missing(6, "годы есть только на балансе книги, не на деньгах стратегии — будут "
                           "после пересчёта")
    return _have(6)


def _numbers(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(7, _no_run_reason(facts))
    trial = _trial_view(facts)
    if trial is None:
        return _missing(7, "исход прогона не записан, чисел нет")
    h = trial.get("headline") or {}
    metrics = trial.get("metrics") or {}
    lacks = []
    if not h.get("on_capital"):
        lacks.append("числа не на деньгах стратегии")
    for key, word in (("ann_return", "доход в год"), ("max_dd", "худшее падение"),
                      ("sharpe", "отношение дохода к колебаниям"),
                      ("ann_return_ex_best_1pct", "доход без 1% лучших дней")):
        if h.get(key) is None:
            lacks.append(f"нет: {word}")
    if lacks:
        return _missing(7, "; ".join(lacks))
    version = facts.rules_version or ""
    if version and version < RULES_WITH_SPLIT:
        return _missing(7, f"прогон по старым правилам ({version}): проверка вперёд не последняя "
                           "треть данных — будет после пересчёта по правилам 2026-11-04.1")
    if metrics.get("judged_on_forward") != 1 or not h.get("fit"):
        return _missing(7, "подбор и проверка вперёд не разделены: у прогона нет двух частей")
    return _have(7)


def _noise(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(8, _no_run_reason(facts))
    trial = _trial_view(facts)
    if trial is None:
        return _missing(8, "исход прогона не записан, планки шума нет")
    h = trial.get("headline") or {}
    m = trial.get("metrics") or {}
    power = m.get("noise_cap_power") if h.get("on_capital") and isinstance(
        m.get("noise_cap_power"), int | float) else m.get("noise_power")
    if h.get("noise_percentile") is None:
        return _missing(8, "планка шума не посчитана для этого прогона")
    if power is None:
        return _missing(8, "планка посчитана, но не сказано, хватает ли данных, чтобы ей верить")
    return _have(8)


def _venue_runs(facts: Facts) -> dict[str, Mapping[str, Any]]:
    """Ladder steps 2 and 3 by venue."""
    out = {}
    for step in (facts.ladder or {}).get("steps") or []:
        if step.get("number") in (2, 3):
            out[step.get("venue") or ("binance" if step["number"] == 2 else "hyperliquid")] = step
    return out


def _money(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(9, _no_run_reason(facts))
    steps = (facts.ladder or {}).get("steps") or []
    money = next((s for s in steps if s.get("number") == 4), None)
    if money is None:
        return _missing(9, "лестница не построена, про деньги нечего сказать")
    runs = {v: s for v, s in _venue_runs(facts).items() if s.get("run")}
    if not runs:
        return _missing(9, "ни у одной площадки нет прогона, нужные деньги не посчитаны")
    rows = {m["venue"]: m for m in money.get("money") or []}
    lacks = []
    for venue, step in runs.items():
        run = step["run"]
        row = rows.get(venue)
        if run.get("min_capital_usd") is None:
            lacks.append(f"{_venue(venue)}: не посчитано, сколько денег нужно на все ордера")
        elif (row is not None and row.get("rule") == "fewer_own"
              and row.get("outcome") == "undecided"):
            # rules 2026-11-04.2: the decisive line is the book on the strategy's own money
            lacks.append(f"{_venue(venue)}: {row.get('detail') or 'на своих деньгах не измерено'}")
        elif row is None or not row.get("money_line_ru"):
            lacks.append(f"{_venue(venue)}: ряд на $1k, $3k, $10k ещё не измерен — будет после "
                         "пересчёта")
    if not lacks:
        return _have(9)
    if money.get("reason_ru") and not money.get("money"):
        lacks = [f"{money['reason_ru']}"]
    return _missing(9, "; ".join(lacks))


def _venues(facts: Facts) -> Item:
    if not facts.has_run and facts.status == "candidate":
        return _na(10, NEVER_RUN_RU)
    steps = _venue_runs(facts)
    if len(steps) < 2:
        return _missing(10, "лестница не показывает обе площадки")
    gaps, notes = [], []
    for venue in ("binance", "hyperliquid"):
        step = steps.get(venue)
        if step is None:
            gaps.append(f"{_venue(venue)}: ступени нет")
        elif step.get("run"):
            notes.append(f"{_venue(venue)}: {step.get('outcome_ru')}")
        else:
            reason = step.get("reason") or ""
            words = step.get("reason_ru") or "причина не записана"
            if reason in LEGIT_ABSENT:
                notes.append(f"{_venue(venue)}: {words}")
            else:
                gaps.append(f"на {_venue(venue)} ещё не прогонялась ({words})")
    if gaps:
        return _missing(10, "; ".join(gaps))
    return _have(10, "; ".join(notes))


def _span(stage: StageFacts) -> tuple[date | None, date | None]:
    """The first day the stage's data is judged from and its last day."""
    start = stage.spec_start
    if start is not None and stage.judged_from is not None:
        start = max(start, stage.judged_from)
    return start, stage.run_end or stage.spec_end


def explain_start(stage: StageFacts, venue_start: date | None) -> tuple[bool, str]:
    """(known reason, words) for the first day of the stage's data. A start no later than the
    venue's first day needs none; a later one needs a reason code can verify: the coin list's own
    listing day, the knowable-coins check, the warm-up the strategy needs, or a reason the spec
    writes under `sources["data.start"]`. A day a person wrote down is not one."""
    start, _ = _span(stage)
    if start is None:
        return False, "начало данных не записано"
    day = start.isoformat()
    if venue_start is None:
        return False, f"начало данных площадки неизвестно (данные с {day})"
    if start <= venue_start + timedelta(days=START_SLACK_DAYS):
        return True, "с начала данных площадки"
    if stage.data_reason:
        return True, f"причина записана в спеке: {stage.data_reason}"
    if stage.knowable_from and abs((start - stage.knowable_from).days) <= WARMUP_SLACK_DAYS:
        return True, ("монеты стали очевидным выбором только с "
                      f"{stage.knowable_from.isoformat()} (код проверил по данным биржи)")
    base = venue_start
    covered = ""
    if stage.coverage_day and stage.coverage_day > venue_start:
        base = stage.coverage_day
        covered = (f"монета {stage.coverage_coin} появилась на площадке только с "
                   f"{stage.coverage_day.isoformat()}")
        if start <= base + timedelta(days=START_SLACK_DAYS):
            return True, covered
    if stage.warmup_days:
        allowed = int(WARMUP_FACTOR * stage.warmup_days) + WARMUP_SLACK_DAYS
        if start <= base + timedelta(days=allowed):
            words = (f"разогрев: стратегии нужно около {stage.warmup_days} дн. истории, чтобы "
                     "посчитать первый сигнал")
            return True, f"{covered}; {words}" if covered else words
    hand = {d for d in (stage.instruments_chosen_at, stage.params_fixed_at) if d}
    if stage.judged_from and stage.judged_from in hand:
        cause = f"оценка идёт с дня, когда числа или монеты записал человек ({day})"
    elif stage.judged_from:
        cause = "в спеке задан день начала оценки без причины"
    else:
        cause = "спек просит данные с этого дня, а причина не записана"
    return False, (f"период короче доступного: начинается с {day}, потому что {cause}; данные "
                   f"площадки есть с {venue_start.isoformat()}")


def _data(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(11, _no_run_reason(facts))
    stages = [s for s in facts.stages if s.trial_id is not None]
    if not stages:
        return _missing(11, "диапазон данных прогона не найден")
    lines, bad = [], []
    for stage in stages:
        start, end = _span(stage)
        venue_start = facts.venue_starts.get(stage.venue or "") or VENUE_START_FALLBACK.get(
            stage.venue or "")
        known, words = explain_start(stage, venue_start)
        name = _venue(stage.venue)
        span = f"с {start.isoformat() if start else '?'} по {end.isoformat() if end else '?'}"
        (lines if known else bad).append(f"{name}: {span} — {words}")
    if bad:
        return _missing(11, "; ".join(bad))
    return _have(11, "; ".join(lines))


def _coins(facts: Facts) -> Item:
    specs = [s for s in facts.stages if s.has_spec and s.coins is not None]
    if not specs:
        if facts.status == "candidate" or not facts.has_run:
            return _na(12, "спека ещё нет" if facts.status == "candidate" else NO_RUN_CLOSED_RU)
        return _missing(12, "файл спека не найден: откуда монеты, проверить нечем")
    defects = [s for s in specs if s.coins.defect]
    if defects:
        first = defects[0]
        if first.coins.kind == "agent":
            return _missing(12, first.coins.text)
        return _missing(12, first.coins.text)
    seen, notes = set(), []
    for stage in specs:
        if stage.coins.text not in seen:
            seen.add(stage.coins.text)
            notes.append(stage.coins.text)
    return _have(12, "; ".join(notes))


def _costs(facts: Facts) -> Item:
    specs = [s for s in facts.stages if s.has_spec or s.costs]
    if not specs:
        if facts.status == "candidate" or not facts.has_run:
            return _na(13, "спека ещё нет" if facts.status == "candidate" else NO_RUN_CLOSED_RU)
        return _missing(13, "издержки не найдены: спека нет")
    lacks = []
    for stage in specs:
        costs = stage.costs or {}
        if costs.get("taker_fee_bps") is None or costs.get("slippage_bps") is None:
            lacks.append(f"{_venue(stage.venue)}: не заданы комиссия и проскальзывание")
    if lacks:
        return _missing(13, "; ".join(lacks))
    first = specs[0].costs or {}
    return _have(13, f"комиссия {first['taker_fee_bps']:g} б.п., проскальзывание "
                     f"{first['slippage_bps']:g} б.п.; плату за удержание контракта берёт стенд "
                     "из данных площадки")


def _double(facts: Facts) -> Item:
    if not facts.has_run:
        return _na(14, _no_run_reason(facts))
    if facts.duplicate:
        return _have(14, f"это двойник {facts.duplicate.get('of')}")
    if facts.double_checked:
        return _have(14, "двойника нет")
    return _missing(14, "проверка по результатам ещё не проводилась — будет при следующем прогоне "
                        "идеи или по `qlab standard --mark-doubles`")


def check(facts: Facts) -> list[Item]:
    """The fourteen items of the standard for one idea, in order."""
    return [_plain(facts), _card(facts), _outcome(facts), _ladder(facts), _chart(facts),
            _years(facts), _numbers(facts), _noise(facts), _money(facts), _venues(facts),
            _data(facts), _coins(facts), _costs(facts), _double(facts)]


def have_count(items: Sequence[Item]) -> int:
    """How many of the fourteen items the page has (`не нужно` is not one it has)."""
    return sum(1 for i in items if i.state == STATE_HAVE)


# --------------------------------------------------------------------------
# Reading the facts of a real idea


@functools.lru_cache(maxsize=16)
def _venue_start_cached(venue: str, signature: int) -> date | None:
    from qlab import knowable_coins

    try:
        cfg = knowable_coins.load_config()
    except Exception:  # noqa: BLE001 - a checkout without the config
        return VENUE_START_FALLBACK.get(venue)
    folder = cfg.raw_dir / venue / cfg.interval
    first: date | None = None
    if folder.is_dir():
        for path in folder.glob("*.parquet"):
            match = knowable_coins._STEM.match(path.stem)
            if match is None or ":" in match["coin"] or match["coin"].endswith(
                    knowable_coins.SPOT_SUFFIX):
                continue
            day = date.fromisoformat(match["start"])
            first = day if first is None else min(first, day)
    return first or VENUE_START_FALLBACK.get(venue)


def venue_start(venue: str) -> date | None:
    """The first day of the data we hold for a venue: the earliest start in the raw cache's file
    names, else the known first day of the venue."""
    try:
        from qlab import knowable_coins

        cfg = knowable_coins.load_config()
        folder = cfg.raw_dir / venue / cfg.interval
        signature = int(folder.stat().st_mtime_ns) if folder.is_dir() else 0
    except Exception:  # noqa: BLE001
        signature = 0
    return _venue_start_cached(venue, signature)


@functools.lru_cache(maxsize=512)
def _first_day(venue: str, coin: str) -> date | None:
    from qlab import knowable_coins

    try:
        cfg = knowable_coins.load_config()
        folder = cfg.raw_dir / venue / cfg.interval
        best = knowable_coins._newest_files(folder) if folder.is_dir() else {}
        path = best.get(coin)
        if path is None:
            return None
        import pandas as pd

        frame = pd.read_parquet(path, columns=["price"]).dropna()
        if frame.empty:
            return None
        return pd.Timestamp(frame.index.min()).date()
    except Exception:  # noqa: BLE001 - no data on disk: coverage is then unknown
        return None


def coverage_of(venue: str | None, coins: Sequence[str]) -> tuple[date | None, str | None]:
    """The first day on which every coin of a short explicit list has data on the venue, and the
    coin that decides it."""
    if not venue or not coins or len(coins) > 12:
        return None, None
    days = [(_first_day(venue, c.upper()), c) for c in coins]
    days = [(d, c) for d, c in days if d is not None]
    if not days:
        return None, None
    day, coin = max(days)
    return day, coin


_WARM_DAYS_KEYS = frozenset({
    "min_history_days", "lookbacks_days", "lookbacks", "lookback_days", "vol_window_days",
    "book_vol_window_days", "book_vol_min_days", "vol_window", "window", "min_observations",
    "window_days", "history_days"})


def _numbers_in(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        return [n for v in value for n in _numbers_in(v)]
    return []


def warmup_days(params: Mapping[str, Any], interval: str | None = None) -> int | None:
    """Days of history a strategy needs before its first signal, read off its own parameters: the
    longest lookback or window it names (a count of bars is turned into days by the interval).
    None when the parameters name none. An estimate to explain a later start, never a rule."""
    found: list[float] = []
    hours = 1.0 if (interval or "").endswith("h") else 24.0

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in _WARM_DAYS_KEYS:
                    found.extend(_numbers_in(value))
                elif key.endswith("_bars") or key == "window_bars":
                    found.extend(n * hours / 24.0 for n in _numbers_in(value))
                elif isinstance(value, Mapping | list):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(params)
    return int(max(found)) if found else None


def coin_facts(spec: Any, *, status: str, view: Any | None, knowable_plain: str | None,
               root: Path | None = None) -> CoinFacts:
    """Where the spec's coin list comes from (`qlab.coin_scope` rules, docs/COIN_SCOPE.md)."""
    from qlab import coin_scope

    listed = coin_scope.coin_list(spec)
    names = tuple(sorted(listed.coins)) if listed.coins else ()
    shown = ", ".join(names[:8]) + (" и др." if len(names) > 8 else "")
    if listed.by_code:
        return CoinFacts("code", "монеты выбирает правило кода (params.universe)", names)
    if listed.coins is None and not listed.where:
        return CoinFacts("code", "монеты — все подходящие на площадке, их отбирает код (по "
                                 "обороту и длине истории)", names)
    if spec.coins_knowable is not None or knowable_plain:
        return CoinFacts("knowable", f"{shown} — очевидный выбор, код проверил по данным биржи",
                         names)
    problems = coin_scope.problems(spec, root=root)
    if problems:
        return CoinFacts("agent", problems[0], names)
    if coin_scope.owner_word(spec):
        return CoinFacts("owner", f"{shown} — слово владельца", names)
    if coin_scope.declared_choice(spec):
        return CoinFacts("choice", f"{shown} — объявленный выбор, выбирает код", names)
    cfg = {}
    try:
        cfg = coin_scope.load_config(coin_scope.CONFIG, root)
    except OSError:
        pass
    parent = coin_scope.parent_path(spec, cfg)
    if parent or status in ("paper", "live") or spec.idea_id.endswith("-live"):
        return CoinFacts("production", f"{shown} — список боевой книги", names)
    if spec.sources.get("coins"):
        return CoinFacts("source", f"{shown} — из источника идеи", names)
    return CoinFacts("agent", f"монеты ({shown}) выбраны агентом без источника: нет слова "
                              "владельца, объявленного выбора и правила кода", names)


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _stage_facts(session: Any, entry: Mapping[str, Any], idea: Any, run_by_venue: Mapping,
                 root: Path | None) -> StageFacts:
    from qlab.plain.specs import view_for_idea
    from qlab.registry.models import Spec, Trial

    trial_id = entry.get("trial_id")
    stage_idea = entry["idea_id"]
    row = None
    if trial_id is not None:
        trial = session.get(Trial, trial_id)
        row = session.get(Spec, trial.spec_id) if trial is not None else None
    if row is None:
        row = (session.query(Spec).filter(Spec.idea_id == stage_idea)
               .order_by(Spec.version.desc()).first())
    view = view_for_idea(stage_idea, row, root=root)
    data = (dict(view.data) if view is not None
            else dict(getattr(row, "data_requirements", {}) or {}))
    extras = view.extras if view is not None else {}
    sources = extras.get("sources") if isinstance(extras.get("sources"), dict) else {}
    venue = data.get("source")
    known = extras.get("coins_knowable") if isinstance(extras.get("coins_knowable"), dict) else None
    remembered = None
    if known is None:
        from qlab import knowable_coins

        remembered = knowable_coins.recall(stage_idea) or knowable_coins.recall(idea.id)
        if remembered is not None:
            known = {"known_from": remembered.known_from, "plain": remembered.plain,
                     "from_start": remembered.from_start}
    costs = dict(view.costs) if view is not None else dict(getattr(row, "costs_model", {}) or {})
    run_end = None
    step = run_by_venue.get(venue)
    if step and step.get("run"):
        run_end = _as_date(step["run"].get("data_end"))
    spec_obj = None
    path = getattr(view, "path", None)
    if path:
        from qlab.pipeline.sources import QLAB_ROOT
        from qlab.pipeline.spec import load_spec

        try:
            spec_obj = load_spec((root or QLAB_ROOT) / path)
        except Exception:  # noqa: BLE001 - an unreadable spec file is a gap, not a crash
            spec_obj = None
    coins = None
    coverage_day = coverage_coin = None
    if spec_obj is not None:
        coins = coin_facts(spec_obj, status=idea.status.value, view=view,
                           knowable_plain=known.get("plain") if known else None, root=root)
        coverage_day, coverage_coin = coverage_of(venue, coins.coins) \
            if coins.kind in ("production", "source", "owner", "choice", "agent") else (None, None)
    elif view is not None and path is None:
        coins = None
    own = extras.get("own_mechanism")
    return StageFacts(
        number=entry["stage"], idea_id=stage_idea, venue=venue, trial_id=trial_id,
        series_window=entry.get("window"), spec_start=_as_date(data.get("start")),
        spec_end=_as_date(data.get("end")), run_end=run_end,
        judged_from=_as_date(extras.get("judged_from")),
        instruments_chosen_at=_as_date(extras.get("instruments_chosen_at")),
        params_fixed_at=_as_date(extras.get("params_fixed_at")),
        data_reason=(sources.get("data.start") or sources.get("data_start")) if sources else None,
        knowable_from=_as_date(known.get("known_from")) if known else None,
        knowable_plain=known.get("plain") if known else None,
        warmup_days=warmup_days(view.params, data.get("interval")) if view is not None else None,
        coverage_day=coverage_day, coverage_coin=coverage_coin, coins=coins,
        has_spec=view is not None and bool(costs or path), costs=costs or None,
        own_source=str(own.get("idea_from")) if isinstance(own, dict) and own.get("idea_from")
        else None)


def _lineage(session: Any, idea_id: str) -> list[str]:
    from qlab.registry.models import Idea

    out = [idea_id]
    while True:
        parent = session.query(Idea.parent_id).filter(Idea.id == out[-1]).scalar()
        if not parent or parent in out:
            return out
        out.append(parent)


@dataclass
class Shared:
    """What is the same for every idea of one pass, read once."""

    cards: Mapping[str, Any]
    ledgers: Any
    texts: Any = None
    duplicates: Mapping[str, Any] = field(default_factory=dict)
    root: Path | None = None


def shared(root: Path | None = None) -> Shared:
    from qlab import duplicates
    from qlab.api.cards import load_cards
    from qlab.ladder import load_ledgers
    from qlab.plain.texts import load_texts

    return Shared(cards={c["idea_id"]: c for c in load_cards()}, ledgers=load_ledgers(),
                  texts=load_texts(root), duplicates=duplicates.load_ledger(), root=root)


def skipped(idea: Any) -> bool:
    """Calibration noise books and ideas the owner put away carry no standard."""
    return idea.id.startswith("noise-") or idea.status.value in SKIPPED_STATUSES


def gather(session: Any, idea: Any, common: Shared | None = None) -> Facts:
    """Read the facts of one idea: only what its page already reads, nothing recomputed."""
    from qlab import forward_series
    from qlab.api.serializers import trial_json
    from qlab.ladder import build_ladder
    from qlab.plain import plain_page
    from qlab.registry.models import Spec, Trial

    common = common or shared()
    forward = forward_series.forward_view(session, idea.id) or {"stages": []}
    spec_ids = [i for (i,) in session.query(Spec.id).filter(Spec.idea_id == idea.id)]
    latest = None
    if spec_ids:
        latest = (session.query(Trial).filter(Trial.spec_id.in_(spec_ids), Trial.route.isnot(None))
                  .order_by(Trial.id.desc()).first())
    outcome = {"trial": trial_json(latest)} if latest is not None else None
    try:
        ladder_obj = build_ladder(session, idea.id, common.ledgers)
        ladder = ladder_obj.to_dict() if ladder_obj is not None else None
    except Exception:  # noqa: BLE001 - a table an old registry lacks
        ladder = None
    steps = {s["venue"]: s for s in (ladder or {}).get("steps", []) if s.get("number") in (2, 3)}
    try:
        plain = plain_page(session, idea, root=common.root, cards=common.cards,
                           texts=common.texts)
        plain_source = plain["what"]["source"]
    except Exception:  # noqa: BLE001
        plain_source = None
    card = next((i for i in _lineage(session, idea.id) if i in common.cards), None)
    entries = [e for e in forward.get("stages", [])]
    stages = tuple(_stage_facts(session, e, idea, steps, common.root) for e in entries)
    has_run = any(e.get("trial_id") is not None for e in entries) or latest is not None
    rules = None
    if latest is not None:
        run_steps = [s for s in steps.values() if s.get("run")]
        rules = next((s["run"].get("rules_version") for s in run_steps
                      if s["run"].get("trial_id") == latest.id), None)
        rules = rules or (run_steps[0]["run"].get("rules_version") if run_steps else None)
    venues = {s.venue: venue_start(s.venue) for s in stages if s.venue}
    return Facts(
        idea_id=idea.id, status=idea.status.value, has_run=has_run, plain_source=plain_source,
        card=card, outcome=outcome, rules_version=rules, ladder=ladder, forward=forward,
        stages=stages, venue_starts=venues,
        duplicate=common.duplicates["results"].get(idea.id),
        double_checked=idea.id in common.duplicates.get("checked", {}))


def standard_for(session: Any, idea: Any, common: Shared | None = None) -> dict[str, Any]:
    """The idea page's block: the fourteen items and how many the page has."""
    items = check(gather(session, idea, common))
    return {"idea_id": idea.id, "total": N_ITEMS, "have": have_count(items),
            "items": [i.to_dict() for i in items]}


def compact(items: Sequence[Item]) -> list[dict[str, Any]]:
    """One row cell per item: state and the short words."""
    return [{"number": i.number, "state": i.state, "text": i.text} for i in items]


def standard_all(session: Any, common: Shared | None = None) -> dict[str, Any]:
    """Every idea the standard applies to, with its items and the totals per item. A second stage
    of an idea (`<id>-hl`) is its parent's second venue and is not a row of its own."""
    from qlab.bench_label import bench_block, bench_reasons, status_key
    from qlab.registry.models import Idea
    from qlab.registry.stages import is_stage2_child
    from qlab.registry.strategies import families_of

    common = common or shared()
    rows = []
    families = families_of(session)
    # The bench in the registry's three words, not the bare «bench» (owner, 2026-11-04: «на странице
    # - Стандарт - такой статус - есть а в реестре - нет»).
    reasons = bench_reasons(session)
    for idea in session.query(Idea).order_by(Idea.id):
        if skipped(idea) or is_stage2_child(idea):
            continue
        items = check(gather(session, idea, common))
        reason = reasons.get(idea.id) if idea.status.value == "bench" else None
        rows.append({"id": idea.id, "title": idea.title, "status": idea.status.value,
                     "status_key": status_key(idea.status.value, reason),
                     "bench": bench_block(reason),
                     "family": families.get(idea.id, ("", "other"))[1], "have": have_count(items),
                     "items": compact(items)})
    return {"items": [{"number": d.number, "key": d.key, "title": d.title, "short": d.short}
                      for d in ITEMS],
            "rows": rows, "totals": totals(rows)}


def totals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per item: how many ideas lack it, and the commonest reasons."""
    out = []
    for d in ITEMS:
        cells = [r["items"][d.number - 1] for r in rows]
        missing = [c for c in cells if c["state"] == STATE_MISSING]
        reasons = Counter(_reason_key(c["text"]) for c in missing)
        out.append({"number": d.number, "key": d.key, "short": d.short, "title": d.title,
                    "have": sum(1 for c in cells if c["state"] == STATE_HAVE),
                    "missing": len(missing),
                    "not_needed": sum(1 for c in cells if c["state"] == STATE_NA),
                    "common_reasons": [{"reason": r, "count": n}
                                       for r, n in reasons.most_common(3)]})
    return out


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:\.\d)?")
_VENUE_PREFIX = re.compile(r"^(?:на )?(?:Binance|Hyperliquid(?: xyz)?)\b:?\s*")


def _reason_key(text: str | None) -> str:
    """The reason with the idea-specific parts (venue, dates, versions) cut off, so that the same
    cause counts as one."""
    if not text:
        return "причина не записана"
    head = _VENUE_PREFIX.sub("", text.split(";")[0].strip())
    for stem in ("список монет сужен без источника", "монеты ("):
        if head.startswith(stem):
            return stem if stem.endswith("источника") else "монеты выбраны агентом без источника"
    head = re.sub(r"\s*\(\d{4}-[^)]*\)", "", head)
    head = _DATE.sub("…", head)
    # «Binance: с … по … — период короче…» -> the part after the dash holds the cause
    if " — " in head and head.startswith("с …"):
        head = head.split(" — ", 1)[1]
    if head.startswith("период короче"):
        head = "период короче доступного, причина: " + head.split("потому что", 1)[-1].strip()
    return head.strip()


def missing_lines(items: Sequence[Item]) -> list[str]:
    return [f"{i.number}. {i.short}: {i.reason}" for i in items if i.state == STATE_MISSING]


def most_common_missing(rows: Sequence[Mapping[str, Any]], count: int = 3) -> list[tuple[str, int]]:
    """The items most ideas lack: (short name, ideas)."""
    found = [(t["short"], t["missing"]) for t in totals(rows) if t["missing"]]
    return sorted(found, key=lambda x: -x[1])[:count]


SYMBOL = {STATE_HAVE: "+", STATE_MISSING: "-", STATE_NA: "."}


def render_text(data: Mapping[str, Any], *, reasons: bool = True) -> str:
    """The standard as a text table: a row per idea, a column per item (`+` есть, `-` нет,
    `.` не нужно), totals under it and the commonest reasons of what is missing."""
    rows, totals_ = data["rows"], data["totals"]
    width = max([len(r["id"]) for r in rows] + [10])
    head = f"{'идея':<{width}}  {'статус':<9} " + " ".join(f"{d.number:>2}" for d in ITEMS)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = " ".join(f"{SYMBOL[c['state']]:>2}" for c in r["items"])
        lines.append(f"{r['id']:<{width}}  {r['status']:<9} {cells}  {r['have']}/{N_ITEMS}")
    lines.append("-" * len(head))
    lines.append(f"{'нет у скольких идей':<{width}}  {'':<9} "
                 + " ".join(f"{t['missing']:>2}" for t in totals_))
    lines.append("")
    for d in ITEMS:
        t = totals_[d.number - 1]
        lines.append(f"{d.number:>2}. {d.short:<9} нет у {t['missing']:>3}, "
                     f"не нужно {t['not_needed']:>3} — {d.title}")
        if reasons:
            lines += [f"      {r['count']:>3} × {r['reason']}" for r in t["common_reasons"]]
    return "\n".join(lines)


def report_line(data: Mapping[str, Any]) -> str | None:
    """The morning report's line: how many ideas lack something and what is lacking most."""
    rows = data["rows"]
    short = [r for r in rows if any(c["state"] == STATE_MISSING for c in r["items"])]
    if not short:
        return None
    common = ", ".join(f"«{name}» — у {n}" for name, n in most_common_missing(rows))
    return (f"стандарт страницы: у {len(short)} из {len(rows)} идей не хватает пунктов, чаще "
            f"всего — {common}. Таблица — страница «Стандарт» в консоли и `qlab standard`.")


__all__ = ["CoinFacts", "Facts", "ITEMS", "Item", "ItemDef", "N_ITEMS", "Shared", "StageFacts",
           "check", "coin_facts", "compact", "explain_start", "gather", "have_count",
           "missing_lines", "most_common_missing", "render_text", "report_line", "shared",
           "skipped", "standard_all", "standard_for", "totals", "venue_start", "warmup_days"]
