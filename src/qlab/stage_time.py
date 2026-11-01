"""A time limit for a night stage that runs ideas on the stand from a list (docs/PIPELINE.md,
«Лимит времени на шаг»).

Night of 2026-11-01 (run 31): `waiting` took 94.7 minutes, 72.5 of them one idea's second stage,
and nothing could stop it: only the restamp and the second-venue stages had a time limit, an
estimate before an idea starts and a guard for the idea that overruns. This module is that same
guard, shared, for the other stages of the calculations lane that walk a list (waiting, twins,
the experiments' twins, overlays) and, without a refusal, for the watch list (recheck).

`StageClock` is made once per call of a stage with the stage's limit (`max_minutes_per_night` in
its night/*.yaml; None: no limit, as for a hand run of named ideas). Before an item starts the stage
asks `refuse(item, estimate)`:

* the time is up: nothing new starts, the item waits for the next night;
* the estimate alone is larger than the stage's whole limit: the item is heavy, the night never
  starts it, the report and the page name it with the command that runs it by hand;
* the estimate is larger than what is left: it waits for the next night.

The estimate is the time the item took the last time (`SecondsLedger`, written by every run, a
hand run too) and, where a stage knows it, the data to fetch (`qlab.hl_fetch`). After the item the
stage calls `finished(item, seconds)`: an item that took the stage past its whole limit is not cut
(it has written its trials), it is recorded as the overrun, and the ledger now holds how long it
took, so the next night does not start it again until a hand run shows it has become faster.

The watch list is not skipped (`refuse` is not asked of it): the owner's list runs every night
whatever it costs. `finished` still records the items that were slow and the overrun, and the
report and the page say so."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

SECONDS_LEDGER = Path("data/stage_seconds.json")
LIMIT_KEY = "max_minutes_per_night"
SLOW_KEY = "slow_item_minutes"

TIME_UP, HEAVY, DEFERRED = "time_up", "heavy", "deferred"


def read_limit(path: Path, key: str = LIMIT_KEY) -> float | None:
    """The stage's time limit in minutes from its night/*.yaml; None when the file or the key is
    missing (no limit: a working directory without the project's night folder, a test)."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return None
    value = raw.get(key)
    if value is None:
        return None
    limit = float(value)
    if limit <= 0:
        raise ValueError(f"{path}: {key} must be positive")
    return limit


class SecondsLedger:
    """How long each item of each stage took the last time it ran: `{stage: {item: {seconds,
    day}}}` in one small file. Written by every run of a stage, by hand too, so a hand run of an
    item the night refuses shows the next night that it has become fast."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SECONDS_LEDGER
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def last(self, stage: str, item: str) -> float | None:
        seconds = ((self._load().get(stage) or {}).get(item) or {}).get("seconds")
        if isinstance(seconds, int | float) and not isinstance(seconds, bool) and seconds > 0:
            return float(seconds)
        return None

    def record(self, stage: str, item: str, seconds: float, day: str) -> None:
        with self._lock:
            data = self._load()
            data.setdefault(stage, {})[item] = {"seconds": round(seconds, 2), "day": day}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True),
                           encoding="utf-8")
            tmp.replace(self.path)


class StageClock:
    """One call of a stage and its time. See the module docstring."""

    def __init__(self, stage: str, limit_minutes: float | None, *, hand_command: str = "",
                 slow_minutes: float | None = None, ledger: SecondsLedger | None = None,
                 day: str = "", now: Callable[[], float] = time.monotonic) -> None:
        self.stage = stage
        self.limit_minutes = limit_minutes
        self.hand_command = hand_command
        self.slow_minutes = slow_minutes
        self.ledger = ledger
        self.day = day
        self._now = now
        self._started = now()
        self.time_up = False
        self.heavy: dict[str, str] = {}
        self.deferred: list[str] = []
        self.overran: dict | None = None
        self.slow: list[dict] = []

    # ------------------------------------------------------------------ the question

    def now(self) -> float:
        """The clock's own time in seconds, for a caller that times an item."""
        return self._now()

    @property
    def spent_minutes(self) -> float:
        return (self._now() - self._started) / 60

    @property
    def left_minutes(self) -> float | None:
        return None if self.limit_minutes is None else self.limit_minutes - self.spent_minutes

    def last_minutes(self, item: str) -> float | None:
        """The minutes the item took the last time, or None when it never ran (or there is no
        ledger)."""
        seconds = None if self.ledger is None else self.ledger.last(self.stage, item)
        return None if seconds is None else seconds / 60

    def refuse(self, item: str, estimate_minutes: float | None = None, why: str = ""
               ) -> tuple[str, str] | None:
        """Whether the item may not start now: None, or (kind, text) with `kind` one of
        `TIME_UP`, `HEAVY`, `DEFERRED`. The estimate defaults to the last time the item took."""
        if self.limit_minutes is None:
            return None
        if estimate_minutes is None:
            estimate_minutes = self.last_minutes(item)
            if estimate_minutes is not None:
                why = why or f"в прошлый раз шла {estimate_minutes:.0f} минут"
        if self.spent_minutes >= self.limit_minutes:
            self.time_up = True
            self.deferred.append(item)
            return TIME_UP, (f"время шага ({self.limit_minutes:g} минут на ночь) вышло: "
                             "ждёт следующей ночи")
        if estimate_minutes is None:
            return None
        if estimate_minutes > self.limit_minutes:
            text = (f"слишком тяжело для ночи: оценка {estimate_minutes:.0f} минут при пределе "
                    f"шага на всю ночь {self.limit_minutes:g}" + (f" ({why})" if why else "")
                    + (f"; вручную: {self.hand_command.format(idea=item)}"
                       if self.hand_command else ""))
            self.heavy[item] = text
            return HEAVY, text
        if estimate_minutes > (self.left_minutes or 0.0):
            self.deferred.append(item)
            return DEFERRED, (f"оценка {estimate_minutes:.0f} минут не уместится в остаток времени "
                              f"шага ({max(0.0, self.left_minutes or 0.0):.0f} из "
                              f"{self.limit_minutes:g}): ждёт следующей ночи")
        return None

    # ------------------------------------------------------------------ the answer

    def finished(self, item: str, seconds: float) -> None:
        """The item has run for `seconds`: kept in the ledger, and noted when it was slow or took
        the stage past its whole limit (it is never cut)."""
        if self.ledger is not None and seconds > 0:
            self.ledger.record(self.stage, item, seconds, self.day)
        minutes = seconds / 60
        if self.slow_minutes is not None and minutes >= self.slow_minutes:
            self.slow.append({"idea": item, "minutes": round(minutes, 1)})
        if (self.limit_minutes is not None and self.overran is None
                and self.spent_minutes > self.limit_minutes):
            self.overran = {"idea": item, "minutes": round(minutes, 1),
                            "limit_minutes": self.limit_minutes}

    def summary(self) -> dict:
        """What the report and the page show; stored in the stage's block of the night's state."""
        out: dict = {"limit_minutes": self.limit_minutes,
                     "minutes_spent": round(self.spent_minutes, 1)}
        if self.time_up:
            out["time_up"] = True
        if self.deferred:
            out["deferred"] = list(dict.fromkeys(self.deferred))
        if self.heavy:
            out["heavy"] = dict(self.heavy)
        if self.overran:
            out["overran"] = dict(self.overran)
        if self.slow:
            out["slow"] = list(self.slow)
        return out


# ---------------------------------------------------------------------------------- the words


def overran_line(over: Mapping, hand: str) -> str:
    """One item that took a stage past its whole limit."""
    return (f"Идея {over['idea']} шла {over['minutes']:g} минут, и шаг вышел за свой предел "
            f"({over['limit_minutes']:g} минут на ночь): её не прервали (испытания уже записаны), "
            "остальное отложено до следующей ночи."
            + (f" Сколько она шла, записано: пока она не станет быстрее, ночь её не начнёт, "
               f"вручную: `{hand.format(idea=over['idea'])}`." if hand else ""))


def time_lines(summary: Mapping | None, hand: str = "", *, watch: bool = False) -> list[str]:
    """The report's lines for a stage's time: what did not start, what was too heavy, what ran
    past the limit. `watch` is the watch list, which is never skipped: it only says it was slow."""
    if not summary:
        return []
    lines: list[str] = []
    limit = summary.get("limit_minutes")
    if watch and limit is not None and summary.get("minutes_spent", 0) > limit:
        lines.append(f"Список наблюдения шёл {summary['minutes_spent']:g} минут при ожидаемых "
                     f"{limit:g} на ночь; его не сокращают, но это дольше, чем рассчитано.")
    if summary.get("time_up"):
        names = f" ({', '.join(summary['deferred'])})" if summary.get("deferred") else ""
        lines.append(f"Время шага ({limit:g} минут на ночь) вышло: остальное ждёт следующей "
                     f"ночи{names}.")
    elif summary.get("deferred"):
        lines.append("Не уместились в остаток времени шага и ждут следующей ночи: "
                     f"{', '.join(summary['deferred'])}.")
    for item, why in (summary.get("heavy") or {}).items():
        lines.append(f"- {item}: не запущена — {why}")
    if summary.get("overran") and not watch:
        lines.append(overran_line(summary["overran"], hand))
    if watch:
        for slow in summary.get("slow") or []:
            lines.append(f"- {slow['idea']}: шла {slow['minutes']:g} минут (это больше, чем "
                         "обычно; список наблюдения не пропускается)")
    return lines


def page_note(summary: Mapping | None, hand: str = "", *, watch: bool = False) -> str | None:
    """What the stage left undone tonight and why, for the step's row on the pipeline page."""
    lines = time_lines(summary, hand, watch=watch)
    return " ".join(line.lstrip("- ") for line in lines) or None


__all__ = ["DEFERRED", "HEAVY", "LIMIT_KEY", "SECONDS_LEDGER", "SLOW_KEY", "SecondsLedger",
           "StageClock", "TIME_UP", "overran_line", "page_note", "read_limit", "time_lines"]
