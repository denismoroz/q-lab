"""The four year signs A / B / V / G, read from stored metrics (docs/YEARS.md).

Owner, 2026-10-28: «а дава пока не будем строить отказы, а просто соберем по признакам
проходи или нет по всем 4м признакам. когда наберем статистику - решим - какое правило -
лучше». The signs decide nothing (rules 2026-10-28.2): each is recorded for every run as
"holds / does not hold / not defined", and this module also counts them over the registry,
so the owner can later see which sign separates what.

* **A** -- without its best calendar year the strategy still earns at least 4% a year.
* **B** -- at least two thirds of the counted years ended above zero.
* **V** -- without the best 1% of days the return is above zero.
* **G** -- the best calendar year gives at most half of the profit.

The lines are the informational readings of the ruleset (`ruleset.informational`, looked up
by `SignSpec.rule_id`); nothing here holds a threshold of its own. A trial stores no sign:
`read_signs` recomputes them from the trial's metrics, so a sign needs no table.

The statistics (`collect`, `summarize`) never fetch data and never write: they read the
registry's trials. Calibration noise books (`noise-*` ideas) are not strategies and are left
out.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from qlab.rules.engine import read_informational
from qlab.rules.schema import RuleSet

HOLDS, FAILS, UNDEFINED = "holds", "fails", "not defined"
"""The three states of a sign in the console and the tables."""
HOLDS_RU, FAILS_RU, UNDEFINED_RU = "держится", "не держится", "не определено"

STAGE_BINANCE, STAGE_HYPERLIQUID, STAGE_OTHER = "binance-long", "hyperliquid-stage2", "other"
"""Which stage of an idea's evaluation a trial is: the long history on Binance, the second
stage on Hyperliquid (docs/TWO_STAGE.md), or any other run (e.g. a spec that runs on
Hyperliquid by itself)."""
STAGES = (STAGE_BINANCE, STAGE_HYPERLIQUID, STAGE_OTHER)

OUTCOME_PASSED, OUTCOME_REJECTED, OUTCOME_OTHER = "passed", "rejected", "other"
OUTCOMES = (OUTCOME_PASSED, OUTCOME_REJECTED, OUTCOME_OTHER)
PASSED_ROUTES = frozenset({"paper", "shelf", "needs-infrastructure"})
"""Routes of a run that failed no strategy rule (`qlab.pipeline.evaluate.decide_route`):
passed and affordable now, passed but short of capital, passed but not runnable here."""
REJECTED_ROUTES = frozenset({"reject"})
"""Every other route -- needs-forward, needs-more-data, not-evaluable, error -- decided
nothing yet and counts as `other`."""

NO_YEARS_RU = "у прогона нет разбивки по годам"
FEW_YEARS_RU = "засчитанных лет меньше двух"
NO_YEAR_RU = "нет ни одного засчитанного года"
LOSS_RU = "прибыль за весь период не положительна"
NO_NUMBER_RU = "в прогоне нет этого числа"


@dataclass(frozen=True)
class SignSpec:
    letter: str
    """A, B, V, G in the owner's Cyrillic letters: А Б В Г."""
    rule_id: str
    """The informational reading of the ruleset that draws the sign's line."""
    title: str
    """What the sign says, in plain Russian (the line itself is in the ruleset)."""


SIGN_A, SIGN_B = "ex_best_year_4pct", "years_positive_share"
SIGN_V, SIGN_G = "ex_best_1pct_positive", "best_year_share_of_profit"
SIGNS = (
    SignSpec("А", SIGN_A, "без лучшего года не меньше 4% в год"),
    SignSpec("Б", SIGN_B, "прибыльных лет не меньше двух третей"),
    SignSpec("В", SIGN_V, "без 1% лучших дней выше нуля"),
    SignSpec("Г", SIGN_G, "лучший год даёт не больше половины прибыли"),
)
LETTERS = tuple(s.letter for s in SIGNS)


@dataclass(frozen=True)
class SignReading:
    letter: str
    rule_id: str
    title: str
    metric: str
    value: float | None
    comparator: str | None
    threshold: float | None
    holds: bool | None
    """None: not defined for this run."""
    reason: str | None = None
    """Why it is not defined, in plain Russian; None when it is defined."""

    @property
    def state(self) -> str:
        return UNDEFINED if self.holds is None else HOLDS if self.holds else FAILS

    @property
    def state_ru(self) -> str:
        return UNDEFINED_RU if self.holds is None else HOLDS_RU if self.holds else FAILS_RU

    def to_json(self) -> dict[str, Any]:
        return {"letter": self.letter, "id": self.rule_id, "title": self.title,
                "metric": self.metric, "value": self.value, "comparator": self.comparator,
                "threshold": self.threshold, "holds": self.holds, "state": self.state,
                "reason": self.reason}


def _number(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


def has_year_view(metrics: Mapping[str, object] | None) -> bool:
    """Whether the run stored a year-by-year view (runs before rules 2026-10-27.2 did not)."""
    return any(k.startswith("year_") or k == "years_counted" for k in (metrics or {}))


def not_defined_reason(spec: SignSpec, metrics: Mapping[str, object]) -> str:
    """Why sign `spec` has no number in `metrics`, in plain Russian."""
    counted = _number(metrics, "years_counted")
    if spec.rule_id == SIGN_V:
        return NO_NUMBER_RU
    if not has_year_view(metrics):
        return NO_YEARS_RU
    if spec.rule_id == SIGN_B:
        return NO_YEAR_RU
    if counted is None or counted < 2:
        return FEW_YEARS_RU
    if spec.rule_id == SIGN_G:
        return LOSS_RU
    return NO_NUMBER_RU


def read_signs(metrics: Mapping[str, object] | None, ruleset: RuleSet) -> list[SignReading]:
    """The signs `ruleset` draws a line for, against `metrics`, in the order А Б В Г. A sign
    the ruleset has no reading for (an older version) is left out. Pure."""
    m: Mapping[str, object] = metrics or {}
    readings = {r.rule_id: r for r in read_informational(m, ruleset)}  # type: ignore[arg-type]
    out = []
    for spec in SIGNS:
        r = readings.get(spec.rule_id)
        if r is None or r.comparator is None:
            continue
        out.append(SignReading(
            letter=spec.letter, rule_id=spec.rule_id, title=spec.title, metric=r.metric,
            value=r.value, comparator=r.comparator, threshold=r.threshold, holds=r.holds,
            reason=None if r.holds is not None else not_defined_reason(spec, m)))
    return out


def sign_line(reading: SignReading) -> str:
    """One sign in words: «А: без лучшего года не меньше 4% в год — держится (13.0%)»."""
    text = f"{reading.letter}: {reading.title} — {reading.state_ru}"
    if reading.holds is None:
        return f"{text} ({reading.reason})"
    return f"{text} ({reading.value * 100:.1f}%)" if reading.value is not None else text


def signs_sentence(metrics: Mapping[str, object] | None, ruleset: RuleSet) -> str | None:
    """The signs as one compact tail for a report line, e.g. «признаки: А держится, Б держится,
    В не держится, Г не определено (засчитанных лет меньше двух)». None when the ruleset draws
    no sign."""
    readings = read_signs(metrics, ruleset)
    if not readings:
        return None
    parts = [f"{r.letter} {r.state_ru}" + (f" ({r.reason})" if r.reason else "")
             for r in readings]
    return "признаки: " + ", ".join(parts)


def latest_ruleset() -> RuleSet | None:
    """The newest ruleset, or None when it cannot be read (a report line then goes without
    the signs rather than failing). Read afresh each time: a few reports call it, and a failed
    read must not stick."""
    from qlab.rules.loader import load_latest

    try:
        return load_latest()
    except Exception:  # noqa: BLE001 - a view must not fail over a missing rules file
        return None


# --------------------------------------------------------------------------
# Statistics over the registry


@dataclass(frozen=True)
class SignRow:
    """One judged trial and its four signs."""

    idea_id: str
    status: str
    trial_id: int
    stage: str
    route: str
    years_counted: int
    signs: dict[str, bool | None] = field(default_factory=dict)
    """letter -> holds / fails / not defined (True / False / None)."""

    @property
    def outcome(self) -> str:
        return (OUTCOME_PASSED if self.route in PASSED_ROUTES
                else OUTCOME_REJECTED if self.route in REJECTED_ROUTES else OUTCOME_OTHER)

    def to_json(self) -> dict[str, Any]:
        return {"idea_id": self.idea_id, "status": self.status, "trial_id": self.trial_id,
                "stage": self.stage, "route": self.route, "outcome": self.outcome,
                "years_counted": self.years_counted,
                "signs": {k: _state(v) for k, v in self.signs.items()}}


def _state(holds: bool | None) -> str:
    return UNDEFINED if holds is None else HOLDS if holds else FAILS


def stage_of(idea: Any, spec: Any) -> str:
    """Which stage of its idea's evaluation a trial is. The second stage is the child idea
    of the stage-2 link (`qlab.registry.stages`); a run on Binance's data is the long history;
    anything else is `other`."""
    from qlab.registry.stages import is_stage2_child

    if is_stage2_child(idea):
        return STAGE_HYPERLIQUID
    return STAGE_BINANCE if (spec.data_requirements or {}).get("source") == "binance" \
        else STAGE_OTHER


def collect(session: Session, ruleset: RuleSet, *, all_trials: bool = False,
            on_capital: bool = False) -> list[SignRow]:
    """The signs of the registry's judged trials, one row per idea (its latest trial) or, with
    `all_trials`, one row per trial -- every trial counts, the deflation way.

    `on_capital` reads the signs off the year summaries on the strategy's own money
    (`cap_year_*`, docs/CAPITAL_BASE.md) instead of the stand's; a trial from before those
    summaries has none and is left out.

    A judged trial is one that carries a route and a year view. Of an idea's trials the
    latest such one is taken (the highest id), so a later run without a year view does not
    hide an earlier one that has it. Calibration noise books are not ideas to judge."""
    from qlab.registry.models import Idea, Spec, Trial
    from qlab.year_view import capital_view, year_rows

    query = (session.query(Trial, Spec, Idea).join(Spec, Trial.spec_id == Spec.id)
             .join(Idea, Spec.idea_id == Idea.id)
             .filter(Trial.route.isnot(None), Trial.metrics.isnot(None),
                     ~Idea.id.like("noise-%"))
             .order_by(Idea.id, Trial.id))
    rows: list[SignRow] = []
    for trial, spec, idea in query:
        metrics = trial.metrics or {}
        if on_capital:
            metrics = capital_view(metrics)
        if not year_rows(metrics):
            continue
        counted = _number(metrics, "years_counted")
        rows.append(SignRow(
            idea_id=idea.id, status=idea.status.value, trial_id=trial.id,
            stage=stage_of(idea, spec), route=trial.route.value,
            years_counted=int(counted) if counted is not None else 0,
            signs={r.letter: r.holds for r in read_signs(metrics, ruleset)}))
    if all_trials:
        return rows
    latest: dict[str, SignRow] = {}
    for row in rows:  # ordered by trial id inside an idea: the last one wins
        latest[row.idea_id] = row
    return list(latest.values())


def summarize(rows: Iterable[SignRow]) -> dict[str, Any]:
    """Per sign: how many hold, fail, are not defined; and the same split against the run's
    outcome (passed / rejected / other) and against its stage."""
    rows = list(rows)
    totals: dict[str, Counter] = {}
    by_outcome: dict[str, dict[str, Counter]] = {}
    by_stage: dict[str, dict[str, Counter]] = {}
    for letter in LETTERS:
        totals[letter] = Counter({HOLDS: 0, FAILS: 0, UNDEFINED: 0})
        by_outcome[letter] = {o: Counter({HOLDS: 0, FAILS: 0, UNDEFINED: 0}) for o in OUTCOMES}
        by_stage[letter] = {s: Counter({HOLDS: 0, FAILS: 0, UNDEFINED: 0}) for s in STAGES}
    for row in rows:
        for letter, holds in row.signs.items():
            state = _state(holds)
            totals[letter][state] += 1
            by_outcome[letter][row.outcome][state] += 1
            by_stage[letter][row.stage][state] += 1
    return {
        "trials": len(rows),
        "ideas": len({r.idea_id for r in rows}),
        "totals": {k: dict(v) for k, v in totals.items()},
        "by_outcome": {k: {o: dict(c) for o, c in v.items()} for k, v in by_outcome.items()},
        "by_stage": {k: {s: dict(c) for s, c in v.items()} for k, v in by_stage.items()},
    }


def render_table(rows: list[SignRow]) -> list[str]:
    """One line per row: idea, status, trial, stage, outcome, counted years, then А Б В Г
    as holds / fails / not defined."""
    width = max([len(r.idea_id) for r in rows] + [4])
    head = (f"{'idea':<{width}} {'status':<11} {'trial':>7} {'stage':<18} {'route':<21} "
            f"{'years':>5}  " + "  ".join(f"{letter:<11}" for letter in LETTERS))
    lines = [head]
    for r in rows:
        cells = "  ".join(f"{_state(r.signs.get(letter)) if letter in r.signs else '-':<11}"
                          for letter in LETTERS)
        lines.append(f"{r.idea_id:<{width}} {r.status:<11} {r.trial_id:>7} {r.stage:<18} "
                     f"{r.route:<21} {r.years_counted:>5}  {cells}")
    return lines


def _cross(title: str, table: dict[str, dict[str, dict[str, int]]], keys: tuple[str, ...]
           ) -> list[str]:
    lines = [title]
    for spec in SIGNS:
        lines.append(f"  {spec.letter}: {spec.title}")
        lines.append(f"      {'':<20}{HOLDS:>7}{FAILS:>7}{UNDEFINED:>13}")
        for key in keys:
            c = table[spec.letter][key]
            lines.append(f"      {key:<20}{c[HOLDS]:>7}{c[FAILS]:>7}{c[UNDEFINED]:>13}")
    return lines


def render_summary(summary: dict[str, Any]) -> list[str]:
    """The totals per sign and the two cross-tables (sign against the run's outcome, and
    against its stage)."""
    lines = [f"totals over {summary['trials']} trial(s) of {summary['ideas']} idea(s):",
             f"      {'':<20}{HOLDS:>7}{FAILS:>7}{UNDEFINED:>13}"]
    for spec in SIGNS:
        t = summary["totals"][spec.letter]
        lines.append(f"  {spec.letter}   {'':<20}{t[HOLDS]:>7}{t[FAILS]:>7}{t[UNDEFINED]:>13}"
                     f"   {spec.title}")
    lines += [""] + _cross(
        "against the run's outcome (passed = paper, shelf, needs-infrastructure; "
        "rejected = reject; other = every undecided route):", summary["by_outcome"], OUTCOMES)
    lines += [""] + _cross("against the stage of the evaluation:", summary["by_stage"], STAGES)
    return lines


SECTION_TITLE = "Четыре признака по годам: статистика"


def report_lines(session: Session, ruleset: RuleSet) -> list[str]:
    """The morning report's few lines: the totals over each idea's latest judged run, and
    what they rest on. The table and the cross-tables are `qlab year-signs`; with
    `--capital` they are read on the strategy's own money (docs/CAPITAL_BASE.md)."""
    latest = collect(session, ruleset)
    every = collect(session, ruleset, all_trials=True)
    if not latest:
        return []
    summary = summarize(latest)
    lines = [
        "Признаки только записываются, ничего не решают (владелец, 2026-10-28: «когда наберем "
        "статистику - решим - какое правило - лучше»). Основа: последний прогон каждой идеи "
        f"с разбивкой по годам — {summary['ideas']} идей (второй этап идеи считается отдельно); "
        f"всех таких прогонов в реестре {len(every)}.", ""]
    for spec in SIGNS:
        t = summary["totals"][spec.letter]
        lines.append(f"- {spec.letter} ({spec.title}): держится {t[HOLDS]}, не держится "
                     f"{t[FAILS]}, не определено {t[UNDEFINED]}")
    capital = summarize(collect(session, ruleset, on_capital=True))
    if capital["trials"]:
        lines += ["", "То же на деньгах стратегии (docs/CAPITAL_BASE.md), для "
                  f"{capital['ideas']} идей, у которых прогон уже посчитан так:"]
        for spec in SIGNS:
            t = capital["totals"][spec.letter]
            lines.append(f"- {spec.letter}: держится {t[HOLDS]}, не держится {t[FAILS]}, "
                         f"не определено {t[UNDEFINED]}")
    lines += ["", "Таблица по идеям и разрезы по исходу и этапу: `uv run qlab year-signs` "
              "(`--capital` — на деньгах стратегии)."]
    return lines


__all__ = [
    "FAILS", "HOLDS", "LETTERS", "OUTCOMES", "SECTION_TITLE", "SIGNS", "STAGES", "SignReading",
    "SignRow", "SignSpec", "UNDEFINED", "collect", "has_year_view", "latest_ruleset",
    "not_defined_reason", "read_signs", "render_summary", "render_table", "report_lines",
    "sign_line", "signs_sentence", "stage_of", "summarize"]
