"""Can the earning be told from zero? A reading of every run that decides nothing
(docs/NOISE_BAR.md, stand 2026-11-03.1).

Owner, 2026-11-03: «да, делай А+Б+Г». The noise bar asks whether a result beats random books of
the same shape; this asks the plainer question beside it: if the strategy's days were drawn again
in the same kind of order, how often would its average day still be above zero?

**The numbers.** Every run (the judged part, the strategy's own money, docs/CAPITAL_BASE.md) gives
its returns per UTC day: each day's share of the capital the strategy started with, the days of a
strategy that does not reinvest summed within a day, those of one that does compounded (the same
way `qlab.overlay_compare` makes its daily series). The days are resampled with a **circular
moving-block bootstrap** -- the very one the comparison of an overlay with its base uses
(`qlab.overlay_compare._block_indices`), so that the dependence between neighbouring days is kept.
From 2000 resamples:

* `cap_mean_positive_p` -- the share of resamples whose average day is at or below zero (a
  one-sided p-value for "the strategy earns nothing"; the smallest it can be is 0, which with 2000
  resamples reads «меньше 1 из 2000»);
* `cap_ann_ci_low`, `cap_ann_ci_high` -- the 95% interval of the yearly return (2.5th and 97.5th
  percentile of the resampled yearly returns, by the same convention as `cap_ann_return`: summed
  for a book that does not reinvest, compounded for one that does).

**Block length.** `n ** (1/3)` days, at least 2 (`qlab.overlay_compare.block_length`): the rule of
thumb for the variance of a mean of dependent observations (Hall, Horowitz and Jing, 1995), the
one the overlay comparison already uses -- a second, different choice would make the two readings
not comparable. **Seed.** One fixed number (`SEED`, the day the reading was introduced): the same
days always give the same p and the same interval, so a run re-measured on the same data changes
nothing. **Resamples.** 2000, as in the overlay comparison.

**Not defined.** Fewer than `MIN_DAYS` days, or a run whose base of percent is undefined (a book
with deposits): no key is written, and the reading says so.

**It decides nothing.** Owner's rule «сначала замер, потом правило»: the three numbers are recorded
in `trial.metrics`, shown beside the noise bar, and counted over the registry (`collect`,
`summarize`: how many ideas would pass and fail at p <= `ALPHA`, against the outcome and against
the noise bar), so that the owner can later make a rule of it. No rules file reads them
(`informational` readings are not used for this one: the line is not drawn).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from qlab.overlay_compare import (
    N_BOOT,
    _annual_rows,
    _block_indices,
    _per_day,
    block_length,
)

SEED = 20261103
"""Fixed on 2026-11-03, the day the reading was introduced: the same days give the same numbers."""
ALPHA = 0.05
"""The line the statistics count at: p at or below it would pass (`overlay_compare.ALPHA`)."""
MIN_DAYS = 30
"""Fewer days than this say nothing about an average day (`overlay_compare.compare_series` refuses
the same)."""

P_KEY, LOW_KEY, HIGH_KEY = "cap_mean_positive_p", "cap_ann_ci_low", "cap_ann_ci_high"
DAYS_KEY, BLOCK_KEY = "cap_earning_days", "cap_earning_block_days"
KEYS = (P_KEY, LOW_KEY, HIGH_KEY, DAYS_KEY, BLOCK_KEY)

PASSES, FAILS, UNDEFINED = "passes", "fails", "not defined"
PASSES_RU, FAILS_RU, UNDEFINED_RU = "отличим от нуля", "не отличим от нуля", "не определено"
SECTION_TITLE = "Заработок отличим от нуля: статистика"


@dataclass(frozen=True)
class Test:
    """The bootstrap of one daily series."""

    p: float
    ann_low: float
    ann_high: float
    days: int
    block_days: int
    n_boot: int
    seed: int


def daily_money(capital_return: pd.Series, *, additive: bool) -> pd.Series:
    """A run's returns on the strategy's own money per UTC day (bars of a day summed when the
    strategy does not reinvest, compounded when it does)."""
    return _per_day(capital_return, additive=additive)


def bootstrap(daily: np.ndarray, *, additive: bool, n_boot: int = N_BOOT,
              seed: int = SEED) -> Test | None:
    """The moving-block bootstrap of one daily series (module docstring); None when it has fewer
    than `MIN_DAYS` days or a day without a number."""
    values = np.asarray(daily, dtype=float)
    n = len(values)
    if n < MIN_DAYS or not np.isfinite(values).all():
        return None
    block = block_length(n)
    idx = _block_indices(n, block, n_boot, np.random.default_rng(seed))
    resampled = values[idx]
    means = resampled.mean(axis=1)
    annual = _annual_rows(resampled, additive)
    return Test(
        p=float(np.mean(means <= 0.0)),
        ann_low=float(np.percentile(annual, 2.5)), ann_high=float(np.percentile(annual, 97.5)),
        days=n, block_days=block, n_boot=n_boot, seed=seed)


def metrics_of(capital_return: pd.Series, *, additive: bool) -> dict[str, float]:
    """The metrics a run stores (`KEYS`); empty when the reading is not defined for it."""
    if len(capital_return) < 2:
        return {}
    found = bootstrap(daily_money(capital_return, additive=additive).to_numpy(dtype=float),
                      additive=additive)
    if found is None:
        return {}
    return {P_KEY: found.p, LOW_KEY: found.ann_low, HIGH_KEY: found.ann_high,
            DAYS_KEY: float(found.days), BLOCK_KEY: float(found.block_days)}


# --------------------------------------------------------------------------
# Reading a stored run


def _number(m: Mapping[str, object] | None, key: str) -> float | None:
    value = (m or {}).get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


@dataclass(frozen=True)
class Reading:
    p: float | None
    ann_low: float | None
    ann_high: float | None
    days: int | None
    block_days: int | None

    @property
    def defined(self) -> bool:
        return self.p is not None

    @property
    def distinct(self) -> bool | None:
        """Whether p is at or below `ALPHA`; None when the run has no reading."""
        return None if self.p is None else self.p <= ALPHA

    @property
    def state(self) -> str:
        return UNDEFINED if self.p is None else PASSES if self.p <= ALPHA else FAILS

    @property
    def state_ru(self) -> str:
        return {PASSES: PASSES_RU, FAILS: FAILS_RU, UNDEFINED: UNDEFINED_RU}[self.state]


def read(metrics: Mapping[str, object] | None) -> Reading:
    days, block = _number(metrics, DAYS_KEY), _number(metrics, BLOCK_KEY)
    return Reading(_number(metrics, P_KEY), _number(metrics, LOW_KEY), _number(metrics, HIGH_KEY),
                   None if days is None else int(days), None if block is None else int(block))


def p_text(p: float, n_boot: int = N_BOOT) -> str:
    """«p = 0.012», or «p меньше 1 из 2000» when no resample reached zero."""
    return f"p меньше 1 из {n_boot}" if p <= 0.0 else f"p = {p:.3f}"


def line_ru(metrics: Mapping[str, object] | None) -> str | None:
    """The reading in one line of plain words, e.g. «заработок отличим от нуля: p = 0.012,
    95% интервал дохода в год от +3.1% до +14.0% (ничего не решает)»; None for a run without it."""
    r = read(metrics)
    if r.p is None:
        return None
    verdict = f"{PASSES_RU}" if r.distinct else f"{FAILS_RU}"
    interval = ""
    if r.ann_low is not None and r.ann_high is not None:
        interval = (f", 95% интервал дохода в год от {_pct(r.ann_low)} до {_pct(r.ann_high)}")
    return f"заработок {verdict}: {p_text(r.p)}{interval} (ничего не решает)"


def _pct(value: float) -> str:
    return f"{value:+.1%}".replace("-", "−")


# --------------------------------------------------------------------------
# Statistics over the registry


OUTCOME_PASSED, OUTCOME_REJECTED, OUTCOME_OTHER = "passed", "rejected", "other"


@dataclass(frozen=True)
class Row:
    """One idea's latest judged run and its reading."""

    idea_id: str
    status: str
    trial_id: int
    stage: str
    route: str
    reading: Reading
    noise_rank: float | None
    noise_passes: bool | None
    """Whether the run's place among the noise books is at or above the bar of the rules."""

    @property
    def outcome(self) -> str:
        from qlab.year_signs import PASSED_ROUTES, REJECTED_ROUTES

        return (OUTCOME_PASSED if self.route in PASSED_ROUTES
                else OUTCOME_REJECTED if self.route in REJECTED_ROUTES else OUTCOME_OTHER)

    def to_json(self) -> dict[str, Any]:
        r = self.reading
        return {"idea_id": self.idea_id, "status": self.status, "trial_id": self.trial_id,
                "stage": self.stage, "route": self.route, "outcome": self.outcome,
                "state": r.state, "p": r.p, "ann_ci_low": r.ann_low, "ann_ci_high": r.ann_high,
                "days": r.days, "block_days": r.block_days, "noise_rank": self.noise_rank,
                "noise_passes": self.noise_passes}


def noise_bar_of(ruleset) -> tuple[str, float] | None:
    """(the metric, the bar) of the ruleset's noise comparison: the rule `noise_power` names."""
    setting = getattr(ruleset, "noise_power", None)
    if setting is None:
        return None
    rule = next((r for r in ruleset.rules if r.metric == setting.percentile_metric), None)
    return None if rule is None else (rule.metric, rule.threshold)


def collect(session: Session, ruleset) -> tuple[list[Row], int]:
    """(rows, ideas with a judged run but no reading yet). A row per idea: its latest routed run
    that carries the numbers of a backtest; the reading may be missing (older runs, waiting for
    their recomputation, `docs/RESTAMP.md`). Calibration noise books are left out."""
    from qlab.registry.models import Idea, Spec, Trial
    from qlab.year_signs import stage_of

    bar = noise_bar_of(ruleset)
    query = (session.query(Trial, Spec, Idea).join(Spec, Trial.spec_id == Spec.id)
             .join(Idea, Spec.idea_id == Idea.id)
             .filter(Trial.route.isnot(None), Trial.metrics.isnot(None),
                     ~Idea.id.like("noise-%"))
             .order_by(Idea.id, Trial.id))
    latest: dict[str, Row] = {}
    for trial, spec, idea in query:
        m = trial.metrics or {}
        if "n_periods" not in m and "ann_return_net" not in m:
            continue  # a not-evaluable run: nothing was tested
        rank = _number(m, bar[0]) if bar else None
        latest[idea.id] = Row(
            idea_id=idea.id, status=idea.status.value, trial_id=trial.id,
            stage=stage_of(idea, spec), route=trial.route.value, reading=read(m),
            noise_rank=rank, noise_passes=None if rank is None or bar is None else rank >= bar[1])
    rows = [r for r in latest.values() if r.reading.defined]
    return rows, len(latest) - len(rows)


def summarize(rows: list[Row]) -> dict[str, Any]:
    """How many ideas would pass and fail at p <= `ALPHA`, overall, against the run's outcome,
    against the stage, and against the noise bar (the two readings side by side)."""
    from qlab.year_signs import STAGES

    outcomes = (OUTCOME_PASSED, OUTCOME_REJECTED, OUTCOME_OTHER)
    total: Counter = Counter({PASSES: 0, FAILS: 0})
    by_outcome = {o: Counter({PASSES: 0, FAILS: 0}) for o in outcomes}
    by_stage = {s: Counter({PASSES: 0, FAILS: 0}) for s in STAGES}
    crossing = {"noise passes": Counter({PASSES: 0, FAILS: 0}),
                "noise does not pass": Counter({PASSES: 0, FAILS: 0}),
                "no noise rank": Counter({PASSES: 0, FAILS: 0})}
    for row in rows:
        state = row.reading.state
        total[state] += 1
        by_outcome[row.outcome][state] += 1
        by_stage[row.stage][state] += 1
        key = ("no noise rank" if row.noise_passes is None
               else "noise passes" if row.noise_passes else "noise does not pass")
        crossing[key][state] += 1
    return {
        "ideas": len(rows), "alpha": ALPHA, "total": dict(total),
        "by_outcome": {k: dict(v) for k, v in by_outcome.items()},
        "by_stage": {k: dict(v) for k, v in by_stage.items()},
        "against_noise_bar": {k: dict(v) for k, v in crossing.items()},
    }


def render_table(rows: list[Row]) -> list[str]:
    """One line per idea: status, trial, stage, outcome, p, the interval, the noise rank."""
    width = max([len(r.idea_id) for r in rows] + [4])
    head = (f"{'idea':<{width}} {'status':<11} {'trial':>7} {'stage':<18} {'route':<21} "
            f"{'p':>7} {'ann 95% low':>12} {'high':>8} {'days':>5} {'noise':>6}")
    lines = [head]
    for r in rows:
        x = r.reading
        lines.append(
            f"{r.idea_id:<{width}} {r.status:<11} {r.trial_id:>7} {r.stage:<18} {r.route:<21} "
            f"{x.p:>7.3f} {x.ann_low:>12.1%} {x.ann_high:>8.1%} {x.days or 0:>5} "
            + (f"{r.noise_rank:>6.1%}" if r.noise_rank is not None else f"{'-':>6}"))
    return lines


def render_summary(summary: dict[str, Any]) -> list[str]:
    def row(label: str, c: Mapping[str, int]) -> str:
        return f"      {label:<22}{c.get(PASSES, 0):>8}{c.get(FAILS, 0):>8}"

    lines = [f"at p <= {summary['alpha']:g}: over {summary['ideas']} idea(s)",
             f"      {'':<22}{'passes':>8}{'fails':>8}", row("all", summary["total"]), "",
             "against the run's outcome (passed = paper, shelf, needs-infrastructure):"]
    lines += [row(k, v) for k, v in summary["by_outcome"].items()]
    lines += ["", "against the stage of the evaluation:"]
    lines += [row(k, v) for k, v in summary["by_stage"].items()]
    lines += ["", "against the noise bar (the rank among random books of the same shape):"]
    lines += [row(k, v) for k, v in summary["against_noise_bar"].items()]
    return lines


def report_lines(session: Session, ruleset) -> list[str]:
    """The morning report's few lines: the totals over each idea's latest judged run, and how they
    sit beside the noise bar. The table and the cross-tables are `qlab earning-test`."""
    rows, missing = collect(session, ruleset)
    if not rows:
        if not missing:
            return []
        return [f"Замера пока нет ни у одной из {missing} идей: он появляется у прогона, "
                "сделанного стендом 2026-11-03.1 или новее (пересчёт старых результатов)."]
    s = summarize(rows)
    t = s["total"]
    cross = s["against_noise_bar"]
    lines = [
        "Замер только записывается, ничего не решает (владелец: «сначала замер, потом "
        f"правило»). Вопрос: если дни стратегии вытянуть заново, как часто средний день "
        f"окажется не выше нуля? Основа: последний прогон каждой идеи, деньги стратегии; "
        f"{s['ideas']} идей" + (f", ещё {missing} ждут пересчёта" if missing else "") + ".", "",
        f"- при p не больше {ALPHA:g}: отличимы от нуля {t.get(PASSES, 0)}, "
        f"не отличимы {t.get(FAILS, 0)}",
        f"- из тех, кого пропускает планка шума: отличимы от нуля {cross['noise passes'][PASSES]}, "
        f"не отличимы {cross['noise passes'][FAILS]}",
        f"- из тех, кого планка шума не пропускает: отличимы от нуля "
        f"{cross['noise does not pass'][PASSES]}, "
        f"не отличимы {cross['noise does not pass'][FAILS]}",
        "", "Таблица по идеям и разрезы по исходу и этапу: `uv run qlab earning-test`."]
    return lines


__all__ = [
    "ALPHA", "FAILS", "KEYS", "MIN_DAYS", "PASSES", "SECTION_TITLE", "SEED", "UNDEFINED",
    "Reading", "Row", "Test", "bootstrap", "collect", "daily_money", "line_ru", "metrics_of",
    "noise_bar_of", "p_text", "read", "render_summary", "render_table", "report_lines",
    "summarize"]
