"""Summarize a calibration run and write `docs/CALIBRATION_<rules-version>.md`
(docs/TASKS.md, T16).

The report is deliberately narrow: admission rate per series, the
distribution of `ann_return_net` noise reaches (median/quartiles/max), the
best `sharpe_net` noise reaches in N tries, and the same figures for the
three real strategies side by side -- exactly what T16 asks for, nothing
inferred or embellished beyond it. Written in Russian (docs/CLAUDE.md:
"Документация в docs/ и досье — по-русски").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from qlab.calibration.noise import STRUCTURAL_BAND
from qlab.calibration.run import GENERATOR_NAMES as _GENERATOR_NAMES
from qlab.calibration.run import REFERENCE_SPEC_PATH, NoiseTrial
from qlab.pipeline.evaluate import Evaluation
from qlab.rules.schema import RuleSet

# Routes `qlab.pipeline.evaluate.decide_route` can return that mean "the
# rules engine let this strategy through" -- affordable now (`paper`) or
# affordable in principle but not with money on hand today (`shelf`). Both
# are an admission for calibration purposes: the ruleset said yes to the
# EDGE claim either way, `shelf` only adds a capital-size caveat on top.
_ADMITTED_ROUTES = frozenset({"paper", "shelf"})

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True, slots=True)
class Distribution:
    """`ann_return_net`-shaped summary: n, min, quartiles, max."""

    n: int
    min: float
    q1: float
    median: float
    q3: float
    max: float

    @classmethod
    def from_values(cls, values: list[float]) -> Distribution | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        return cls(
            n=len(values),
            min=float(arr.min()),
            q1=float(q1),
            median=float(median),
            q3=float(q3),
            max=float(arr.max()),
        )


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    """One noise series' (`dollar_neutral` or `unconstrained`) calibration
    result: how often the ruleset admitted a book that is known to have no
    edge, and what noise's own return/Sharpe distribution looked like.
    """

    series: str
    n_trials: int
    n_admitted: int
    n_rejected: int
    n_needs_more_data: int
    n_error: int
    ann_return_net: Distribution | None
    best_sharpe_net: float | None
    best_sharpe_trial: NoiseTrial | None

    @property
    def admission_rate(self) -> float:
        if self.n_trials == 0:
            return float("nan")
        return self.n_admitted / self.n_trials


def summarize_series(trials: list[NoiseTrial]) -> SeriesSummary:
    """Reduce a list of `NoiseTrial`s (all from the same series) to a
    `SeriesSummary`. Every trial is counted somewhere -- admitted, rejected,
    needs-more-data, or errored -- regardless of whether it errored out
    before producing metrics, so `n_trials` is always the true number of
    attempts, never just the number that happened to complete cleanly.
    """
    if not trials:
        raise ValueError("cannot summarize an empty trial list")
    series = trials[0].series
    if any(t.series != series for t in trials):
        raise ValueError("all trials passed to summarize_series must share the same series")

    n_admitted = n_rejected = n_needs_more_data = n_error = 0
    ann_returns: list[float] = []
    best_sharpe: float | None = None
    best_sharpe_trial: NoiseTrial | None = None

    for trial in trials:
        route = trial.evaluation.routing.route
        if route in _ADMITTED_ROUTES:
            n_admitted += 1
        elif route == "reject":
            n_rejected += 1
        elif route == "needs-more-data":
            n_needs_more_data += 1
        else:  # "error"
            n_error += 1

        metrics = trial.evaluation.metrics
        if metrics is None:
            continue
        ann_returns.append(float(metrics["ann_return_net"]))
        sharpe = metrics.get("sharpe_net")
        if sharpe is not None and not math.isnan(sharpe):
            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = float(sharpe)
                best_sharpe_trial = trial

    return SeriesSummary(
        series=series,
        n_trials=len(trials),
        n_admitted=n_admitted,
        n_rejected=n_rejected,
        n_needs_more_data=n_needs_more_data,
        n_error=n_error,
        ann_return_net=Distribution.from_values(ann_returns),
        best_sharpe_net=best_sharpe,
        best_sharpe_trial=best_sharpe_trial,
    )


def summarize_by_generator(trials: list[NoiseTrial]) -> dict[str, SeriesSummary]:
    """Break one series' trials down by generator (`t.generator`).

    A single generator dominating the aggregate admission rate or best-Sharpe
    figure is exactly the failure mode that made an earlier version of this
    report wrong (`bootstrap_time`'s look-ahead bug inflated both the
    unconstrained admission rate and the best-Sharpe headline, and the
    aggregate table alone did not show it). This breakdown is rendered
    unconditionally in `render_report` so the composition behind any
    aggregate number is always visible, not just when something is already
    suspected to be wrong.
    """
    by_generator: dict[str, list[NoiseTrial]] = {}
    for trial in trials:
        by_generator.setdefault(trial.generator, []).append(trial)
    return {generator: summarize_series(ts) for generator, ts in by_generator.items()}


@dataclass(frozen=True, slots=True)
class RealResult:
    """One real strategy's outcome, for the report's side-by-side table."""

    idea_id: str
    title: str
    route: str
    reason: str
    ann_return_net: float | None
    sharpe_net: float | None
    min_capital_usd: float | None


def summarize_real(evaluations: dict[str, Evaluation], titles: dict[str, str]) -> list[RealResult]:
    """`RealResult` rows for the three real strategies, in `evaluations`'
    insertion order (docs/TASKS.md T16 wants them "for comparison" — order
    doesn't carry meaning, but a stable one makes runs diffable)."""
    rows = []
    for idea_id, evaluation in evaluations.items():
        metrics = evaluation.metrics or {}
        rows.append(
            RealResult(
                idea_id=idea_id,
                title=titles.get(idea_id, idea_id),
                route=evaluation.routing.route,
                reason=evaluation.routing.reason,
                ann_return_net=metrics.get("ann_return_net"),
                sharpe_net=metrics.get("sharpe_net"),
                min_capital_usd=metrics.get("min_capital_usd"),
            )
        )
    return rows


def _is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fmt_pct(value: float | None) -> str:
    return "н/д" if _is_missing(value) else f"{value:+.2%}"


def _fmt_num(value: float | None, digits: int = 3) -> str:
    return "н/д" if _is_missing(value) else f"{value:.{digits}f}"


def _fmt_usd(value: float | None) -> str:
    return "н/д" if value is None else f"${value:,.0f}"


def _dominant_generator_note(
    breakdown: dict[str, SeriesSummary], total_admitted: int
) -> str | None:
    """If one generator accounts for >= 50% of a series' admissions, say so
    by name. This is the exact shape of the `bootstrap_time` look-ahead
    problem (see `_REVISION_NOTE`): an aggregate admission rate can describe
    one generator's behaviour, not "noise" in general. Returns `None` when
    there is nothing concentrated enough to flag (including when nothing was
    admitted at all -- an empty numerator has no "dominant" contributor).
    """
    if not breakdown or total_admitted == 0:
        return None
    generator, summary = max(breakdown.items(), key=lambda kv: kv[1].n_admitted)
    if summary.n_admitted == 0:
        return None
    share = summary.n_admitted / total_admitted
    if share < 0.5:
        return None
    return (
        f"из {total_admitted} допущенных в этой серии {summary.n_admitted} ({share:.0%}) — "
        f"от одного генератора, `{generator}`; агрегатная доля допуска для этой серии в "
        "основном описывает его поведение, не «шум вообще» — см. разбивку по генераторам ниже."
    )


_SERIES_TITLES = {
    "dollar_neutral": "dollar-neutral (Σ весов = 0)",
    "unconstrained": "без ограничения на нетто (бета + carry)",
}

# Documents a real defect found and fixed in this exact calibration tool on
# 2026-09-21, and the numbers it produced before the fix -- kept as a fixed
# historical record (docs/TASKS.md T16's own review loop asked for this to
# be stated plainly, not silently corrected). `bootstrap_time` sampled a
# source row `j` uniformly over the WHOLE index for every decision row `t`,
# so `t` could be filled from a row `j > t` -- computed from, and for a
# momentum-style reference ENCODING, price history strictly after `t`. That
# is look-ahead, and it inflated exactly the two headline numbers this
# report exists to produce.
_REVISION_NOTE = """## Ревизия 2026-09-21: генератор `bootstrap_time` содержал look-ahead

Первая версия этого отчёта (тот же снапшот, те же 200+200 попыток) сообщала:

| серия | доля допуска | лучший sharpe_net |
|---|---:|---:|
| dollar-neutral | 3.0% | 1.847 |
| без ограничения на нетто | 13.0% | 1.684 |

и вывод «серия без ограничения на нетто допускается заметно чаще — планка
ловит бету и carry».

**Обе цифры были искажены.** `bootstrap_time` (`qlab.calibration.noise`)
сэмплировал строку-источник `j` равномерно по ВСЕМУ ряду для каждого `t`,
поэтому решению на день `t` мог достаться вектор весов, посчитанный
трендовой эталонной стратегией по ценам ПОСЛЕ `t` — например, по 90-дневному
лукбэку, захватывающему `t+1..t+300`. Для моментум-сигнала это прямая
утечка: книга «знала» на день `t`, куда пойдёт рынок после `t`, и именно
поэтому `bootstrap_time` оказался лучшим генератором в обеих сериях (лучший
Шарп 1.847/1.684 против ≤0.94 у трёх остальных, причинных генераторов —
разбивка по генераторам ниже). Проверка на бесплатный проезд по костам
(оборот 200–247 годовых против 36 у trend) была верной, но она ловит только
нулевой оборот, а не утечку по времени — отсюда и разные проверки нужны
для разных дефектов.

Заявленный разрыв 13.0% против 3.0% почти целиком держался на
`bootstrap_time`: в серии без ограничения на нетто он один допускался в
51.9% попыток, тогда как остальные три генератора — в 0.6%–1.9%. На причинных
генераторах разница между сериями оказалась в 1–2 процентных пункта на
~150 попытках каждый — то есть статистически неразличима, а не «планка ловит
бету» в том виде, как было заявлено.

**Исправление:** `bootstrap_time` теперь сэмплирует `j` только из прошлого
или настоящего (`j <= t`); строка 0 не ресэмплируется вовсе (истории до неё
нет). Добавлен регрессионный тест на причинность
(`qlab.calibration.test_noise.test_generator_does_not_use_future_reference_rows`),
общий для всех генераторов, а не только для `bootstrap_time` — он ловит
именно этот класс дефекта у ЛЮБОГО будущего генератора: возмущает эталонные
строки СТРОГО ПОСЛЕ отметки времени и проверяет, что вывод генератора до и
включительно этой отметки не изменился.

Числа ниже — повторный прогон после исправления."""


def render_report(
    *,
    ruleset: RuleSet,
    series_summaries: dict[str, SeriesSummary],
    real_results: list[RealResult],
    reference_idea_id: str,
    generated_on: date,
    generator_breakdowns: dict[str, dict[str, SeriesSummary]] | None = None,
    include_revision_note: bool = True,
) -> str:
    """Render the full Russian-language calibration report as Markdown."""
    lines: list[str] = []
    a = lines.append

    a(f"# Калибровка правил `{ruleset.version}` шумом (T16)")
    a("")
    a(
        f"Дата: {generated_on.isoformat()}. Эталонная стратегия для структурного "
        f"соответствия шума — `{reference_idea_id}` (`{REFERENCE_SPEC_PATH.name}`). "
        "Каждый прогон ниже — настоящий вызов `evaluate_spec`: те же правила, косты, "
        "accrual и универс, что и для любой реальной стратегии; каждый прогон записан "
        "в журнал испытаний (`trial`)."
    )
    a("")
    a(
        "**Вопрос:** если в отбор подать заведомо беспредметный шум — стратегии без "
        "какого-либо сигнала, но с сопоставимой гросс-экспозицией, оборотом и числом "
        "позиций, — как часто отбор говорит «да», и какой Шарп шум достигает в лучшем "
        "случае из N попыток?"
    )
    a("")

    if include_revision_note:
        a(_REVISION_NOTE)
        a("")

    # ---- Headline table -------------------------------------------------
    a("## Итог")
    a("")
    a(
        "| серия | N | допущено | доля допуска | лучший sharpe_net | медиана ann_return_net "
        "| Q1 | Q3 | max |"
    )
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for series, summary in series_summaries.items():
        dist = summary.ann_return_net
        a(
            f"| {_SERIES_TITLES.get(series, series)} "
            f"| {summary.n_trials} "
            f"| {summary.n_admitted} "
            f"| {summary.admission_rate:.1%} "
            f"| {_fmt_num(summary.best_sharpe_net)} "
            f"| {_fmt_pct(dist.median if dist else None)} "
            f"| {_fmt_pct(dist.q1 if dist else None)} "
            f"| {_fmt_pct(dist.q3 if dist else None)} "
            f"| {_fmt_pct(dist.max if dist else None)} |"
        )
    a("")
    for series, summary in series_summaries.items():
        a(
            f"- **{_SERIES_TITLES.get(series, series)}**: из {summary.n_trials} прогонов "
            f"{summary.n_admitted} допущено (`paper`/`shelf`), {summary.n_rejected} отклонено, "
            f"{summary.n_needs_more_data} «нужно больше данных», {summary.n_error} упали с ошибкой "
            "(структурное несоответствие или сбой стратегии — см. `error` в trial)."
        )
    a("")

    dn = series_summaries.get("dollar_neutral")
    un = series_summaries.get("unconstrained")

    # ---- The main result: is the real strategy distinguishable from noise? ---
    trend_row = next((r for r in real_results if r.idea_id == reference_idea_id), None)
    if dn is not None and trend_row is not None:
        a("## Главный результат: trend против лучшего чистого шума")
        a("")
        a(
            "Ради этого числа калибровка и делается: цифра «Шарп X» из бэктеста ничего "
            "не доказывает сама по себе, пока не известно, чего перебором достигает "
            f"стратегия без всякого сигнала. Ниже — эталонная стратегия `{reference_idea_id}` "
            f"против лучшего результата {dn.n_trials} книг серии "
            f"«{_SERIES_TITLES.get('dollar_neutral', 'dollar_neutral')}» (чистый шум отбора "
            "инструментов, без беты и carry)."
        )
        a("")
        if trend_row.sharpe_net is not None and dn.best_sharpe_net is not None:
            a(
                f"- **sharpe_net**: шум — **{dn.best_sharpe_net:.3f}** (лучший из "
                f"{dn.n_trials}); `{reference_idea_id}` — **{trend_row.sharpe_net:.3f}**."
            )
        if trend_row.ann_return_net is not None and dn.ann_return_net is not None:
            a(
                f"- **ann_return_net**: шум — **{dn.ann_return_net.max:+.2%}** (максимум из "
                f"{dn.n_trials}); `{reference_idea_id}` — **{trend_row.ann_return_net:+.2%}**."
            )
        a("")
        if (
            trend_row.sharpe_net is not None
            and dn.best_sharpe_net is not None
            and trend_row.ann_return_net is not None
            and dn.ann_return_net is not None
        ):
            sharpe_gap = trend_row.sharpe_net - dn.best_sharpe_net
            return_gap = trend_row.ann_return_net - dn.ann_return_net.max
            if sharpe_gap <= 0 or abs(return_gap) <= 0.01:
                a(
                    f"**{reference_idea_id} практически неотличим от лучшего результата чистого "
                    f"шума на этом окне и числе попыток** (разница по sharpe_net "
                    f"{sharpe_gap:+.3f}, по ann_return_net {return_gap:+.2%}). Это не значит, "
                    "что trend случаен -- значит, что ни одна из этих двух цифр САМА ПО СЕБЕ, "
                    "без сравнения с этим прогоном, не отличает найденный эдж от того, что "
                    f"даёт {dn.n_trials} попыток без всякого сигнала."
                )
            else:
                a(
                    f"`{reference_idea_id}` превосходит лучший результат чистого шума на этом "
                    f"числе попыток (разница по sharpe_net {sharpe_gap:+.3f}, по "
                    f"ann_return_net {return_gap:+.2%}) -- но этот разрыв стоит сверять с "
                    "числом испытаний, потраченных на поиск и настройку самого trend "
                    "(docs/ACCEPTANCE_M2.md: шесть разведочных прогонов по концентрации trend), "
                    "не только с числом попыток шума здесь."
                )
        a("")

    if dn is not None and un is not None:
        a("### Разница между сериями")
        a("")
        gap = un.admission_rate - dn.admission_rate
        # A small epsilon guards this threshold against float artifacts --
        # e.g. 12/200 - 2/200 evaluates to 0.049999999999999996 in binary
        # floating point, one ULP shy of the literal 0.05 the rate values
        # would suggest; without the epsilon that single ULP silently
        # flipped this into the "no notable gap" branch on a real run.
        _EPS = 1e-9
        un_notably_higher = un.admission_rate > dn.admission_rate * 1.5 and gap >= 0.05 - _EPS
        dn_notably_higher = dn.admission_rate > un.admission_rate * 1.5 and -gap >= 0.05 - _EPS
        if un_notably_higher:
            a(
                f"Серия без ограничения на нетто допускается заметно чаще "
                f"({un.admission_rate:.1%} против {dn.admission_rate:.1%} у dollar-neutral, "
                f"разница {gap:+.1%}). **Это значит, что планка `net_edge_positive` в текущем "
                "виде отчасти ловит рыночную бету и carry, а не отбор инструментов** — тот же "
                "класс проблемы, что и декорреляция у XSMOM: критерий, отбирающий не то. "
                "Случайная книга без нетто-ограничения в исследуемом окне не была нейтральна "
                "к рынку, и часть её доходности — это цена риска, который отбор не должен "
                "путать с эджем."
            )
        elif dn_notably_higher:
            a(
                f"Неожиданно, dollar-neutral серия допускается чаще ({dn.admission_rate:.1%} "
                f"против {un.admission_rate:.1%}). Это НЕ ожидавшийся результат "
                "docs/TASKS.md T16 (там бета/carry предполагались источником ложных допусков "
                "в серии без ограничения на нетто) — приводится как есть, без подгонки под "
                "ожидание; см. раздел «Оговорки» о возможных причинах (окно, знак funding, "
                "конкретные сиды)."
            )
        else:
            a(
                f"Доля допуска у двух серий близка ({un.admission_rate:.1%} без ограничения на "
                f"нетто против {dn.admission_rate:.1%} dollar-neutral, разница {gap:+.1%}) — "
                "на этом окне бета/carry не создают систематически более лёгкий проход по "
                "сравнению с чистым шумом отбора инструментов. Это НЕ означает, что риск "
                "бета/carry исключён в принципе — только что на измеренных 2025-01-01 -- "
                "2026-09-20 разница не выражена достаточно, чтобы её утверждать."
            )
        a("")

        if generator_breakdowns:
            for series, summary in (("dollar_neutral", dn), ("unconstrained", un)):
                note = _dominant_generator_note(
                    generator_breakdowns.get(series, {}), summary.n_admitted
                )
                if note is not None:
                    a(f"- {note}")
            a("")

    # ---- Per-generator breakdown -------------------------------------------
    if generator_breakdowns:
        a("## Разбивка по генераторам")
        a("")
        a(
            "Публикуется всегда, а не только когда что-то уже вызывает подозрение: "
            "агрегатная цифра по серии может целиком держаться на одном генераторе "
            "(конкретный прошлый случай — раздел «Ревизия» выше)."
        )
        a("")
        for series in ("dollar_neutral", "unconstrained"):
            breakdown = generator_breakdowns.get(series)
            if not breakdown:
                continue
            a(f"**{_SERIES_TITLES.get(series, series)}**")
            a("")
            a("| генератор | N | допущено | доля допуска | лучший sharpe_net |")
            a("|---|---:|---:|---:|---:|")
            for generator in sorted(breakdown):
                gen_summary = breakdown[generator]
                a(
                    f"| `{generator}` | {gen_summary.n_trials} | {gen_summary.n_admitted} "
                    f"| {gen_summary.admission_rate:.1%} "
                    f"| {_fmt_num(gen_summary.best_sharpe_net)} |"
                )
            a("")

    # ---- Real strategies --------------------------------------------------
    a("## Три реальные стратегии — для сравнения")
    a("")
    a("| стратегия | вердикт | ann_return_net | sharpe_net | min_capital_usd |")
    a("|---|---|---:|---:|---:|")
    for row in real_results:
        a(
            f"| {row.title} | {row.route} ({row.reason}) | {_fmt_pct(row.ann_return_net)} "
            f"| {_fmt_num(row.sharpe_net)} | {_fmt_usd(row.min_capital_usd)} |"
        )
    a("")
    if dn is not None and un is not None:
        trend_row = next((r for r in real_results if r.idea_id == reference_idea_id), None)
        if trend_row is not None and trend_row.ann_return_net is not None:
            for series, summary in series_summaries.items():
                dist = summary.ann_return_net
                if dist is None:
                    continue
                rank = (
                    "выше медианы шума"
                    if trend_row.ann_return_net > dist.median
                    else "не выше медианы шума"
                )
                a(
                    f"- `{reference_idea_id}` (ann_return_net={trend_row.ann_return_net:+.2%}) "
                    f"{rank} серии «{_SERIES_TITLES.get(series, series)}» "
                    f"(медиана {_fmt_pct(dist.median)}, максимум {_fmt_pct(dist.max)} из "
                    f"{summary.n_trials} попыток)."
                )
            a("")

    # ---- Methodology --------------------------------------------------
    a("## Как строился шум")
    a("")
    a(
        f"Четыре генератора (`qlab.calibration.noise`), каждый детерминирован по "
        f"явному сиду: {', '.join(f'`{g}`' for g in _GENERATOR_NAMES)}. Каждый использует "
        "веса эталонной стратегии как источник формы (гросс-экспозиция, оборот, число "
        "позиций, минимальный размер ноги), но не как источник сигнала — связь между "
        "инструментом и решением разрушается на разный лад в каждом генераторе (см. "
        "докстринги функций)."
    )
    a("")
    a(
        f"Структурное соответствие проверяется автоматически "
        f"(`qlab.calibration.noise.check_structural_match`) для каждой сгенерированной "
        f"книги: отношение gross/turnover/positions кандидата к эталону должно попасть в "
        f"полосу {STRUCTURAL_BAND[0]}x–{STRUCTURAL_BAND[1]}x, и оборот не может быть "
        "нулевым или близким к нулю ни при каких обстоятельствах — иначе книга не платит "
        "костов и допуск ничего не измеряет. Полоса асимметрична (нижняя граница туже "
        "верхней): три из четырёх генераторов перерисовывают шум заново на каждой "
        "перебалансировке, что структурно даёт БОЛЬШИЙ оборот, чем у медленного "
        "трендового сигнала — это делает допуск ТРУДНЕЕ, а не легче, поэтому широкая "
        "верхняя граница безопасна; риск свободного прохода живёт только у нижней "
        "границы, и она держится узкой."
    )
    a("")
    a(
        "Две серии — dollar-neutral и без ограничения на нетто — построены из ОДНОГО и "
        "того же случайного черновика на пару (генератор, сид): dollar-neutral серия это "
        "тот же черновик после `neutralize()` (обнуление суммы по строке с сохранением "
        "гросс-экспозиции и минимального размера ноги). Это изолирует именно эффект "
        "ограничения на нетто, а не какое-то другое различие между сериями."
    )
    a("")

    a("## Оговорки")
    a("")
    a(
        "- Единственная эталонная стратегия для структурного соответствия — trend "
        "(`trend-tsmom-crypto`): это единственный из трёх спеков, который проходит "
        "весь путь без блокеров слоя данных на этом снапшоте (Bv2 и FRAB требуют "
        "спотовую ногу с неполным покрытием — docs/ACCEPTANCE_M2.md, docs/TASKS.md T19). "
        "Числа для Bv2/FRAB в таблице выше — реальные вердикты сегодняшнего прогона, "
        "но шум против ИХ формы (карри со слотами, спот+хедж) не строился в этом заходе."
    )
    a(
        "- «Допущено» здесь означает `paper` ИЛИ `shelf` — обе означают, что правила "
        "приняли заявку на эдж; `shelf` лишь добавляет оговорку по капиталу. Если "
        "считать допуском только `paper`, доля будет ниже — обе цифры разумны в "
        "зависимости от вопроса."
    )
    a(
        "- Лучший `sharpe_net` в таблице — максимум по ВСЕМ попыткам серии, а не только "
        "по допущенным: вопрос «что достижимо перебором» не ограничен исходом отбора."
    )
    a(
        "- Ничего в этом отчёте не подгонялось под ожидаемый результат: генераторы, "
        "полоса структурного соответствия и правило `net_edge_positive` не менялись "
        "по итогам прогона, чтобы шум прошёл чаще или реже задуманного."
    )
    a("")

    return "\n".join(lines) + "\n"


def calibration_report_path(rules_version: str, *, docs_dir: Path = DOCS_DIR) -> Path:
    return docs_dir / f"CALIBRATION_{rules_version}.md"


def write_report(content: str, rules_version: str, *, docs_dir: Path = DOCS_DIR) -> Path:
    """Persist the calibration alongside the ruleset version (docs/TASKS.md
    T16: "результат сохраняется вместе с версией правил"). One file per
    rules version -- re-running calibration for the same version overwrites
    its own file, never a different version's."""
    path = calibration_report_path(rules_version, docs_dir=docs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


__all__ = [
    "Distribution",
    "RealResult",
    "SeriesSummary",
    "calibration_report_path",
    "render_report",
    "summarize_by_generator",
    "summarize_real",
    "summarize_series",
    "write_report",
]
