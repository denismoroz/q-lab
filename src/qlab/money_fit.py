"""The money step on the strategy's own money, in the «fewer positions» form (rules 2026-11-04.2).

Owner, 2026-11-04, on a page whose money step read «нужно $37 591, своих $3 000: не помещается»:
«опять вижу нужно 37000$ - мы же решили вчера проверять стратегию по доступному бюджету а не на
таких космических суммах». Asked how, the owner chose: the money step judges on the strategy's own
money, in the «fewer positions» form; it passes when that book keeps at least 70% of the
unconstrained book's return and its daily returns correlate with it at 0.9 or more, on the last
third of the data (the forward test); the strict minimum stays only as a small line.

Two halves, the measurement and the judgement:

* `at_evaluation` runs inside `qlab.pipeline.evaluate.evaluate_spec` under a ruleset with
  `money_fit`, on the run's judged part (the forward third under rules 2026-11-04.x). A strategy
  that only produces weights is run on its own money through `qlab.money_ladder.constrained_run`
  in the «fewer» mode, against the stand's own result of the same rows (the unconstrained book). A
  strategy that drives its own book (Bv2, FRAB: it already skips the orders under the minimum in
  frab's code) is its own code run on its own money against the same code on
  `qlab.money_ladder.REFERENCE_BOOK_USD`. The numbers are written as `money_fit_*` metrics on the
  trial; a measurement that cannot be made writes `money_fit_unmeasured` with a code from a CLOSED
  list (`Unmeasured`), never nothing. It never raises: the verdict of the run does not depend on it.
* `judge` is a pure function of those metrics, the rule block of the run's ruleset and the
  strategy's money as the ladder knows it. It is the whole of step 4 under the new rule.

The prefix is not `ml_`: the night's money ladder (`qlab.money_ladder.write_metrics`) replaces
every `ml_*` key of a trial with its own reading, which would erase these.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

PREFIX = "money_fit_"


class Unmeasured(IntEnum):
    """Why the own-money book was not measured (a closed list, stored as its number)."""

    NO_MONEY = 1
    NO_CAPITAL_BASE = 2
    BOOK_WITHOUT_MONEY = 3
    FAILED = 4
    NO_JUDGED_PART = 5


UNMEASURED_RU: dict[Unmeasured, str] = {
    Unmeasured.NO_MONEY: "в спеке не названы деньги стратегии",
    Unmeasured.NO_CAPITAL_BASE: "у прогона нет чисел на деньгах стратегии (стенд отказался считать "
                                "основу)",
    Unmeasured.BOOK_WITHOUT_MONEY: "книга сама ставит заявки, а в её параметрах нет суммы денег: "
                                   "запустить её на своих деньгах нечем",
    Unmeasured.FAILED: "расчёт упал",
    Unmeasured.NO_JUDGED_PART: "у прогона нет судимой части",
}


def _key(name: str) -> str:
    return PREFIX + name


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _cap_ann(result) -> float | None:
    from qlab.harness.metrics import cap_key, capital_metrics

    return _finite(capital_metrics(result).get(cap_key("ann_return_net")))


def _avg_positions(weights: np.ndarray) -> float:
    from qlab.harness.metrics import ZERO_WEIGHT_TOL

    if not weights.size:
        return 0.0
    return float((np.abs(weights) > ZERO_WEIGHT_TOL).sum(axis=1).mean())


def _numbers(*, own_usd: float, fewer_result, reference_result, held: np.ndarray,
             wanted: np.ndarray, book: bool) -> dict[str, float]:
    from qlab import money_ladder
    from qlab.harness.metrics import ZERO_WEIGHT_TOL, cap_key, capital_metrics

    out: dict[str, float] = {_key("own_usd"): float(own_usd), _key("book"): 1.0 if book else 0.0,
                             _key("fewer_avg_positions"): _avg_positions(held),
                             _key("reference_avg_positions"): _avg_positions(wanted)}
    values = {
        "fewer_ann_return": _cap_ann(fewer_result),
        "reference_ann_return": _cap_ann(reference_result),
        "fewer_max_dd": _finite(capital_metrics(fewer_result).get(cap_key("max_dd"))),
        "fewer_correlation": money_ladder.correlation(fewer_result.net_return,
                                                      reference_result.net_return),
    }
    wanted_cells = np.abs(wanted) > ZERO_WEIGHT_TOL
    if wanted_cells.any() and held.shape == wanted.shape:
        values["fewer_missing_position_share"] = float(
            (wanted_cells & ~(np.abs(held) > ZERO_WEIGHT_TOL)).sum() / wanted_cells.sum())
    out.update({_key(k): float(v) for k, v in values.items() if _finite(v) is not None})
    return out


def measure_weights(*, part_panel, part_weights, reference_result, costs, extra_cost, capital,
                    minimums: np.ndarray, trims_allowed: bool, own_usd: float) -> dict[str, float]:
    """The «fewer positions» book on `own_usd` against `reference_result` (the stand's result of
    the same rows, unconstrained), for a strategy that only produces weights."""
    from qlab import money_ladder

    result, held, _ = money_ladder.constrained_run(
        own_usd, "fewer", part_panel=part_panel, weights=part_weights, costs=costs,
        extra_cost=extra_cost, capital_kwargs=capital.kwargs, base=capital.base,
        reinvested=capital.reinvested, minimums=minimums, trims_allowed=trims_allowed,
        reference_result=reference_result)
    return _numbers(own_usd=own_usd, fewer_result=result, reference_result=reference_result,
                    held=held, wanted=part_weights.to_numpy(dtype=float), book=False)


def measure_book(run_at, own_usd: float) -> dict[str, float]:
    """A book that drives itself: its code on `own_usd` against the same code on
    `REFERENCE_BOOK_USD` (`run_at` as `qlab.money_ladder.book_runner` makes it)."""
    from qlab import money_ladder

    reference, ref_weights, _ = run_at(money_ladder.REFERENCE_BOOK_USD)
    result, weights, _ = run_at(own_usd)
    held = weights.reindex_like(ref_weights).fillna(0.0).to_numpy(dtype=float)
    return _numbers(own_usd=own_usd, fewer_result=result, reference_result=reference, held=held,
                    wanted=ref_weights.to_numpy(dtype=float), book=True)


def unmeasured(code: Unmeasured, own_usd: float | None = None) -> dict[str, float]:
    out = {_key("unmeasured"): float(int(code))}
    if own_usd is not None:
        out[_key("own_usd")] = float(own_usd)
    return out


def minimums_of(columns, floors: Mapping[str, float] | None, min_leg_notional: float
                ) -> np.ndarray:
    """Each instrument's minimum order: the venue's per-contract floor, else the spec's one."""
    return np.array([(floors or {}).get(str(c), min_leg_notional) for c in columns], dtype=float)


def trims_allowed(venue) -> bool:
    return venue is not None and venue.reduce_only_exempt == "any_reduction"


def at_evaluation(*, spec, panel, judged: tuple[int, int], judged_run: Mapping[str, Any], costs,
                  extra_cost, capital, orders, floors: Mapping[str, float] | None
                  ) -> dict[str, float]:
    """The `money_fit_*` metrics of one run, measured on its judged rows (`judged`, the rows
    `judged_run` holds the stand's result of). Never raises: a failure is `Unmeasured.FAILED`."""
    from qlab import money_ladder
    from qlab.capital_floor import with_money
    from qlab.venues.config import load_venue

    own = _finite(spec.strategy_capital_usd)
    if own is None or own <= 0:
        return unmeasured(Unmeasured.NO_MONEY)
    if capital.refused is not None:
        return unmeasured(Unmeasured.NO_CAPITAL_BASE, own)
    reference = judged_run.get("result")
    part_panel = judged_run.get("panel")
    part_weights = judged_run.get("weights")
    if reference is None or part_panel is None or part_weights is None or not len(part_weights):
        return unmeasured(Unmeasured.NO_JUDGED_PART, own)
    try:
        if orders is not None:
            _, has_money = with_money(spec.params, 1.0)
            if not has_money:
                return unmeasured(Unmeasured.BOOK_WITHOUT_MONEY, own)
            lo, hi = judged
            return measure_book(money_ladder.book_runner(
                spec.code_ref, spec.params, panel, part_panel, costs, lo, hi), own)
        venue = load_venue(spec.execution_venue or spec.data.source)
        return measure_weights(
            part_panel=part_panel, part_weights=part_weights, reference_result=reference,
            costs=costs, extra_cost=extra_cost, capital=capital,
            minimums=minimums_of(part_weights.columns, floors, spec.min_leg_notional),
            trims_allowed=trims_allowed(venue), own_usd=own)
    except Exception:  # noqa: BLE001 - a stated reason, never a failed run
        return unmeasured(Unmeasured.FAILED, own)


def measure_stored(session, trial_id: int, own_usd: float | None = None) -> dict[str, float]:
    """The same measurement for a stored trial, re-created from its spec and snapshot
    (`qlab.anomalies.rerun.load_run`, its judged rows). Reads the registry, writes nothing; for
    a dry check of the rule on runs judged before it. Raises `RunUnavailable` as `load_run`."""
    from qlab import money_ladder
    from qlab.anomalies.rerun import load_run
    from qlab.capital_floor import with_money
    from qlab.harness.strategy import capital_setup, own_orders
    from qlab.pipeline.evaluate import resolve_strategy
    from qlab.venues.config import leg_floors, load_venue

    run = load_run(session, trial_id)
    spec = run.spec
    own = _finite(own_usd or spec.strategy_capital_usd)
    if own is None or own <= 0:
        return unmeasured(Unmeasured.NO_MONEY)
    strategy = resolve_strategy(spec.code_ref)
    capital = capital_setup(strategy, run.panel, spec.params)
    if capital.refused is not None:
        return unmeasured(Unmeasured.NO_CAPITAL_BASE, own)
    part_panel, part_weights = run.part()
    lo, hi = run.judged
    if own_orders(strategy, run.panel, spec.params) is not None:
        _, has_money = with_money(spec.params, 1.0)
        if not has_money:
            return unmeasured(Unmeasured.BOOK_WITHOUT_MONEY, own)
        return measure_book(money_ladder.book_runner(
            spec.code_ref, spec.params, run.panel, part_panel, run.costs, lo, hi), own)
    venue = load_venue(run.judged_for)
    from qlab.harness.run import run_backtest

    reference = run_backtest(part_panel, part_weights, run.costs, part_panel.funding,
                             extra_cost=run.extra_cost, **capital.kwargs)
    floors = leg_floors(venue, list(run.weights.columns))
    return measure_weights(
        part_panel=part_panel, part_weights=part_weights, reference_result=reference,
        costs=run.costs, extra_cost=run.extra_cost, capital=capital,
        minimums=minimums_of(part_weights.columns, floors, spec.min_leg_notional),
        trims_allowed=trims_allowed(venue), own_usd=own)


# --------------------------------------------------------------------------
# The judgement (pure)

PASSED, FAILED, NO_EDGE, NOT_MEASURED = "passed", "failed", "no_edge", "unmeasured"


@dataclass(frozen=True)
class Judgement:
    """What step 4 says under the rule `money_fit` for one venue's run."""

    state: str  # passed | failed | no_edge | unmeasured
    own_usd: float | None
    fewer_ann_return: float | None
    reference_ann_return: float | None
    correlation: float | None
    min_return_share: float
    min_correlation: float
    book: bool = False
    why_ru: str = ""
    """The reason of an `unmeasured` judgement, in words."""
    return_ok: bool | None = None
    correlation_ok: bool | None = None
    no_positions: bool = False

    @property
    def detail_ru(self) -> str:
        return detail_ru(self)


def _number(m: Mapping[str, Any], name: str) -> float | None:
    return _finite(m.get(_key(name)))


def judge(metrics: Mapping[str, Any], rule, money: float | None) -> Judgement:
    """Step 4 under `rule` (`qlab.rules.schema.MoneyFit`) from a run's `money_fit_*` metrics, with
    `money` the strategy's own money as the ladder knows it.

    * not measured (no keys, a stated code, money moved since): `unmeasured`, with the reason;
    * the unconstrained book does not earn (its yearly return on the strategy's money is not above
      zero): `no_edge` -- nothing to keep, the edge rule of the venue's step fails the run;
    * no position fits the strategy's money at all: `failed`;
    * else `passed` when the yearly return kept is at least `min_return_share` of the
      unconstrained one AND the daily correlation at least `min_correlation`, `failed` if not.
    """
    share, corr_line = float(rule.min_return_share), float(rule.min_correlation)
    own = _number(metrics, "own_usd")
    fewer = _number(metrics, "fewer_ann_return")
    ref = _number(metrics, "reference_ann_return")
    corr = _number(metrics, "fewer_correlation")
    book = (_number(metrics, "book") or 0.0) > 0

    def out(state: str, why: str = "", **kw) -> Judgement:
        return Judgement(state, own, fewer, ref, corr, share, corr_line, book=book, why_ru=why,
                         **kw)

    code = _number(metrics, "unmeasured")
    if code is not None:
        try:
            why = UNMEASURED_RU[Unmeasured(int(code))]
        except ValueError:
            why = f"причина с номером {int(code)} не известна этой версии кода"
        return out(NOT_MEASURED, why)
    if own is None:
        return out(NOT_MEASURED, "прогон сделан без этого замера (до пересчёта)")
    if money is not None and abs(own - float(money)) > 0.005:
        return out(NOT_MEASURED, f"замер сделан на ${own:,.0f}, а деньги стратегии теперь "
                                 f"${float(money):,.0f} — до пересчёта")
    if ref is None:
        return out(NOT_MEASURED, "доход книги без ограничений не определён")
    if ref <= 0:
        return out(NO_EDGE)
    positions = _number(metrics, "fewer_avg_positions")
    if positions is not None and positions <= 0:
        return out(FAILED, return_ok=False, correlation_ok=False, no_positions=True)
    if fewer is None:
        return out(NOT_MEASURED, "доход книги на своих деньгах не определён")
    if corr is None:
        return out(NOT_MEASURED, "совпадение по дням не определено (у одной из книг доход "
                                 "не меняется)")
    return_ok = fewer >= share * ref
    corr_ok = corr >= corr_line
    return out(PASSED if return_ok and corr_ok else FAILED, return_ok=return_ok,
               correlation_ok=corr_ok)


def _usd(value: float) -> str:
    return "$" + f"{value:,.0f}".replace(",", " ")


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.1f}".replace("-", "−") + "%"


def detail_ru(j: Judgement) -> str:
    """The decisive line of step 4, in plain words: «на своих $3 000 при меньшем числе позиций:
    доход +4.1% в год против +3.6% без ограничений, день в день совпадает на 0.997 —
    помещается»."""
    own = "своих" if j.own_usd is None else f"своих {_usd(j.own_usd)}"
    how = "своим кодом книги" if j.book else "при меньшем числе позиций"
    against = (f"той же книги на {_usd(100_000)}" if j.book else "без ограничений")
    if j.state == NOT_MEASURED:
        return f"на {own} {how} не измерено: {j.why_ru}"
    if j.state == NO_EDGE:
        return (f"не решает: без ограничений стратегия не зарабатывает (доход "
                f"{_pct(j.reference_ann_return)} в год на деньгах стратегии), сохранять нечего; "
                "провал ставит правило дохода на ступени площадки")
    if j.no_positions:
        return f"на {own} {how} не помещается ни одной позиции — не помещается"
    ret = f"доход {_pct(j.fewer_ann_return)} в год против {_pct(j.reference_ann_return)} {against}"
    if j.return_ok is False:
        ret += (f" (нужно не меньше {j.min_return_share:.0%} от него: "
                f"{_pct(j.min_return_share * (j.reference_ann_return or 0.0))})")
    corr = f"день в день совпадает на {j.correlation:.3f}"
    if j.correlation_ok is False:
        corr += f" (нужно не меньше {j.min_correlation:.2f})"
    word = "помещается" if j.state == PASSED else "не помещается"
    return f"на {own} {how}: {ret}, {corr} — {word}"


def strict_ru(need: float | None, ladder_reading=None) -> str | None:
    """The strict minimum as a small line beside the decisive one: «чтобы держать все позиции как
    есть, нужно $37 591 (задаёт H на Binance, позиция 0.0133% денег 2026-06-11)». The position that
    sets it comes from the night's money ladder when its number is the run's."""
    if need is None:
        return None
    line = f"чтобы держать все позиции как есть, нужно {_usd(need)}"
    s = getattr(ladder_reading, "strict", None)
    if (s is not None and s.min_capital_usd is not None and s.instrument is not None
            and s.weight is not None and abs(s.min_capital_usd - need) <= max(1.0, 1e-3 * need)):
        from qlab.money_ladder import venue_ru

        what = "заявка" if s.from_orders else "позиция"
        line += (f" (задаёт {s.instrument} на {venue_ru(s.venue or '')}, {what} "
                 f"{s.weight * 100:.3g}% денег {s.day})")
    return line + "; это показание, ступень решает строка выше"


__all__ = ["FAILED", "NO_EDGE", "NOT_MEASURED", "PASSED", "PREFIX", "UNMEASURED_RU", "Judgement",
           "Unmeasured", "at_evaluation", "detail_ru", "judge", "measure_book", "measure_stored",
           "measure_weights", "strict_ru", "unmeasured"]
