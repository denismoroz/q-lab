"""Known-good strategies with a planted edge: how many does the ruleset
reject? (docs/TASKS.md T26, docs/T26_FALSE_REJECTIONS.md)

Owner, 2026-09-21: "я не хочу поиметь систему которая будет резать валидные
стратегии". Noise calibration (T16) measures one of the two errors -- a
strategy with no edge admitted. This module measures the other -- a strategy
WITH an edge rejected -- which needs strategies whose edge is known. Real
ones are too few (FRAB, Bv2, trend) and their true edge is unknown, so the
known-good inputs are built:

1. Take the noise books of a reference strategy's own shape -- the same
   generators, neutralisation and structural match as T16
   (`qlab.calibration.noise`), on the same panel, costs and funding. They
   have no edge by construction; their spread is the luck of that shape.
2. Plant a known edge in each: `alpha` a year, added to the book's net return
   on every bar the book holds positions (`alpha / periods_per_year` per bar).
   The planted book's true edge over its noise twin is exactly `alpha`.
3. Judge every planted book like a candidate: its metrics, its percentile
   against the OTHER noise books (its own twin left out), the ruleset, and
   the route -- as a forward test with no selection period, since nothing in
   it was fitted.

Read the result as: "a strategy of this shape with a true edge of alpha a
year, over this window, is admitted / told 'too early' / rejected this often".
At alpha = 0 the admitted share is the false-admission rate T16 measured.

These are measurements of a ruleset, not trials of any idea: nothing here is
a candidate, so nothing is written to `trial` (deflation counts candidates).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.calibration.noise import (
    GENERATORS,
    StructuralMismatchError,
    check_structural_match,
    compute_shape,
    neutralize,
    reference_min_position,
)
from qlab.calibration.percentile import noise_percentile
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics, min_capital_usd, periods_per_year
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest
from qlab.pipeline.evaluate import decide_fit_forward_route, decide_route, forward_years_needed
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.schema import RuleSet

GENERATOR_NAMES = tuple(GENERATORS)
ADMITTED = frozenset({"paper", "shelf", "needs-infrastructure"})
EARLY = frozenset({"needs-forward", "needs-more-data"})

# Facts about the venue and the pipeline that a planted book shares with the
# reference run; they are copied, not recomputed.
SHARED_FACTS = (
    "venue_supported",
    "data_forward_available",
    "atomic_execution",
    "point_in_time_universe",
    "accrual_applied",
)


@dataclass(frozen=True)
class NoiseBook:
    generator: str
    seed: int
    weights: pd.DataFrame
    result: RunResult


@dataclass(frozen=True)
class PlantedOutcome:
    alpha: float
    window_days: float
    n: int
    admitted: int
    early: int
    rejected: int
    mean_return: float
    median_sharpe: float
    reject_rules: Counter


def noise_books(
    panel: MarketPanel, reference: pd.DataFrame, costs: CostModel, n: int, seed_offset: int = 0
) -> list[NoiseBook]:
    """The dollar-neutral noise series of T16 for `reference`'s shape.
    A draw that fails the structural match is skipped, as an ERROR noise
    trial is in T16."""
    floor = reference_min_position(reference)
    ref_shape = compute_shape(reference)
    books = []
    for i in range(n):
        generator = GENERATOR_NAMES[i % len(GENERATOR_NAMES)]
        seed = seed_offset + i
        weights = neutralize(GENERATORS[generator](reference, panel, seed=seed),
                             min_position_floor=floor)
        try:
            check_structural_match(compute_shape(weights), ref_shape)
        except StructuralMismatchError:
            continue
        books.append(NoiseBook(generator, seed, weights,
                               run_backtest(panel, weights, costs, panel.funding)))
    return books


def _window(result: RunResult, start: pd.Timestamp) -> RunResult:
    keep = result.net_return.index >= start
    return dataclasses.replace(
        result,
        net_return=result.net_return[keep],
        gross_return=result.gross_return[keep],
        turnover=result.turnover[keep],
        cost=result.cost[keep],
        accrual=result.accrual[keep],
    )


def planted_outcomes(
    *,
    panel: MarketPanel,
    reference: pd.DataFrame,
    books: Sequence[NoiseBook],
    ruleset: RuleSet,
    shared_facts: Mapping[str, float],
    alphas: Sequence[float],
    window_starts: Mapping[str, pd.Timestamp],
    min_leg_notional: float,
    deployable_capital_usd: float,
) -> dict[str, list[PlantedOutcome]]:
    """{window label: one outcome per alpha}. Each window runs from its start
    to the panel's end."""
    ppy = periods_per_year(panel.prices.index)
    out: dict[str, list[PlantedOutcome]] = {}
    for label, start in window_starts.items():
        windowed = [_window(b.result, start) for b in books]
        days = (windowed[0].net_return.index[-1] - start) / pd.Timedelta(days=1) + 1
        noise_ann = [compute_metrics(w)["ann_return_net"] for w in windowed]
        capital = [
            min_capital_usd(b.weights.loc[b.weights.index >= start], min_leg_notional)
            for b in books
        ]
        rows = []
        for alpha in alphas:
            routes, returns, sharpes, reject_rules = Counter(), [], [], Counter()
            for k, (book, w) in enumerate(zip(books, windowed, strict=True)):
                active = book.weights.loc[w.net_return.index].abs().sum(axis=1) > 0
                planted = w.net_return + np.where(active, alpha / ppy, 0.0)
                metrics = compute_metrics(dataclasses.replace(w, net_return=planted))
                others = noise_ann[:k] + noise_ann[k + 1 :]
                metrics["noise_return_percentile"] = noise_percentile(
                    metrics["ann_return_net"], others
                )
                metrics["min_capital_usd"] = capital[k]
                metrics.update({key: shared_facts[key] for key in SHARED_FACTS
                                if key in shared_facts})
                result = evaluate_rules(metrics, ruleset)
                if ruleset.forward_resolution is not None:
                    metrics["forward_days"] = days
                    years = forward_years_needed(metrics["sharpe_net"],
                                                 ruleset.forward_resolution)
                    if years is not None:
                        metrics["forward_days_needed"] = years * 365.0
                    routing = decide_fit_forward_route(
                        ruleset=ruleset, selection=None, forward=result, metrics=metrics,
                        deployable_capital_usd=deployable_capital_usd, params_fixed_at=None,
                    )
                else:
                    routing = decide_route(result, metrics, deployable_capital_usd)
                routes[routing.route] += 1
                if routing.route == "reject":
                    reject_rules.update(result.failed_rule_ids)
                returns.append(metrics["ann_return_net"])
                sharpes.append(metrics["sharpe_net"])
            rows.append(PlantedOutcome(
                alpha=alpha,
                window_days=days,
                n=len(books),
                admitted=sum(v for r, v in routes.items() if r in ADMITTED),
                early=sum(v for r, v in routes.items() if r in EARLY),
                rejected=routes.get("reject", 0),
                mean_return=float(np.mean(returns)),
                median_sharpe=float(np.nanmedian(sharpes)),
                reject_rules=reject_rules,
            ))
        out[label] = rows
    return out


__all__ = ["NoiseBook", "PlantedOutcome", "noise_books", "planted_outcomes"]
