"""Wire a candidate's matched-noise percentile into its OWN verdict
(docs/TASKS.md T27/T28).

`qlab.calibration.percentile` turns (a candidate's metric, a matched noise
sample) into a percentile; `qlab.calibration.run.run_noise_series` produces
the matched noise sample; `qlab.pipeline.evaluate.evaluate_spec` runs the
candidate and hands its metrics to the rules engine. This module is the
glue between the three, because none of them can see the whole picture on
its own: the noise run needs the candidate's SPEC (to structurally match
against) before the candidate itself has been evaluated, and the rules
engine needs the percentile ALONGSIDE the candidate's own backtest metrics
in one mapping, in one trial, for one verdict.

`evaluate_spec_with_shape_aware_bar` is the ONE function this task adds for
the owner to call per candidate. Order of operations, and why it is this
order and not the more obvious "evaluate the candidate, then calibrate,
then decide":

    1. Run `n_trials` matched noise trials against the candidate's spec.
       This does not need the candidate's own metrics -- each noise trial
       re-derives the candidate's weights internally
       (`qlab.calibration.noise.NoiseStrategy`, from `reference_code_ref`/
       `reference_params`) and perturbs them, so it can run before, after,
       or interleaved with the candidate's own trial with no ordering
       constraint from that side.
    2. Evaluate the candidate itself through `evaluate_spec`, passing an
       `extra_metrics` callback that computes the percentile from step 1's
       noise sample against THIS run's own `ann_return_net`/`sharpe_net`
       once they exist. This keeps everything -- backtest metrics, venue
       facts, and the percentile -- in the ONE metrics mapping the rules
       engine evaluates and the ONE trial row that gets persisted, rather
       than writing a trial without the percentile and a second one with
       it (two trials for one evaluation would double-count in the
       deflation accounting CLAUDE.md requires: "trial written for every
       run", not for every attempt at a run).

Evaluating the candidate BEFORE running the 200 matched noise trials would
work too (nothing here truly forces this order) -- noise-first is chosen
only so a caller who only wants the noise-sample's own shape (e.g. for a
report) does not have to run the candidate to get it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from qlab.calibration.percentile import (
    CANONICAL_SERIES,
    ShapeAwarePercentiles,
    compute_shape_aware_percentiles,
)
from qlab.calibration.run import NoiseTrial, run_noise_series
from qlab.pipeline.evaluate import Evaluation, evaluate_spec
from qlab.pipeline.spec import StrategySpec
from qlab.rules.schema import RuleSet


@dataclass(frozen=True, slots=True)
class ShapeAwareResult:
    """Everything one `evaluate_spec_with_shape_aware_bar` call produced:
    the candidate's own `Evaluation` (trial, metrics, rules verdict,
    routing -- exactly what `qlab evaluate` would have produced, plus the
    percentile keys in `metrics`), the `ShapeAwarePercentiles` summary, and
    the raw noise trials the percentile was computed from (so a caller can
    build a report, e.g. `docs/T28_SHAPE_AWARE_BAR.md`'s before/after
    table, without re-querying the registry).

    `percentiles` is `None` only when `candidate.error is not None` -- the
    candidate's own run failed before it ever produced an
    `ann_return_net`/`sharpe_net` to rank, so there is nothing to compute a
    percentile against. The noise trials themselves are still returned:
    that calibration work happened and was recorded regardless of whether
    the candidate's own run succeeded.
    """

    candidate: Evaluation
    percentiles: ShapeAwarePercentiles | None
    noise_trials: tuple[NoiseTrial, ...]


def evaluate_spec_with_shape_aware_bar(
    spec: StrategySpec,
    *,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    n_trials: int,
    seed_offset: int = 0,
    series: str = CANONICAL_SERIES,
) -> ShapeAwareResult:
    """Evaluate `spec` under `ruleset`, with a matched-noise percentile of
    its own computed and folded into the SAME verdict.

    `n_trials` has no default on purpose (unlike `qlab.calibration.run.
    run_noise_series`, which defaults nothing either, and
    `qlab.cli.calibrate_cmd`, which defaults to 200 only at the CLI layer):
    a calibration run is genuinely expensive (docs/TASKS.md T16: "не меньше
    200... порядка десяти минут"), so a caller must say explicitly how much
    of that cost it is paying, rather than inheriting a number it never
    chose. `qlab.calibration.run.run_noise_series` itself still rejects
    `n_trials <= 0`.

    `series` defaults to `qlab.calibration.percentile.CANONICAL_SERIES`
    (`"dollar_neutral"`) -- see that module's docstring for why the
    dollar-neutral series is the one every T27/T28 figure is quoted
    against. Passing `"unconstrained"` computes the percentile against the
    other series instead; nothing about this function is series-specific
    besides the default.

    Every one of the `n_trials` noise trials, and the one candidate trial,
    is a real `evaluate_spec` call -- a real recorded `trial` row, per
    CLAUDE.md ("`trial` пишется всегда"). This function writes exactly
    `n_trials + 1` trials, never fewer (a caller who wants to reuse an
    already-computed noise sample should call `compute_shape_aware_percentiles`
    directly against trials it already has, not this function).
    """
    noise_trials = run_noise_series(
        series=series,
        n_trials=n_trials,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd,
        reference=spec,
        seed_offset=seed_offset,
    )

    def _extra_metrics(base_metrics: dict[str, float]) -> dict[str, float]:
        percentiles = compute_shape_aware_percentiles(
            base_metrics,
            noise_trials,
            reference_idea_id=spec.idea_id,
            series=series,
        )
        extra: dict[str, float] = {}
        if percentiles.return_percentile is not None:
            extra["noise_return_percentile"] = percentiles.return_percentile
        if percentiles.sharpe_percentile is not None:
            extra["noise_sharpe_percentile"] = percentiles.sharpe_percentile
        return extra

    candidate = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd,
        extra_metrics=_extra_metrics,
    )

    percentiles = (
        compute_shape_aware_percentiles(
            candidate.metrics, noise_trials, reference_idea_id=spec.idea_id, series=series
        )
        if candidate.metrics is not None
        else None
    )

    return ShapeAwareResult(
        candidate=candidate, percentiles=percentiles, noise_trials=tuple(noise_trials)
    )


__all__ = ["ShapeAwareResult", "evaluate_spec_with_shape_aware_bar"]
