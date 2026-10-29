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
    _usable_metric_values,
    compute_shape_aware_percentiles,
    noise_book_counts,
    noise_percentile,
    noise_power,
)
from qlab.calibration.run import NoiseTrial, start_noise_series
from qlab.pipeline.evaluate import (
    Evaluation,
    _get_or_create_spec_row,
    evaluate_spec,
    not_evaluable_reasons,
    resolve_panel,
)
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

    `percentiles` is `None` when the candidate produced no
    `ann_return_net`/`sharpe_net` to rank: its run failed
    (`candidate.error is not None`) or it was not evaluable (T24). In the
    second case `noise_trials` is empty -- no noise is run for a strategy
    that cannot be tested. The noise trials themselves are still returned:
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
    review_route=None,
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

    `review_route` is handed to the candidate's `evaluate_spec` as is (stage 2 of the
    idea generator, docs/TWO_STAGE.md); the noise books never get one.

    `series` defaults to `qlab.calibration.percentile.CANONICAL_SERIES`
    (`"dollar_neutral"`) -- see that module's docstring for why the
    dollar-neutral series is the one every T27/T28 figure is quoted
    against. Passing `"unconstrained"` computes the percentile against the
    other series instead; nothing about this function is series-specific
    besides the default.

    Every one of the `n_trials` noise trials, and the one candidate trial,
    is a real `evaluate_spec` call -- a real recorded `trial` row, per
    CLAUDE.md ("`trial` пишется всегда"). This function writes exactly
    `n_trials + 1` trials -- or exactly one, when the spec declares that it
    cannot test its idea's strategy (see above) (a caller who wants to reuse an
    already-computed noise sample should call `compute_shape_aware_percentiles`
    directly against trials it already has, not this function).
    """
    if spec.exploratory is None and not_evaluable_reasons(
        spec, None, resolve_panel(session, spec).meta
    ):
        # Declared before any data is read: this run cannot test the
        # strategy its idea names (T24), so there is nothing to rank against
        # noise and 200 noise backtests would be wasted. `evaluate_spec`
        # writes the one not-evaluable trial and says why.
        candidate = evaluate_spec(
            spec,
            session=session,
            ruleset=ruleset,
            deployable_capital_usd=deployable_capital_usd,
        )
        return ShapeAwareResult(candidate=candidate, percentiles=None, noise_trials=())

    # The candidate's registry row and its data are settled and committed before the noise books
    # start in the worker pool: a write lock this session held while it waits for them would
    # stall the workers' writes.
    _get_or_create_spec_row(session, spec)
    session.commit()
    noise_run = start_noise_series(
        series=series,
        n_trials=n_trials,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd,
        reference=spec,
        seed_offset=seed_offset,
        replace_unusable=True,
    )

    def _extra_metrics(base_metrics: dict[str, float]) -> dict[str, float]:
        noise_trials = noise_run.result()  # waits for the books still running in the pool
        percentiles = compute_shape_aware_percentiles(
            base_metrics,
            noise_trials,
            reference_idea_id=spec.idea_id,
            series=series,
        )
        # How many books the bar stands on: asked for, usable, drawn again for the ones that
        # gave no number (docs/T28_SHAPE_AWARE_BAR.md, "Учёт шумовых книг").
        extra: dict[str, float] = dict(noise_book_counts(noise_trials, series=series))
        if percentiles.return_percentile is not None:
            extra["noise_return_percentile"] = percentiles.return_percentile
        if percentiles.sharpe_percentile is not None:
            extra["noise_sharpe_percentile"] = percentiles.sharpe_percentile
        # The same rank on the strategy's own money (docs/CAPITAL_BASE.md): the noise books are
        # read on the base and the reinvest choice of the strategy they imitate, so both sides
        # of the comparison are on one base. Read by no rule yet; the owner decides which
        # return metric decides.
        cap_noise = _usable_metric_values(noise_trials, series=series, metric="cap_ann_return")
        cap_candidate = base_metrics.get("cap_ann_return")
        if cap_noise and isinstance(cap_candidate, int | float) and cap_candidate == cap_candidate:
            extra["noise_cap_return_percentile"] = noise_percentile(cap_candidate, cap_noise)
        # Can the bar tell a good strategy from luck on this window at all
        # (rules 2026-10-24.2)? Asked of the same noise books it ranks against.
        bar = next((r.threshold for r in ruleset.rules
                    if r.metric == "noise_return_percentile"), None)
        if ruleset.noise_power is not None and bar is not None:
            power = noise_power(
                _usable_metric_values(noise_trials, series=series, metric="ann_return_net"),
                ruleset.noise_power.good_return, bar)
            if power is not None:
                extra["noise_power"] = power
        # Inside each market regime too (docs/REGIMES.md): the candidate's
        # return on those days against the same noise books' on the same days.
        for key, value in base_metrics.items():
            if not (key.startswith("regime_") and key.endswith("_ann_return")):
                continue
            noise = [t.evaluation.metrics[key] for t in noise_trials
                     if t.series == series and t.evaluation.metrics
                     and key in t.evaluation.metrics]
            if noise and value == value:
                extra[key.replace("_ann_return", "_noise_percentile")] = noise_percentile(
                    value, noise)
        return extra

    candidate = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd,
        extra_metrics=_extra_metrics,
        review_route=review_route,
    )

    # Also when the candidate failed before it asked: the books are recorded either way, and a
    # failure of theirs is raised here as it was when they ran first.
    noise_trials = noise_run.result()
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
