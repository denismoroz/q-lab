"""How a candidate ranks against its OWN matched noise (docs/TASKS.md T27/T28).

The problem this module exists to fix: `rules/2026-09-21.1.yaml` admits a
strategy on `ann_return_net >= 0.04`. That threshold is the owner's economic
floor (beat stablecoin staking) and it is a fixed number on an absolute
return axis. What 4% net annual return MEANS -- how hard it is for pure
noise to clear it by accident -- depends entirely on the shape of the book:

    noise matched to trend's ~150-leg book clears 4% net       1.0% of the time
    noise matched to a 20-leg book, same panel, same rules     23.5% of the time

(`docs/CALIBRATION_2026-09-21.1.md`, `docs/CALIBRATION_2026-09-21.1__xsmom-honest-20legs.md`,
both dollar-neutral series, 200 matched noise trials each.) A fixed return
threshold is not comparable across book shapes; a PERCENTILE is, by
construction: "the candidate beats N% of noise books structurally matched to
IT" means the same thing whether the book has 20 legs or 150.

This module is deliberately narrow -- it only turns (a candidate's own
metric, a matched noise sample of that same metric) into a percentile. It
does NOT run the noise calibration itself (`qlab.calibration.run.
run_noise_series` already does that, and it is not cheap: 200 trials, ~10
minutes) and it does NOT decide a verdict (`qlab.rules.engine.evaluate`
does that, from whatever ruleset references the metric this module
produces). Orchestrating "run the matched noise, then compute the
percentile, then feed it to the rules engine alongside the candidate's own
backtest" is `qlab.calibration.shape_aware`'s job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from qlab.calibration.run import NoiseTrial

# The one series this module's percentile is computed against. Every figure
# quoted in docs/TASKS.md T27/T28 and in `rules/2026-09-22.1.yaml`'s own
# rationale (1.0% / 23.5%) is the dollar-neutral series -- "чистый шум
# отбора инструментов, без беты и carry" -- not the unconstrained one, so
# this is the series the admission-rate reasoning was actually calibrated
# on. The unconstrained series remains available from `run_noise_series`
# for anyone who wants it, this module just does not default to it.
CANONICAL_SERIES = "dollar_neutral"


@dataclass(frozen=True, slots=True)
class ShapeAwarePercentiles:
    """One candidate's position in its own matched-noise distribution.

    `return_percentile`/`sharpe_percentile` are `None`, never `NaN`, when
    they could not be computed (no usable noise sample, or the candidate's
    own metric is missing/NaN) -- `None` is what a missing dict key becomes
    when read with `.get()`, which is exactly how
    `qlab.rules.engine.evaluate` decides a metric is "unknown" rather than
    failing or passing a rule on it (docs/REGISTRY.md's `unknown` verdict
    kind). A `NaN` value would instead compare `False` against every
    threshold and silently read as an ordinary FAILURE, not as "we could not
    tell" -- exactly the confusion this task's discipline section warns
    against ("a missing percentile must NOT read as a pass" -- and, by the
    same logic, must not silently read as a distinct kind of fail either).
    """

    reference_idea_id: str
    series: str
    n_requested: int
    n_usable_return: int
    n_usable_sharpe: int
    return_percentile: float | None
    sharpe_percentile: float | None


def noise_percentile(candidate_value: float, noise_values: Sequence[float]) -> float:
    """Share of `noise_values` that `candidate_value` beats.

    "Beats" means strictly greater than -- a noise draw exactly equal to the
    candidate is not counted as beaten, matching the fact that a candidate
    tied with the noise sample's own value is exactly AT that percentile,
    not above it. With a continuous metric like `ann_return_net` or
    `sharpe_net` and O(200) noise draws, exact ties are a measure-zero event
    in practice; this convention only matters for degenerate inputs (e.g. a
    constant noise sample), where it is the conservative reading: a tie does
    not count in the candidate's favour.

    Returns a fraction in `[0, 1]`: `1.0` means the candidate beat every
    noise draw, `0.0` means it beat none.

    Raises:
        ValueError: `noise_values` is empty -- there is nothing to rank
            against. Callers must decide what "no usable noise sample"
            means for their metric (`ShapeAwarePercentiles` returns `None`
            rather than raising, for exactly this reason) — this function
            itself refuses to guess a percentile from zero data points.
    """
    if not noise_values:
        raise ValueError("cannot compute a percentile against an empty noise sample")
    arr = np.asarray(noise_values, dtype=float)
    return float(np.mean(arr < candidate_value))


def _usable_metric_values(
    noise_trials: Sequence[NoiseTrial], *, series: str, metric: str
) -> list[float]:
    """Pull `metric` out of every `noise_trials` entry in `series` that
    actually produced a number.

    Skips: trials from a different series, trials whose run errored (a
    `StructuralMismatchError` or any other failure -- `evaluation.metrics is
    None`, see `qlab.pipeline.evaluate.Evaluation`), and trials where the
    metric itself came back `NaN` (e.g. `sharpe_net` on fewer than 2
    periods or zero variance, `qlab.harness.metrics.compute_metrics`). The
    resulting count can be smaller than `n_trials` requested -- that
    shrinkage is exactly what `docs/XSMOM_T21.md` found happen to 150 of
    200 trials before the rebalance-cadence fix (T27, gap 5), and it must
    be reported honestly (`ShapeAwarePercentiles.n_usable_return` /
    `n_usable_sharpe`), never silently backfilled.
    """
    values: list[float] = []
    for trial in noise_trials:
        if trial.series != series:
            continue
        metrics = trial.evaluation.metrics
        if metrics is None:
            continue
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        values.append(float(value))
    return values


def compute_shape_aware_percentiles(
    candidate_metrics: Mapping[str, float],
    noise_trials: Sequence[NoiseTrial],
    *,
    reference_idea_id: str,
    series: str = CANONICAL_SERIES,
) -> ShapeAwarePercentiles:
    """Where `candidate_metrics` sits in the `noise_trials` distribution.

    `noise_trials` is expected to be the OUTPUT of
    `qlab.calibration.run.run_noise_series` called with `reference` set to
    the candidate's OWN spec (not trend's, not any other book's) -- a
    percentile is only shape-aware if the noise it is measured against was
    structurally matched to the same book. This function does not check
    that itself (it has no way to: `NoiseTrial` does not carry the
    reference spec's identity) -- it is the caller's responsibility, see
    `qlab.calibration.shape_aware.evaluate_spec_with_shape_aware_bar`.

    `n_requested` counts every `noise_trials` entry in `series`, regardless
    of whether it produced a usable number -- the honest denominator for
    "how many trials did this calibration cost", independent of how many
    were usable for either percentile.
    """
    n_requested = sum(1 for trial in noise_trials if trial.series == series)

    return_values = _usable_metric_values(noise_trials, series=series, metric="ann_return_net")
    sharpe_values = _usable_metric_values(noise_trials, series=series, metric="sharpe_net")

    candidate_return = candidate_metrics.get("ann_return_net")
    candidate_sharpe = candidate_metrics.get("sharpe_net")

    return_percentile = (
        noise_percentile(candidate_return, return_values)
        if return_values and candidate_return is not None and not math.isnan(candidate_return)
        else None
    )
    sharpe_percentile = (
        noise_percentile(candidate_sharpe, sharpe_values)
        if sharpe_values and candidate_sharpe is not None and not math.isnan(candidate_sharpe)
        else None
    )

    return ShapeAwarePercentiles(
        reference_idea_id=reference_idea_id,
        series=series,
        n_requested=n_requested,
        n_usable_return=len(return_values),
        n_usable_sharpe=len(sharpe_values),
        return_percentile=return_percentile,
        sharpe_percentile=sharpe_percentile,
    )


__all__ = [
    "CANONICAL_SERIES",
    "ShapeAwarePercentiles",
    "compute_shape_aware_percentiles",
    "noise_percentile",
]
