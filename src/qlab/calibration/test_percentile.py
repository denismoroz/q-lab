"""Tests for `qlab.calibration.percentile` (docs/TASKS.md T27/T28).

Pure-function tests: `NoiseTrial`/`Evaluation` are plain dataclasses, so a
matched-noise sample can be hand-built here without a database, a snapshot,
or a real backtest -- exactly the same reason `qlab.rules.test_rules`
exercises `evaluate()` with hand-built metrics dicts instead of full
pipeline runs.
"""

from __future__ import annotations

import pytest

from qlab.calibration.percentile import (
    CANONICAL_SERIES,
    compute_shape_aware_percentiles,
    noise_percentile,
)
from qlab.calibration.run import NoiseTrial
from qlab.pipeline.evaluate import Evaluation, RoutingDecision

# --------------------------------------------------------------------------
# noise_percentile
# --------------------------------------------------------------------------


def test_noise_percentile_beats_everything() -> None:
    assert noise_percentile(100.0, [1.0, 2.0, 3.0]) == 1.0


def test_noise_percentile_beats_nothing() -> None:
    assert noise_percentile(-100.0, [1.0, 2.0, 3.0]) == 0.0


def test_noise_percentile_exact_tie_does_not_count_as_beaten() -> None:
    # 2.0 beats only the one value strictly below it (1.0), not the tie.
    assert noise_percentile(2.0, [1.0, 2.0, 3.0]) == pytest.approx(1 / 3)


def test_noise_percentile_matches_hand_count() -> None:
    noise = [-0.1, -0.05, 0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
    # candidate beats the 6 values strictly below 0.04
    assert noise_percentile(0.04, noise) == pytest.approx(6 / 8)


def test_noise_percentile_rejects_empty_sample() -> None:
    with pytest.raises(ValueError):
        noise_percentile(0.05, [])


# --------------------------------------------------------------------------
# compute_shape_aware_percentiles
# --------------------------------------------------------------------------


def _noise_trial(
    *, series: str = CANONICAL_SERIES, ann_return_net: float | None, sharpe_net: float | None
) -> NoiseTrial:
    metrics = None
    if ann_return_net is not None:
        metrics = {"ann_return_net": ann_return_net}
        if sharpe_net is not None:
            metrics["sharpe_net"] = sharpe_net
    return NoiseTrial(
        series=series,
        generator="shuffled_instruments",
        seed=0,
        evaluation=Evaluation(
            trial_id=1,
            metrics=metrics,
            rules_result=None,
            routing=RoutingDecision(route="reject", reason="noise"),
            error=None if metrics is not None else "structural mismatch",
        ),
    )


def test_matches_known_admission_rate_shape() -> None:
    # 198 noise draws below 0.04, 2 at/above it -- the exact 1.0% shape
    # docs/CALIBRATION_2026-09-21.1.md reports for trend's own book.
    noise = [_noise_trial(ann_return_net=-0.05, sharpe_net=-0.5) for _ in range(198)]
    noise += [_noise_trial(ann_return_net=0.10, sharpe_net=1.0) for _ in range(2)]

    percentiles = compute_shape_aware_percentiles(
        {"ann_return_net": 0.045, "sharpe_net": 0.3},
        noise,
        reference_idea_id="trend-tsmom-crypto",
    )

    assert percentiles.n_requested == 200
    assert percentiles.n_usable_return == 200
    # candidate (0.045) beats all 198 low draws but neither of the 2 high ones
    assert percentiles.return_percentile == pytest.approx(198 / 200)


def test_structural_mismatch_failures_are_excluded_from_the_denominator() -> None:
    noise = [_noise_trial(ann_return_net=-0.05, sharpe_net=-0.5) for _ in range(50)]
    noise += [_noise_trial(ann_return_net=None, sharpe_net=None) for _ in range(150)]

    percentiles = compute_shape_aware_percentiles(
        {"ann_return_net": 0.05, "sharpe_net": 0.5}, noise, reference_idea_id="xsmom-honest-20legs"
    )

    assert percentiles.n_requested == 200
    assert percentiles.n_usable_return == 50
    assert percentiles.return_percentile == pytest.approx(1.0)


def test_nan_sharpe_excluded_but_return_still_usable() -> None:
    noise = [_noise_trial(ann_return_net=0.01, sharpe_net=float("nan")) for _ in range(5)]

    percentiles = compute_shape_aware_percentiles(
        {"ann_return_net": 0.05, "sharpe_net": 1.0}, noise, reference_idea_id="ref"
    )

    assert percentiles.n_usable_return == 5
    assert percentiles.n_usable_sharpe == 0
    assert percentiles.return_percentile == pytest.approx(1.0)
    assert percentiles.sharpe_percentile is None


def test_no_usable_noise_returns_none_not_an_exception() -> None:
    noise = [_noise_trial(ann_return_net=None, sharpe_net=None) for _ in range(10)]

    percentiles = compute_shape_aware_percentiles(
        {"ann_return_net": 0.05, "sharpe_net": 0.5}, noise, reference_idea_id="ref"
    )

    assert percentiles.n_usable_return == 0
    assert percentiles.return_percentile is None
    assert percentiles.sharpe_percentile is None


def test_other_series_is_ignored_by_default() -> None:
    dollar_neutral = [_noise_trial(ann_return_net=-0.1, sharpe_net=-1.0) for _ in range(5)]
    unconstrained = [
        _noise_trial(series="unconstrained", ann_return_net=1.0, sharpe_net=5.0) for _ in range(5)
    ]

    percentiles = compute_shape_aware_percentiles(
        {"ann_return_net": 0.0, "sharpe_net": 0.0},
        dollar_neutral + unconstrained,
        reference_idea_id="ref",
    )

    # candidate at 0.0 beats every dollar-neutral draw (-0.1) and none of
    # the unconstrained ones -- if unconstrained leaked in, this would drop.
    assert percentiles.n_requested == 5
    assert percentiles.return_percentile == pytest.approx(1.0)


def test_candidate_missing_its_own_metric_returns_none() -> None:
    noise = [_noise_trial(ann_return_net=0.01, sharpe_net=0.1) for _ in range(10)]

    percentiles = compute_shape_aware_percentiles({}, noise, reference_idea_id="ref")

    assert percentiles.return_percentile is None
    assert percentiles.sharpe_percentile is None
