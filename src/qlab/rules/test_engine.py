"""The rules engine reads a NaN metric as unknown."""

from __future__ import annotations


def test_a_nan_metric_is_unknown_not_a_failure() -> None:
    """2026-10-04: a matched-noise book's two-day forward test had a NaN
    annual return; read as a failure it broke the registry's verdict check."""
    from qlab.rules.engine import evaluate
    from qlab.rules.loader import load_latest

    result = evaluate({"ann_return_net": float("nan")}, load_latest())
    row = next(r for r in result.rows if r.metric == "ann_return_net")
    assert row.passed is None and row.value is None
    assert "ann_return_net" in result.unknown_metrics
