"""The rules engine reads a NaN metric as unknown."""

from __future__ import annotations


def test_a_nan_metric_is_unknown_not_a_failure() -> None:
    """2026-10-04: a matched-noise book's two-day forward test had a NaN
    annual return; read as a failure it broke the registry's verdict check."""
    from qlab.rules.engine import evaluate
    from qlab.rules.loader import load_latest

    result = evaluate({"cap_ann_return": float("nan")}, load_latest())
    row = next(r for r in result.rows if r.metric == "cap_ann_return")
    assert row.passed is None and row.value is None
    assert "cap_ann_return" in result.unknown_metrics
