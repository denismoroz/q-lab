"""Tests for the screening-rules engine: schema, loader, evaluate, nearness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qlab.rules.engine import evaluate
from qlab.rules.loader import DEFAULT_RULES_DIR, list_versions, load, load_latest
from qlab.rules.nearness import NearnessVerdict, classify_nearness, is_near, nearness
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage, parse_version

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_version_format_accepted() -> None:
    RuleSet(version="2026-09-20.1")


@pytest.mark.parametrize(
    "bad_version",
    ["2026-09-20", "2026-9-20.1", "20260920.1", "2026-09-20.1a", "not-a-version"],
)
def test_version_format_rejected(bad_version: str) -> None:
    with pytest.raises(ValueError):
        RuleSet(version=bad_version)


def test_duplicate_rule_id_rejected() -> None:
    dupe = {
        "id": "dupe",
        "stage": "edge",
        "metric": "x",
        "comparator": ">",
        "threshold": 0,
    }
    with pytest.raises(ValueError):
        RuleSet(version="2026-09-20.1", rules=[dupe, dupe])


def test_comparator_is_strict_enum() -> None:
    with pytest.raises(ValueError):
        Rule(id="r", stage="edge", metric="x", comparator="=", threshold=0)


def test_stage_is_strict_enum() -> None:
    with pytest.raises(ValueError):
        Rule(id="r", stage="not-a-stage", metric="x", comparator=">", threshold=0)


def test_near_margin_abs_defaults_to_none() -> None:
    rule = Rule(id="r", stage="edge", metric="x", comparator=">", threshold=0)
    assert rule.near_margin_abs is None


def test_near_margin_abs_rejects_negative() -> None:
    with pytest.raises(ValueError):
        Rule(id="r", stage="edge", metric="x", comparator=">", threshold=0, near_margin_abs=-0.1)


def test_near_margin_abs_accepts_non_negative() -> None:
    rule = Rule(id="r", stage="edge", metric="x", comparator=">", threshold=0, near_margin_abs=0.1)
    assert rule.near_margin_abs == 0.1


@pytest.mark.parametrize(
    "version,expected",
    [
        ("2026-09-20.1", ((2026, 9, 20), 1)),
        ("2026-09-20.10", ((2026, 9, 20), 10)),
    ],
)
def test_parse_version(version: str, expected: tuple) -> None:
    d, n = parse_version(version)
    assert (d.year, d.month, d.day) == expected[0]
    assert n == expected[1]


def test_version_sorts_numerically_not_lexicographically() -> None:
    versions = ["2026-09-20.9", "2026-09-20.10", "2026-09-20.2"]
    versions.sort(key=parse_version)
    assert versions == ["2026-09-20.2", "2026-09-20.9", "2026-09-20.10"]
    # sanity: plain string sort would get this wrong, which is the whole point
    assert sorted(versions) != versions or sorted(["9", "10", "2"]) == ["10", "2", "9"]


# --------------------------------------------------------------------------
# engine: evaluate()
# --------------------------------------------------------------------------


def _rule(**overrides) -> Rule:
    base = dict(id="r", stage=Stage.EDGE, metric="m", comparator=Comparator.GT, threshold=0.0)
    base.update(overrides)
    return Rule(**base)


def test_evaluate_is_pure_same_input_same_output() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(
                id="a",
                stage=Stage.PREFLIGHT,
                metric="cap",
                comparator=Comparator.LE,
                threshold=100,
            ),
            _rule(id="b", stage=Stage.EDGE, metric="ret", comparator=Comparator.GT, threshold=0),
        ],
    )
    metrics = {"cap": 50.0, "ret": 0.1}

    result_1 = evaluate(metrics, ruleset)
    result_2 = evaluate(metrics, ruleset)

    assert result_1 == result_2
    # inputs must not be mutated
    assert metrics == {"cap": 50.0, "ret": 0.1}
    assert [r.id for r in ruleset.rules] == ["a", "b"]


def test_evaluate_no_side_effects_across_calls() -> None:
    ruleset = RuleSet(version="2026-09-20.1", rules=[_rule(id="a")])
    evaluate({"m": 1.0}, ruleset)
    # a second, independent call with different metrics must not be
    # affected by the first call in any way
    result = evaluate({"m": -1.0}, ruleset)
    assert result.rows[0].passed is False


def test_fatal_stops_next_stage_but_not_current_stage() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="preflight_fatal_fail", stage=Stage.PREFLIGHT, metric="a",
                  comparator=Comparator.GT, threshold=10, fatal=True),
            _rule(id="preflight_other", stage=Stage.PREFLIGHT, metric="b",
                  comparator=Comparator.GT, threshold=0, fatal=False),
            _rule(id="edge_should_not_run", stage=Stage.EDGE, metric="c",
                  comparator=Comparator.GT, threshold=0, fatal=True),
        ],
    )
    metrics = {"a": 1.0, "b": 1.0, "c": 1.0}

    result = evaluate(metrics, ruleset)

    ids_seen = [row.rule_id for row in result.rows]
    assert "preflight_fatal_fail" in ids_seen
    # same-stage sibling still evaluated despite the fatal failure
    assert "preflight_other" in ids_seen
    # next stage never processed at all — no row emitted for it
    assert "edge_should_not_run" not in ids_seen

    assert result.overall_passed is False
    assert result.failed_fatal_rule_id == "preflight_fatal_fail"


def test_non_fatal_failure_does_not_stop_next_stage() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="soft_fail", stage=Stage.PREFLIGHT, metric="a",
                  comparator=Comparator.GT, threshold=10, fatal=False),
            _rule(id="edge_runs", stage=Stage.EDGE, metric="b",
                  comparator=Comparator.GT, threshold=0, fatal=False),
        ],
    )
    result = evaluate({"a": 1.0, "b": 1.0}, ruleset)
    ids_seen = [row.rule_id for row in result.rows]
    # non-fatal failure does not stop the next stage from being evaluated
    assert "soft_fail" in ids_seen
    assert "edge_runs" in ids_seen
    # ... but it is still fail-closed: a non-fatal failure is a failure
    assert result.overall_passed is False
    assert result.failed_fatal_rule_id is None
    assert result.failed_rule_ids == ("soft_fail",)
    assert result.decisive is True


def test_missing_metric_is_unknown_not_a_fatal_failure_but_is_fail_closed() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[_rule(id="a", metric="missing", fatal=True)],
    )
    result = evaluate({}, ruleset)
    assert result.rows[0].passed is None
    assert result.rows[0].value is None
    assert "missing" in result.unknown_metrics
    # unknown must not trip the fatal short-circuit ...
    assert result.failed_fatal_rule_id is None
    assert result.failed_rule_ids == ()
    # ... but a candidate that wasn't fully computed is not "passed" either
    assert result.decisive is False
    assert result.overall_passed is False


def test_empty_metrics_against_real_ruleset_is_not_promoted() -> None:
    # This is the regression the fail-open bug produced: a candidate we
    # computed nothing for must never come back as having passed.
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="a", metric="m1", fatal=True),
            _rule(id="b", metric="m2", fatal=False),
        ],
    )
    result = evaluate({}, ruleset)
    assert result.decisive is False
    assert result.overall_passed is False
    assert result.unknown_metrics == ("m1", "m2")


def test_none_metric_value_is_unknown() -> None:
    ruleset = RuleSet(version="2026-09-20.1", rules=[_rule(id="a", metric="m")])
    result = evaluate({"m": None}, ruleset)
    assert result.rows[0].passed is None
    assert "m" in result.unknown_metrics
    assert result.decisive is False


def test_overall_passed_true_requires_no_failures_and_no_unknowns() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="a", metric="m1", comparator=Comparator.GT, threshold=0, fatal=True),
            _rule(id="b", metric="m2", comparator=Comparator.GE, threshold=0, fatal=False),
        ],
    )
    result = evaluate({"m1": 1.0, "m2": 0.0}, ruleset)
    assert result.decisive is True
    assert result.failed_rule_ids == ()
    assert result.overall_passed is True


def test_vacuous_pass_on_empty_ruleset() -> None:
    # No rules at all -> nothing was computed, nothing failed, nothing is
    # unknown. Vacuously true, same as `all(())`. Documented in
    # EvaluationResult's docstring precisely so this isn't mistaken for a
    # fail-open bug later.
    ruleset = RuleSet(version="2026-09-20.1", rules=[])
    result = evaluate({"anything": 1.0}, ruleset)
    assert result.rows == ()
    assert result.decisive is True
    assert result.overall_passed is True
    assert result.failed_rule_ids == ()
    assert result.failed_fatal_rule_id is None


def test_stages_filter_restricts_evaluation() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="a", stage=Stage.PREFLIGHT, metric="a"),
            _rule(id="b", stage=Stage.EDGE, metric="b"),
        ],
    )
    result = evaluate({"a": 1.0, "b": 1.0}, ruleset, stages=[Stage.EDGE])
    ids_seen = [row.rule_id for row in result.rows]
    assert ids_seen == ["b"]


def test_stage_processing_order_is_fixed_regardless_of_declaration_order() -> None:
    # declared edge-then-preflight in the rules list
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[
            _rule(id="edge_rule", stage=Stage.EDGE, metric="e", fatal=True, threshold=10),
            _rule(id="preflight_rule", stage=Stage.PREFLIGHT, metric="p", fatal=True, threshold=10),
        ],
    )
    result = evaluate({"e": -1.0, "p": -1.0}, ruleset)
    # preflight fails fatally first -> edge stage must never run
    ids_seen = [row.rule_id for row in result.rows]
    assert ids_seen == ["preflight_rule"]
    assert result.failed_fatal_rule_id == "preflight_rule"


# --------------------------------------------------------------------------
# loader: based_on inheritance, retired exclusion, version listing
# --------------------------------------------------------------------------


def _write_ruleset(directory: Path, payload: dict) -> None:
    path = directory / f"{payload['version']}.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_based_on_inherits_and_overrides_by_id(tmp_path: Path) -> None:
    _write_ruleset(
        tmp_path,
        {
            "version": "2026-09-01.1",
            "based_on": None,
            "rules": [
                {"id": "a", "stage": "edge", "metric": "m_a", "comparator": ">", "threshold": 0},
                {"id": "b", "stage": "edge", "metric": "m_b", "comparator": ">", "threshold": 0},
            ],
        },
    )
    _write_ruleset(
        tmp_path,
        {
            "version": "2026-09-10.1",
            "based_on": "2026-09-01.1",
            "rules": [
                # override threshold of "a", leave "b" untouched, add "c"
                {"id": "a", "stage": "edge", "metric": "m_a", "comparator": ">", "threshold": 5},
                {"id": "c", "stage": "tail", "metric": "m_c", "comparator": "<=", "threshold": 1},
            ],
        },
    )

    child = load("2026-09-10.1", rules_dir=tmp_path)

    by_id = {r.id: r for r in child.rules}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].threshold == 5  # overridden
    assert by_id["b"].metric == "m_b"  # inherited untouched
    assert by_id["c"].stage == Stage.TAIL  # newly added


def test_retired_rule_excluded_from_inheritance(tmp_path: Path) -> None:
    _write_ruleset(
        tmp_path,
        {
            "version": "2026-09-01.1",
            "based_on": None,
            "rules": [
                {"id": "a", "stage": "edge", "metric": "m_a", "comparator": ">", "threshold": 0},
                {
                    "id": "b",
                    "stage": "correlation",
                    "metric": "m_b",
                    "comparator": ">",
                    "threshold": 0,
                },
            ],
        },
    )
    _write_ruleset(
        tmp_path,
        {
            "version": "2026-09-12.1",
            "based_on": "2026-09-01.1",
            "rules": [],
            "retired": [
                {"id": "b", "retired_in": "2026-09-12.1", "reason": "no longer a filter"},
            ],
        },
    )

    child = load("2026-09-12.1", rules_dir=tmp_path)

    assert {r.id for r in child.rules} == {"a"}
    assert {r.id for r in child.retired} == {"b"}


def test_list_versions_sorted_numerically(tmp_path: Path) -> None:
    for v in ["2026-09-20.9", "2026-09-20.10", "2026-09-20.2", "2026-09-19.1"]:
        _write_ruleset(tmp_path, {"version": v, "based_on": None, "rules": []})

    versions = list_versions(tmp_path)

    assert versions == [
        "2026-09-19.1",
        "2026-09-20.2",
        "2026-09-20.9",
        "2026-09-20.10",
    ]


def test_load_latest_picks_highest_version(tmp_path: Path) -> None:
    for v in ["2026-09-20.1", "2026-09-20.10", "2026-09-20.2"]:
        _write_ruleset(tmp_path, {"version": v, "based_on": None, "rules": []})

    latest = load_latest(tmp_path)

    assert latest.version == "2026-09-20.10"


def test_load_missing_version_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load("2026-01-01.1", rules_dir=tmp_path)


def test_filename_must_match_version_field(tmp_path: Path) -> None:
    (tmp_path / "2026-09-20.1.yaml").write_text(
        yaml.safe_dump({"version": "2026-09-20.2", "rules": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load("2026-09-20.1", rules_dir=tmp_path)


def test_based_on_cycle_is_rejected(tmp_path: Path) -> None:
    # a -> b -> a
    _write_ruleset(
        tmp_path,
        {"version": "2026-09-01.1", "based_on": "2026-09-05.1", "rules": []},
    )
    _write_ruleset(
        tmp_path,
        {"version": "2026-09-05.1", "based_on": "2026-09-01.1", "rules": []},
    )

    with pytest.raises(ValueError, match="cycle"):
        load("2026-09-01.1", rules_dir=tmp_path)


def test_based_on_self_cycle_is_rejected(tmp_path: Path) -> None:
    # a -> a
    _write_ruleset(
        tmp_path,
        {"version": "2026-09-01.1", "based_on": "2026-09-01.1", "rules": []},
    )

    with pytest.raises(ValueError, match="cycle"):
        load("2026-09-01.1", rules_dir=tmp_path)


def test_based_on_longer_cycle_is_rejected(tmp_path: Path) -> None:
    # a -> b -> c -> a
    _write_ruleset(tmp_path, {"version": "2026-09-01.1", "based_on": "2026-09-02.1", "rules": []})
    _write_ruleset(tmp_path, {"version": "2026-09-02.1", "based_on": "2026-09-03.1", "rules": []})
    _write_ruleset(tmp_path, {"version": "2026-09-03.1", "based_on": "2026-09-01.1", "rules": []})

    with pytest.raises(ValueError, match="cycle"):
        load("2026-09-01.1", rules_dir=tmp_path)


def test_load_legacy_sentinel_raises_clear_error_not_file_not_found(tmp_path: Path) -> None:
    from qlab.rules.loader import LEGACY_SENTINEL_VERSION

    with pytest.raises(ValueError, match="sentinel"):
        load(LEGACY_SENTINEL_VERSION, rules_dir=tmp_path)


# --------------------------------------------------------------------------
# nearness
# --------------------------------------------------------------------------


def test_nearness_basic() -> None:
    assert nearness(9.0, 10.0) == pytest.approx(0.1)
    assert nearness(11.0, 10.0) == pytest.approx(0.1)
    assert nearness(10.0, 10.0) == 0.0


def test_nearness_threshold_zero_uses_absolute_difference() -> None:
    assert nearness(0.05, 0.0) == pytest.approx(0.05)
    assert nearness(-0.05, 0.0) == pytest.approx(0.05)
    assert nearness(0.0, 0.0) == 0.0


# --- classify_nearness(): the five branches -------------------------------


def test_classify_nearness_undefined_for_none_value() -> None:
    assert classify_nearness(None, 0.8, Comparator.GE, margin=1.0) is NearnessVerdict.UNDEFINED
    # even with threshold == 0 and margin_abs given, a missing value is
    # still undefined, not "not near"
    assert classify_nearness(None, 0.0, Comparator.GT, margin_abs=0.1) is NearnessVerdict.UNDEFINED


def test_classify_nearness_not_near_for_passing_value() -> None:
    # value satisfies the comparator -> it's a pass, not a near-miss
    assert classify_nearness(0.9, 0.8, Comparator.GE, margin=0.5) is NearnessVerdict.NOT_NEAR
    assert classify_nearness(0.1, 0.0, Comparator.GT, margin_abs=0.01) is NearnessVerdict.NOT_NEAR


def test_classify_nearness_relative_threshold_nonzero() -> None:
    # sharpe_net >= 0.8, value 0.79 is a failure close to the threshold
    assert classify_nearness(0.79, 0.8, Comparator.GE, margin=0.05) is NearnessVerdict.NEAR
    # value 0.1 is far below 0.8 -> outside a 5% relative margin
    assert classify_nearness(0.1, 0.8, Comparator.GE, margin=0.05) is NearnessVerdict.NOT_NEAR


def test_classify_nearness_threshold_zero_with_margin_abs() -> None:
    # comparator ">" 0.0, value slightly negative (a failure) and close to 0
    assert classify_nearness(-0.001, 0.0, Comparator.GT, margin_abs=0.01) is NearnessVerdict.NEAR
    assert classify_nearness(-1.0, 0.0, Comparator.GT, margin_abs=0.01) is NearnessVerdict.NOT_NEAR


def test_classify_nearness_threshold_zero_without_margin_abs_is_undefined() -> None:
    # This is the regression the incident report describes: sharpe_net
    # -0.22 against threshold 0.0 must NOT come back "near" just because a
    # *relative* margin happens to be set — a relative margin is
    # meaningless at threshold == 0, and no margin_abs was supplied.
    result = classify_nearness(-0.22, 0.0, Comparator.GT, margin=0.25)
    assert result is NearnessVerdict.UNDEFINED
    assert result is not NearnessVerdict.NOT_NEAR  # "undefined" != "definitely not near"

    # a real annual-return-losing candidate from the incident report
    result_fx = classify_nearness(-0.14, 0.0, Comparator.GT, margin=0.25)
    assert result_fx is NearnessVerdict.UNDEFINED


def test_classify_nearness_relative_without_margin_is_undefined() -> None:
    # threshold != 0 but no relative margin supplied -> nothing to compare
    # against, same "don't guess" principle as the threshold == 0 case
    assert classify_nearness(0.79, 0.8, Comparator.GE) is NearnessVerdict.UNDEFINED


def test_classify_nearness_accepts_comparator_as_string() -> None:
    assert classify_nearness(0.79, 0.8, ">=", margin=0.05) is NearnessVerdict.NEAR


# --- is_near(): thin wrapper over classify_nearness() ----------------------


def test_is_near_true_only_for_near_verdict() -> None:
    assert is_near(0.79, 0.8, Comparator.GE, margin=0.05) is True


def test_is_near_false_for_passing_value() -> None:
    assert is_near(0.9, 0.8, Comparator.GE, margin=0.5) is False


def test_is_near_false_when_outside_margin() -> None:
    assert is_near(0.1, 0.8, Comparator.GE, margin=0.05) is False


def test_is_near_false_for_none_value() -> None:
    assert is_near(None, 0.8, Comparator.GE, margin=1.0) is False


def test_is_near_collapses_undefined_to_false() -> None:
    # This is exactly the pitfall the docstring warns about: is_near()
    # cannot distinguish "definitely not near" from "undefined because no
    # margin_abs was configured for a threshold == 0 rule". Both are False.
    assert is_near(-0.22, 0.0, Comparator.GT, margin=0.25) is False
    assert classify_nearness(-0.22, 0.0, Comparator.GT, margin=0.25) is NearnessVerdict.UNDEFINED


# --------------------------------------------------------------------------
# rules/2026-09-20.1.yaml — the actual first shipped version
# --------------------------------------------------------------------------


def test_first_ruleset_file_exists() -> None:
    assert (DEFAULT_RULES_DIR / "2026-09-20.1.yaml").is_file()


def test_first_ruleset_loads_and_validates() -> None:
    ruleset = load("2026-09-20.1")

    assert ruleset.version == "2026-09-20.1"
    assert ruleset.based_on is None
    assert len(ruleset.rules) > 0

    rule_ids = {r.id for r in ruleset.rules}
    # preflight must be checked first per docs/PLAN.md, deliberately ahead
    # of the SCREENING.md order
    assert "capital_fit" in rule_ids
    assert "net_edge_positive" in rule_ids

    # profile/correlation are classification, not filters: no active rules
    stages_with_rules = {r.stage for r in ruleset.rules}
    assert Stage.PROFILE not in stages_with_rules
    assert Stage.CORRELATION not in stages_with_rules

    retired_ids = {r.id for r in ruleset.retired}
    assert "decorrelation_required" in retired_ids
    # SCREENING.md §6: PBO is invalid on a menu of variants of one idea
    # (null ~0.605); trend TSMOM and the XSMOM overlays were previously
    # rejected partly on that uninformative number.
    assert "pbo_threshold_on_homogeneous_menu" in retired_ids


def test_first_ruleset_evaluates_without_crashing() -> None:
    ruleset = load("2026-09-20.1")
    metrics = {
        "venue_supported": 1.0,
        "data_forward_available": 1.0,
        "atomic_execution": 1.0,
        "min_notional_usd": 100.0,
        "point_in_time_universe": 1.0,
        "accrual_applied": 1.0,
        "ann_return_net": 0.05,
    }
    result = evaluate(metrics, ruleset)
    assert result.overall_passed is True
    assert result.unknown_metrics == ()


# --------------------------------------------------------------------------
# rules/2026-09-22.1.yaml -- the shape-aware admission bar (docs/TASKS.md T28)
# --------------------------------------------------------------------------


def test_shape_aware_ruleset_file_exists() -> None:
    assert (DEFAULT_RULES_DIR / "2026-09-22.1.yaml").is_file()


def test_shape_aware_ruleset_inherits_net_edge_positive_unchanged() -> None:
    base = load("2026-09-21.1")
    shape_aware = load("2026-09-22.1")

    base_edge = next(r for r in base.rules if r.id == "net_edge_positive")
    inherited_edge = next(r for r in shape_aware.rules if r.id == "net_edge_positive")

    # `based_on` inheritance carries this rule over byte-for-byte -- T28's
    # whole point is that the ECONOMIC floor is untouched, only a second,
    # independent floor is added alongside it.
    assert inherited_edge == base_edge

    rule_ids = {r.id for r in shape_aware.rules}
    assert "shape_aware_edge" in rule_ids


def test_shape_aware_rule_is_fatal_edge_stage_with_threshold_at_99th_percentile() -> None:
    ruleset = load("2026-09-22.1")
    rule = next(r for r in ruleset.rules if r.id == "shape_aware_edge")

    assert rule.stage is Stage.EDGE
    assert rule.metric == "noise_return_percentile"
    assert rule.comparator is Comparator.GE
    assert rule.threshold == pytest.approx(0.99)
    assert rule.fatal is True


def test_shape_aware_rule_missing_percentile_is_unknown_not_a_pass() -> None:
    """The whole discipline point of T28: a candidate evaluated without its
    own matched calibration must never read as admitted just because the
    percentile rule saw nothing to fail."""
    ruleset = load("2026-09-22.1")
    metrics = {
        "venue_supported": 1.0,
        "data_forward_available": 1.0,
        "atomic_execution": 1.0,
        "min_capital_usd": 100.0,
        "point_in_time_universe": 1.0,
        "accrual_applied": 1.0,
        "ann_return_net": 0.32,  # comfortably clears net_edge_positive alone
        # noise_return_percentile deliberately absent: calibration not run.
    }
    result = evaluate(metrics, ruleset)

    assert result.overall_passed is False
    assert result.decisive is False
    assert "noise_return_percentile" in result.unknown_metrics
    # missing is unknown, never a fatal failure in its own right
    assert result.failed_fatal_rule_id is None


def test_shape_aware_rule_rejects_candidate_below_99th_percentile() -> None:
    ruleset = load("2026-09-22.1")
    metrics = {
        "venue_supported": 1.0,
        "data_forward_available": 1.0,
        "atomic_execution": 1.0,
        "min_capital_usd": 100.0,
        "point_in_time_universe": 1.0,
        "accrual_applied": 1.0,
        "ann_return_net": 0.32,
        "noise_return_percentile": 0.84,  # e.g. XSMOM-20's own measured spot
    }
    result = evaluate(metrics, ruleset)

    assert result.overall_passed is False
    assert result.decisive is True
    assert "shape_aware_edge" in result.failed_rule_ids


def test_shape_aware_rule_admits_candidate_at_or_above_99th_percentile() -> None:
    ruleset = load("2026-09-22.1")
    metrics = {
        "venue_supported": 1.0,
        "data_forward_available": 1.0,
        "atomic_execution": 1.0,
        "min_capital_usd": 100.0,
        "point_in_time_universe": 1.0,
        "accrual_applied": 1.0,
        "ann_return_net": 0.05,
        "noise_return_percentile": 0.99,
    }
    result = evaluate(metrics, ruleset)

    assert result.overall_passed is True
    assert result.decisive is True
