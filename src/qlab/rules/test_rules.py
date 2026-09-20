"""Tests for the screening-rules engine: schema, loader, evaluate, nearness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qlab.rules.engine import evaluate
from qlab.rules.loader import DEFAULT_RULES_DIR, list_versions, load, load_latest
from qlab.rules.nearness import is_near, nearness
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
    assert "soft_fail" in ids_seen
    assert "edge_runs" in ids_seen
    assert result.overall_passed is True
    assert result.failed_fatal_rule_id is None


def test_missing_metric_is_unknown_not_failure() -> None:
    ruleset = RuleSet(
        version="2026-09-20.1",
        rules=[_rule(id="a", metric="missing", fatal=True)],
    )
    result = evaluate({}, ruleset)
    assert result.rows[0].passed is None
    assert result.rows[0].value is None
    assert "missing" in result.unknown_metrics
    # unknown must not trip the fatal short-circuit
    assert result.overall_passed is True
    assert result.failed_fatal_rule_id is None


def test_none_metric_value_is_unknown() -> None:
    ruleset = RuleSet(version="2026-09-20.1", rules=[_rule(id="a", metric="m")])
    result = evaluate({"m": None}, ruleset)
    assert result.rows[0].passed is None
    assert "m" in result.unknown_metrics


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


def test_is_near_true_for_close_failure() -> None:
    # sharpe_net >= 0.8, value 0.79 is a failure close to the threshold
    assert is_near(0.79, 0.8, Comparator.GE, margin=0.05) is True


def test_is_near_false_for_passing_value() -> None:
    # value satisfies the comparator -> not a "near miss", it's a pass
    assert is_near(0.9, 0.8, Comparator.GE, margin=0.5) is False


def test_is_near_false_when_outside_margin() -> None:
    assert is_near(0.1, 0.8, Comparator.GE, margin=0.05) is False


def test_is_near_false_for_none_value() -> None:
    assert is_near(None, 0.8, Comparator.GE, margin=1.0) is False


def test_is_near_accepts_comparator_as_string() -> None:
    assert is_near(0.79, 0.8, ">=", margin=0.05) is True


def test_is_near_threshold_zero() -> None:
    # comparator ">" 0.0, value slightly negative (a failure) and close to 0
    assert is_near(-0.001, 0.0, Comparator.GT, margin=0.01) is True
    assert is_near(-1.0, 0.0, Comparator.GT, margin=0.01) is False


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
