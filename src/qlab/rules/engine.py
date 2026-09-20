"""Pure rule-evaluation engine.

`evaluate()` is the only thing that produces a screening verdict — no agent
decides pass/fail, per docs/REGISTRY.md and CLAUDE.md. It takes a metrics
mapping and a `RuleSet` and returns a list of verdict rows plus an aggregate
summary. It performs no I/O and has no side effects: same inputs, same
outputs, every time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qlab.rules.schema import STAGE_ORDER, Rule, RuleSet, Stage


@dataclass(frozen=True, slots=True)
class VerdictRow:
    """One rule's verdict against one set of metrics.

    Mirrors the columns of the `verdict` table in docs/REGISTRY.md, minus the
    identifiers (idea/spec/trial/data-range) that only the caller — not the
    pure engine — knows about. This is a plain dataclass, not an ORM model;
    persisting it is someone else's job.
    """

    stage: Stage
    rule_id: str
    rules_version: str
    metric: str
    value: float | None
    comparator: str
    threshold: float
    passed: bool | None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate result of evaluating a full ruleset against one metrics set."""

    rows: tuple[VerdictRow, ...]
    overall_passed: bool
    failed_fatal_rule_id: str | None
    unknown_metrics: tuple[str, ...]


def evaluate(
    metrics: Mapping[str, float | None],
    ruleset: RuleSet,
    stages: Sequence[Stage] | None = None,
) -> EvaluationResult:
    """Evaluate `ruleset` against `metrics`.

    - A rule whose metric is missing from `metrics` or is `None` gets a
      verdict of `passed=None` ("unknown") — this is neither a pass nor a
      failure and never triggers a fatal stop.
    - Stages are always processed in the fixed order preflight -> edge ->
      tail -> profile -> capacity -> correlation, regardless of the order
      rules appear in `ruleset.rules` or the order given in `stages`.
    - If `stages` is given, only those stages are evaluated (still in the
      fixed relative order); rules belonging to other stages are skipped
      entirely (no row is emitted for them).
    - If a fatal rule fails, every other rule in that *same* stage is still
      evaluated (a full picture of the stage is needed to reconsider
      graveyard verdicts later), but no further stage is processed.
    """
    allowed_stages = set(stages) if stages is not None else None

    rules_by_stage: dict[Stage, list[Rule]] = {stage: [] for stage in STAGE_ORDER}
    for rule in ruleset.rules:
        rules_by_stage[rule.stage].append(rule)

    rows: list[VerdictRow] = []
    failed_fatal_rule_id: str | None = None
    unknown_metrics: set[str] = set()

    for stage in STAGE_ORDER:
        if allowed_stages is not None and stage not in allowed_stages:
            continue

        stage_rules = rules_by_stage[stage]
        if not stage_rules:
            continue

        stage_has_fatal_failure = False
        for rule in stage_rules:
            raw_value = metrics.get(rule.metric)

            if raw_value is None:
                rows.append(
                    VerdictRow(
                        stage=stage,
                        rule_id=rule.id,
                        rules_version=ruleset.version,
                        metric=rule.metric,
                        value=None,
                        comparator=rule.comparator.value,
                        threshold=rule.threshold,
                        passed=None,
                        note="metric missing or None: verdict unknown",
                    )
                )
                unknown_metrics.add(rule.metric)
                continue

            value = float(raw_value)
            passed = rule.comparator.compare(value, rule.threshold)
            rows.append(
                VerdictRow(
                    stage=stage,
                    rule_id=rule.id,
                    rules_version=ruleset.version,
                    metric=rule.metric,
                    value=value,
                    comparator=rule.comparator.value,
                    threshold=rule.threshold,
                    passed=passed,
                )
            )

            if not passed and rule.fatal:
                stage_has_fatal_failure = True
                if failed_fatal_rule_id is None:
                    failed_fatal_rule_id = rule.id

        if stage_has_fatal_failure:
            break

    return EvaluationResult(
        rows=tuple(rows),
        overall_passed=failed_fatal_rule_id is None,
        failed_fatal_rule_id=failed_fatal_rule_id,
        unknown_metrics=tuple(sorted(unknown_metrics)),
    )
