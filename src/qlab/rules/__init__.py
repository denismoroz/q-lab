"""Versioned screening-rules engine.

Rules are a versioned YAML config (`rules/<version>.yaml` at the repo root,
format in `docs/REGISTRY.md`); `evaluate()` is the pure function that turns
metrics + a ruleset into verdicts. No agent decides pass/fail — the code
does.
"""

from qlab.rules.engine import EvaluationResult, VerdictRow, evaluate
from qlab.rules.loader import list_versions, load, load_latest
from qlab.rules.nearness import is_near, nearness
from qlab.rules.schema import (
    STAGE_ORDER,
    Comparator,
    RetiredRule,
    Rule,
    RuleSet,
    Stage,
    parse_version,
)

__all__ = [
    "STAGE_ORDER",
    "Comparator",
    "EvaluationResult",
    "Rule",
    "RuleSet",
    "RetiredRule",
    "Stage",
    "VerdictRow",
    "evaluate",
    "is_near",
    "list_versions",
    "load",
    "load_latest",
    "nearness",
    "parse_version",
]
