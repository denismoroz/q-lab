"""Versioned screening-rules engine.

Rules are a versioned YAML config (`rules/<version>.yaml` at the repo root,
format in `docs/REGISTRY.md`); `evaluate()` is the pure function that turns
metrics + a ruleset into verdicts. No agent decides pass/fail — the code
does.
"""

from qlab.rules.engine import EvaluationResult, VerdictRow, evaluate
from qlab.rules.loader import LEGACY_SENTINEL_VERSION, list_versions, load, load_latest
from qlab.rules.nearness import NearnessVerdict, classify_nearness, is_near, nearness
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
    "LEGACY_SENTINEL_VERSION",
    "STAGE_ORDER",
    "Comparator",
    "EvaluationResult",
    "NearnessVerdict",
    "Rule",
    "RuleSet",
    "RetiredRule",
    "Stage",
    "VerdictRow",
    "classify_nearness",
    "evaluate",
    "is_near",
    "list_versions",
    "load",
    "load_latest",
    "nearness",
    "parse_version",
]
