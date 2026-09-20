"""The `qlab evaluate` pipeline: spec -> data -> backtest -> metrics -> verdict.

The single entry point every strategy candidate goes through
(docs/PLAN.md, "Что такое фреймворк"). See `qlab.pipeline.spec` for the
input file format and `qlab.pipeline.evaluate` for the pipeline itself.
"""

from qlab.pipeline.evaluate import (
    Evaluation,
    RoutingDecision,
    StrategyResolutionError,
    decide_route,
    evaluate_spec,
    resolve_strategy,
)
from qlab.pipeline.spec import SpecCosts, SpecData, StrategySpec, load_spec

__all__ = [
    "Evaluation",
    "RoutingDecision",
    "SpecCosts",
    "SpecData",
    "StrategyResolutionError",
    "StrategySpec",
    "decide_route",
    "evaluate_spec",
    "load_spec",
    "resolve_strategy",
]
