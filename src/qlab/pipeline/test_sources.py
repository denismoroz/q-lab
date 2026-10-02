"""Tests for `qlab.pipeline.sources`: every number cites a checkable source."""

from __future__ import annotations

import pytest

from qlab.pipeline.sources import SpecSourceError, require_sources, source_problems
from qlab.pipeline.spec import StrategySpec

FIXTURE = "owner 2026-01-01: test fixture"


def _spec(params: dict, sources: dict) -> StrategySpec:
    base_sources = {"costs": FIXTURE, "min_leg_notional": FIXTURE}
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": params,
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2026-01-01",
                 "end": "2026-02-01"},
        "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "sources": {**base_sources, **sources},
    })


def test_uncovered_number_is_named() -> None:
    problems = source_problems(_spec({"window": 30, "other": None}, {}))
    assert problems == ["no source for window = 30"]


def test_number_must_appear_in_the_cited_file() -> None:
    ok = _spec({"lookbacks": [30, 60, 90, 120]},
               {"lookbacks": "src/frab/strategy/trend/params.py (TrendParams.lookbacks)"})
    assert source_problems(ok) == []
    invented = _spec({"lookbacks": [30, 61]},
                     {"lookbacks": "src/frab/strategy/trend/params.py"})
    [problem] = source_problems(invented)
    assert "61 does not appear" in problem


def test_missing_file_and_uncheckable_citation_are_refused() -> None:
    problems = source_problems(_spec({"a": 1, "b": 2},
                                     {"a": "docs/NO_SUCH_FILE.md", "b": "common knowledge"}))
    assert any("does not exist" in p for p in problems)
    assert any("names nothing checkable" in p for p in problems)


def test_derived_formula_must_give_the_value() -> None:
    assert source_problems(_spec({"slip": 0.9}, {"slip": "derived: 4.4 - 3.5; SCREENING"})) == []
    [problem] = source_problems(_spec({"slip": 1.0}, {"slip": "derived: 4.4 - 3.5; SCREENING"}))
    assert "derives 0.9, not 1" in problem


def test_an_ancestor_covers_everything_under_it_and_owner_decisions_count() -> None:
    spec = _spec({"base": {"x": 0.137, "y": {"z": True}}}, {"base": FIXTURE})
    assert source_problems(spec) == []


def test_require_sources_raises_with_every_problem() -> None:
    with pytest.raises(SpecSourceError, match="no source for window"):
        require_sources(_spec({"window": 30}, {}))
