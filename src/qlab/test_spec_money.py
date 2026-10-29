"""The strategy's own money is a field of the spec (owner 2026-10-29, В1): where it comes from
and that every place that states it agrees."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qlab import ladder
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec, load_spec

ROOT = Path(__file__).resolve().parents[2]
MONEY_PARAMS = ("capital_usd", "budget_cap_usdc")


def _spec(**extra) -> dict:
    return {"idea_id": "t", "title": "t", "code_ref": "a:b", "params": {},
            "data": {"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1),
                     "end": date(2026, 1, 1)},
            "costs": {"taker_fee_bps": 4.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
            "unexpressed_mechanisms": [], **extra}


def test_a_spec_may_leave_the_money_unset_and_nothing_is_guessed() -> None:
    assert StrategySpec.model_validate(_spec()).strategy_capital_usd is None


def test_a_sum_needs_a_citation_and_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="needs sources"):
        StrategySpec.model_validate(_spec(strategy_capital_usd=120))
    with pytest.raises(ValidationError, match="must be > 0"):
        StrategySpec.model_validate(_spec(
            strategy_capital_usd=0, sources={"strategy_capital_usd": "owner 2026-10-28"}))
    ok = StrategySpec.model_validate(_spec(
        strategy_capital_usd=120, sources={"strategy_capital_usd": "owner 2026-10-28: «да я»"}))
    assert ok.strategy_capital_usd == 120


def test_the_sum_is_checked_like_every_other_number(tmp_path) -> None:
    # a cited file must hold the number: 7 is nowhere in CLAUDE.md's budget lines as the budget,
    # but "120" is
    good = StrategySpec.model_validate(_spec(
        strategy_capital_usd=120, sources={"strategy_capital_usd": "CLAUDE.md, the FRAB budget"}))
    assert not [p for p in source_problems(good) if "strategy_capital_usd" in p]
    invented = StrategySpec.model_validate(_spec(
        strategy_capital_usd=123456, sources={"strategy_capital_usd": "CLAUDE.md"}))
    assert any("strategy_capital_usd = 123456" in p for p in source_problems(invented))


def _specs() -> list[tuple[Path, StrategySpec]]:
    return [(p, load_spec(p)) for p in sorted((ROOT / "specs").rglob("*.yaml"))]


def _money_params(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            here = f"{prefix}.{key}" if prefix else key
            if key in MONEY_PARAMS and isinstance(value, int | float):
                yield here, float(value)
            yield from _money_params(value, here)


def test_every_money_in_the_spec_files_agrees_with_the_spec_own_parameter() -> None:
    """One source of truth: where a strategy takes its money as a parameter (Bv2 `capital_usd`,
    trend `capital_usd`, FRAB `budget_cap_usdc`) the field says the same number."""
    stated = 0
    for path, spec in _specs():
        if spec.strategy_capital_usd is None:
            continue
        stated += 1
        for where, value in _money_params(spec.params):
            assert value == spec.strategy_capital_usd, (path, where, value)
        assert not [p for p in source_problems(spec) if "strategy_capital_usd" in p], path
    assert stated >= 35


def test_every_paper_and_live_book_runs_a_spec_that_states_the_same_money() -> None:
    books = yaml.safe_load((ROOT / "night" / "paper_books.yaml").read_text(encoding="utf-8"))
    for block in ("books", "live_books"):
        for idea, cfg in books[block].items():
            spec = load_spec(ROOT / cfg["spec"])
            assert spec.strategy_capital_usd == float(cfg["strategy_capital_usd"]), idea


def test_the_ladder_reads_the_money_of_the_spec_files() -> None:
    money = ladder.spec_money(ROOT / "specs")
    assert money["strategy-b-v2"] == 257.0 and money["frab"] == 120.0
    assert money["trend-retune-weekly"] == 3000.0
    assert "xsmom-research-liq1m" not in money  # nothing declared: not set, not guessed
    assert "frab-live-own-spot" in money and money["frab-live-own-spot"] == 90.0


def test_a_second_stage_without_a_file_of_its_own_has_its_first_stages_money() -> None:
    from qlab.ladder import Ledgers, strategy_money
    from qlab.registry.models import Idea
    from qlab.registry.stages import NOTES_PREFIX

    child = Idea(id="x-hl", parent_id="x", title="t", notes=f"{NOTES_PREFIX} x")
    assert strategy_money(child, Ledgers(money={"x": 257.0})) == 257.0
    assert strategy_money(child, Ledgers(money={"x-hl": 90.0, "x": 257.0})) == 90.0
    assert strategy_money("y", Ledgers(money={"x": 257.0})) is None
