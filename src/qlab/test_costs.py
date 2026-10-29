"""The cost table (night/costs.yaml, `qlab.costs`, docs/COSTS.md) and the specs against it."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from qlab import costs
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec, load_spec

pytestmark = pytest.mark.cost_table
ROOT = Path(__file__).resolve().parents[2]


def _spec(*, source="hyperliquid", taker=4.5, slippage=0.9, execution_venue=None,
          sources=None) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "t", "title": "t", "code_ref": "a:b", "params": {},
        "data": {"source": source, "interval": "1d", "start": date(2025, 1, 1),
                 "end": date(2026, 1, 1)},
        "costs": {"taker_fee_bps": taker, "slippage_bps": slippage}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "execution_venue": execution_venue,
        "sources": sources or {}})


def test_the_table_has_the_numbers_the_owner_named() -> None:
    hl = costs.venue_costs("hyperliquid")
    assert (hl.taker_fee_bps, hl.slippage_bps) == (4.5, 0.9)  # docs/TWO_STAGE.md «Комиссия»
    binance = costs.venue_costs("binance")
    assert (binance.taker_fee_bps, binance.slippage_bps) == (5.0, 0.9)
    assert costs.venue_costs("nowhere") is None


def test_every_number_of_the_table_is_cited_where_the_checker_finds_it() -> None:
    # the cited files are q-lab's own and frab's sibling repository: the same checks as a spec's
    assert costs.table_citation_problems() == []


def test_a_row_without_a_source_or_a_key_is_refused(tmp_path) -> None:
    broken = tmp_path / "costs.yaml"
    broken.write_text("venues:\n  x: {taker_fee_bps: 1, slippage_bps: 1, sources: "
                      "{taker_fee_bps: a}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no source for slippage_bps"):
        costs.table(broken)
    broken.write_text("venues:\n  x: {taker_fee_bps: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        costs.table(broken)


def test_a_spec_with_its_venues_row_is_fine() -> None:
    assert costs.problems(_spec()) == []
    assert costs.problems(_spec(source="binance", taker=5.0)) == []


def test_the_row_is_that_of_the_venue_where_the_orders_go() -> None:
    proxy = _spec(source="binance", taker=4.5, execution_venue="hyperliquid")
    assert costs.order_venue(proxy) == "hyperliquid" and costs.problems(proxy) == []
    wrong = _spec(source="binance", taker=4.5)  # orders on binance: its row is 5.0
    assert any("not the table's row of binance" in p for p in costs.problems(wrong))


def test_other_numbers_need_a_sentence_that_says_why() -> None:
    other = _spec(taker=3.5)
    assert any("say why" in p for p in costs.problems(other))
    said = _spec(taker=3.5, sources={costs.DEVIATION_KEY: "owner 2026-10-29: frab's own number"})
    assert costs.problems(said) == []
    assert not [p for p in source_problems(said) if "differs_from_table" in p]


def test_the_sentence_must_itself_name_something_checkable() -> None:
    vague = _spec(taker=3.5, sources={costs.DEVIATION_KEY: "because we felt like it"})
    assert any("names nothing checkable" in p for p in source_problems(vague))


def test_a_sentence_for_costs_that_are_the_rows_is_refused() -> None:
    needless = _spec(sources={costs.DEVIATION_KEY: "owner 2026-10-29: no need"})
    assert any("drop the sentence" in p for p in costs.problems(needless))


def test_a_venue_with_no_row_is_named() -> None:
    assert any("no row for venue 'aster'" in p for p in costs.problems(_spec(source="aster")))


def test_every_spec_of_the_lab_has_its_venues_row_or_says_why() -> None:
    """The 164 specs of specs/: the old numbers were kept, each with its sentence."""
    checked = 0
    for path in sorted((ROOT / "specs").rglob("*.yaml")):
        spec = load_spec(path)
        assert costs.problems(spec) == [], path
        checked += 1
    assert checked > 100


def test_the_second_stage_config_takes_its_costs_from_the_table() -> None:
    from qlab.pipeline import stage2

    config = stage2.load_config(ROOT / "night" / "stage2.yaml")
    row = costs.venue_costs(config.venue)
    assert (config.taker_fee_bps, config.slippage_bps) == (row.taker_fee_bps, row.slippage_bps)
    assert config.cost_sources["costs.taker_fee_bps"] == row.sources["taker_fee_bps"]
    raw = yaml.safe_load((ROOT / "night" / "stage2.yaml").read_text(encoding="utf-8"))
    assert "taker_fee_bps" not in raw["costs"]  # one place for the number


def test_a_stage_two_spec_made_now_has_the_row_without_a_sentence() -> None:
    from qlab.pipeline import stage2

    parent = _spec(source="binance", taker=5.0)
    config = stage2.load_config(ROOT / "night" / "stage2.yaml")
    child = stage2.translate(parent, config).spec
    assert costs.problems(child) == []
    assert costs.DEVIATION_KEY not in child.sources
