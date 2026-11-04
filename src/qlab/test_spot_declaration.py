"""A spec that reads the coin's own spot closes declares `spot_prices: own`
(qlab.spot_declaration)."""

from __future__ import annotations

import pytest
import yaml

from qlab.pipeline.spec import StrategySpec
from qlab.spot_declaration import (
    declare_own_in_spec_text,
    names_the_perp_mark,
    spot_declaration_problems,
)

REFUSAL = ("The card reads the coin's spot closes for the fast return; the stand's default marks "
           "a spot column at its perp's price (data.spot_prices perp, as the base), so the fast "
           "return is read at that mark and the spot-perp gap does not enter it.")
OTHER = "The lot table is today's szDecimals applied to the whole history."


def _spec(unexpressed: list[str], spot_prices: str = "perp") -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a.b:C", "params": {},
        "data": {"source": "binance", "interval": "1h", "start": "2026-01-01",
                 "end": "2026-02-01", "instruments": ["BTC", "BTC-SPOT"],
                 "spot_prices": spot_prices},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": unexpressed,
        "sources": {"costs": "owner: test", "min_leg_notional": "owner: test"}})


def test_the_stands_own_refusal_is_recognised_and_a_lot_table_is_not() -> None:
    assert names_the_perp_mark(REFUSAL)
    assert names_the_perp_mark("the stand marks a spot column at its perp's price")
    assert not names_the_perp_mark(OTHER)


def test_declaring_the_gap_as_unexpressed_while_marking_at_the_perp_is_a_problem() -> None:
    problems = spot_declaration_problems(_spec([REFUSAL]))
    assert len(problems) == 1 and "declare data.spot_prices: own" in problems[0]


def test_a_stale_line_beside_own_is_a_problem_too() -> None:
    problems = spot_declaration_problems(_spec([REFUSAL], "own"))
    assert len(problems) == 1 and "remove the line" in problems[0]


def test_own_without_the_line_and_other_lines_are_fine() -> None:
    assert spot_declaration_problems(_spec([], "own")) == []
    assert spot_declaration_problems(_spec([OTHER])) == []
    assert spot_declaration_problems(_spec([OTHER], "own")) == []


TEXT = f"""# comment kept
idea_id: x
data:
  source: binance
  interval: "1h"
  instruments: [BTC, BTC-SPOT]

unexpressed_mechanisms:
  - "{REFUSAL}"
  - "{OTHER}"

required_instruments: [BTC]
"""


def test_the_edit_declares_own_and_removes_only_the_perp_mark_line() -> None:
    new, removed = declare_own_in_spec_text(TEXT)
    assert removed == [REFUSAL]
    loaded = yaml.safe_load(new)
    assert loaded["data"]["spot_prices"] == "own"
    assert loaded["unexpressed_mechanisms"] == [OTHER]
    assert new.startswith("# comment kept\n") and "required_instruments: [BTC]" in new
    # nothing else moved
    expected = yaml.safe_load(TEXT)
    expected["data"]["spot_prices"] = "own"
    expected["unexpressed_mechanisms"] = [OTHER]
    assert loaded == expected


def test_an_emptied_list_is_written_as_an_empty_list() -> None:
    text = TEXT.replace(f'  - "{OTHER}"\n', "")
    new, _ = declare_own_in_spec_text(text)
    assert yaml.safe_load(new)["unexpressed_mechanisms"] == []
    assert "unexpressed_mechanisms: []" in new


def test_a_perp_line_in_the_data_block_is_replaced_and_the_edit_is_repeatable() -> None:
    text = TEXT.replace('  interval: "1h"\n', '  interval: "1h"\n  spot_prices: perp\n')
    new, _ = declare_own_in_spec_text(text)
    assert yaml.safe_load(new)["data"]["spot_prices"] == "own"
    assert new.count("spot_prices:") == 1
    again, removed = declare_own_in_spec_text(new)
    assert removed == [] and yaml.safe_load(again) == yaml.safe_load(new)


def test_a_file_without_a_data_block_is_refused() -> None:
    with pytest.raises(ValueError, match="no data block"):
        declare_own_in_spec_text("idea_id: x\n")
