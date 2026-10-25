"""The outcome-market mechanisms added to docs/MECHANISMS.md on 2026-10-25: the generator's
parser reads them, the gap table marks them unusable (there is no binary-contract block), and
the director's `outcome-markets` family holds exactly them.

These tests read the real catalogue, so they pin what was written there."""

from __future__ import annotations

from pathlib import Path

import pytest

import qlab.agents.generator as gen
from qlab.agents.director import parse_gap_table
from qlab.directions import load_config

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = ROOT / "docs" / "MECHANISMS.md"

# parameter -> values the generator reads from the rows of each mechanism
PINNED = {
    "outcome-favorite-longshot": {
        "signal.buy_above_price": (0.5, 0.61), "signal.buy_band_low": (0.08,),
        "signal.buy_band_high": (0.1,), "signal.price_bucket_edge": ()},
    "outcome-horizon-miscalibration": {
        "signal.max_days_to_expiry": (), "signal.minutes_to_expiry": ()},
    "outcome-maker-taker-split": {"costs.taker_fee_coefficient": (0.07,)},
    "outcome-complete-set-arbitrage": {"signal.min_gap": ()},
}


@pytest.fixture(scope="module")
def mechanisms() -> list[gen.Mechanism]:
    return gen.catalogue(CATALOGUE)


def test_every_outcome_mechanism_is_parsed_with_its_rows(mechanisms) -> None:
    by_id = {m.id: m for m in mechanisms}
    for mech_id, rows in PINNED.items():
        m = by_id[mech_id]
        assert {r.param: r.values for r in m.params} == rows, mech_id
        assert m.summary and m.blocks and m.checked and m.unexpressed and m.sources, mech_id
        assert not m.fully_expressed, mech_id  # no block expresses a binary contract yet


def test_numbers_in_the_notes_of_a_row_are_not_values(mechanisms) -> None:
    # The 10 and 90 cent edges of the reporting buckets are bins, not entry rules, and the
    # ten-minute window is an analysis window: neither may become a usable number.
    by_id = {m.id: m for m in mechanisms}
    rows = {r.param: r for r in by_id["outcome-favorite-longshot"].params}
    assert rows["signal.price_bucket_edge"].values == ()
    rows = {r.param: r for r in by_id["outcome-horizon-miscalibration"].params}
    assert rows["signal.minutes_to_expiry"].values == ()


def test_the_gap_table_marks_every_outcome_mechanism_unusable() -> None:
    gaps = parse_gap_table(CATALOGUE.read_text(encoding="utf-8"))
    for mech_id in PINNED:
        gap, usable = gaps[mech_id]
        assert gap and usable.startswith("нет"), mech_id


def test_the_outcome_family_holds_exactly_these_mechanisms(mechanisms) -> None:
    config = load_config(ROOT / "night" / "director.yaml")
    assert config is not None
    assert set(config.families["outcome-markets"]) == set(PINNED)
    for mech_id in PINNED:
        assert config.family_of(mech_id) == "outcome-markets"
    listed = [i for members in config.families.values() for i in members]
    assert set(listed) <= {m.id for m in mechanisms} and len(listed) == len(set(listed))
