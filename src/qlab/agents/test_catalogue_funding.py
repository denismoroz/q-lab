"""The funding / venue mechanisms added to docs/MECHANISMS.md on 2026-10-25: the generator's
parser reads every new row, the director's gap table and families know them, and the guard
accepts exactly the numbers their rows give (and none for a parameter without a value).

These tests read the real catalogue, so they pin what was written there: a source that
gives a number later must change the pinned value here on purpose, not by accident."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import qlab.agents.generator as gen
from qlab.agents.director import parse_gap_table
from qlab.directions import load_config

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = ROOT / "docs" / "MECHANISMS.md"
DATA_END = date(2026, 10, 24)

NEW = ("perp-carry-crash-timing", "perp-funding-extreme-rebound", "cross-venue-funding-spread",
       "classic-carry", "crypto-variance-risk-premium")

# parameter -> values the generator reads from the rows of each new mechanism
PINNED = {
    "perp-carry-crash-timing": {
        "signal.window_days": (), "rebalance_days": (), "min_history_days": ()},
    "perp-funding-extreme-rebound": {
        "signal.history_window_days": (180.0,), "signal.tail_fraction": (0.1,),
        "rebalance_days": (1.0,), "min_history_days": (180.0,)},
    "cross-venue-funding-spread": {
        "signal.min_spread_bps": (20.0,), "rebalance_days": (), "min_history_days": ()},
    # Month windows in days since the owner's «месяц = 30 дней» (2026-10-26).
    "classic-carry": {
        "portfolio.fraction": (0.2, 0.25, 0.3), "signal.window_days": (360.0,),
        "rebalance_days": (30.0,), "min_history_days": ()},
    "crypto-variance-risk-premium": {"signal.window_days": ()},
}


@pytest.fixture(scope="module")
def mechanisms() -> list[gen.Mechanism]:
    return gen.catalogue(CATALOGUE)


def test_every_new_mechanism_is_parsed_with_its_rows(mechanisms) -> None:
    by_id = {m.id: m for m in mechanisms}
    assert len(by_id) == len(mechanisms) >= 15
    for mech_id, rows in PINNED.items():
        m = by_id[mech_id]
        assert {r.param: r.values for r in m.params} == rows, mech_id
        assert m.summary and m.blocks and m.checked and m.unexpressed and m.sources, mech_id
        assert not m.fully_expressed, mech_id  # the blocks express none of them yet


def test_the_parameter_of_a_block_that_does_not_exist_is_not_a_parameter_of_one_that_does(
        mechanisms) -> None:
    # 180 days is the history the percentile is ranked in, not the window of the existing
    # `funding` signal; it must not be usable under signal.window_days.
    m = next(m for m in mechanisms if m.id == "perp-funding-extreme-rebound")
    assert not any(r.param == "signal.window_days" for r in m.params)


def test_the_gap_table_marks_every_new_mechanism_unusable() -> None:
    gaps = parse_gap_table(CATALOGUE.read_text(encoding="utf-8"))
    for mech_id in NEW:
        gap, usable = gaps[mech_id]
        assert gap and usable.startswith("нет"), mech_id


def test_the_director_families_hold_the_new_mechanisms(mechanisms) -> None:
    config = load_config(ROOT / "night" / "director.yaml")
    assert config is not None
    for mech_id in NEW[:4]:
        assert config.family_of(mech_id) == "funding-and-basis", mech_id
    assert config.family_of("crypto-variance-risk-premium") == "volatility-premium"
    listed = [i for members in config.families.values() for i in members]
    assert set(listed) <= {m.id for m in mechanisms} and len(listed) == len(set(listed))


def _spec(mech: str, name: str, *, signal: str, portfolio: str, rebalance: int,
          min_history: int, rows: dict[str, str]) -> str:
    cited = "".join(f"  {path}: 'docs/MECHANISMS.md, mechanism {mech}, row {path}: {text}'\n"
                    for path, text in rows.items())
    return f"""# mechanism: {mech}
idea_id: {name}
title: Пример для проверки охраны
code_ref: qlab.strategies.blocks:Composed
params:
  universe:
    rule: all
  signal:
{signal}
  portfolio:
{portfolio}
  sizing:
    block: equal
    gross: 1.0
  rebalance_days: {rebalance}
  min_history_days: {min_history}
data:
  source: binance
  interval: 1d
  start: '2019-09-08'
  end: '2026-10-24'
costs:
  taker_fee_bps: 3.5
  slippage_bps: 0.9
min_leg_notional: 10.0
simultaneous_legs: 1
noise_universe: panel
unexpressed_mechanisms: []
differs_from_source:
- example difference written by hand
sources:
  universe: 'docs/MECHANISMS.md, mechanism {mech}, blocks: universe: all'
  signal: 'docs/MECHANISMS.md, mechanism {mech}, blocks: signal: funding'
  portfolio: 'docs/MECHANISMS.md, mechanism {mech}, blocks: portfolio'
  sizing: 'docs/MECHANISMS.md, mechanism {mech}, equal shares'
  sizing.gross: 'docs/MECHANISMS.md, framework value: gross 1 = the whole capital, no leverage'
{cited}  costs.taker_fee_bps: 'owner 2026-10-25: test fixture'
  costs.slippage_bps: 'owner 2026-10-25: test fixture'
  min_leg_notional: 'owner 2026-10-25: test fixture'
"""


def _check(tmp_path: Path, mechanisms, text: str, name: str) -> list[str]:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    _, problems = gen.check_generated(path, mechanisms=mechanisms, data_end=DATA_END,
                                      reference=None)
    return [p for p in problems if not p.startswith("fixation:")]  # not what these tests ask


def _blocks_problems(problems: list[str]) -> list[str]:
    """Problems about numbers; the fixture's costs sources are not under test."""
    return [p for p in problems if "costs" not in p and "min_leg_notional" not in p]


def test_a_funding_spec_of_a_mechanism_without_values_is_refused_number_by_number(
        mechanisms, tmp_path) -> None:
    mech = "perp-carry-crash-timing"
    text = _spec(mech, "gen-ex-carry-crash", signal="    block: funding\n    window_days: 7",
                 portfolio="    block: time_series_sign", rebalance=7, min_history=182,
                 rows={"signal.window_days": "7", "rebalance_days": "7",
                       "min_history_days": "182"})
    problems = _blocks_problems(_check(tmp_path, mechanisms, text, "gen-ex-carry-crash"))
    for path in ("signal.window_days", "rebalance_days", "min_history_days"):
        refused = f"the catalogue gives no value for {path} in {mech}"
        assert any(refused in p for p in problems), path


def test_the_rebound_mechanism_lends_its_numbers_only_to_the_rows_that_give_them(
        mechanisms, tmp_path) -> None:
    mech = "perp-funding-extreme-rebound"
    text = _spec(mech, "gen-ex-rebound", signal="    block: funding\n    window_days: 180",
                 portfolio="    block: time_series_sign", rebalance=1, min_history=180,
                 rows={"signal.window_days": "180", "rebalance_days": "1",
                       "min_history_days": "180"})
    problems = _blocks_problems(_check(tmp_path, mechanisms, text, "gen-ex-rebound"))
    # rebalance_days = 1 and min_history_days = 180 are values of their rows; the 180 under
    # signal.window_days is not (it belongs to the percentile window of a block to come).
    assert len(problems) == 1
    assert "signal.window_days" in problems[0] and "no value" in problems[0]


@pytest.mark.parametrize(("fraction", "accepted"), [(0.3, True), (0.2, True), (0.35, False)])
def test_classic_carry_accepts_the_published_fractions_only(
        mechanisms, tmp_path, fraction, accepted) -> None:
    mech = "classic-carry"
    name = f"gen-ex-carry-{str(fraction).replace('.', '')}"
    text = _spec(mech, name, signal="    block: funding\n    window_days: 7",
                 portfolio=f"    block: long_short_fraction\n    fraction: {fraction}",
                 rebalance=30, min_history=7,
                 rows={"signal.window_days": "7", "portfolio.fraction": str(fraction),
                       "rebalance_days": "30", "min_history_days": "7"})
    problems = _blocks_problems(_check(tmp_path, mechanisms, text, name))
    about_fraction = [p for p in problems if "portfolio.fraction" in p]
    assert (not about_fraction) is accepted
    # a month is 30 days (owner, 2026-10-26): rebalance 30 is a value of its row now, while
    # a window of 7 is still not one (the row gives 360, the 12-month average)
    assert not any("rebalance_days" in p for p in problems)
    assert any("signal.window_days" in p and "7" in p for p in problems)
