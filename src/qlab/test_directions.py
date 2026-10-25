"""Tests for `qlab.directions`: the research director's block is validated by
code, the file waits for the owner's review, and the others read it safely."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from qlab import directions as dirs

DAY = date(2026, 10, 26)
MECHANISMS = {"crypto-xs-momentum", "crypto-low-vol", "perp-funding-carry"}
STRATEGIES = {"bv2", "trend", "frab", "xsmom"}
FAMILIES = {"momentum", "low-volatility", "funding-and-basis"}

GOOD = """
generator:
  prefer_mechanisms: [crypto-xs-momentum]
  pause_mechanisms:
    - {id: crypto-low-vol, why: "every idea of it fails the same rule"}
researcher:
  bv2: {priority: high, note: "look at when the hedge switches, not at its level"}
  trend: {priority: low}
catalogue_requests:
  - {family: momentum, look_for: "papers on volume-weighted signals", why: "a gap in the blocks"}
data_requests:
  - {what: "a second venue's funding history", why: "FRAB has one venue only"}
propose_close_family:
  - {family: low-volatility, why: "the one mechanism is a published negative result"}
"""


def _parse(text: str, **kw):
    kw.setdefault("day", DAY)
    return dirs.parse_block(text, mechanism_ids=MECHANISMS, strategy_names=STRATEGIES,
                            family_names=FAMILIES, **kw)


# ---- the block ----------------------------------------------------------------------


def test_a_good_block_is_accepted_whole() -> None:
    result = _parse(GOOD, memo="docs/research/director-2026-10-26.md")
    d = result.directions
    assert result.problems == [] and d is not None
    assert d.prefer_mechanisms == ("crypto-xs-momentum",)
    assert d.paused() == {"crypto-low-vol": "every idea of it fails the same rule"}
    assert d.researcher["bv2"] == dirs.StrategyDirection(
        "high", "look at when the hedge switches, not at its level")
    assert d.researcher["trend"] == dirs.StrategyDirection("low", None)
    assert d.catalogue_requests[0].look_for == "papers on volume-weighted signals"
    assert d.data_requests[0].what == "a second venue's funding history"
    assert d.propose_close_family[0].family == "low-volatility"


def test_unknown_mechanism_ids_are_rejected_not_passed_on() -> None:
    block = """
generator:
  prefer_mechanisms: [crypto-xs-momentum, made-up-mechanism]
  pause_mechanisms:
    - {id: another-made-up, why: "x"}
    - {id: crypto-low-vol}
"""
    result = _parse(block)
    d = result.directions
    assert d.prefer_mechanisms == ("crypto-xs-momentum",)
    assert d.pause_mechanisms == ()  # an unknown id, and a pause without a reason
    assert any("made-up-mechanism" in p for p in result.problems)
    assert any("another-made-up" in p for p in result.problems)
    assert any("crypto-low-vol" in p and "no reason" in p for p in result.problems)


def test_a_number_in_a_researcher_note_is_flagged_and_the_note_dropped() -> None:
    block = """
researcher:
  bv2: {priority: high, note: "try a 4h timeframe"}
  trend: {priority: low, note: "fewer coins, not more"}
  xsmom: {priority: urgent}
  nonexistent: {priority: high}
"""
    result = _parse(block)
    r = result.directions.researcher
    assert r["bv2"] == dirs.StrategyDirection("high", None)  # the priority stays, the note goes
    assert r["trend"].note == "fewer coins, not more"
    assert r["xsmom"].priority == "normal"
    assert "nonexistent" not in r
    assert any("bv2" in p and "digit" in p for p in result.problems)
    assert any("urgent" in p for p in result.problems)
    assert any("nonexistent" in p for p in result.problems)


def test_a_paused_mechanism_cannot_also_be_preferred() -> None:
    block = """
generator:
  prefer_mechanisms: [crypto-low-vol]
  pause_mechanisms:
    - {id: crypto-low-vol, why: "negative result"}
"""
    result = _parse(block)
    assert result.directions.prefer_mechanisms == ()
    assert "crypto-low-vol" in result.directions.paused()
    assert any("also paused" in p for p in result.problems)


def test_closing_proposals_name_a_known_family_or_strategy_and_carry_a_reason() -> None:
    block = """
propose_close_family:
  - {family: momentum, why: "keeps failing the same rule"}
  - {family: bv2, why: "a strategy may be named too"}
  - {family: astrology, why: "x"}
  - {family: reversal}
"""
    result = _parse(block)
    assert [c.family for c in result.directions.propose_close_family] == ["momentum", "bv2"]
    assert len(result.problems) == 2


def test_a_broken_block_gives_no_directions_and_leaves_the_file_alone() -> None:
    assert _parse("generator: [unclosed").directions is None
    assert _parse("- just\n- a list").directions is None
    odd = _parse("surprise: 1\ngenerator: {prefer_mechanisms: crypto-xs-momentum}")
    assert odd.directions is not None
    assert any("surprise" in p for p in odd.problems)
    assert any("prefer_mechanisms" in p for p in odd.problems)


def test_the_block_is_found_in_the_memo_by_its_fence() -> None:
    memo = "# Записка\n\nТекст.\n\n```directions\ngenerator: {}\n```\n\nКонец.\n"
    assert dirs.extract_block(memo).strip() == "generator: {}"
    assert dirs.extract_block("```yaml\ngenerator: {}\n```") is None
    assert dirs.extract_block("no block") is None


# ---- the file and the owner's review --------------------------------------------------


def _write(root: Path, **kw) -> dirs.Directions:
    result = dirs.parse_block(GOOD, mechanism_ids=MECHANISMS, strategy_names=STRATEGIES,
                              family_names=FAMILIES, day=kw.pop("day", DAY),
                              pending_owner_review=kw.pop("pending", False), **kw)
    dirs.write_directions(result.directions, root)
    return result.directions


def test_the_first_directions_ever_wait_for_the_owner(tmp_path) -> None:
    assert dirs.read_directions(tmp_path).status == "missing"
    assert dirs.pending_for_new(tmp_path) is True  # nothing before: the first run
    _write(tmp_path, pending=True)
    found = dirs.read_directions(tmp_path)
    assert found.status == "pending" and found.directions.pending_owner_review
    # While the flag is true nobody obeys the file.
    assert dirs.active_directions(tmp_path) is None
    # A newer run replacing an unreviewed file stays unreviewed.
    assert dirs.pending_for_new(tmp_path) is True
    # The owner clears the flag: now it applies, and the next run needs no review.
    assert dirs.approve(tmp_path).pending_owner_review is False
    assert dirs.read_directions(tmp_path).status == "active"
    assert dirs.active_directions(tmp_path) is not None
    assert dirs.pending_for_new(tmp_path) is False


def test_history_keeps_every_run_by_date(tmp_path) -> None:
    _write(tmp_path, day=date(2026, 10, 26), pending=True)
    _write(tmp_path, day=date(2026, 10, 28))
    names = sorted(p.name for p in (tmp_path / dirs.HISTORY_DIR).glob("*.yaml"))
    assert names == ["2026-10-26.yaml", "2026-10-28.yaml"]
    assert dirs.has_previous(tmp_path)
    # History alone (the file deleted by hand) still counts as a previous run.
    (tmp_path / dirs.DIRECTIONS).unlink()
    assert dirs.has_previous(tmp_path)


def test_a_missing_or_damaged_file_is_todays_behaviour(tmp_path) -> None:
    assert dirs.active_directions(tmp_path) is None
    (tmp_path / "night").mkdir()
    (tmp_path / dirs.DIRECTIONS).write_text("date: [not a date", encoding="utf-8")
    found = dirs.read_directions(tmp_path)
    assert found.status == "corrupt" and found.error
    assert dirs.active_directions(tmp_path) is None
    assert dirs.approve(tmp_path) is None


def test_the_file_round_trips_and_carries_its_date_and_memo(tmp_path) -> None:
    written = _write(tmp_path, memo="docs/research/director-2026-10-26.md")
    raw = yaml.safe_load((tmp_path / dirs.DIRECTIONS).read_text(encoding="utf-8"))
    assert raw["date"] == "2026-10-26" or raw["date"] == DAY
    assert raw["memo"] == "docs/research/director-2026-10-26.md"
    assert dirs.read_directions(tmp_path).directions == written


# ---- how the others read them ----------------------------------------------------------


def test_strategies_are_ordered_high_first_and_unfinished_first_inside_a_priority() -> None:
    d = _parse(GOOD).directions  # bv2 high, trend low
    names = ["bv2", "frab", "trend", "xsmom"]
    assert dirs.order_strategies(names, d) == ["bv2", "frab", "xsmom", "trend"]
    # A strategy a budget stop left unfinished goes first, but only within its priority.
    assert dirs.order_strategies(names, d, unfinished=["xsmom", "trend"]) == [
        "bv2", "xsmom", "frab", "trend"]
    # No directions: today's order (unfinished first, then by name).
    assert dirs.order_strategies(names, None, unfinished=["xsmom"]) == [
        "xsmom", "bv2", "frab", "trend"]


def test_researcher_focus_carries_priority_and_note() -> None:
    d = _parse(GOOD).directions
    focus = dirs.researcher_focus("bv2", d)
    assert "priority high" in focus and "look at when the hedge switches" in focus
    assert "priority low" in dirs.researcher_focus("trend", d)
    assert dirs.researcher_focus("frab", d) is None  # normal and no note: nothing to add
    assert dirs.researcher_focus("bv2", None) is None


def test_generator_text_names_preferred_and_paused_mechanisms() -> None:
    text = dirs.generator_text(_parse(GOOD).directions)
    assert "crypto-xs-momentum" in text and "crypto-low-vol -- every idea of it" in text
    assert dirs.generator_text(None) == ""
    assert dirs.generator_text(_parse("researcher: {bv2: {priority: high}}").directions) == ""


# ---- night/director.yaml -----------------------------------------------------------------


def test_the_shipped_config_loads_and_every_family_member_is_in_the_catalogue() -> None:
    from qlab.agents.generator import catalogue

    repo = Path(__file__).resolve().parents[2]
    config = dirs.load_config(repo / "night/director.yaml")
    assert config is not None and config.enabled and config.cadence_days == 2
    ids = {m.id for m in catalogue(repo / "docs/MECHANISMS.md")}
    listed = [i for members in config.families.values() for i in members]
    assert set(listed) <= ids and len(listed) == len(set(listed))
    assert config.family_of("crypto-xs-momentum") == "momentum"
    assert config.family_of("not-listed") == "not-listed"
    text = (repo / "night/director.yaml").read_text(encoding="utf-8")
    assert "запуск руководителя раз в 2 дня - добавляй" in text  # the owner's words


def test_a_bad_cadence_is_refused(tmp_path) -> None:
    config = tmp_path / "director.yaml"
    config.write_text("cadence_days: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cadence_days"):
        dirs.load_config(config)
    assert dirs.load_config(tmp_path / "missing.yaml") is None
