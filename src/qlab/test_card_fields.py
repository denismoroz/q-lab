"""The optional fields of an inbox card (qlab.card_fields, cards/SCHEMA.md) and the register of
blocks the implementer writes (qlab.agent_blocks)."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from qlab import agent_blocks
from qlab.card_fields import CardFieldError, param_at, read_card_fields

INBOX = {
    "idea_id": "toy-idea", "title": "Игрушка", "found_at": "2026-10-27",
    "plain_summary": "Покупаем то,   что выросло.\nПродаём то, что упало.",
    "idea_text": "купить то, что выросло за неделю",
    "baseline_spec": "base.yaml",
    "grid": {"params": {"signal.window_days": [7, 14, 30], "top_n": [3, 5]}},
    "new_block": {
        "name": "weekly-momentum",
        "params": [{"name": "window_days", "meaning": "сколько суток считать рост"},
                   {"name": "top_n", "meaning": "сколько монет держать"}],
        "rule": "держим монеты, которые сильнее всего выросли за окно"},
    "data": {"venue": "Binance", "interval": "1d", "coins": ["BTC", "ETH"],
             "honest_from": "2022-01-01"},
}


def _card(tmp_path, **over):
    (tmp_path / "base.yaml").write_text("idea_id: base\n", encoding="utf-8")
    raw = {**INBOX, **over}
    raw = {k: v for k, v in raw.items() if v is not ...}
    path = tmp_path / "card.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_a_hand_written_card_has_none_of_the_fields(tmp_path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("idea_id: old\ntitle: Старая\nplain_summary: Просто так.\n", encoding="utf-8")
    fields = read_card_fields(path, tmp_path)
    assert not fields.inbox and fields.idea_id == "old"
    assert fields.grid is None and fields.new_block is None and fields.data is None
    assert fields.idea_text == "Просто так."  # the summary stands in for the owner's words


def test_an_inbox_card_is_read_whole(tmp_path) -> None:
    fields = read_card_fields(_card(tmp_path), tmp_path)
    assert fields.inbox and fields.idea_text == "купить то, что выросло за неделю"
    assert fields.found_at == date(2026, 10, 27) and str(fields.baseline_spec) == "base.yaml"
    assert fields.grid.count == 6 and fields.grid.paths == ["signal.window_days", "top_n"]
    assert fields.grid.first_variant() == {"signal.window_days": 7, "top_n": 3}
    block = fields.new_block
    assert block.name == "weekly-momentum" and block.module == "weekly_momentum"
    assert [p.name for p in block.params] == ["window_days", "top_n"]
    assert fields.data.venue == "binance" and fields.data.coins == ("BTC", "ETH")
    assert fields.data.honest_from == date(2022, 1, 1)


def test_named_candidates_and_the_short_form(tmp_path) -> None:
    named = read_card_fields(_card(tmp_path, grid={"candidates": {
        "fast": {"signal.window_days": 7}, "slow": {"signal.window_days": 30}}}), tmp_path)
    assert named.grid.count == 2 and named.grid.first_variant() == {"signal.window_days": 7}
    short = read_card_fields(_card(tmp_path, grid={"top_n": [3, 5, 8]}), tmp_path)
    assert short.grid.axes == {"top_n": (3, 5, 8)}


def test_block_params_may_be_a_mapping_of_meanings(tmp_path) -> None:
    fields = read_card_fields(_card(tmp_path, new_block={
        "name": "m", "params": {"window_days": "окно в сутках"},
        "rule": "держим монеты, которые выросли"}), tmp_path)
    assert fields.new_block.params[0].meaning == "окно в сутках"


@pytest.mark.parametrize("over,words", [
    ({"grid": {"params": {"a": [1]}}}, "at least two variants"),
    ({"grid": {"params": {"a": 3}}}, "non-empty lists"),
    ({"grid": {"params": {"a": [1, 2]}, "candidates": {"x": {"a": 1}}}}, "not both"),
    ({"new_block": {"name": "Bad Name", "params": [], "rule": "x"}}, "not a slug"),
    ({"new_block": {"name": "ok", "params": [{"name": "a"}], "rule": "x" * 30}},
     "needs a name and a meaning"),
    ({"data": {"venue": "binance", "interval": "1d"}}, "give the coins"),
    ({"data": {"venue": "binance", "coins": ["BTC"], "honest_from": "soon"}}, "not a date"),
    ({"baseline_spec": "missing.yaml"}, "is not a file"),
])
def test_a_wrong_field_is_refused_with_every_problem_named(tmp_path, over, words) -> None:
    with pytest.raises(CardFieldError) as err:
        read_card_fields(_card(tmp_path, **over), tmp_path)
    assert words in str(err.value)


def test_param_at_reads_a_dotted_path() -> None:
    params = {"signal": {"window_days": 7}, "top_n": 3}
    assert param_at(params, "signal.window_days") == (True, 7)
    assert param_at(params, "top_n") == (True, 3)
    assert param_at(params, "signal.nothing") == (False, None)
    assert param_at(params, "top_n.deeper") == (False, None)


# ---- the register of blocks ------------------------------------------------------------


def test_a_block_is_registered_once_and_later_cards_are_added(tmp_path) -> None:
    path = tmp_path / "blocks.yaml"
    kw = dict(name="weekly-momentum", code_ref="qlab.strategies.agent.weekly_momentum:Mom",
              rule="держим выросшие", params=[{"name": "window_days", "meaning": "окно"}],
              path=path, today=date(2026, 10, 27))
    agent_blocks.register(card="cards/a.yaml", spec="specs/agent/a.yaml", **kw)
    again = agent_blocks.register(card="cards/a.yaml", spec="specs/agent/a.yaml", **kw)
    assert len(agent_blocks.load(path)) == 1 and again.cards == ["cards/a.yaml"]
    kw["rule"] = "другое описание"  # a later card does not rewrite what the block is
    later = agent_blocks.register(card="cards/b.yaml", spec="specs/agent/b.yaml", **kw)
    assert later.rule == "держим выросшие" and later.cards == ["cards/a.yaml", "cards/b.yaml"]
    assert later.specs == ["specs/agent/a.yaml", "specs/agent/b.yaml"]
    assert later.registered == "2026-10-27"
    assert path.read_text(encoding="utf-8").startswith("# Blocks the implementer agent")


def test_the_registered_blocks_become_lines_for_the_agents_brief(tmp_path) -> None:
    path = tmp_path / "blocks.yaml"
    assert agent_blocks.block_lines(path) == []  # no file, no blocks
    agent_blocks.register(name="m", code_ref="qlab.strategies.agent.m:M", rule="rule",
                          params=[{"name": "w", "meaning": "window"}], card="c", spec="s",
                          path=path)
    assert agent_blocks.block_lines(path) == [
        "- m (code_ref qlab.strategies.agent.m:M): rule Parameters: w = window"]
