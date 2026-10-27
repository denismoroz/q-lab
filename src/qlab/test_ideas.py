"""The ideas inbox (qlab.ideas, docs/IDEAS.md): an idea is kept in the words it came in, the same
words are not added twice, the status moves only by code and the owner's commands, and an agent's
```ideas block is read the way its memo wrote it. The conftest points the inbox at an empty
temporary place, so nothing here touches `ideas/inbox.yaml`."""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from qlab import ideas
from qlab.cli import app

DAY = date(2026, 10, 27)
runner = CliRunner()

TRICKY = ("Bv3: страховка по объёму —\n  «когда торгов много, а цена стоит»: 'ёлочки', "
          'кавычки "двойные", двоеточие: и # решётка.  Два пробела подряд.\n')


def test_an_idea_is_stored_word_for_word_and_read_back_the_same() -> None:
    inbox: list[dict] = []
    entry, added = ideas.add(inbox, TRICKY, source="owner", day=DAY)
    assert added and entry["id"] == "idea-0001" and entry["status"] == "new"
    ideas.save(inbox)
    back = ideas.load()
    assert back[0]["text"] == TRICKY  # not trimmed, not re-wrapped, not paraphrased
    assert back[0]["date"] == "2026-10-27" and back[0]["card"] is None


def test_the_same_words_are_not_added_twice_but_other_words_are() -> None:
    inbox: list[dict] = []
    first, _ = ideas.add(inbox, "Страховка по объёму торгов", source="owner", day=DAY)
    again, added = ideas.add(inbox, "  страховка   по ОБЪЁМУ торгов ", source="agent:director",
                             day=DAY)
    assert not added and again is first and len(inbox) == 1
    other, added = ideas.add(inbox, "Страховка по ставке финансирования", source="owner", day=DAY)
    assert added and other["id"] == "idea-0002"


def test_a_source_is_the_owner_or_an_agent_and_an_idea_needs_words() -> None:
    inbox: list[dict] = []
    with pytest.raises(ValueError, match="source must be"):
        ideas.add(inbox, "some idea here", source="somebody", day=DAY)
    with pytest.raises(ValueError, match="needs words"):
        ideas.add(inbox, "   ", source="owner", day=DAY)
    assert ideas.add(inbox, "an idea of a researcher", source="agent:researcher", day=DAY)[1]


def test_the_carder_takes_the_oldest_new_idea_that_has_attempts_left() -> None:
    inbox: list[dict] = []
    for text in ("the first idea of the day", "the second idea of the day",
                 "the third idea of the day"):
        ideas.add(inbox, text, source="owner", day=DAY)
    ideas.set_status(inbox[0], "carded")
    inbox[1]["attempts"] = 2
    assert ideas.oldest_new(inbox, max_attempts=2)["id"] == "idea-0003"
    assert ideas.oldest_new(inbox, max_attempts=3)["id"] == "idea-0002"
    ideas.set_status(inbox[2], "duplicate")
    assert ideas.oldest_new(inbox, max_attempts=2) is None
    with pytest.raises(ValueError, match="status must be"):
        ideas.set_status(inbox[0], "done")


def test_the_owner_rejects_and_retries() -> None:
    inbox: list[dict] = []
    ideas.add(inbox, "an idea the owner will close", source="owner", day=DAY)
    ideas.reject(inbox, "idea-0001", "не наш рынок")
    assert inbox[0]["status"] == "rejected" and "не наш рынок" in inbox[0]["notes"]
    inbox[0]["attempts"] = 2
    ideas.retry(inbox, "idea-0001")
    assert (inbox[0]["status"], inbox[0]["attempts"]) == ("new", 0)
    with pytest.raises(KeyError):
        ideas.reject(inbox, "idea-9999", "no such idea")


MEMO = """# Записка

Текст записки.

```ideas
- "Страховка Bv2 по объёму торгов: включать, когда оборот растёт, а цена стоит"
- {text: "Выход из страховки по числу монет корзины, которые уже растут"}
- "коротко"
- 42
```

Дальше идёт текст.
"""


def test_the_ideas_block_of_a_memo_is_read_and_what_does_not_fit_is_dropped(tmp_path) -> None:
    body = ideas.extract_block(MEMO)
    kept, dropped = ideas.parse_block(body)
    assert kept == ["Страховка Bv2 по объёму торгов: включать, когда оборот растёт, а цена стоит",
                    "Выход из страховки по числу монет корзины, которые уже растут"]
    assert len(dropped) == 2 and "коротко" in dropped[0]
    assert ideas.extract_block("no block here") is None
    assert ideas.parse_block("not: [valid")[1][0].startswith("the ideas block is not YAML")
    assert ideas.parse_block("a: b")[1] == ["the ideas block must be a list"]
    many = "\n".join(f'- "an idea number {i} with enough words"' for i in range(5))
    kept, dropped = ideas.parse_block(many)
    assert len(kept) == ideas.MAX_PER_MEMO and "over the cap" in dropped[0]


def test_a_memo_puts_its_ideas_in_the_inbox_once(tmp_path) -> None:
    memo = tmp_path / "bv2-2026-10-27.md"
    memo.write_text(MEMO, encoding="utf-8")
    first = ideas.ingest_memo(memo, source="agent:researcher", day=DAY)
    assert len(first["added"]) == 2 and first["known"] == [] and len(first["dropped"]) == 2
    inbox = ideas.load()
    assert [e["source"] for e in inbox] == ["agent:researcher"] * 2
    assert inbox[0]["text"].startswith("Страховка Bv2 по объёму")
    assert "bv2-2026" in inbox[0]["notes"]
    again = ideas.ingest_memo(memo, source="agent:researcher", day=DAY)
    assert again["added"] == [] and len(again["known"]) == 2 and len(ideas.load()) == 2


def test_a_memo_without_the_block_or_without_a_file_adds_nothing(tmp_path) -> None:
    memo = tmp_path / "memo.md"
    memo.write_text("# no ideas block\n", encoding="utf-8")
    assert ideas.ingest_memo(memo, source="agent:director", day=DAY) == {
        "added": [], "known": [], "dropped": []}
    assert ideas.ingest_memo(tmp_path / "missing.md", source="agent:director", day=DAY)[
        "added"] == []
    assert ideas.load() == []


def test_the_agents_are_asked_for_an_ideas_block_and_nothing_else_changed() -> None:
    from qlab.agents.director import INSTRUCTIONS as DIRECTOR
    from qlab.agents.researcher import INSTRUCTIONS as RESEARCHER

    for text in (RESEARCHER, DIRECTOR):
        assert "```ideas" in text and "needs NEW code" in text
    # the existing machine-read blocks stay where they were
    assert "```directions" in DIRECTOR and "DROP: <idea_id>" in RESEARCHER
    assert ideas.extract_block(RESEARCHER) is not None  # the example in the prompt is a block


def test_the_cli_adds_lists_rejects_and_retries(tmp_path) -> None:
    out = runner.invoke(app, ["idea", "add", "Страховка по объёму торгов, а не по цене"])
    assert out.exit_code == 0 and "idea-0001: added" in out.output
    out = runner.invoke(app, ["idea", "add", "страховка по объёму торгов, а не по цене"])
    assert "already in the inbox (new)" in out.output
    out = runner.invoke(app, ["idea", "add", "an agent's idea in the inbox", "--source",
                              "agent:researcher"])
    assert out.exit_code == 0
    assert runner.invoke(app, ["idea", "add", "a stranger's idea here", "--source", "x"]
                         ).exit_code == 1
    listing = runner.invoke(app, ["idea", "list"]).output
    assert "idea-0001" in listing and "owner" in listing and "2 ideas" in listing
    assert runner.invoke(app, ["idea", "reject", "idea-0002", "--why", "не нужно"]).exit_code == 0
    only = runner.invoke(app, ["idea", "list", "--status", "rejected"]).output
    assert "idea-0002" in only and "idea-0001" not in only
    assert runner.invoke(app, ["idea", "list", "--status", "bogus"]).exit_code == 1
    assert runner.invoke(app, ["idea", "retry", "idea-0002"]).exit_code == 0
    assert ideas.find(ideas.load(), "idea-0002")["status"] == "new"
    assert runner.invoke(app, ["idea", "reject", "idea-0077", "--why", "x"]).exit_code == 1
