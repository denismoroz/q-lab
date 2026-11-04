"""The inbox page and the one place for an idea (owner, 2026-11-04): a fixture inbox, cards and
registry give the stage of each idea; a registered idea is a link, not a second copy of its card;
`candidate` is read as «ещё не прогонялась»; calibration noise is no idea in any count."""

from __future__ import annotations

import pytest
import yaml

from qlab import test_bench_label as tb
from qlab.api import test_api
from qlab.api.inbox import STAGE_RU, STAGES, card_page, idea_path, inbox_view
from qlab.registry.models import IdeaStatus
from qlab.status_words import NOT_RUN_RU, status_ru

client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory


def _entry(n, status, *, card=None, attempts=0, notes="", **extra) -> dict:
    return {"id": f"idea-{n:04d}", "date": f"2026-11-0{n}", "source": "owner",
            "text": f"words of idea {n}", "status": status, "card": card, "notes": notes,
            "attempts": attempts, **extra}


def _card(idea_id: str) -> dict:
    return {"idea_id": idea_id, "title": f"Title {idea_id}", "found_at": "2026-11-01",
            "plain_summary": "plain words", "sources": []}


ENTRIES = [
    _entry(1, "new"),
    _entry(2, "new", attempts=2, problems=["no baseline"],
           review_blocks=[{"kind": "narrowed", "ref": "grid", "text": "drops a reading"}]),
    _entry(3, "needs-owner", attempts=1,
           notes="вопрос владельцу: Нужна ли сверка по каждой сделке?"),
    _entry(4, "duplicate", notes="повтор: bv2-sig-momentum-sign. Это то же правило."),
    _entry(5, "carded", card="cards/waits-for-code.yaml", notes="карточка написана"),
    _entry(6, "implemented", card="cards/already-registered.yaml"),
]


@pytest.fixture()
def inbox_files(tmp_path, monkeypatch):
    """The fixture inbox and cards: one card waits for code, one is registered, one has no inbox
    entry at all."""
    inbox = tmp_path / "inbox.yaml"
    inbox.write_text(yaml.safe_dump({"ideas": ENTRIES}, allow_unicode=True), encoding="utf-8")
    cards = tmp_path / "cards"
    cards.mkdir()
    for idea_id in ("waits-for-code", "already-registered", "hand-written"):
        (cards / f"{idea_id}.yaml").write_text(
            yaml.safe_dump(_card(idea_id), allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("QLAB_INBOX", str(inbox))
    monkeypatch.setenv("QLAB_CARDS", str(cards))


@pytest.fixture()
def registered(session_factory, client):
    s = session_factory()
    tb._idea(s, "already-registered", IdeaStatus.CANDIDATE)
    tb._idea(s, "noise-book-1", IdeaStatus.CANDIDATE)
    s.commit()


def _by_ref(client) -> dict:
    items = client.get("/api/inbox").json()["items"]
    return {i["id"] or i["idea_id"]: i for i in items}


def test_every_inbox_stage_is_read_from_the_entry(client, inbox_files, registered) -> None:
    items = _by_ref(client)

    assert items["idea-0001"]["stage"] == "waiting"
    assert items["idea-0001"]["stage_ru"] == "ждёт карточника"
    assert items["idea-0002"]["stage"] == "writing"
    assert "попытка 2" in items["idea-0002"]["stage_ru"]
    assert items["idea-0002"]["problems"] == ["no baseline", "narrowed: grid: drops a reading"]
    assert items["idea-0003"]["stage"] == "owner"
    assert items["idea-0003"]["question"] == "Нужна ли сверка по каждой сделке?"
    assert items["idea-0004"]["stage"] == "duplicate"
    assert items["idea-0004"]["repeats"] == "bv2-sig-momentum-sign"
    assert items["idea-0004"]["reason"] == "Это то же правило."
    assert items["idea-0005"]["stage"] == "card_ready"
    assert items["idea-0005"]["stage_ru"] == "карточка готова, ждёт реализации"
    assert items["idea-0005"]["page"] == "/inbox/waits-for-code"
    assert items["idea-0005"]["card_title"] == "Title waits-for-code"


def test_a_registered_idea_is_a_link_to_its_page_not_a_second_card(
        client, inbox_files, registered) -> None:
    items = _by_ref(client)

    assert items["idea-0006"]["stage"] == "registered"
    assert items["idea-0006"]["page"] == "/ideas/already-registered"
    # no card of a registered idea comes back a second time without an inbox entry
    assert "already-registered" not in items


def test_a_card_without_an_inbox_entry_waits_for_code_until_registered(
        client, inbox_files, registered) -> None:
    item = _by_ref(client)["hand-written"]

    assert item["id"] is None
    assert item["stage"] == "card_ready"
    assert item["page"] == "/inbox/hand-written"


def test_the_counts_cover_every_stage_and_agree_with_the_items(
        client, inbox_files, registered) -> None:
    body = client.get("/api/inbox").json()

    assert list(body["counts"]) == list(STAGES)
    assert body["counts"] == {"waiting": 1, "writing": 1, "owner": 1, "duplicate": 1,
                              "card_ready": 2, "registered": 1, "closed": 0}
    assert sum(body["counts"].values()) == len(body["items"])
    assert body["stage_ru"] == STAGE_RU


def test_an_owner_closed_idea_is_closed_even_with_a_card() -> None:
    view = inbox_view([_entry(1, "rejected", card="cards/x.yaml", notes="закрыто владельцем: нет")],
                      [_card("x")], set())

    assert view["items"][0]["stage"] == "closed"
    assert view["items"][0]["reason"] == "закрыто владельцем: нет"


def test_a_missing_inbox_file_is_an_empty_page(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QLAB_INBOX", str(tmp_path / "nothing.yaml"))
    monkeypatch.setenv("QLAB_CARDS", str(tmp_path / "no-cards"))

    body = client.get("/api/inbox").json()

    assert body["items"] == []
    assert sum(body["counts"].values()) == 0


# --------------------------------------------------------------------------
# The redirect of an old /cards/<id> link


def test_card_link_points_to_the_idea_page_once_registered(client, inbox_files, registered) -> None:
    done = client.get("/api/cards/already-registered").json()
    waiting = client.get("/api/cards/waits-for-code").json()

    assert done["registered"] is True and done["page"] == "/ideas/already-registered"
    assert waiting["registered"] is False and waiting["page"] == "/inbox/waits-for-code"
    assert card_page("a", True) == "/ideas/a"
    assert card_page("a", False) == "/inbox/a"


def test_the_idea_page_carries_its_card(client, inbox_files, registered) -> None:
    body = client.get("/api/ideas/already-registered").json()

    assert body["card"]["title"] == "Title already-registered"
    assert body["card"]["plain_summary"] == "plain words"
    assert client.get("/api/ideas/alpha").json()["card"] is None


# --------------------------------------------------------------------------
# One word per stage, noise out of the counts, a coherent funnel


def test_candidate_is_read_as_not_run_yet() -> None:
    assert NOT_RUN_RU == "ещё не прогонялась"
    assert status_ru("candidate") == "ещё не прогонялась"
    assert status_ru("bench") == "bench"


def test_noise_books_are_no_ideas_in_any_count(client, inbox_files, registered) -> None:
    funnel = client.get("/api/funnel").json()
    ideas = client.get("/api/ideas").json()

    # the registry holds two candidates (already-registered and a noise book); only one is an idea
    assert funnel["ideas_by_status"].get("candidate") == 1
    assert funnel["calibration_ideas"] == 1
    assert sum(f["count"] for f in ideas["family_counts"]) == sum(
        1 for i in ideas["items"] if not i["id"].startswith("noise-"))


def test_the_funnel_path_is_inbox_card_registered_judged(client, inbox_files, registered) -> None:
    path = {step["key"]: step for step in client.get("/api/funnel").json()["idea_path"]}

    assert list(path) == ["inbox", "card_ready", "not_run", "run"]
    assert path["inbox"]["count"] == 4  # waiting, writing, owner, duplicate
    assert path["card_ready"]["count"] == 2
    assert path["not_run"]["label_ru"] == "ещё не прогонялась"
    assert path["not_run"]["count"] == 1
    assert path["run"]["count"] == 2  # alpha and the decayed idea of the fixture


def test_idea_path_counts_everything_but_not_run_as_run() -> None:
    path = idea_path({"waiting": 2, "card_ready": 1}, {"candidate": 3, "bench": 4, "paper": 1})

    assert [s["count"] for s in path] == [2, 1, 3, 5]
