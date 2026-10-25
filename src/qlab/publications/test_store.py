"""The `publication` table: dedupe, the new tail, re-judging, the scout's queue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import sessionmaker

from qlab.publications import store
from qlab.publications.filter import load_rules
from qlab.publications.items import Item
from qlab.registry.db import make_engine
from qlab.registry.models import Base, Publication

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
LONG = "We study how the funding rate of perpetual futures is set and who pays it. " * 4


@pytest.fixture()
def session():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as s:
        yield s


def item(pid: str, title: str, days_ago: float = 1, source: str = "arxiv",
         abstract: str = LONG) -> Item:
    return Item(id=pid, source=source, title=title, authors=["A. Author"],
                published=NOW - timedelta(days=days_ago), abstract=abstract,
                link=f"https://example.org/{pid}", categories=["q-fin.TR"])


def test_a_paper_is_stored_once_by_id_and_by_title_across_sources(session) -> None:
    first = store.add_new(session, [item("arxiv:1", "Perpetual Futures: A Study")], now=NOW)
    assert [r.id for r in first] == ["arxiv:1"]
    again = store.add_new(session, [
        item("arxiv:1", "Perpetual Futures: A Study"),            # same id
        item("doi:10.2139/ssrn.9", "Perpetual futures - a study", source="ssrn"),  # same paper
        item("arxiv:2", "Another Paper"), item("arxiv:2", "Another Paper"),     # twice in a batch
    ], now=NOW)
    assert [r.id for r in again] == ["arxiv:2"]
    assert session.query(Publication).count() == 2


def test_the_stored_row_is_left_as_it_was(session) -> None:
    store.add_new(session, [item("arxiv:1", "T", abstract="old " * 80)], now=NOW)
    store.add_new(session, [item("arxiv:1", "T", abstract="new " * 80)], now=NOW)
    assert session.get(Publication, "arxiv:1").abstract.startswith("old")


def test_the_next_read_starts_at_the_newest_paper_less_the_overlap(session) -> None:
    kwargs = dict(now=NOW, lookback_days=90, overlap_days=3)
    assert store.since_for(session, "arxiv", **kwargs) == NOW - timedelta(days=90)
    store.add_new(session, [item("arxiv:1", "A", days_ago=10), item("arxiv:2", "B", days_ago=2),
                            item("doi:1", "C", days_ago=60, source="ssrn")], now=NOW)
    assert store.since_for(session, "arxiv", **kwargs) == NOW - timedelta(days=5)
    assert store.since_for(session, "ssrn", **kwargs) == NOW - timedelta(days=63)
    # A source silent for longer than the lookback does not reach past it.
    assert store.since_for(session, "ssrn", now=NOW + timedelta(days=200), lookback_days=90,
                           overlap_days=3) == NOW + timedelta(days=110)


def test_the_rule_marks_and_a_new_rule_version_judges_again_but_not_handed_papers(session) -> None:
    rules = load_rules()
    store.add_new(session, [item("arxiv:1", "Funding rates of perpetual futures"),
                            item("arxiv:2", "Quantum walks", abstract="Physics. " * 40),
                            item("arxiv:3", "Perpetual futures basis")], now=NOW)
    marked = store.apply_filter(session, rules)
    assert sorted(r.id for r in marked) == ["arxiv:1", "arxiv:3"]
    assert session.get(Publication, "arxiv:1").matched["strong"]
    assert store.apply_filter(session, rules) == []  # judged under this version already

    session.get(Publication, "arxiv:3").handed_at = NOW
    session.get(Publication, "arxiv:3").filter_version = "old"
    session.get(Publication, "arxiv:1").filter_version = "old"
    session.get(Publication, "arxiv:1").candidate = False
    again = store.apply_filter(session, rules)
    assert [r.id for r in again] == ["arxiv:1"]  # the handed paper keeps its old judgement
    assert session.get(Publication, "arxiv:3").filter_version == "old"


def test_the_scouts_queue_is_capped_best_match_first_and_needs_an_abstract(session) -> None:
    rules = load_rules()
    store.add_new(session, [
        item("arxiv:1", "Funding rates of perpetual futures", days_ago=5),
        item("arxiv:2", "Perpetual swaps: funding rate and basis trade", days_ago=9,
             abstract=LONG + " arbitrage"),
        item("arxiv:3", "Perpetual futures", days_ago=1),
        item("arxiv:4", "Perpetual futures funding", days_ago=1, abstract="too short"),
    ], now=NOW)
    store.apply_filter(session, rules)
    picked = store.pick_for_scout(session, cap=2, min_abstract_chars=200)
    # arxiv:2 has the most terms; arxiv:3 and arxiv:1 tie, the newer goes first.
    assert [r.id for r in picked] == ["arxiv:2", "arxiv:3"]
    everyone = store.pick_for_scout(session, cap=10, min_abstract_chars=200)
    assert [r.id for r in everyone] == ["arxiv:2", "arxiv:3", "arxiv:1"]  # no arxiv:4: no abstract
    assert store.pick_for_scout(session, cap=0, min_abstract_chars=200) == []

    store.mark_handed(session, picked, at=NOW, notes={"arxiv:2": "NO CANDIDATE: x"}, drafts={})
    left = store.pick_for_scout(session, cap=10, min_abstract_chars=200)
    assert not {"arxiv:2"} & {r.id for r in left}
    assert session.get(Publication, "arxiv:2").scout_note == "NO CANDIDATE: x"
