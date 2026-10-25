"""Tests for the researcher's memos in the registry (research_notes)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Profile, ResearchNote, SourceType
from qlab.registry.research_notes import (
    import_memos,
    parse_experiments,
    spec_index,
    upsert_research_notes,
)

MEMO = """# frab, 2026-10-25

Some analysis.

- `not-an-experiment` (прогон 1): reject — written by the agent, outside the stand section

## Эксперименты на стенде (записано кодом, не исследователем)

- `frab-research-live-vs-default-retune` (прогон 76715): needs-forward — в год +9.4%, Шарп 41.58
- `frab-research-broken`: прогон упал: ValueError: boom
- `specs/research/frab/frab-research-late.yaml`: не запущен — числа без источника

## Another section

- `after-section` (прогон 2): reject — not an experiment
"""


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for idea_id in ("frab", "bv2-a", "bv2-b"):
        repo.upsert_idea(s, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    s.commit()
    yield s
    s.close()


def test_experiments_are_parsed_only_from_the_stand_section() -> None:
    specs = {"frab-research-live-vs-default-retune":
             "specs/research/frab/frab-research-live-vs-default-retune.yaml"}
    assert parse_experiments(MEMO, specs) == [
        {"idea_id": "frab-research-live-vs-default-retune", "trial_id": 76715,
         "route": "needs-forward",
         "spec": "specs/research/frab/frab-research-live-vs-default-retune.yaml"},
        {"idea_id": "frab-research-broken", "trial_id": None, "route": "error", "spec": None},
        {"idea_id": None, "trial_id": None, "route": "not-run",
         "spec": "specs/research/frab/frab-research-late.yaml"},
    ]


def test_a_restarted_memo_has_every_stand_section_parsed() -> None:
    memo = ("## Эксперименты на стенде\n\n- `a` (прогон 1): reject — x\n\n## Restart\n\n"
            "## Эксперименты на стенде\n\n- `a` (прогон 2): needs-forward — y\n")
    assert [e["trial_id"] for e in parse_experiments(memo)] == [1, 2]


def test_spec_index_finds_specs_by_idea_id(tmp_path: Path) -> None:
    (tmp_path / "frab").mkdir()
    (tmp_path / "frab" / "frab-x.yaml").write_text("a: 1")
    assert spec_index(tmp_path) == {"frab-x": (tmp_path / "frab" / "frab-x.yaml").as_posix()}
    assert spec_index(tmp_path, extra=["elsewhere/frab-x.yaml"]) == {
        "frab-x": "elsewhere/frab-x.yaml"}
    assert spec_index(tmp_path / "missing") == {}


def test_upsert_writes_one_note_per_known_idea_and_is_idempotent(session, tmp_path) -> None:
    memo_path = tmp_path / "frab-2026-10-25.md"
    memo_path.write_text(MEMO, encoding="utf-8")
    at = datetime(2026, 10, 25, 3, tzinfo=UTC)
    for _ in range(2):
        notes = upsert_research_notes(
            session, name="frab", idea_ids=["frab", "bv2-a", "not-in-registry"],
            memo_path=memo_path, at=at, problems=["p1"], tokens=1234, model="opus", specs={})
        session.commit()
        assert [n.idea_id for n in notes] == ["frab", "bv2-a"]
    rows = session.query(ResearchNote).order_by(ResearchNote.idea_id).all()
    assert [r.idea_id for r in rows] == ["bv2-a", "frab"]
    row = rows[1]
    assert (row.name, row.memo, row.problems, row.tokens, row.model) == (
        "frab", MEMO, ["p1"], 1234, "opus")
    assert row.memo_path == memo_path.as_posix()
    assert len(row.experiments) == 3


def test_a_second_upsert_refreshes_the_memo_and_keeps_tokens(session, tmp_path) -> None:
    path = tmp_path / "frab-2026-10-25.md"
    upsert_research_notes(session, name="frab", idea_ids=["frab"], memo_path=path, memo="v1",
                          at=datetime(2026, 10, 25, 3, tzinfo=UTC), tokens=10, model="opus")
    upsert_research_notes(session, name="frab", idea_ids=["frab"], memo_path=path, memo="v2",
                          at=datetime(2026, 10, 26, tzinfo=UTC), refresh_at=False)
    session.commit()
    row = session.query(ResearchNote).one()
    assert (row.memo, row.tokens, row.model) == ("v2", 10, "opus")
    assert row.at.replace(tzinfo=UTC) == datetime(2026, 10, 25, 3, tzinfo=UTC)


def test_one_memo_cannot_have_two_notes_for_the_same_idea(session) -> None:
    now = datetime.now(UTC)
    for _ in range(2):
        session.add(ResearchNote(idea_id="frab", name="frab", at=now, memo_path="m.md", memo="x",
                                 experiments=[], problems=[]))
    with pytest.raises(IntegrityError):
        session.flush()


def test_import_maps_file_names_to_ideas_and_skips_unknown_strategies(session, tmp_path) -> None:
    (tmp_path / "frab-2026-10-25.md").write_text(MEMO, encoding="utf-8")
    (tmp_path / "bv2-2026-10-03.md").write_text("old memo", encoding="utf-8")
    (tmp_path / "mystery-2026-10-03.md").write_text("who?", encoding="utf-8")
    (tmp_path / "README.md").write_text("no date", encoding="utf-8")
    strategies = {"frab": ["frab"], "bv2": ["bv2-a", "bv2-b"]}

    for _ in range(2):
        report = import_memos(session, strategies=strategies, memo_dir=tmp_path,
                              spec_dir=tmp_path / "no-specs")
        session.commit()
    assert report.imported == ["bv2-2026-10-03.md: 2 ideas", "frab-2026-10-25.md: 1 ideas"]
    assert len(report.skipped) == 2
    assert any("mystery" in line for line in report.skipped)
    assert any("README.md" in line for line in report.skipped)
    rows = session.query(ResearchNote).order_by(ResearchNote.idea_id, ResearchNote.id).all()
    assert [(r.idea_id, r.name) for r in rows] == [("bv2-a", "bv2"), ("bv2-b", "bv2"),
                                                   ("frab", "frab")]
    frab = rows[2]
    assert frab.at.replace(tzinfo=UTC) == datetime(2026, 10, 25, tzinfo=UTC)
    assert frab.tokens is None and frab.model is None
