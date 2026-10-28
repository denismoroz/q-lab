"""Tests for the card importer (T32): cards become ideas and drivers, the
graveyard references are checked, and a second run changes nothing. Cards are
hand-written fixtures in tmp_path; the real `cards/` directory is read once, to
prove that what is shipped imports cleanly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

from qlab.cli import app
from qlab.registry import repo
from qlab.registry.card_importer import (
    CardImportError,
    import_cards,
    unreachable_reason,
)
from qlab.registry.models import (
    AssetClass,
    Base,
    Driver,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    StageTransition,
)

REPO_CARDS = Path(__file__).resolve().parents[3] / "cards"


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as s:
        yield s
    engine.dispose()


def card(idea_id: str, **overrides) -> dict:
    base = {
        "idea_id": idea_id,
        "title": f"Title of {idea_id}",
        "found_at": "2026-10-25",
        "plain_summary": "Plain words.",
        "sources": [{"url": "https://example.org/paper", "title": "p", "published": "2026-01"}],
        "driver": {"who_pays": "A payer.", "why": "A reason.", "kill_condition": "It ends.",
                   "observable": "A number."},
        "feasibility": {"venue_supported": True},
        "prior": {"graveyard_matches": [], "note": "n"},
    }
    base.update(overrides)
    return base


def write(directory: Path, *cards: dict) -> Path:
    directory.mkdir(exist_ok=True)
    for c in cards:
        (directory / f"{c['idea_id']}.yaml").write_text(
            yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    return directory


def seed_graveyard(session) -> None:
    repo.upsert_driver(session, id="perp-funding-premium", title="t", description="d",
                       kill_condition="k", observable="o")
    repo.upsert_idea(session, id="frab", title="FRAB", source_type=SourceType.GRAVEYARD,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.CARRY,
                     driver_id="perp-funding-premium", status=IdeaStatus.LIVE)


def test_card_becomes_candidate_idea_with_its_own_driver(session, tmp_path):
    report = import_cards(session, write(tmp_path / "cards", card("alpha")))

    idea = session.get(Idea, "alpha")
    assert idea.status == IdeaStatus.CANDIDATE
    assert idea.driver_id == "alpha"
    assert idea.source_type == SourceType.PAPER
    assert "Plain words." in idea.notes and "2026-10-25" in idea.notes
    assert session.get(Driver, "alpha").kill_condition == "It ends."
    assert report.ideas_inserted == ["alpha"] and report.drivers_inserted == ["alpha"]
    assert report.retired == []


def test_second_run_changes_nothing(session, tmp_path):
    directory = write(tmp_path / "cards", card("alpha"), card("beta"))
    import_cards(session, directory)
    again = import_cards(session, directory)

    assert again.ideas_inserted == [] and again.drivers_inserted == []
    assert sorted(again.ideas_existing) == ["alpha", "beta"]
    assert session.query(Idea).count() == 2 and session.query(Driver).count() == 2


@pytest.mark.parametrize("feasibility", [
    {"venue_supported": False, "note": "second leg is at a broker"},
    {"venue_supported": True,
     "owner_can_open": {"all_legs_reachable": False, "unreachable_legs": ["options on Deribit"]}},
])
def test_card_the_owner_cannot_trade_is_retired_with_a_ledger_row(session, tmp_path, feasibility):
    directory = write(tmp_path / "cards", card("dead", feasibility=feasibility))
    report = import_cards(session, directory)

    assert session.get(Idea, "dead").status == IdeaStatus.RETIRED
    moves = session.query(StageTransition).filter_by(idea_id="dead").all()
    assert [(m.from_status, m.to_status) for m in moves] == [
        (IdeaStatus.CANDIDATE, IdeaStatus.RETIRED)]
    assert "cards" in moves[0].reason and report.retired[0][0] == "dead"


def test_card_without_a_feasibility_verdict_stays_a_candidate():
    assert unreachable_reason({"feasibility": {"venue_supported": True}}) is None
    assert unreachable_reason({}) is None


def test_graveyard_matches_resolve_to_drivers_ideas_and_the_batch(session, tmp_path):
    seed_graveyard(session)
    directory = write(
        tmp_path / "cards",
        card("alpha", prior={"graveyard_matches": ["perp-funding-premium", "frab", "beta"],
                             "note": "n"}),
        card("beta"))
    report = import_cards(session, directory)

    assert report.matches_checked == 3
    assert "graveyard_matches: perp-funding-premium, frab, beta" in session.get(Idea, "alpha").notes


def test_a_typo_in_graveyard_matches_fails_before_any_write(session, tmp_path):
    seed_graveyard(session)
    directory = write(
        tmp_path / "cards", card("good"),
        card("typo", prior={"graveyard_matches": ["perp-fundng-premium"], "note": "n"}))

    with pytest.raises(CardImportError, match="perp-fundng-premium"):
        import_cards(session, directory)
    assert session.get(Idea, "good") is None and session.get(Driver, "good") is None


@pytest.mark.parametrize("broken, message", [
    ({"plain_summary": ""}, "plain summary"),
    ({"driver": {"who_pays": "x"}}, "driver.why"),
    ({"title": None}, "`title` is missing"),
])
def test_malformed_card_is_refused(session, tmp_path, broken, message):
    directory = write(tmp_path / "cards", card("good"), card("bad", **broken))
    with pytest.raises(CardImportError, match=message):
        import_cards(session, directory)
    assert session.get(Idea, "good") is None


def test_existing_idea_keeps_its_status_and_only_gets_a_missing_driver(session, tmp_path):
    seed_graveyard(session)
    repo.upsert_idea(session, id="old", title="Old title", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_SPOT, profile=Profile.CARRY,
                     status=IdeaStatus.PAPER)
    directory = write(tmp_path / "cards", card("old", feasibility={"venue_supported": False}))
    report = import_cards(session, directory)

    idea = session.get(Idea, "old")
    assert idea.status == IdeaStatus.PAPER and idea.title == "Old title"
    assert idea.asset_class == AssetClass.CRYPTO_SPOT
    assert idea.driver_id == "old" and report.drivers_attached == ["old"]
    assert report.retired == []


def test_existing_idea_with_a_driver_gets_no_new_driver(session, tmp_path):
    seed_graveyard(session)
    directory = write(tmp_path / "cards", card("frab"))
    report = import_cards(session, directory)

    assert session.get(Idea, "frab").driver_id == "perp-funding-premium"
    assert session.get(Driver, "frab") is None
    assert report.drivers_inserted == [] and report.drivers_attached == []


def test_drafts_directory_is_not_read(session, tmp_path):
    directory = write(tmp_path / "cards", card("alpha"))
    write(directory / "drafts", card("draft-only"))
    import_cards(session, directory)
    assert session.get(Idea, "draft-only") is None


def test_missing_directory_is_an_error(session, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_cards(session, tmp_path / "nope")


def test_shipped_cards_import_and_every_match_resolves(session, tmp_path):
    """The hand-written cards in the repository: each loads, the dead ones are retired, and
    their graveyard references resolve against the real graveyard seed. A card the carder
    agent wrote (it has a `carder` block) names ideas the registry buried later than the seed
    (2026-10-28: the first one cites retired bv2 research ideas); the carder checks those
    against the live registry when it writes the card, so they are left out here."""
    from qlab.registry.importer import import_graveyard

    hand = tmp_path / "cards"
    hand.mkdir()
    for card in REPO_CARDS.glob("*.yaml"):
        if "carder" not in (yaml.safe_load(card.read_text(encoding="utf-8")) or {}):
            (hand / card.name).write_bytes(card.read_bytes())
    import_graveyard(session, REPO_CARDS.parent / "seed" / "graveyard.yaml")
    report = import_cards(session, hand)

    expected = {p.stem for p in hand.glob("*.yaml")}
    assert set(report.ideas_inserted) | set(report.ideas_existing) == expected
    assert {i for i, _ in report.retired} >= {
        "hip3-offhours-basis", "venue-liquidity-subsidy", "vrp-crypto-delta-hedged"}
    assert session.get(Idea, "liq-cascade-fade-hl").status == IdeaStatus.CANDIDATE


@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    """A throwaway registry file; the CLI runs in-process, so the module-level
    engine cache is reset around the test."""
    from qlab.registry import db as db_module

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab_cards_test.db"))
    db_module._engine = None
    db_module._SessionLocal = None
    yield
    db_module._engine = None
    db_module._SessionLocal = None


def test_cli_import_cards_on_a_temp_database(cli_db, tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["init-db"]).exit_code == 0
    directory = write(tmp_path / "cards", card("alpha"),
                      card("dead", feasibility={"venue_supported": False}))

    first = runner.invoke(app, ["import-cards", "--dir", str(directory)])
    second = runner.invoke(app, ["import-cards", "--dir", str(directory)])

    assert first.exit_code == 0, first.output
    assert "ideas inserted" in first.output and "retired dead" in first.output
    assert second.exit_code == 0, second.output
    assert "ideas already there         2" in second.output


def test_cli_reports_a_bad_card_and_exits_nonzero(cli_db, tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init-db"])
    directory = write(tmp_path / "cards", card("bad", plain_summary=""))
    result = runner.invoke(app, ["import-cards", "--dir", str(directory)])
    assert result.exit_code == 1
