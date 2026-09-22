"""Tests for the read-only registry API.

The fixture database is hand-built through `registry.repo` against
in-memory SQLite and deliberately contains all three legitimate verdict
kinds (docs/REGISTRY.md): a decision, an unknown and a measurement. The
kind tagging is what the UI's whole rendering contract stands on, so it is
asserted per row rather than in aggregate.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.api import cards as cards_module
from qlab.api.app import create_app
from qlab.api.deps import get_session
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    IdeaStatus,
    Profile,
    ShutdownCause,
    SourceType,
    TrialSource,
    TrialStatus,
    VerdictStage,
)


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def populated(session_factory):
    """One live idea with a spec and three trials, one decayed idea, and
    one verdict of each legitimate kind."""
    session = session_factory()
    repo.upsert_driver(
        session,
        id="perp-funding-premium",
        title="Премия за funding",
        description="who pays and why",
        kill_condition="funding flattens",
        observable="funding rate",
    )
    repo.upsert_idea(
        session,
        id="alpha",
        title="Alpha idea",
        source_type=SourceType.PAPER,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.CARRY,
        driver_id="perp-funding-premium",
        status=IdeaStatus.VALIDATED,
        claimed_edge="claims something",
    )
    repo.upsert_idea(
        session,
        id="omega",
        title="Omega idea",
        source_type=SourceType.GRAVEYARD,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.MOMENTUM,
        status=IdeaStatus.DECAYED,
        shutdown_cause=ShutdownCause.EDGE_DECAYED,
    )

    spec = repo.add_spec(
        session,
        idea_id="alpha",
        version=1,
        params={"lookback": 30},
        data_requirements={"bars": "1d"},
        rebalance="daily",
        costs_model={"taker_bps": 4.5},
        code_ref="qlab.strategies.trend",
    )
    repo.add_data_snapshot(
        session,
        id="snap-1",
        source="hyperliquid",
        instruments={"symbols": ["BTC"]},
        range_start=date(2025, 1, 1),
        range_end=date(2026, 1, 1),
        path="data/snapshots/snap-1.parquet",
        rows=365,
    )
    for day, kept in ((3, True), (2, False), (1, True)):
        repo.add_trial(
            session,
            spec_id=spec.id,
            config_hash=f"hash-{day}",
            snapshot_id="snap-1",
            code_sha="abc123",
            params={"lookback": 30},
            status=TrialStatus.OK,
            started_at=datetime(2026, 9, day, tzinfo=UTC),
            metrics={"sharpe_net": 0.9 + day},
            kept=kept,
        )

    repo.add_verdicts(
        session,
        [
            # decision
            dict(
                idea_id="alpha",
                spec_id=spec.id,
                stage=VerdictStage.EDGE,
                rule_id="sharpe_floor",
                rules_version="2026-09-20.1",
                metric="sharpe_net",
                value=1.2,
                comparator=">=",
                threshold=0.8,
                passed=True,
                data_range_start=date(2025, 1, 1),
                data_range_end=date(2026, 1, 1),
                decided_at=datetime(2026, 9, 3, tzinfo=UTC),
                source=TrialSource.QLAB,
            ),
            # unknown: rule exists, metric never computed
            dict(
                idea_id="alpha",
                stage=VerdictStage.TAIL,
                rule_id="max_dd_floor",
                rules_version="2026-09-20.1",
                metric="max_dd",
                value=None,
                comparator=">=",
                threshold=-0.25,
                passed=None,
                data_range_start=date(2025, 1, 1),
                data_range_end=date(2026, 1, 1),
                decided_at=datetime(2026, 9, 2, tzinfo=UTC),
                source=TrialSource.QLAB,
            ),
            # measurement: a number with no criterion behind it
            dict(
                idea_id="alpha",
                stage=VerdictStage.CORRELATION,
                rule_id="unidentified",
                rules_version="frab-legacy",
                metric="corr_to_frab_proxy",
                value=0.86,
                comparator=None,
                threshold=None,
                passed=None,
                data_range_start=None,
                data_range_end=None,
                decided_at=datetime(2026, 9, 1, tzinfo=UTC),
                source=TrialSource.IMPORTED,
            ),
        ],
    )
    session.commit()
    session.close()


@pytest.fixture()
def client(session_factory, populated):
    app = create_app()

    def _override():
        session = session_factory()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Funnel
# --------------------------------------------------------------------------


def test_funnel_returns_the_four_stored_breakdowns(client):
    body = client.get("/api/funnel").json()

    assert body["ideas_by_status"] == {"validated": 1, "decayed": 1}
    assert body["verdicts_by_stage"] == {"edge": 1, "tail": 1, "correlation": 1}
    assert body["verdicts_by_outcome"] == {
        "passed": 1,
        "failed": 0,
        "unknown": 1,
        "measurement": 1,
    }
    assert body["decayed_by_shutdown_cause"] == {"edge-decayed": 1}


# --------------------------------------------------------------------------
# Registry and graveyard
# --------------------------------------------------------------------------


def test_ideas_list_carries_status_driver_profile_and_source(client):
    items = client.get("/api/ideas").json()["items"]

    by_id = {item["id"]: item for item in items}
    assert set(by_id) == {"alpha", "omega"}
    assert by_id["alpha"]["status"] == "validated"
    assert by_id["alpha"]["driver_id"] == "perp-funding-premium"
    assert by_id["alpha"]["driver_title"] == "Премия за funding"
    assert by_id["alpha"]["profile"] == "carry"
    assert by_id["alpha"]["source_type"] == "paper"
    assert by_id["omega"]["driver_id"] is None
    assert by_id["omega"]["shutdown_cause"] == "edge-decayed"


def test_idea_detail_counts_verdicts_by_the_three_kinds(client):
    body = client.get("/api/ideas/alpha").json()

    assert body["driver"]["kill_condition"] == "funding flattens"
    assert body["counts"] == {
        "specs": 1,
        "trials": 3,
        "verdicts": 3,
        "verdicts_decision": 1,
        "verdicts_unknown": 1,
        "verdicts_measurement": 1,
    }


def test_missing_idea_is_404(client):
    assert client.get("/api/ideas/nope").status_code == 404
    assert client.get("/api/ideas/nope/verdicts").status_code == 404


def test_idea_specs_are_paginated(client):
    body = client.get("/api/ideas/alpha/specs").json()

    assert body["total"] == 1
    assert body["items"][0]["code_ref"] == "qlab.strategies.trend"
    assert body["items"][0]["params"] == {"lookback": 30}


def test_idea_trials_are_newest_first_and_keep_discarded_runs(client):
    items = client.get("/api/ideas/alpha/trials").json()["items"]

    assert [t["started_at"][:10] for t in items] == ["2026-09-03", "2026-09-02", "2026-09-01"]
    # A discarded run is still in the ledger: docs/REGISTRY.md counts every
    # trial for deflation, kept or not.
    assert [t["kept"] for t in items] == [True, False, True]
    assert items[0]["spec_version"] == 1


# --------------------------------------------------------------------------
# The three verdict kinds
# --------------------------------------------------------------------------


def test_each_verdict_row_is_tagged_with_its_kind(client):
    items = client.get("/api/ideas/alpha/verdicts").json()["items"]
    by_kind = {item["kind"]: item for item in items}

    assert set(by_kind) == {"decision", "unknown", "measurement"}

    decision = by_kind["decision"]
    assert (decision["value"], decision["comparator"], decision["threshold"]) == (1.2, ">=", 0.8)
    assert decision["passed"] is True

    # An unknown must not be able to read as a pass or a fail anywhere:
    # both `value` and `passed` are null while the rule's threshold stands.
    unknown = by_kind["unknown"]
    assert unknown["value"] is None
    assert unknown["passed"] is None
    assert unknown["threshold"] == -0.25

    # A measurement must never look like a rule was applied: no comparator,
    # no threshold, no verdict.
    measurement = by_kind["measurement"]
    assert measurement["value"] == 0.86
    assert measurement["comparator"] is None
    assert measurement["threshold"] is None
    assert measurement["passed"] is None
    assert measurement["rule_id"] == "unidentified"


def test_api_never_invents_a_threshold_for_a_measurement(client):
    items = client.get("/api/ideas/alpha/verdicts").json()["items"]

    for item in items:
        if item["kind"] == "measurement":
            assert item["threshold"] is None and item["comparator"] is None
        if item["kind"] != "decision":
            assert item["passed"] is None


# --------------------------------------------------------------------------
# Trials ledger
# --------------------------------------------------------------------------


def test_trials_ledger_paginates_and_reports_total(client):
    first = client.get("/api/trials?limit=2&offset=0").json()

    assert first["total"] == 3
    assert first["limit"] == 2
    assert len(first["items"]) == 2
    assert [t["started_at"][:10] for t in first["items"]] == ["2026-09-03", "2026-09-02"]

    second = client.get("/api/trials?limit=2&offset=2").json()
    assert [t["started_at"][:10] for t in second["items"]] == ["2026-09-01"]


def test_trials_ledger_carries_routing_and_metrics(client):
    item = client.get("/api/trials?limit=1").json()["items"][0]

    assert item["idea_id"] == "alpha"
    assert item["idea_title"] == "Alpha idea"
    assert item["code_ref"] == "qlab.strategies.trend"
    assert item["snapshot_id"] == "snap-1"
    assert item["metrics"]["sharpe_net"] == pytest.approx(3.9)


def test_trials_page_size_is_capped(client):
    assert client.get("/api/trials?limit=10000").status_code == 422


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------


@pytest.fixture()
def cards_dir(tmp_path, monkeypatch):
    card = {
        "idea_id": "naked-tails",
        "title": "Кандидат без защиты",
        "found_at": date(2026, 9, 22),
        "sources": [{"url": "https://example.org/a", "title": "A", "published": None}],
        "driver": {
            "who_pays": "forced seller",
            "why": "why",
            "kill_condition": "kc",
            "observable": "obs",
            "quote": {"text": "verbatim one", "source": "https://example.org/a"},
            "quote_venue": {"text": "verbatim two", "source": "https://example.org/b"},
        },
        "signal": {"entry": None, "exit": "time", "quote": {"text": "q", "source": "u"}},
        "sizing": {"rule": None, "binding_constraint": "costs"},
        "protection": {
            "crash": None,
            "pump": None,
            "costs": None,
            "structural": False,
            "note": "ХВОСТЫ НЕ ЗАКРЫТЫ НИЧЕМ",
        },
        "execution": {"legs": 1, "atomic_required": False, "venue": "HL"},
        "death": {"how": "flow dries up", "visible_in_advance": True, "observable": None},
        "feasibility": {"data_free": True, "blockers": ["no history"]},
        "prior": {"graveyard_matches": ["short-term-overreaction"], "note": "different question"},
    }
    protected = dict(card)
    protected["idea_id"] = "hedged"
    protected["title"] = "Кандидат с защитой"
    protected["protection"] = {
        "crash": "вторая нога",
        "pump": None,
        "costs": None,
        "structural": True,
        "note": "структурная",
    }

    for item in (card, protected):
        path = tmp_path / f"{item['idea_id']}.yaml"
        path.write_text(yaml.safe_dump(item, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("QLAB_CARDS", str(tmp_path))
    return tmp_path


def test_cards_index_flags_an_empty_protection_row(client, cards_dir):
    items = client.get("/api/cards").json()["items"]
    by_id = {item["idea_id"]: item for item in items}

    assert by_id["naked-tails"]["protection_is_empty"] is True
    assert by_id["hedged"]["protection_is_empty"] is False


def test_card_detail_exposes_six_parts_and_every_quote(client, cards_dir):
    body = client.get("/api/cards/naked-tails").json()

    assert list(body["parts"]) == list(cards_module.CARD_PARTS)
    # Both quotes on the driver section survive, with their sources.
    quotes = body["parts"]["driver"]["quotes"]
    assert [q["text"] for q in quotes] == ["verbatim one", "verbatim two"]
    assert quotes[1]["source"] == "https://example.org/b"
    # A null stays null; the UI labels it, the API does not fill it in.
    assert body["parts"]["signal"]["fields"]["entry"] is None
    assert body["parts"]["sizing"]["fields"]["rule"] is None
    assert body["protection_is_empty"] is True
    assert body["prior"]["graveyard_matches"] == ["short-term-overreaction"]


def test_missing_card_is_404(client, cards_dir):
    assert client.get("/api/cards/nope").status_code == 404


# --------------------------------------------------------------------------
# Read-only by construction
# --------------------------------------------------------------------------


def test_every_route_is_a_get(client):
    app = client.app
    for route in app.routes:
        methods = getattr(route, "methods", set())
        assert methods <= {"GET", "HEAD"}, f"{route.path} exposes {methods}"
