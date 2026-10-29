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
    assert body["ideas_by_latest_route"] == {}
    assert body["calibration_ideas"] == 0


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



def test_idea_detail_without_routed_trials_says_so(client):
    """All three fixture trials predate stored routes: the outcome block
    must report that, not invent a decision."""
    body = client.get("/api/ideas/alpha").json()

    assert body["latest_outcome"] == {"trial": None, "unrouted_trials": 3}


def test_idea_detail_leads_with_the_latest_routed_trial(client, session_factory):
    from qlab.registry.models import Spec, TrialRoute

    session = session_factory()
    spec_id = session.query(Spec.id).filter(Spec.idea_id == "alpha").scalar()
    for route, reason in (
        (TrialRoute.REJECT, "older decision"),
        (TrialRoute.NOT_EVALUABLE, "implementation does not express: X"),
    ):
        repo.add_trial(
            session,
            spec_id=spec_id,
            config_hash="h",
            snapshot_id="snap-1",
            code_sha="abc123",
            params={},
            status=TrialStatus.NOT_EVALUABLE
            if route == TrialRoute.NOT_EVALUABLE
            else TrialStatus.OK,
            route=route,
            route_reason=reason,
        )
    session.commit()
    session.close()

    outcome = client.get("/api/ideas/alpha").json()["latest_outcome"]

    assert outcome["trial"]["route"] == "not-evaluable"
    assert outcome["trial"]["route_reason"] == "implementation does not express: X"
    assert outcome["unrouted_trials"] == 3


def test_idea_detail_carries_the_latest_five_research_notes_newest_first(client, session_factory):
    from qlab.registry.models import ResearchNote

    session = session_factory()
    for day in range(1, 8):
        session.add(ResearchNote(
            idea_id="alpha", name="alpha", at=datetime(2026, 10, day, tzinfo=UTC),
            memo_path=f"docs/research/alpha-2026-10-0{day}.md", memo=f"memo {day}",
            experiments=[{"idea_id": "alpha-x", "trial_id": day, "route": "reject", "spec": None}],
            problems=["a problem"], tokens=day * 100, model="opus"))
    session.add(ResearchNote(idea_id="omega", name="omega", at=datetime(2026, 10, 9, tzinfo=UTC),
                             memo_path="docs/research/omega-2026-10-09.md", memo="other",
                             experiments=[], problems=[]))
    session.commit()
    session.close()

    notes = client.get("/api/ideas/alpha").json()["research_notes"]

    assert [n["memo"] for n in notes] == [f"memo {d}" for d in (7, 6, 5, 4, 3)]
    first = notes[0]
    assert first["at"].startswith("2026-10-07")
    assert first["memo_path"] == "docs/research/alpha-2026-10-07.md"
    assert first["experiments"] == [
        {"idea_id": "alpha-x", "trial_id": 7, "route": "reject", "spec": None}]
    assert first["problems"] == ["a problem"] and first["tokens"] == 700
    assert [n["memo"] for n in client.get("/api/ideas/omega").json()["research_notes"]] == ["other"]


def test_idea_detail_without_research_notes_has_an_empty_list(client):
    assert client.get("/api/ideas/alpha").json()["research_notes"] == []


def test_idea_detail_names_the_strategy_of_a_variant_and_lists_a_roots_variants(
        client, session_factory):
    from qlab.registry.models import AssetClass, Idea, IdeaStatus, Profile, SourceType

    session = session_factory()
    session.add(Idea(id="alpha-variant", title="v", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     status=IdeaStatus.BENCH, parent_id="alpha"))
    session.commit()
    session.close()

    assert client.get("/api/ideas/alpha-variant").json()["strategy"] == {
        "root": "alpha", "is_variant": True, "active_variants": []}
    root = client.get("/api/ideas/alpha").json()["strategy"]
    assert root["is_variant"] is False and root["active_variants"] == ["alpha-variant"]


def test_idea_detail_puts_both_stages_of_a_generated_idea_side_by_side(client, session_factory):
    """docs/TWO_STAGE.md: stage 1 under the idea, stage 2 under its child; the parent's page
    and the child's page show the same pair, with the data each was judged on."""
    from qlab.registry.models import Idea, Spec, TrialRoute

    assert client.get("/api/ideas/alpha").json()["two_stage"] is None  # no second stage

    session = session_factory()
    session.add(Idea(id="alpha-hl", title="Alpha, stage 2", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     status=IdeaStatus.REJECTED, parent_id="alpha",
                     notes="Second stage of alpha (docs/TWO_STAGE.md): the same mechanism"))
    session.flush()
    stage1_spec = session.query(Spec).filter(Spec.idea_id == "alpha").one()
    stage2_spec = repo.add_spec(
        session, idea_id="alpha-hl", version=1, params={"lookback": 30},
        data_requirements={"source": "hyperliquid"}, rebalance="1d", costs_model={},
        code_ref="qlab.strategies.trend")
    stage1_spec.data_requirements = {"source": "binance"}
    trials = {}
    for key, spec, route, reason in (("1", stage1_spec, TrialRoute.NEEDS_FORWARD, "waits"),
                                     ("2", stage2_spec, TrialRoute.REJECT, "capital_fit")):
        trials[key] = repo.add_trial(
            session, spec_id=spec.id, config_hash=f"h{key}", snapshot_id="snap-1",
            code_sha="x", params={}, status=TrialStatus.OK, route=route, route_reason=reason,
            metrics={"ann_return_net": 0.1 if key == "1" else -0.2})
    for key, idea_id, start in (("1", "alpha", date(2019, 9, 8)), ("2", "alpha-hl",
                                                                    date(2025, 1, 30))):
        repo.add_verdicts(session, [dict(
            idea_id=idea_id, trial_id=trials[key].id, stage=VerdictStage.EDGE, rule_id="r",
            rules_version="v", metric="m", value=1.0, comparator=">=", threshold=0.0,
            passed=key == "1", data_range_start=start, data_range_end=date(2026, 10, 24),
            decided_at=datetime(2026, 10, 25, tzinfo=UTC), source=TrialSource.QLAB)])
    session.commit()
    session.close()

    parent = client.get("/api/ideas/alpha").json()["two_stage"]
    child = client.get("/api/ideas/alpha-hl").json()["two_stage"]
    assert parent["role"] == "parent" and child["role"] == "child"
    for view in (parent, child):
        assert (view["parent_id"], view["child_id"]) == ("alpha", "alpha-hl")
        assert view["stage1"]["venue"] == "binance" and view["stage2"]["venue"] == "hyperliquid"
        assert view["stage1"]["trial"]["route"] == "needs-forward"
        assert view["stage2"]["trial"]["route"] == "reject"
        assert view["stage1"]["data_range"] == {"start": "2019-09-08", "end": "2026-10-24"}
        assert view["stage2"]["data_range"] == {"start": "2025-01-30", "end": "2026-10-24"}
        assert view["parent_status"] == "validated"
    # The child is a variant of its parent: it takes no place of its own.
    assert client.get("/api/ideas/alpha-hl").json()["strategy"]["root"] == "alpha"


def test_idea_detail_shows_the_causal_twins_two_stages_on_the_originals_page(
        client, session_factory):
    """Owner, 2026-10-28: the original idea's page carries its causal twin's Binance-then-
    Hyperliquid answer; an idea with no twin, or a same-named idea that is not its child, has
    none."""
    from qlab.registry.models import Idea

    assert client.get("/api/ideas/alpha").json()["causal_twin"] is None

    session = session_factory()
    for idea_id, parent in (("alpha-causal", "alpha"), ("alpha-causal-hl", "alpha-causal"),
                            ("beta-causal", "alpha")):
        session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                         status=IdeaStatus.VALIDATED, parent_id=parent,
                         notes="Second stage of alpha-causal (docs/TWO_STAGE.md)"
                         if idea_id.endswith("-hl") else None))
    session.add(Idea(id="beta", title="beta", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     status=IdeaStatus.BENCH))
    session.commit()
    session.close()

    twin = client.get("/api/ideas/alpha").json()["causal_twin"]
    assert twin["id"] == "alpha-causal" and twin["status"] == "validated"
    assert (twin["two_stage"]["parent_id"], twin["two_stage"]["child_id"]) == (
        "alpha-causal", "alpha-causal-hl")
    assert client.get("/api/ideas/beta").json()["causal_twin"] is None


def test_idea_detail_carries_the_latest_paper_check_or_null(client, session_factory):
    from qlab.registry.models import PaperCheck

    assert client.get("/api/ideas/alpha").json()["paper_check"] is None

    session = session_factory()
    for day, paper in ((24, -0.01), (25, -0.018)):
        session.add(PaperCheck(
            idea_id="alpha", book="b2_cold", at=date(2026, 10, day),
            window_start=date(2026, 10, 3), window_end=date(2026, 10, 24), paper_return=paper,
            stand_return=-0.024, stand_trial_id=None, replay_return=None,
            stand_window_start=date(2026, 10, 3), stand_window_end=date(2026, 10, 24), note=""))
    session.commit()
    session.close()

    check = client.get("/api/ideas/alpha").json()["paper_check"]

    assert check["at"] == "2026-10-25" and check["paper_return"] == -0.018
    assert check["stand_return"] == -0.024 and check["replay_return"] is None
    assert check["window_start"] == "2026-10-03" and check["stand_window_end"] == "2026-10-24"
    assert client.get("/api/ideas/omega").json()["paper_check"] is None


def test_agent_funnel_with_no_agent_ideas_has_no_price_of_a_survivor(client, monkeypatch, tmp_path):
    from qlab import funnel

    monkeypatch.setattr(funnel, "NIGHT_DIR", tmp_path / "no-nights")

    body = client.get("/api/agent-funnel").json()

    assert [p["label"] for p in body["periods"]] == ["last-7-days", "since-agents-started"]
    for period in body["periods"]:
        # `alpha` was not created by an agent: it is not in the agents' funnel.
        assert period["created"] == 0 and period["survivors"] == 0
        assert period["tokens_per_survivor"] is None
        assert [row["source"] for row in period["by_source"]] == [
            "generator", "researcher", "implementer", "scout"]


def test_agent_funnel_counts_agent_ideas_survivors_and_the_price(
        client, session_factory, monkeypatch, tmp_path):
    from qlab import funnel
    from qlab.registry.models import TokenSpend, TrialRoute

    monkeypatch.setattr(funnel, "NIGHT_DIR", tmp_path / "no-nights")
    session = session_factory()
    idea = repo.upsert_idea(
        session, id="bv2-research-x", title="x", source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
        status=IdeaStatus.VALIDATED, notes="Declared by the researcher (bv2).")
    spec = repo.add_spec(session, idea_id=idea.id, version=1, params={}, data_requirements={},
                         rebalance="1d", costs_model={}, code_ref="a:b")
    repo.add_trial(session, spec_id=spec.id, config_hash="r", code_sha="x", params={},
                   status=TrialStatus.OK, route=TrialRoute.PAPER, snapshot_id="snap-1")
    now = datetime.now(UTC)
    session.add(TokenSpend(at=now, stage="implement", agent="researcher", tokens_in=300,
                           tokens_out=200, usd_est=0.0))
    session.commit()
    session.close()

    body = client.get("/api/agent-funnel").json()

    # Created just now: inside both periods.
    for period in body["periods"]:
        researcher = next(r for r in period["by_source"] if r["source"] == "researcher")
        assert (researcher["created"], researcher["evaluated"]) == (1, 1)
        assert researcher["routes"] == {"paper": 1}
        assert period["survivor_ids"] == ["bv2-research-x"]
        assert period["tokens_total"] == 500 and period["tokens_per_survivor"] == 500
        assert period["tokens_by_stage"] == {"implement": 500}


def test_idea_detail_carries_the_paper_rule_or_null(client, monkeypatch, tmp_path):
    import json

    import pandas as pd

    from qlab import paper_check, paper_rule

    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n  alpha:\n    book: trend\n    label: Тренд\n    paper_file: trend_equity.csv\n"
        "    strategy_id: 5\n    sum_coins: false\n    strategy_capital_usd: 3000\n"
        "    replay_block: trend\n", encoding="utf-8")
    export = tmp_path / "paper" / "2026-10-25"
    export.mkdir(parents=True)
    first = int(datetime(2026, 9, 14, tzinfo=UTC).timestamp() * 1000)
    ts = [first + h * 3_600_000 for h in range(24 * 30 + 1)]
    pd.DataFrame({"strategy_id": 5, "ts_ms": ts, "equity": 3000.0}).to_csv(
        export / "trend_equity.csv", index=False)
    nights = tmp_path / "nights"
    nights.mkdir()
    lines = ["**trend**", "    paper  1.0 -> 1.0 (-1.20%)", "    replay 1.0 -> 1.0 (-1.22%)",
             "    daily return correlation 0.9998 (n=41)"]
    (nights / "2026-10-25.json").write_text(json.dumps({"paper": lines}), encoding="utf-8")
    monkeypatch.setattr(paper_check, "BOOKS", books)
    monkeypatch.setattr(paper_rule, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper_rule, "NIGHT_DIR", nights)

    rule = client.get("/api/ideas/alpha").json()["paper_rule"]

    assert rule["whole_weeks"] == 4 and rule["min_weeks"] == 4 and rule["max_weeks"] == 6
    assert rule["replay"] == "match" and rule["ready"] is True
    assert rule["text"].startswith("Тренд: на бумаге 4 недели из минимум 4–6")
    assert "бумага не судит о доходности — судит стенд" in rule["text"]
    assert client.get("/api/ideas/omega").json()["paper_rule"] is None


def test_idea_detail_carries_the_decay_check_or_null(client, session_factory, monkeypatch,
                                                     tmp_path):
    import json

    from qlab import decay
    from qlab.registry.models import DecayCheck

    monkeypatch.setattr(decay, "NIGHT_DIR", tmp_path / "nights")
    assert client.get("/api/ideas/alpha").json()["decay"] is None

    detail = {"idea_id": "alpha", "book": "b2", "window_days": 42, "alert": None,
              "ret": {"realised": 0.04, "stand": {"p10": -0.03, "p50": -0.01, "p90": 0.09},
                      "percentile": 73.0}}
    session = session_factory()
    for day in (24, 25):
        session.add(DecayCheck(
            idea_id="alpha", book="b2", at=date(2026, 10, day), window_start=date(2026, 9, 13),
            window_end=date(2026, 10, day), realised_return=0.04, return_percentile=73.0,
            detail=dict(detail, window_days=day)))
    session.commit()
    session.close()

    found = client.get("/api/ideas/alpha").json()["decay"]

    assert found["at"] == "2026-10-25" and found["window_days"] == 25
    assert found["ret"]["percentile"] == 73.0 and found["alert"] is None
    assert client.get("/api/ideas/omega").json()["decay"] is None

    # No table row: the newest night's state still answers.
    nights = tmp_path / "nights"
    nights.mkdir()
    (nights / "2026-10-25.json").write_text(json.dumps(
        {"decay": {"lines": [], "results": [dict(detail, idea_id="omega")]}}), encoding="utf-8")
    assert client.get("/api/ideas/omega").json()["decay"]["book"] == "b2"


def test_allocation_endpoint_returns_the_latest_saved_proposal_or_null(client, monkeypatch,
                                                                       tmp_path):
    import json

    from qlab import pool_allocation

    monkeypatch.setattr(pool_allocation, "NIGHT_DIR", tmp_path)
    assert client.get("/api/allocation").json()["proposal"] is None
    state = {"allocation": {"lines": ["x"], "proposal": {"capital": 3000.0, "lines": []},
                            "snapshot": {}, "changes": []}}
    (tmp_path / "2026-10-25.json").write_text(json.dumps(state), encoding="utf-8")

    body = client.get("/api/allocation").json()

    assert body["at"] == "2026-10-25" and body["proposal"]["capital"] == 3000.0


def test_a_trial_carries_its_year_view_or_null(client, session_factory):
    """docs/YEARS.md: the metrics' per-year keys regrouped for the console, with the newest
    ruleset's informational readings; null for a run from before the view existed."""
    from qlab.registry.models import Spec, TrialRoute

    assert client.get("/api/ideas/alpha/trials").json()["items"][0]["year_view"] is None

    session = session_factory()
    spec = session.query(Spec).filter(Spec.idea_id == "alpha").one()
    metrics = {"ann_return_net": 0.2, "ann_return_net_ex_best_1pct": 0.03,
               "year_2021_return": 0.9, "year_2021_days": 365, "year_2021_max_dd": -0.2,
               "year_2021_sharpe": 1.4,
               "year_2022_return": -0.1, "year_2022_days": 365, "year_2022_max_dd": -0.3,
               "year_2023_return": 0.01, "year_2023_days": 90, "year_2023_max_dd": -0.05,
               "years_counted": 2, "years_positive": 1, "years_positive_share": 0.5,
               "ann_return_net_ex_best_year": -0.1, "best_year_share_of_profit": 1.2}
    repo.add_trial(
        session, spec_id=spec.id, config_hash="years", snapshot_id="snap-1", code_sha="x",
        params={}, status=TrialStatus.OK, route=TrialRoute.REJECT, route_reason="r",
        started_at=datetime(2026, 10, 27, tzinfo=UTC), metrics=metrics)
    session.commit()
    session.close()

    trial = client.get("/api/ideas/alpha").json()["latest_outcome"]["trial"]
    view = trial["year_view"]
    assert [y["year"] for y in view["years"]] == [2021, 2022, 2023]
    assert view["years"][0]["return"] == 0.9 and view["years"][0]["sharpe"] == 1.4
    assert view["years"][1]["sharpe"] is None
    assert [y["partial"] for y in view["years"]] == [False, False, True]
    assert [y["counted"] for y in view["years"]] == [True, True, False]
    assert view["best_year"] == 2021
    assert view["summary"]["years_positive"] == 1 and view["summary"]["years_counted"] == 2
    assert view["line"].startswith("по годам: 2021 +90%, 2022 −10%, 2023 (90 дн.) +1%")
    readings = {r["id"]: r for r in view["readings"]}
    assert readings["ex_best_year_positive"]["holds"] is False
    assert readings["ex_best_1pct_positive"]["holds"] is True
    # The four signs А Б В Г (rules 2026-10-28.2) with their state and, when not defined, why.
    assert readings["years_positive_share"]["threshold"] == 0.667
    assert readings["years_positive_share"]["holds"] is False
    signs = {s["letter"]: s for s in view["signs"]}
    assert list(signs) == ["А", "Б", "В", "Г"]
    assert signs["А"]["state"] == "fails" and signs["А"]["holds"] is False
    assert signs["Б"]["state"] == "fails" and signs["Б"]["value"] == 0.5
    assert signs["В"]["state"] == "holds"
    assert signs["Г"]["state"] == "fails" and signs["Г"]["value"] == 1.2
    assert signs["А"]["reason"] is None
    # The raw metrics are still all there, unchanged.
    assert trial["metrics"]["year_2021_return"] == 0.9
    assert client.get("/api/trials?limit=1").json()["items"][0]["year_view"]["best_year"] == 2021


def test_a_trial_carries_its_protection_readings_or_null(client, session_factory):
    """docs/PROTECTION.md: the `prot_*` metrics of a Bv2-family run as the four readings
    П1 - П4 against the lines of the newest ruleset; null for any other run."""
    from qlab.registry.models import Spec, TrialRoute

    assert client.get("/api/ideas/alpha/trials").json()["items"][0]["protection"] is None

    session = session_factory()
    spec = session.query(Spec).filter(Spec.idea_id == "alpha").one()
    metrics = {"ann_return_net": 0.2, "prot_weeks": 24.0, "prot_days": 170.0,
               "prot_total_return": 0.08, "prot_worst_fall": -0.04,
               "prot_long_worst_fall": -0.13, "prot_fall_ratio": 0.31,
               "prot_fall_capture": 0.45, "prot_rise_capture": 0.5, "prot_fall_weeks": 10.0,
               "prot_rise_weeks": 11.0}
    repo.add_trial(
        session, spec_id=spec.id, config_hash="prot", snapshot_id="snap-1", code_sha="x",
        params={}, status=TrialStatus.OK, route=TrialRoute.REJECT, route_reason="r",
        started_at=datetime(2026, 10, 29, tzinfo=UTC), metrics=metrics)
    session.commit()
    session.close()

    view = client.get("/api/ideas/alpha").json()["latest_outcome"]["trial"]["protection"]
    readings = {r["letter"]: r for r in view["readings"]}
    assert list(readings) == ["П1", "П2", "П3", "П4"]
    assert readings["П1"]["state"] == "holds" and readings["П1"]["value"] == 0.31
    assert readings["П2"]["state"] == "fails" and readings["П2"]["threshold"] == 0.3
    assert readings["П3"]["state"] == "number" and readings["П3"]["threshold"] is None
    assert readings["П4"]["state"] == "holds"
    assert view["numbers"]["prot_weeks"] == 24.0 and view["min_weeks"] == 8


def test_a_trial_leads_with_the_strategys_own_money_and_carries_the_stand_beside_it(
        client, session_factory):
    """docs/CAPITAL_BASE.md, owner 2026-10-29: the figures the console leads with (`headline`) are
    the run on the strategy's own money; the stand's share of the book's balance is beside them;
    the year view leads with the money's years and the year signs are read off them."""
    from qlab.registry.models import Spec, TrialRoute

    session = session_factory()
    spec = session.query(Spec).filter(Spec.idea_id == "alpha").one()
    metrics = {
        "ann_return_net": 0.0202, "sharpe_net": 0.55, "max_dd": -0.0448,
        "noise_return_percentile": 0.99,
        "cap_ann_return": 0.0841, "cap_sharpe": 0.53, "cap_max_dd": -0.1934,
        "cap_total_return": 0.118, "cap_reinvested": 0.0, "noise_cap_return_percentile": 0.98,
        "year_2024_return": 0.02, "year_2024_days": 366, "year_2025_return": 0.02,
        "year_2025_days": 365, "years_counted": 2, "years_positive": 2,
        "years_positive_share": 1.0, "ann_return_net_ex_best_year": 0.02,
        "best_year_share_of_profit": 0.5,
        "cap_year_2024_return": 0.09, "cap_year_2024_days": 366, "cap_year_2025_return": 0.08,
        "cap_year_2025_days": 365, "cap_years_counted": 2, "cap_years_positive": 2,
        "cap_years_positive_share": 1.0, "cap_ann_return_ex_best_year": 0.08,
        "cap_best_year_share_of_profit": 0.53}
    repo.add_trial(
        session, spec_id=spec.id, config_hash="money", snapshot_id="snap-1", code_sha="x",
        params={}, status=TrialStatus.OK, route=TrialRoute.REJECT, route_reason="r",
        started_at=datetime(2026, 10, 29, 12, tzinfo=UTC), metrics=metrics)
    session.commit()
    session.close()

    trial = client.get("/api/ideas/alpha").json()["latest_outcome"]["trial"]
    h = trial["headline"]
    assert h["on_capital"] and h["reinvested"] is False and h["differs"] and not h["base_refused"]
    assert (h["ann_return"], h["sharpe"], h["max_dd"], h["noise_percentile"]) == (
        0.0841, 0.53, -0.1934, 0.98)
    assert h["stand"]["ann_return"] == 0.0202 and h["stand"]["noise_percentile"] == 0.99
    assert "от денег стратегии" in h["basis"]
    view = trial["year_view"]
    assert view["line"].startswith("по годам (от денег стратегии")
    assert "Как считает стенд (от баланса книги)" in view["line"]
    assert view["capital"]["years"][0]["return"] == 0.09 and view["years"][0]["return"] == 0.02
    # the signs the console shows first are the money's (А: without the best year 8% >= 4%)
    assert {s["letter"]: s["state"] for s in view["capital"]["signs"]}["А"] == "holds"
    listed = client.get("/api/ideas/alpha/trials").json()["items"][0]
    assert listed["headline"]["ann_return"] == 0.0841
    assert listed["metrics"]["ann_return_net"] == 0.0202
    older = client.get("/api/ideas/alpha/trials").json()["items"][-1]
    assert older["headline"] is None or not older["headline"]["on_capital"]


def test_idea_detail_carries_the_ladder_of_seven_steps(client, session_factory, monkeypatch):
    """Owner, 2026-10-29 (docs/LADDER.md): every idea's page answers the same seven steps; the
    runs of steps 2 and 3 carry their numbers as the stage cards did."""
    import qlab.ladder as ladder_module
    from qlab.registry.models import Spec, TrialRoute

    monkeypatch.setattr(ladder_module, "load_ledgers", lambda *a, **k: ladder_module.Ledgers())
    session = session_factory()
    spec = session.query(Spec).filter(Spec.idea_id == "alpha").one()
    spec.data_requirements = {"source": "hyperliquid"}
    repo.add_trial(session, spec_id=spec.id, config_hash="ladder", snapshot_id="snap-1",
                   code_sha="x", params={}, status=TrialStatus.OK, route=TrialRoute.NEEDS_FORWARD,
                   route_reason="waits", metrics={"ann_return_net": 0.1, "forward_days": 10.0})
    session.commit()
    session.close()
    body = client.get("/api/ideas/alpha").json()["ladder"]

    assert body["idea_id"] == "alpha"
    assert [s["number"] for s in body["steps"]] == [1, 2, 3, 4, 5, 6, 7]
    for step in body["steps"]:
        assert step["outcome"] in {"passed", "failed", "undecided", "not_run"}
        assert (step["outcome"] == "not_run") == (step["reason"] is not None)
        assert step["outcome_ru"] and step["title"]
    with_run = [s for s in body["steps"] if s["number"] in (2, 3) and s["run"]]
    assert with_run and all(s["trial"]["id"] == s["run"]["trial_id"] for s in with_run)
