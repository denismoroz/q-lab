"""Tests for the registry layer: schema, repo functions, FK/constraint
enforcement, and db.session_scope. Runs entirely against in-memory SQLite.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.registry import repo
from qlab.registry.db import get_database_url, get_db_path, make_engine, session_scope
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Driver,
    Idea,
    IdeaStatus,
    Profile,
    RevivalOutcome,
    SourceType,
    Spec,
    StageTransition,
    TokenSpend,
    Trial,
    TrialSource,
    TrialStatus,
    Verdict,
    VerdictStage,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """A fresh in-memory SQLite DB per test, with FK enforcement on (as
    qlab.registry.db does for the real DB), sharing one connection via
    StaticPool so the schema and data are visible across sessions."""
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
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()


def _idea_kwargs(**overrides):
    kwargs = dict(
        id="perp-funding-carry",
        title="Perp funding carry",
        source_type=SourceType.PAPER,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.CARRY,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_schema_creates_all_tables(engine):
    expected = {
        "driver",
        "idea",
        "spec",
        "data_snapshot",
        "trial",
        "verdict",
        "revival_check",
        "stage_transition",
        "token_spend",
        "dossier",
    }
    tables = set(inspect(engine).get_table_names())
    assert expected <= tables


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def test_upsert_driver_roundtrip_and_update(session):
    repo.upsert_driver(
        session,
        id="perp-funding-premium",
        title="Perp funding premium",
        description="longs pay shorts when perp trades above spot",
        kill_condition="funding compresses structurally, e.g. venue fee change",
        observable="8h funding rate, from exchange API",
    )
    session.commit()

    fetched = session.get(Driver, "perp-funding-premium")
    assert fetched is not None
    assert fetched.title == "Perp funding premium"

    # upsert again with new title -> update, not duplicate row
    repo.upsert_driver(
        session,
        id="perp-funding-premium",
        title="Perp funding premium (updated)",
        description="longs pay shorts when perp trades above spot",
        kill_condition="funding compresses structurally",
        observable="8h funding rate",
    )
    session.commit()

    assert session.query(Driver).count() == 1
    assert session.get(Driver, "perp-funding-premium").title == "Perp funding premium (updated)"


# --------------------------------------------------------------------------
# idea
# --------------------------------------------------------------------------


def test_upsert_idea_roundtrip_and_update(session):
    idea = repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    assert idea.status == IdeaStatus.CANDIDATE
    assert idea.created_at is not None
    assert idea.updated_at is not None

    created_at_before = idea.created_at
    repo.upsert_idea(session, **_idea_kwargs(title="Perp funding carry v2"))
    session.commit()

    fetched = session.get(Idea, "perp-funding-carry")
    assert fetched.title == "Perp funding carry v2"
    assert fetched.created_at == created_at_before  # created_at must not change on update
    assert session.query(Idea).count() == 1


def test_idea_optional_driver_fk(session):
    repo.upsert_driver(
        session,
        id="perp-funding-premium",
        title="t",
        description="d",
        kill_condition="k",
        observable="o",
    )
    idea = repo.upsert_idea(session, **_idea_kwargs(driver_id="perp-funding-premium"))
    session.commit()
    assert idea.driver_id == "perp-funding-premium"


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


def test_add_spec_and_unique_idea_version(session):
    repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    spec = repo.add_spec(
        session,
        idea_id="perp-funding-carry",
        version=1,
        params={"lookback": 30},
        data_requirements={"instruments": ["BTC-PERP"]},
        rebalance="daily",
        costs_model={"taker_bps": 5},
        code_ref="qlab.strategies.perp_funding_carry",
    )
    session.commit()
    assert spec.id is not None

    # same (idea_id, version) again must violate the unique constraint
    # (repo functions flush, so the IntegrityError surfaces at the call site)
    with pytest.raises(IntegrityError):
        repo.add_spec(
            session,
            idea_id="perp-funding-carry",
            version=1,
            params={"lookback": 60},
            data_requirements={"instruments": ["BTC-PERP"]},
            rebalance="daily",
            costs_model={"taker_bps": 5},
            code_ref="qlab.strategies.perp_funding_carry",
        )
    session.rollback()

    # a new version for the same idea is fine
    spec2 = repo.add_spec(
        session,
        idea_id="perp-funding-carry",
        version=2,
        params={"lookback": 60},
        data_requirements={"instruments": ["BTC-PERP"]},
        rebalance="daily",
        costs_model={"taker_bps": 5},
        code_ref="qlab.strategies.perp_funding_carry",
    )
    session.commit()
    assert spec2.version == 2


def test_spec_requires_existing_idea_fk(session):
    with pytest.raises(IntegrityError):
        repo.add_spec(
            session,
            idea_id="does-not-exist",
            version=1,
            params={},
            data_requirements={},
            rebalance="daily",
            costs_model={},
            code_ref="qlab.strategies.x",
        )
    session.rollback()


# --------------------------------------------------------------------------
# data_snapshot + trial
# --------------------------------------------------------------------------


def _make_idea_and_spec(session) -> Spec:
    repo.upsert_idea(session, **_idea_kwargs())
    spec = repo.add_spec(
        session,
        idea_id="perp-funding-carry",
        version=1,
        params={"lookback": 30},
        data_requirements={},
        rebalance="daily",
        costs_model={},
        code_ref="qlab.strategies.perp_funding_carry",
    )
    session.commit()
    return spec


def test_data_snapshot_roundtrip(session):
    repo.add_data_snapshot(
        session,
        id="sha256:abc123",
        source="binance",
        instruments={"symbols": ["BTCUSDT"]},
        range_start=date(2024, 1, 1),
        range_end=date(2024, 6, 1),
        path="data/snapshots/abc123.parquet",
        rows=1000,
    )
    session.commit()
    fetched = session.get(DataSnapshot, "sha256:abc123")
    assert fetched is not None
    assert fetched.rows == 1000


def test_add_trial_with_snapshot(session):
    spec = _make_idea_and_spec(session)
    snap = repo.add_data_snapshot(
        session,
        id="sha256:abc123",
        source="binance",
        instruments={"symbols": ["BTCUSDT"]},
        range_start=date(2024, 1, 1),
        range_end=date(2024, 6, 1),
        path="data/snapshots/abc123.parquet",
        rows=1000,
    )
    session.commit()

    trial = repo.add_trial(
        session,
        spec_id=spec.id,
        config_hash="cfg-hash-1",
        code_sha="deadbeef",
        params={"lookback": 30},
        status=TrialStatus.OK,
        snapshot_id=snap.id,
        metrics={"sharpe_net": 1.2},
        kept=True,
    )
    session.commit()
    assert trial.id is not None
    assert trial.source == TrialSource.QLAB


def test_trial_requires_snapshot_unless_imported(session):
    spec = _make_idea_and_spec(session)

    # qlab-sourced trial without a snapshot violates the check constraint
    with pytest.raises(IntegrityError):
        repo.add_trial(
            session,
            spec_id=spec.id,
            config_hash="cfg-hash-1",
            code_sha="deadbeef",
            params={},
            status=TrialStatus.OK,
            snapshot_id=None,
            source=TrialSource.QLAB,
        )
    session.rollback()

    # an imported historical trial may omit the snapshot
    trial = repo.add_trial(
        session,
        spec_id=spec.id,
        config_hash="cfg-hash-1",
        code_sha="deadbeef",
        params={},
        status=TrialStatus.OK,
        snapshot_id=None,
        source=TrialSource.IMPORTED,
    )
    session.commit()
    assert trial.source == TrialSource.IMPORTED


def test_trial_is_append_only_by_convention(session):
    """Repo has no update/delete for trial — every add_trial call inserts a
    new row, even for what would conceptually be a "rerun"."""
    spec = _make_idea_and_spec(session)
    for _ in range(3):
        repo.add_trial(
            session,
            spec_id=spec.id,
            config_hash="cfg-hash-1",
            code_sha="deadbeef",
            params={},
            status=TrialStatus.ERROR,
            source=TrialSource.IMPORTED,
            kept=False,
        )
    session.commit()
    assert session.query(Trial).count() == 3


# --------------------------------------------------------------------------
# verdict (batch)
# --------------------------------------------------------------------------


def test_add_verdicts_batch(session):
    spec = _make_idea_and_spec(session)
    trial = repo.add_trial(
        session,
        spec_id=spec.id,
        config_hash="cfg-hash-1",
        code_sha="deadbeef",
        params={},
        status=TrialStatus.OK,
        source=TrialSource.IMPORTED,
        metrics={"sharpe_net": 0.5, "ann_return_net": -0.01},
    )
    session.commit()

    rows = repo.add_verdicts(
        session,
        [
            dict(
                idea_id="perp-funding-carry",
                spec_id=spec.id,
                trial_id=trial.id,
                stage=VerdictStage.EDGE,
                rule_id="net_edge_positive",
                rules_version="2026-09-20.1",
                metric="ann_return_net",
                value=-0.01,
                comparator=">",
                threshold=0.0,
                passed=False,
                data_range_start=date(2024, 1, 1),
                data_range_end=date(2024, 6, 1),
            ),
            dict(
                idea_id="perp-funding-carry",
                spec_id=spec.id,
                trial_id=trial.id,
                stage=VerdictStage.EDGE,
                rule_id="sharpe_floor",
                rules_version="2026-09-20.1",
                metric="sharpe_net",
                value=0.5,
                comparator=">=",
                threshold=0.8,
                passed=False,
                data_range_start=date(2024, 1, 1),
                data_range_end=date(2024, 6, 1),
            ),
        ],
    )
    session.commit()

    assert len(rows) == 2
    assert all(r.id is not None for r in rows)
    assert all(r.decided_at is not None for r in rows)
    assert session.query(Verdict).count() == 2


def test_verdict_requires_existing_idea_fk(session):
    with pytest.raises(IntegrityError):
        repo.add_verdicts(
            session,
            [
                dict(
                    idea_id="does-not-exist",
                    stage=VerdictStage.PREFLIGHT,
                    rule_id="capital_fit",
                    rules_version="2026-09-20.1",
                    metric="min_notional_usd",
                    value=100.0,
                    comparator="<=",
                    threshold=2500.0,
                    passed=True,
                    data_range_start=date(2024, 1, 1),
                    data_range_end=date(2024, 6, 1),
                )
            ],
        )
    session.rollback()


# --------------------------------------------------------------------------
# set_status / stage_transition atomicity
# --------------------------------------------------------------------------


def test_set_status_writes_transition_atomically(session):
    repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    idea = repo.set_status(
        session,
        idea_id="perp-funding-carry",
        new_status=IdeaStatus.SPECCING,
        reason="spec drafted",
        rules_version="2026-09-20.1",
    )
    session.commit()

    assert idea.status == IdeaStatus.SPECCING
    transitions = session.query(StageTransition).filter_by(idea_id="perp-funding-carry").all()
    assert len(transitions) == 1
    t = transitions[0]
    assert t.from_status == IdeaStatus.CANDIDATE
    assert t.to_status == IdeaStatus.SPECCING
    assert t.reason == "spec drafted"
    assert t.rules_version == "2026-09-20.1"


def test_set_status_unknown_idea_raises(session):
    with pytest.raises(ValueError):
        repo.set_status(
            session,
            idea_id="does-not-exist",
            new_status=IdeaStatus.REJECTED,
            reason="n/a",
            rules_version=None,
        )


def test_set_status_rolls_back_both_writes_together(session):
    """If the transaction containing set_status is rolled back (e.g. because
    a later statement in the same unit of work fails), neither the idea's
    status nor the stage_transition row should stick — proving the two
    writes are part of one atomic transaction, not two independent ones."""
    repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    repo.set_status(
        session,
        idea_id="perp-funding-carry",
        new_status=IdeaStatus.REJECTED,
        reason="failed edge test",
        rules_version="2026-09-20.1",
    )
    # something else in the same transaction fails before commit
    session.add(Spec(idea_id="does-not-exist", version=1, params={}, data_requirements={},
                      rebalance="daily", costs_model={}, code_ref="x",
                      created_at=datetime.now(UTC)))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    idea = session.get(Idea, "perp-funding-carry")
    assert idea.status == IdeaStatus.CANDIDATE  # unchanged
    assert session.query(StageTransition).count() == 0  # transition not persisted


# --------------------------------------------------------------------------
# FK restrict on delete (append-only / audit invariant)
# --------------------------------------------------------------------------


def test_deleting_idea_with_dependent_spec_is_restricted(session):
    _make_idea_and_spec(session)
    session.delete(session.get(Idea, "perp-funding-carry"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# --------------------------------------------------------------------------
# revival_check, stage_transition, token_spend, dossier
# --------------------------------------------------------------------------


def test_revival_check_roundtrip(session):
    repo.upsert_idea(session, **_idea_kwargs(status=IdeaStatus.REJECTED))
    session.commit()
    verdict = repo.add_verdicts(
        session,
        [
            dict(
                idea_id="perp-funding-carry",
                stage=VerdictStage.EDGE,
                rule_id="sharpe_floor",
                rules_version="2026-09-20.1",
                metric="sharpe_net",
                value=0.5,
                comparator=">=",
                threshold=0.8,
                passed=False,
                data_range_start=date(2024, 1, 1),
                data_range_end=date(2024, 6, 1),
            )
        ],
    )[0]
    session.commit()

    from qlab.registry.models import RevivalCheck

    rc = RevivalCheck(
        idea_id="perp-funding-carry",
        verdict_id=verdict.id,
        rejected_data_end=date(2024, 6, 1),
        eligible_at=date(2024, 9, 1),
        outcome=RevivalOutcome.PENDING,
    )
    session.add(rc)
    session.commit()

    fetched = session.get(RevivalCheck, rc.id)
    assert fetched.outcome == RevivalOutcome.PENDING


def test_token_spend_roundtrip(session):
    repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    ts = TokenSpend(
        at=datetime.now(UTC),
        stage="scout",
        idea_id="perp-funding-carry",
        agent="scout-agent",
        tokens_in=1000,
        tokens_out=200,
        usd_est=0.03,
    )
    session.add(ts)
    session.commit()
    assert session.query(TokenSpend).count() == 1


def test_dossier_roundtrip(session):
    repo.upsert_idea(session, **_idea_kwargs())
    session.commit()

    from qlab.registry.models import Dossier

    d = Dossier(
        idea_id="perp-funding-carry",
        path="dossiers/perp-funding-carry.md",
        rendered_at=datetime.now(UTC),
        rules_version="2026-09-20.1",
    )
    session.add(d)
    session.commit()
    assert session.query(Dossier).count() == 1


# --------------------------------------------------------------------------
# db.py
# --------------------------------------------------------------------------


def test_get_db_path_default_and_env(monkeypatch):
    monkeypatch.delenv("QLAB_DB", raising=False)
    assert get_db_path() == "data/qlab.db"

    monkeypatch.setenv("QLAB_DB", "/tmp/custom.db")
    assert get_db_path() == "/tmp/custom.db"
    assert get_database_url() == "sqlite+pysqlite:////tmp/custom.db"


def test_session_scope_commits_on_success(tmp_path):
    db_file = tmp_path / "qlab_test.db"
    eng = make_engine(f"sqlite+pysqlite:///{db_file}")
    Base.metadata.create_all(eng)

    factory = sessionmaker(bind=eng, expire_on_commit=False)

    import qlab.registry.db as db_module

    original_sessionmaker = db_module._SessionLocal
    original_engine = db_module._engine
    db_module._SessionLocal = factory
    db_module._engine = eng
    try:
        with session_scope() as s:
            repo.upsert_driver(
                s,
                id="perp-funding-premium",
                title="t",
                description="d",
                kill_condition="k",
                observable="o",
            )
    finally:
        db_module._SessionLocal = original_sessionmaker
        db_module._engine = original_engine

    with factory() as s:
        assert s.get(Driver, "perp-funding-premium") is not None
    eng.dispose()


def test_session_scope_rolls_back_on_exception(tmp_path):
    db_file = tmp_path / "qlab_test.db"
    eng = make_engine(f"sqlite+pysqlite:///{db_file}")
    Base.metadata.create_all(eng)

    factory = sessionmaker(bind=eng, expire_on_commit=False)

    import qlab.registry.db as db_module

    original_sessionmaker = db_module._SessionLocal
    original_engine = db_module._engine
    db_module._SessionLocal = factory
    db_module._engine = eng
    try:
        with pytest.raises(RuntimeError):
            with session_scope() as s:
                repo.upsert_driver(
                    s,
                    id="perp-funding-premium",
                    title="t",
                    description="d",
                    kill_condition="k",
                    observable="o",
                )
                raise RuntimeError("boom")
    finally:
        db_module._SessionLocal = original_sessionmaker
        db_module._engine = original_engine

    with factory() as s:
        assert s.get(Driver, "perp-funding-premium") is None
    eng.dispose()


def test_session_scope_enforces_foreign_keys(tmp_path):
    db_file = tmp_path / "qlab_test.db"
    eng = make_engine(f"sqlite+pysqlite:///{db_file}")
    Base.metadata.create_all(eng)

    factory = sessionmaker(bind=eng, expire_on_commit=False)

    import qlab.registry.db as db_module

    original_sessionmaker = db_module._SessionLocal
    original_engine = db_module._engine
    db_module._SessionLocal = factory
    db_module._engine = eng
    try:
        with pytest.raises(IntegrityError):
            with session_scope() as s:
                repo.add_spec(
                    s,
                    idea_id="does-not-exist",
                    version=1,
                    params={},
                    data_requirements={},
                    rebalance="daily",
                    costs_model={},
                    code_ref="x",
                )
    finally:
        db_module._SessionLocal = original_sessionmaker
        db_module._engine = original_engine
    eng.dispose()
