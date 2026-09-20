"""Tests for the graveyard importer: structural validation, the three
verdict kinds (decision/unknown/measurement), idempotency, and the
resulting ImportReport. Runs entirely against in-memory SQLite; never
touches seed/graveyard.yaml (that file is produced by another agent and
may be absent or mid-rewrite) — every fixture here is hand-written.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.registry.importer import (
    UNIDENTIFIED_RULE_ID,
    GraveyardImportError,
    import_graveyard,
)
from qlab.registry.models import (
    Base,
    Driver,
    Idea,
    IdeaStatus,
    StageTransition,
    TrialSource,
    Verdict,
)
from qlab.rules import LEGACY_SENTINEL_VERSION

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


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
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()


def _minimal_graveyard() -> dict:
    return {
        "drivers": [
            {
                "id": "perp-funding-premium",
                "title": "Perp funding premium",
                "description": "longs pay shorts",
                "kill_condition": "funding compresses to zero",
                "observable": "8h funding rate",
            }
        ],
        "ideas": [
            {
                "id": "cross-exchange-spread",
                "title": "Cross exchange funding spread",
                "source_type": "graveyard",
                "source_url": None,
                "claimed_edge": "NET +9.26%/year",
                "asset_class": "crypto-perp",
                "driver_id": "perp-funding-premium",
                "profile": "carry",
                "status": "rejected",
                "notes": "rejected on stale correlation criterion",
                "verdicts": [
                    {
                        "stage": "edge",
                        "metric": "ann_return_net",
                        "value": 0.0926,
                        "comparator": ">",
                        "threshold": 0.0,
                        "passed": True,
                        "data_range_start": None,
                        "data_range_end": None,
                        "decided_at": date(2026, 6, 16),
                        "rule_id": "net_edge_positive",
                        "source_file": "research/cross_exchange/FINDINGS.md",
                        "source_quote": "NET +9.26%/year, turnover 62/year",
                    }
                ],
            }
        ],
    }


def _write_yaml(tmp_path, data: dict, name: str = "graveyard.yaml"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return path


# --------------------------------------------------------------------------
# Basic round-trip
# --------------------------------------------------------------------------


def test_import_basic_counts(session, tmp_path):
    path = _write_yaml(tmp_path, _minimal_graveyard())
    report = import_graveyard(session, path)
    session.commit()

    assert report.drivers_inserted == 1
    assert report.drivers_updated == 0
    assert report.ideas_inserted == 1
    assert report.verdicts_inserted == 1
    assert report.verdicts_skipped_duplicate == 0
    assert report.data_errors == []

    assert session.get(Idea, "cross-exchange-spread") is not None
    verdict = session.query(Verdict).one()
    assert verdict.source == TrialSource.IMPORTED
    assert verdict.rules_version == LEGACY_SENTINEL_VERSION
    assert verdict.rule_id == "net_edge_positive"
    assert "import_key=" in verdict.note
    assert "source_file=research/cross_exchange/FINDINGS.md" in verdict.note
    assert "NET +9.26" in verdict.note


def test_import_is_idempotent(session, tmp_path):
    path = _write_yaml(tmp_path, _minimal_graveyard())

    import_graveyard(session, path)
    session.commit()
    assert session.query(Verdict).count() == 1
    assert session.query(Idea).count() == 1
    assert session.query(Driver).count() == 1

    report2 = import_graveyard(session, path)
    session.commit()

    assert report2.verdicts_inserted == 0
    assert report2.verdicts_skipped_duplicate == 1
    assert report2.drivers_updated == 1
    assert report2.ideas_updated == 1
    assert report2.status_changes == []  # unchanged seed -> no phantom transitions
    assert session.query(Verdict).count() == 1
    assert session.query(Idea).count() == 1
    assert session.query(Driver).count() == 1


# --------------------------------------------------------------------------
# Status changes on re-import (repo.set_status, not a silent assignment)
# --------------------------------------------------------------------------


def test_reimport_with_changed_seed_status_applies_transition(session, tmp_path):
    """A status edit made directly in the seed file must not be silently
    dropped: upsert_idea never touches status on an existing row (by
    design — status changes go through set_status so stage_transition
    stays authoritative), so the importer itself must detect the mismatch
    and apply it via set_status, and report it."""
    data = _minimal_graveyard()
    data["ideas"][0]["status"] = "bench"
    path = _write_yaml(tmp_path, data)

    report1 = import_graveyard(session, path)
    session.commit()
    assert report1.status_changes == []  # first sighting of the idea, not a change
    assert session.get(Idea, "cross-exchange-spread").status == IdeaStatus.BENCH

    data["ideas"][0]["status"] = "paper"
    path2 = _write_yaml(tmp_path, data, name="graveyard2.yaml")
    report2 = import_graveyard(session, path2)
    session.commit()

    idea = session.get(Idea, "cross-exchange-spread")
    assert idea.status == IdeaStatus.PAPER
    assert report2.status_changes == [("cross-exchange-spread", "bench", "paper")]
    assert report2.status_changes_count == 1

    transitions = (
        session.query(StageTransition).filter_by(idea_id="cross-exchange-spread").all()
    )
    assert len(transitions) == 1
    assert transitions[0].from_status == IdeaStatus.BENCH
    assert transitions[0].to_status == IdeaStatus.PAPER
    assert transitions[0].reason == f"seed re-import: {path2}"


def test_reimport_without_status_edit_creates_no_phantom_transitions(session, tmp_path):
    """Re-running the same (unchanged) file twice in a row must not add any
    stage_transition rows — only a genuine mismatch between seed and DB
    should ever produce one."""
    data = _minimal_graveyard()
    data["ideas"][0]["status"] = "bench"
    path = _write_yaml(tmp_path, data)

    import_graveyard(session, path)
    session.commit()

    data["ideas"][0]["status"] = "paper"
    path2 = _write_yaml(tmp_path, data, name="graveyard2.yaml")
    import_graveyard(session, path2)
    session.commit()
    assert session.query(StageTransition).count() == 1

    # re-import the same (already-applied) status again: no new transition
    report3 = import_graveyard(session, path2)
    session.commit()

    assert report3.status_changes == []
    assert session.query(StageTransition).count() == 1


# --------------------------------------------------------------------------
# Structural failures: nothing gets written
# --------------------------------------------------------------------------


def test_missing_file_raises(session, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_graveyard(session, tmp_path / "does-not-exist.yaml")


def test_top_level_must_be_a_mapping(session, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(GraveyardImportError):
        import_graveyard(session, path)


def test_bad_enum_value_fails_before_writing(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["asset_class"] = "not-a-real-asset-class"
    path = _write_yaml(tmp_path, data)

    with pytest.raises(GraveyardImportError) as exc_info:
        import_graveyard(session, path)

    assert "cross-exchange-spread" in str(exc_info.value)
    assert session.query(Idea).count() == 0
    assert session.query(Driver).count() == 0
    assert session.query(Verdict).count() == 0


def test_unknown_field_is_rejected(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["not_a_real_field"] = "typo"
    path = _write_yaml(tmp_path, data)

    with pytest.raises(GraveyardImportError):
        import_graveyard(session, path)
    assert session.query(Idea).count() == 0


def test_missing_required_field_fails_before_writing(session, tmp_path):
    data = _minimal_graveyard()
    del data["ideas"][0]["verdicts"][0]["source_quote"]
    path = _write_yaml(tmp_path, data)

    with pytest.raises(GraveyardImportError) as exc_info:
        import_graveyard(session, path)
    assert "verdict[0]" in str(exc_info.value)
    assert session.query(Idea).count() == 0


# --------------------------------------------------------------------------
# Data errors: per-row, reported and skipped, rest still imports
# --------------------------------------------------------------------------


def test_value_without_passed_is_data_error_and_skipped(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"].append(
        {
            "stage": "correlation",
            "metric": "corr_to_frab_proxy",
            "value": 0.86,
            "comparator": "<",
            "threshold": 0.30,
            "passed": None,  # inconsistent: a criterion exists but no decision recorded
            "data_range_start": None,
            "data_range_end": None,
            "decided_at": date(2026, 6, 16),
            "rule_id": None,
            "source_file": "research/cross_exchange/FINDINGS.md",
            "source_quote": "SPREAD correlated with FRAB-proxy = +0.86",
        }
    )
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert len(report.data_errors) == 1
    assert report.data_errors[0].idea_id == "cross-exchange-spread"
    assert report.data_errors[0].verdict_index == 1
    assert "value and passed" in report.data_errors[0].message
    # the first (good) verdict still imports
    assert report.verdicts_inserted == 1
    assert session.query(Verdict).count() == 1


def test_measurement_with_rule_id_is_a_data_error(session, tmp_path):
    """A measurement (no comparator/threshold) with a non-null rule_id is
    self-contradictory per docs/REGISTRY.md: a measurement means no rule
    was ever identified."""
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"] = [
        {
            "stage": "edge",
            "metric": "sharpe_net",
            "value": 0.77,
            "comparator": None,
            "threshold": None,
            "passed": None,
            "data_range_start": None,
            "data_range_end": None,
            "decided_at": date(2026, 9, 13),
            "rule_id": "sharpe_floor",
            "source_file": "research/GRAVEYARD_REVIEW_2026_09.md",
            "source_quote": "full book (46 positions) 0.77",
        }
    ]
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert len(report.data_errors) == 1
    assert report.verdicts_inserted == 0


def test_comparator_without_threshold_is_a_data_error(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"][0]["threshold"] = None
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert len(report.data_errors) == 1
    assert report.verdicts_inserted == 0


# --------------------------------------------------------------------------
# The three verdict kinds
# --------------------------------------------------------------------------


def test_measurement_rule_id_null_becomes_unidentified(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"] = [
        {
            "stage": "edge",
            "metric": "sharpe_net",
            "value": 0.77,
            "comparator": None,
            "threshold": None,
            "passed": None,
            "data_range_start": None,
            "data_range_end": None,
            "decided_at": date(2026, 9, 13),
            "rule_id": None,
            "source_file": "research/GRAVEYARD_REVIEW_2026_09.md",
            "source_quote": "full book (46 positions) 0.77",
        }
    ]
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert report.data_errors == []
    assert report.verdicts_measurement == 1
    assert report.verdicts_unidentified_rule == 1
    assert report.verdicts_null_value == 0  # "measurement", not "unknown"

    verdict = session.query(Verdict).one()
    assert verdict.rule_id == UNIDENTIFIED_RULE_ID
    assert verdict.comparator is None
    assert verdict.threshold is None
    assert verdict.passed is None
    assert verdict.value == 0.77
    assert verdict.source == TrialSource.IMPORTED


def test_unknown_kind_imports_and_counts_null_value(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"] = [
        {
            "stage": "edge",
            "metric": "sharpe_net",
            "value": None,
            "comparator": ">=",
            "threshold": 0.8,
            "passed": None,
            "data_range_start": None,
            "data_range_end": None,
            "decided_at": date(2026, 9, 13),
            "rule_id": "sharpe_floor",
            "source_file": "research/x.md",
            "source_quote": "not computed",
        }
    ]
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert report.data_errors == []
    assert report.verdicts_null_value == 1
    assert report.verdicts_measurement == 0
    assert report.verdicts_unidentified_rule == 0

    verdict = session.query(Verdict).one()
    assert verdict.value is None
    assert verdict.passed is None
    assert verdict.comparator == ">="
    assert verdict.threshold == 0.8
    assert verdict.rule_id == "sharpe_floor"


def test_decayed_idea_missing_shutdown_cause_fails_before_writing(session, tmp_path):
    """A decayed idea without shutdown_cause is a structural problem, not a
    per-verdict data error: the whole import must fail loudly, before a
    single row is written, and the error must name the offending idea."""
    data = _minimal_graveyard()
    data["ideas"][0]["status"] = "decayed"
    path = _write_yaml(tmp_path, data)

    with pytest.raises(GraveyardImportError) as exc_info:
        import_graveyard(session, path)

    assert "cross-exchange-spread" in str(exc_info.value)
    assert "shutdown_cause" in str(exc_info.value)
    assert session.query(Idea).count() == 0


def test_decayed_idea_with_shutdown_cause_imports(session, tmp_path):
    data = _minimal_graveyard()
    data["ideas"][0]["status"] = "decayed"
    data["ideas"][0]["shutdown_cause"] = "false-discovery"
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert report.ideas_inserted == 1
    idea = session.get(Idea, "cross-exchange-spread")
    assert idea.status.value == "decayed"
    assert idea.shutdown_cause.value == "false-discovery"


def test_null_rule_id_on_a_decision_still_becomes_unidentified(session, tmp_path):
    """rule_id: null can also appear on an ordinary decision row (a
    graveyard document that recorded pass/fail against a threshold but
    never named the rule) — the substitution applies regardless of kind."""
    data = _minimal_graveyard()
    data["ideas"][0]["verdicts"][0]["rule_id"] = None
    path = _write_yaml(tmp_path, data)

    report = import_graveyard(session, path)
    session.commit()

    assert report.verdicts_unidentified_rule == 1
    verdict = session.query(Verdict).one()
    assert verdict.rule_id == UNIDENTIFIED_RULE_ID
    assert verdict.passed is True
    assert verdict.value == 0.0926
