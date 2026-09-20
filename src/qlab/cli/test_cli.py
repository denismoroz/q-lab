"""Smoke tests for the `qlab` CLI: every command exits 0 and produces the
expected shape of output, exercised against a throwaway SQLite file (never
the real data/qlab.db) and a hand-written graveyard fixture (never
seed/graveyard.yaml).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import qlab.registry.db as db_module
from qlab.cli import app

runner = CliRunner()

_SAMPLE_GRAVEYARD_YAML = """
drivers:
  - id: perp-funding-premium
    title: Perp funding premium
    description: longs pay shorts
    kill_condition: funding compresses to zero
    observable: 8h funding rate

ideas:
  - id: cross-exchange-spread
    title: Cross exchange funding spread
    source_type: graveyard
    source_url: null
    claimed_edge: "NET +9.26%/year"
    asset_class: crypto-perp
    driver_id: perp-funding-premium
    profile: carry
    status: rejected
    notes: rejected on stale correlation criterion
    verdicts:
      - stage: edge
        metric: ann_return_net
        value: 0.0926
        comparator: ">"
        threshold: 0.0
        passed: true
        data_range_start: null
        data_range_end: null
        decided_at: 2026-06-16
        rule_id: net_edge_positive
        source_file: research/cross_exchange/FINDINGS.md
        source_quote: "NET +9.26%/year"
      - stage: edge
        metric: sharpe_net
        value: 0.77
        comparator: null
        threshold: null
        passed: null
        data_range_start: null
        data_range_end: 2024-01-01
        decided_at: 2026-09-13
        rule_id: null
        source_file: research/GRAVEYARD_REVIEW_2026_09.md
        source_quote: "full book Sharpe 0.77"
      - stage: capacity
        metric: ann_return_on_capital
        value: 0.031
        comparator: "<"
        threshold: 0.10
        passed: false
        data_range_start: null
        data_range_end: null
        decided_at: 2026-09-13
        rule_id: null
        source_file: research/GRAVEYARD_REVIEW_2026_09.md
        source_quote: "3.1% on capital, all 9 coins"
"""


@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    """Point qlab.registry.db at a throwaway file for the duration of a test
    and reset its module-level engine/sessionmaker cache around it, since
    the CLI runs in-process (CliRunner does not spawn a subprocess)."""
    db_file = tmp_path / "qlab_cli_test.db"
    monkeypatch.setenv("QLAB_DB", str(db_file))
    db_module._engine = None
    db_module._SessionLocal = None
    yield db_file
    db_module._engine = None
    db_module._SessionLocal = None


def test_init_db_creates_tables(cli_db):
    result = runner.invoke(app, ["init-db"])
    assert result.exit_code == 0, result.output
    assert cli_db.exists()


def test_import_graveyard_reports_and_exits_zero(cli_db, tmp_path):
    runner.invoke(app, ["init-db"])
    yaml_path = tmp_path / "graveyard.yaml"
    yaml_path.write_text(_SAMPLE_GRAVEYARD_YAML, encoding="utf-8")

    result = runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert "verdicts inserted" in result.output
    assert "total:" in result.output

    # idempotent from the CLI too
    result2 = runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])
    assert result2.exit_code == 0, result2.output


def test_import_graveyard_missing_file_reports_error(cli_db, tmp_path):
    runner.invoke(app, ["init-db"])
    result = runner.invoke(app, ["import-graveyard", "--file", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


@pytest.fixture()
def imported_db(cli_db, tmp_path):
    runner.invoke(app, ["init-db"])
    yaml_path = tmp_path / "graveyard.yaml"
    yaml_path.write_text(_SAMPLE_GRAVEYARD_YAML, encoding="utf-8")
    result = runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])
    assert result.exit_code == 0, result.output
    return cli_db


def test_graveyard_retired_exits_zero(imported_db):
    result = runner.invoke(app, ["graveyard", "retired"])
    assert result.exit_code == 0, result.output
    assert "total:" in result.output


def test_graveyard_near_exits_zero(imported_db):
    result = runner.invoke(app, ["graveyard", "near", "--margin", "0.5"])
    assert result.exit_code == 0, result.output
    assert "total:" in result.output


def test_graveyard_ripe_exits_zero_and_shows_unknown_group(imported_db):
    result = runner.invoke(app, ["graveyard", "ripe", "--min-new-days", "1"])
    assert result.exit_code == 0, result.output
    assert "ripe for revival:" in result.output
    # the header (with both counts) must come before the detail groups
    header_pos = result.output.index("ripe for revival:")
    cannot_revive_pos = result.output.index("cannot auto-revive")
    assert header_pos < cannot_revive_pos
    # the imported fixture has verdicts with a null data_range_end/start
    assert "cannot auto-revive" in result.output


def test_funnel_exits_zero_and_shows_three_outcome_kinds(imported_db):
    result = runner.invoke(app, ["funnel"])
    assert result.exit_code == 0, result.output
    assert "passed" in result.output
    assert "failed" in result.output
    assert "unknown" in result.output
    assert "measurement" in result.output
    assert "total:" in result.output
