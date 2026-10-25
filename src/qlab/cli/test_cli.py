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
    assert "status changes" in result2.output


def test_import_graveyard_reports_status_change_on_seed_edit(cli_db, tmp_path):
    """Editing an idea's status in the seed and re-importing must surface
    the change in the report, not silently leave the old status in place."""
    runner.invoke(app, ["init-db"])
    yaml_path = tmp_path / "graveyard.yaml"
    yaml_path.write_text(_SAMPLE_GRAVEYARD_YAML, encoding="utf-8")
    runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])

    yaml_path.write_text(
        _SAMPLE_GRAVEYARD_YAML.replace("status: rejected", "status: bench", 1),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert "status changes (seed re-import):" in result.output
    assert "cross-exchange-spread: rejected -> bench" in result.output


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


def test_funnel_shows_shutdown_cause_breakdown(imported_db):
    """The fixture graveyard has no decayed ideas, so the section prints but
    is empty; a separate case below covers the populated section."""
    result = runner.invoke(app, ["funnel"])
    assert result.exit_code == 0, result.output
    assert "decayed ideas by shutdown cause:" in result.output
    assert "(none)" in result.output


def test_funnel_shutdown_cause_breakdown_with_decayed_idea(cli_db, tmp_path):
    runner.invoke(app, ["init-db"])
    yaml_path = tmp_path / "graveyard.yaml"
    yaml_path.write_text(
        _SAMPLE_GRAVEYARD_YAML.replace("status: rejected", "status: decayed", 1).replace(
            "notes: rejected on stale correlation criterion",
            "shutdown_cause: false-discovery\n"
            "    notes: rejected on stale correlation criterion",
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["import-graveyard", "--file", str(yaml_path)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["funnel"])
    assert result.exit_code == 0, result.output
    assert "decayed ideas by shutdown cause:" in result.output
    assert "false-discovery" in result.output


def test_retire_needs_a_reason_and_writes_the_transition(cli_db):
    """Owner, 2026-10-10: «да, переводи все три» -- control measurements
    leave the bench by a recorded decision, never silently."""
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import (
        AssetClass,
        Idea,
        IdeaStatus,
        Profile,
        SourceType,
        StageTransition,
    )

    runner.invoke(app, ["init-db"])
    with session_scope() as session:
        repo.upsert_idea(session, id="control-a", title="control", source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    assert runner.invoke(app, ["retire", "control-a"]).exit_code != 0  # no reason given
    assert runner.invoke(app, ["retire", "nope", "--reason", "x"]).exit_code == 1
    result = runner.invoke(app, ["retire", "control-a", "--reason", "a control, done its job"])
    assert result.exit_code == 0, result.output
    with session_scope() as session:
        assert session.get(Idea, "control-a").status == IdeaStatus.RETIRED
        row = session.query(StageTransition).filter_by(idea_id="control-a").order_by(
            StageTransition.id.desc()).first()
        assert row.to_status == IdeaStatus.RETIRED and row.reason == "a control, done its job"
    again = runner.invoke(app, ["retire", "control-a", "--reason", "again"])
    assert "already retired" in again.output


def test_idea_set_parent_links_a_variant_and_refuses_a_cycle_or_unknown_ids(cli_db):
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    runner.invoke(app, ["init-db"])
    with session_scope() as session:
        for idea_id in ("root", "variant", "grandchild"):
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    result = runner.invoke(app, ["idea", "set-parent", "variant", "root"])
    assert result.exit_code == 0, result.output
    assert "variant: (none) -> root" in result.output
    again = runner.invoke(app, ["idea", "set-parent", "variant", "root"])
    assert "variant: root -> root" in again.output
    assert runner.invoke(app, ["idea", "set-parent", "grandchild", "variant"]).exit_code == 0

    for idea_id, parent in (("root", "grandchild"), ("root", "variant"), ("root", "root")):
        refused = runner.invoke(app, ["idea", "set-parent", idea_id, parent])
        assert refused.exit_code == 1 and "loop" in refused.output
    assert runner.invoke(app, ["idea", "set-parent", "nope", "root"]).exit_code == 1
    assert runner.invoke(app, ["idea", "set-parent", "root", "nope"]).exit_code == 1
    with session_scope() as session:
        assert session.get(Idea, "root").parent_id is None
        assert session.get(Idea, "grandchild").parent_id == "variant"


def test_research_notes_import_is_idempotent_and_reports_skipped_strategies(cli_db, tmp_path):
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Profile, ResearchNote, SourceType

    runner.invoke(app, ["init-db"])
    with session_scope() as session:
        for idea_id in ("frab", "bv2-a"):
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    memos = tmp_path / "research"
    memos.mkdir()
    (memos / "frab-2026-10-25.md").write_text(
        "memo\n\n## Эксперименты на стенде\n\n- `frab-x` (прогон 7): reject — bad\n",
        encoding="utf-8")
    (memos / "ghost-2026-10-01.md").write_text("memo", encoding="utf-8")
    config = tmp_path / "research.yaml"
    config.write_text("strategies:\n  frab:\n    ideas: [frab]\n  bv2:\n    ideas: [bv2-a]\n",
                      encoding="utf-8")
    args = ["research-notes", "import", "--memo-dir", str(memos), "--config", str(config),
            "--spec-dir", str(tmp_path / "specs")]

    for _ in range(2):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    assert "imported frab-2026-10-25.md: 1 ideas" in result.output
    assert "skipped ghost-2026-10-01.md" in result.output
    assert "1 memos imported, 1 skipped" in result.output
    with session_scope() as session:
        row = session.query(ResearchNote).one()
        assert row.idea_id == "frab" and row.experiments[0]["trial_id"] == 7


def test_generate_dry_run_prints_the_catalogue_without_spending(tmp_path, monkeypatch):
    catalogue = tmp_path / "MECHANISMS.md"
    catalogue.write_text(
        "### xs-reversal-1w\n\n**Суть:** Откат.\n\n**Блоки:** signal reversal\n\n"
        "**Параметры:**\n\n| параметр блока | значения из источника | источник |\n|---|---|---|\n"
        "| signal.lookback_days | 7, 14, 28 | [Author 2020](https://example.org) |\n\n"
        "**Чего блоки не выражают:** Сезонность.\n\n**Источники:** Author 2020.\n",
        encoding="utf-8")
    monkeypatch.setenv("QLAB_CLAUDE_BIN", "/nonexistent/claude")  # an agent call would raise
    result = runner.invoke(app, ["generate", "--dry-run", "--catalogue", str(catalogue)])
    assert result.exit_code == 0, result.output
    assert "xs-reversal-1w: 1 parameter rows, partly expressed" in result.output
    assert "signal.lookback_days: 7, 14, 28" in result.output
    assert "1 mechanisms in" in result.output


def test_director_dry_run_prints_the_evidence_and_approve_clears_the_flag(
        cli_db, tmp_path, monkeypatch):
    from datetime import date

    import qlab.agents.director as director_module
    from qlab import directions as dirs

    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "MECHANISMS.md").write_text(
        "### xs-mom\n\n**Суть:** s\n\n**Блоки:** b\n\n**Параметры:**\n\n"
        "| параметр блока | значения из источника | источник |\n|---|---|---|\n"
        "| rebalance_days | 7 | [A](https://e.org) |\n\n**Чего блоки не выражают:** x\n",
        encoding="utf-8")
    config = tmp_path / "director.yaml"
    config.write_text("enabled: true\ncadence_days: 2\nfamilies:\n  momentum: [xs-mom]\n",
                      encoding="utf-8")
    research = tmp_path / "research.yaml"
    research.write_text("strategies:\n  bv2: {ideas: [i], specs: []}\n", encoding="utf-8")
    monkeypatch.setattr(director_module, "QLAB_ROOT", root)
    monkeypatch.setenv("QLAB_CLAUDE_BIN", "/nonexistent/claude")  # an agent call would raise
    assert runner.invoke(app, ["init-db"]).exit_code == 0

    result = runner.invoke(app, ["director", "--dry-run", "--config", str(config),
                                 "--research-config", str(research)])
    assert result.exit_code == 0, result.output
    assert "# Evidence for the research director" in result.output
    assert "xs-mom: buildable now" in result.output and "researcher strategies" in result.output

    # Nothing to approve yet; then the owner clears the flag of the first directions.
    assert runner.invoke(app, ["director", "--approve"]).exit_code == 1
    parsed = dirs.parse_block("generator: {}", mechanism_ids=set(), strategy_names=set(),
                              family_names=set(), day=date(2026, 10, 26),
                              pending_owner_review=True)
    dirs.write_directions(parsed.directions, root)
    assert dirs.active_directions(root) is None
    result = runner.invoke(app, ["director", "--approve"])
    assert result.exit_code == 0 and "approved" in result.output
    assert dirs.active_directions(root) is not None
