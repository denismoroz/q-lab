"""`qlab anomalies` (flags, dossiers, the learning loop) and `qlab anomaly` (the investigator's
read-only toolkit) through the command line (docs/ANOMALIES.md)."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

import qlab.anomalies.config as anomaly_config
import qlab.registry.db as db
from qlab.anomalies import flags as fl
from qlab.anomalies import store, toolkit
from qlab.anomalies.scan import ScanResult
from qlab.anomalies.test_investigator import CALL_ONE, CALL_TWO, GOOD, cfg_with, stream
from qlab.anomalies.test_investigator import fake_claude as fake_claude  # noqa: F401, PLC0414
from qlab.anomalies.test_stage import world as world  # noqa: F401, PLC0414
from qlab.cli import app

runner = CliRunner()
FIRED = fl.FlagResult("implausible_day", fl.FIRES, 1.8, 1.0, detail={
    "days_over": 3, "days": 100, "top": [{"day": "2026-01-21", "return": 0.2, "day_line": 0.1,
                                          "positions": 6, "gross": 0.6, "net": 0.2,
                                          "times_line": 1.8}]})


def _scan(world, name: str = "alpha", flags=None) -> int:  # noqa: F811
    trial_id = world.ids[name]
    with db.session_scope() as session:
        store.save_scan(session, ScanResult(trial_id, name, "paper", flags or [FIRED]), "test.1")
    return trial_id


def _counts(world) -> dict[str, int]:  # noqa: F811
    with world.engine.connect() as conn:
        tables = [r[0] for r in conn.execute(text(
            "select name from sqlite_master where type='table'"))]
        return {t: conn.execute(text(f'select count(*) from "{t}"')).scalar() for t in tables}


def test_the_list_says_when_nothing_was_scanned_and_otherwise_shows_runs(world):  # noqa: F811
    empty = runner.invoke(app, ["anomalies"])
    assert empty.exit_code == 0 and "no scanned run" in empty.output

    _scan(world)
    _scan(world, "gamma", [fl.FlagResult("implausible_day", fl.QUIET, 0.2, 1.0)])

    listing = runner.invoke(app, ["anomalies"])
    assert listing.exit_code == 0
    assert "alpha" in listing.output and "implausible_day" in listing.output
    assert "totals over 2 scanned run(s) of 2 idea(s); 1 with at least one flag fired" \
        in listing.output
    only = runner.invoke(app, ["anomalies", "--fired"])
    assert "gamma" not in only.output.split("totals")[0]
    data = json.loads(runner.invoke(app, ["anomalies", "--json"]).output)
    assert data["summary"]["flagged"] == 1 and len(data["rows"]) == 2


def test_show_prints_the_flags_in_words_and_the_dossier(world):  # noqa: F811
    trial_id = _scan(world)
    with db.session_scope() as session:
        store.save_investigation(session, store.NewInvestigation(
            idea_id="alpha", trial_id=trial_id, model="opus", flags_fingerprint="x",
            cause="data-artifact", summary="s", dossier="# Разбор аномалии: alpha",
            evidence=[], rejected=[], proposals=[], caught_by=[], new_check=None,
            programmer_requests=[], tokens=0, calls=0))

    out = runner.invoke(app, ["anomalies", "show", str(trial_id)])

    assert out.exit_code == 0
    assert "сработал" in out.output and "# Разбор аномалии: alpha" in out.output
    assert "data-artifact" in runner.invoke(app, ["anomalies"]).output
    missing = runner.invoke(app, ["anomalies", "show", "999"])
    assert missing.exit_code == 1 and "no stored scan" in missing.output


def test_the_owner_decides_on_a_proposed_check_from_the_command_line(world):  # noqa: F811
    trial_id = _scan(world)
    with db.session_scope() as session:
        row = store.save_investigation(session, store.NewInvestigation(
            idea_id="alpha", trial_id=trial_id, model="opus", flags_fingerprint="x",
            cause="stand-bug", summary="s", dossier="d", evidence=[], rejected=[], proposals=[],
            caught_by=[], new_check="Сверять списки монет.", programmer_requests=[], tokens=0,
            calls=0))
        number = row.id

    learning = runner.invoke(app, ["anomalies", "learning"])
    assert "caught by no flag" in learning.output and "Сверять списки монет." in learning.output

    done = runner.invoke(app, ["anomalies", "check", str(number), "adopted"])
    assert done.exit_code == 0 and "adopted" in done.output
    assert "awaiting" not in runner.invoke(app, ["anomalies", "learning"]).output
    bad = runner.invoke(app, ["anomalies", "check", str(number), "maybe"])
    assert bad.exit_code == 1 and "status" in bad.output


def test_scan_stores_the_code_detectors_flags_without_any_agent(world, monkeypatch):  # noqa: F811
    from qlab.anomalies import stage

    def fake(session, trial_id, cfg, **kw):
        return ScanResult(trial_id, "alpha", "paper", [FIRED], {})

    monkeypatch.setattr(stage, "scan_trial", fake)
    trial_id = world.ids["alpha"]

    out = runner.invoke(app, ["anomalies", "scan", "--trial", str(trial_id)])
    again = runner.invoke(app, ["anomalies", "scan", "--trial", str(trial_id)])

    assert out.exit_code == 0 and "fired implausible_day" in out.output
    assert "already scanned" in again.output
    with db.session_scope() as session:
        assert session.query(store.AnomalyScan).count() == 1
    assert runner.invoke(app, ["anomalies", "scan"]).exit_code == 1  # names nothing


def test_investigate_by_hand_goes_through_the_guard_and_refuses_a_quiet_run(
        world, monkeypatch, fake_claude, tmp_path):  # noqa: F811
    from qlab.anomalies import investigator as inv
    from qlab.anomalies import stage

    cfg = cfg_with(tmp_path)
    cfg = type(cfg)(version="test.1", enabled=True, scan=cfg.scan, flags=cfg.flags,
                    investigator=cfg.investigator)
    monkeypatch.setattr(anomaly_config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(inv, "programmer_repos", lambda: set())
    trial_id = _scan(world, flags=[fl.FlagResult("implausible_day", fl.QUIET, 0.2, 1.0)])
    monkeypatch.setattr(stage, "scan_one", lambda *a, **k: "")
    quiet = runner.invoke(app, ["anomalies", "investigate", str(trial_id)])
    assert quiet.exit_code == 1 and "no flag fired" in quiet.output

    _scan(world, flags=[FIRED])
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    out = runner.invoke(app, ["anomalies", "investigate", str(trial_id)])

    assert out.exit_code == 0, out.output
    assert "# Разбор аномалии: alpha" in out.output
    with db.session_scope() as session:
        assert session.query(store.AnomalyInvestigation).count() == 1
    again = runner.invoke(app, ["anomalies", "investigate", str(trial_id)])
    assert again.exit_code == 1 and "investigated under the same flags" in again.output


# --- qlab anomaly -------------------------------------------------------------------------


def test_the_toolkit_commands_write_nothing_to_the_registry(world):  # noqa: F811
    trial_id = _scan(world)
    before = _counts(world)

    for argv in (["anomaly", "flags", str(trial_id)],
                 ["anomaly", "flags", "999"],
                 ["anomaly", "compare-venues", "alpha"],
                 ["anomaly", "contributions", str(trial_id)],
                 ["anomaly", "coin", "BTC", "--trial", str(trial_id)],
                 ["anomaly", "counterfactual", str(trial_id), "--exclude-volume-below", "1"],
                 ["anomaly", "battery", str(trial_id)]):
        runner.invoke(app, argv)

    assert _counts(world) == before


def test_a_toolkit_answer_is_numbered_and_logged_for_the_dossier_to_cite(
        world, monkeypatch, tmp_path):  # noqa: F811
    trial_id = _scan(world)
    log = tmp_path / "calls" / "1.jsonl"
    monkeypatch.setenv(toolkit.LOG_ENV, str(log))

    first = runner.invoke(app, ["anomaly", "flags", str(trial_id)])
    second = runner.invoke(app, ["anomaly", "flags", str(trial_id), "--json"])

    assert first.output.startswith("[call 1] qlab anomaly flags")
    assert "implausible_day: fires" in first.output
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["cmd"] for r in rows] == [f"qlab anomaly flags {trial_id}"] * 2
    assert rows[0]["output"].startswith("[call 1]") and "implausible_day" in rows[0]["output"]
    assert json.loads(second.output)["fired"] == ["implausible_day"]


def test_without_a_log_the_toolkit_prints_plain_text(world, monkeypatch):  # noqa: F811
    trial_id = _scan(world)
    monkeypatch.delenv(toolkit.LOG_ENV, raising=False)

    out = runner.invoke(app, ["anomaly", "flags", str(trial_id)])

    assert out.output.startswith(f"trial {trial_id}")


def test_a_counterfactual_takes_exactly_one_rule(world):  # noqa: F811
    none = runner.invoke(app, ["anomaly", "counterfactual", "1"])
    two = runner.invoke(app, ["anomaly", "counterfactual", "1", "--exclude-volume-below", "1",
                              "--cap-instrument-share", "0.1"])

    for out in (none, two):
        assert out.exit_code == 1 and "exactly one rule" in out.output


def test_asking_the_strategy_again_applies_to_a_ban_not_to_a_cap(world):  # noqa: F811
    out = runner.invoke(app, ["anomaly", "counterfactual", "1", "--cap-leg-contribution", "0.02",
                              "--reselect"])

    assert out.exit_code == 1 and "ban on coin-days" in out.output


def test_a_trial_that_cannot_be_re_run_is_an_error_line_not_a_traceback(world):  # noqa: F811
    trial_id = world.ids["alpha"]  # a registry row with no spec file on disk

    out = runner.invoke(app, ["anomaly", "contributions", str(trial_id)])

    assert out.exit_code == 1 and "error:" in out.output and "spec file" in out.output


@pytest.mark.parametrize("command", ["contributions", "coin", "counterfactual", "battery",
                                     "compare-venues", "flags"])
def test_the_toolkits_commands_are_the_ones_the_prompt_teaches(command):
    from qlab.anomalies.investigator import INSTRUCTIONS

    assert f"uv run qlab anomaly {command}" in INSTRUCTIONS
    assert runner.invoke(app, ["anomaly", command, "--help"]).exit_code == 0


# --- qlab evaluate -------------------------------------------------------------------------


def test_evaluate_shows_and_stores_the_flags_of_a_run_that_passed(world, monkeypatch):  # noqa: F811
    from qlab.anomalies import cli as anomaly_cli
    from qlab.anomalies import scan as scan_module

    trial_id = world.ids["alpha"]
    monkeypatch.setattr(scan_module, "scan_trial", lambda session, tid, cfg, **kw: ScanResult(
        tid, "alpha", "paper", [FIRED], {}))

    lines = anomaly_cli.evaluate_lines(trial_id, "paper")

    assert lines[0].startswith("anomaly flags (recorded, they decide nothing")
    assert "сработал" in lines[1] and lines[-1] == ""
    with db.session_scope() as session:
        assert session.query(store.AnomalyScan).count() == 1
    # A run that did not pass is not scanned, and a scan that fails never fails the evaluation.
    assert anomaly_cli.evaluate_lines(trial_id, "reject") == []

    def boom(*a, **k):
        raise RuntimeError("no snapshot")

    monkeypatch.setattr(scan_module, "scan_trial", boom)
    failed = anomaly_cli.evaluate_lines(trial_id, "paper")
    assert failed[0].startswith("anomaly flags: not computed") and "no snapshot" in failed[0]
