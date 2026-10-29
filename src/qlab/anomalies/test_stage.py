"""The night's anomaly stage: the code part scans, the agent part investigates a bounded number of
flagged runs, passed ideas first, idempotently, through the budget guard (docs/ANOMALIES.md).
The scan and the claude binary are faked; the registry, the ledger and the guard are real."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import qlab.registry.db as db
from qlab import night
from qlab.anomalies import flags as fl
from qlab.anomalies import investigator as inv
from qlab.anomalies import report, stage, store
from qlab.anomalies.scan import ScanResult
from qlab.anomalies.test_flags import RAW
from qlab.anomalies.test_investigator import (
    CALL_ONE,
    CALL_TWO,
    GOOD,
    NOW,
    RESET,
    _reading,
    fake_claude,  # noqa: F401 - a fixture
    stream,
)
from qlab.anomalies.test_scan import _idea, _trial
from qlab.budget import NightBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.registry.models import (
    AnomalyInvestigation,
    AnomalyScan,
    Base,
    IdeaStatus,
    TokenSpend,
    TrialRoute,
)

DAY = date(2026, 10, 28)
END = date(2026, 10, 27)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A file registry behind the global session, a night folder and a config file in tmp."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'qlab.db'}")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "night")
    raw = {**RAW, "version": "test.1", "scan": {**RAW["scan"], "routes": ["paper", "shelf"]},
           "investigator": {**RAW["investigator"], "calls_dir": str(tmp_path / "calls"),
                            "max_per_night": 2}}
    config = tmp_path / "anomalies.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(inv, "programmer_repos", lambda: {"qlab"})
    with db.session_scope() as session:
        ids = {}
        for name, route in (("alpha", TrialRoute.PAPER), ("beta", TrialRoute.SHELF),
                            ("gamma", TrialRoute.PAPER), ("delta", TrialRoute.REJECT),
                            ("noise-1", TrialRoute.PAPER)):
            _idea(session, name, status=IdeaStatus.PAPER)
            ids[name] = _trial(session, name, route=route, metrics={"ann_return_net": 0.1})
    return type("World", (), {"config": config, "ids": ids, "tmp": tmp_path,
                              "ledger": tmp_path / "ledger.json", "queue": tmp_path / "q.json",
                              "engine": engine})


def fire(monkeypatch, world, fired_by_idea: dict[str, list[str]]):
    """Make the scan of each idea's trial fire the named flags."""
    by_trial = {world.ids[name]: names for name, names in fired_by_idea.items()}

    def fake_scan(session, trial_id, cfg, **kw):
        from qlab.registry.models import Spec, Trial

        trial = session.get(Trial, trial_id)
        idea = session.get(Spec, trial.spec_id).idea_id
        names = by_trial.get(trial_id, [])
        flags = [fl.FlagResult(n, fl.FIRES, 2.0, 1.0, detail={"top": [{"day": "2026-01-21",
                                                              "return": 0.2}]}) for n in names]
        flags += [fl.FlagResult(n, fl.QUIET, 0.1, 1.0) for n in
                  ("implausible_day", "venue_divergence") if n not in names]
        return ScanResult(trial_id, idea, str(trial.route.value), flags, {"days": 5})

    monkeypatch.setattr(stage, "scan_trial", fake_scan)


def run_stage(world, state=None, **kwargs):
    state = {"done": {}} if state is None else state
    kwargs.setdefault("budget_factory", lambda session: NightBudget.open(
        session, _reading(), now=NOW))
    stage.anomalies_stage(DAY, END, state, config_path=world.config, ledger_path=world.ledger,
                          queue_path=world.queue, **kwargs)
    return state


# --- the code part ------------------------------------------------------------------------


def test_the_code_part_scans_the_runs_that_passed_and_nothing_else(world, monkeypatch) -> None:
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})

    state = run_stage(world, agent=False)

    scanned = {e["idea"] for e in state["anomalies"]["scans"]}
    assert scanned == {"alpha", "beta", "gamma"}  # not delta (reject) and not noise-1
    with db.session_scope() as session:
        rows = {r.idea_id: r for r in session.query(AnomalyScan)}
        assert set(rows) == scanned and rows["alpha"].fired == ["implausible_day"]
        assert rows["alpha"].config_version == "test.1" and rows["beta"].fired == []
    assert state["done"][f"anomalies:scan:{world.ids['alpha']}"]


def test_a_scan_is_made_once_per_trial_and_version_even_in_a_new_night(world, monkeypatch) -> None:
    fire(monkeypatch, world, {})
    run_stage(world, agent=False)
    calls = []
    monkeypatch.setattr(stage, "scan_trial",
                        lambda *a, **k: calls.append(a) or pytest.fail("scanned again"))

    again = run_stage(world, agent=False)  # a new night's state, the same registry

    assert calls == [] and again["anomalies"]["scans"] == []
    with db.session_scope() as session:
        assert session.query(AnomalyScan).count() == 3


def test_a_changed_version_of_the_lines_scans_again(world, monkeypatch) -> None:
    fire(monkeypatch, world, {})
    run_stage(world, agent=False)
    raw = yaml.safe_load(world.config.read_text())
    world.config.write_text(yaml.safe_dump({**raw, "version": "test.2"}), encoding="utf-8")

    state = run_stage(world, agent=False)

    assert len(state["anomalies"]["scans"]) == 3
    with db.session_scope() as session:
        assert {r.config_version for r in session.query(AnomalyScan)} == {"test.1", "test.2"}


def test_at_most_the_nightly_number_of_scans_and_the_rest_wait(world, monkeypatch) -> None:
    fire(monkeypatch, world, {})

    state = run_stage(world, agent=False, max_scans=2)

    assert len(state["anomalies"]["scans"]) == 2 and len(state["anomalies"]["deferred"]) == 1


def test_a_scan_that_fails_is_a_report_line_and_the_night_goes_on(world, monkeypatch) -> None:
    fire(monkeypatch, world, {})
    good = stage.scan_trial

    def flaky(session, trial_id, cfg, **kw):
        if trial_id == world.ids["alpha"]:
            raise RuntimeError("snapshot gone")
        return good(session, trial_id, cfg)

    monkeypatch.setattr(stage, "scan_trial", flaky)

    state = run_stage(world, agent=False)

    errors = [e for e in state["anomalies"]["scans"] if e.get("error")]
    assert [e["idea"] for e in errors] == ["alpha"] and "snapshot gone" in errors[0]["error"]
    assert len(state["anomalies"]["scans"]) == 3
    assert f"anomalies:scan:{world.ids['alpha']}" not in state["done"]  # tried again tomorrow


def test_the_stage_is_off_without_its_config_or_when_disabled(world, monkeypatch) -> None:
    assert run_stage(world, agent=False) is not None
    state = {"done": {}}
    stage.anomalies_stage(DAY, END, state, config_path=world.tmp / "missing.yaml", agent=False)
    assert "anomalies" not in state

    raw = yaml.safe_load(world.config.read_text())
    world.config.write_text(yaml.safe_dump({**raw, "enabled": False}), encoding="utf-8")
    state = {"done": {}}
    stage.anomalies_stage(DAY, END, state, config_path=world.config, agent=False)
    assert "anomalies" not in state


def test_a_dry_run_writes_and_runs_nothing(world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    args_log = fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])

    state = run_stage(world, dry_run=True)

    assert {e["idea"] for e in state["anomalies"]["scans"]} == {"alpha", "beta", "gamma"}
    assert all(e["action"] == "would-scan" for e in state["anomalies"]["scans"])
    assert not args_log.exists() and not world.ledger.exists()
    with db.session_scope() as session:
        assert session.query(AnomalyScan).count() == 0


# --- the agent part -----------------------------------------------------------------------


def test_passed_ideas_are_investigated_first_and_only_up_to_the_nightly_limit(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"],
                              "gamma": ["implausible_day", "venue_divergence"],
                              "beta": ["implausible_day", "venue_divergence", "stand_vs_paper"]})
    fake_claude(stream(GOOD, tokens=60_000), calls=[CALL_ONE, CALL_TWO])

    state = run_stage(world)

    done = state["anomalies"]["investigations"]
    # alpha and gamma passed (route paper): gamma has more flags so goes first; beta (shelf,
    # three flags) comes after every passed idea and the limit of two leaves it for tomorrow.
    assert [e["idea"] for e in done] == ["gamma", "alpha"]
    assert all(e["cause"] == "unrealistic-tradability" for e in done)
    with db.session_scope() as session:
        assert session.query(AnomalyInvestigation).count() == 2
        spends = session.query(TokenSpend).all()
        assert {s.stage for s in spends} == {"recheck"} and len(spends) == 2
    ledger = json.loads(world.ledger.read_text())
    assert set(ledger) == {str(world.ids["gamma"]), str(world.ids["alpha"])}
    assert all(v["status"] == "done" for v in ledger.values())


def test_a_run_waiting_for_forward_days_is_scanned_but_never_investigated(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    """Owner, 2026-10-29 (docs/EVALUATION_PATHS.md Р11): the code scans `needs-forward` runs as
    well; the agent (tokens) reads only the routes named for it."""
    with db.session_scope() as session:
        _idea(session, "waiting", status=IdeaStatus.BENCH)
        world.ids["waiting"] = _trial(session, "waiting", route=TrialRoute.NEEDS_FORWARD,
                                      metrics={"ann_return_net": 0.1})
    raw = yaml.safe_load(world.config.read_text(encoding="utf-8"))
    raw["scan"]["routes"] = ["paper", "shelf", "needs-forward"]
    raw["investigator"]["investigate_routes"] = ["paper", "shelf"]
    world.config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    fire(monkeypatch, world, {"waiting": ["implausible_day", "venue_divergence"]})
    fake_claude(stream(GOOD, tokens=60_000), calls=[CALL_ONE, CALL_TWO])

    state = run_stage(world)

    assert "waiting" in {e["idea"] for e in state["anomalies"]["scans"]}
    assert "waiting" not in {e["idea"] for e in state["anomalies"].get("investigations", [])}
    with db.session_scope() as session:
        assert session.query(AnomalyScan).filter_by(idea_id="waiting").count() == 1
        assert session.query(AnomalyInvestigation).count() == 0


def test_a_trial_is_not_investigated_again_under_the_same_flags(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    args_log = fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    run_stage(world)
    first_calls = args_log.read_text().count("--model")

    second_night = run_stage(world)  # a new night, nothing changed

    assert second_night["anomalies"]["investigations"] == []
    assert args_log.read_text().count("--model") == first_calls
    with db.session_scope() as session:
        assert session.query(AnomalyInvestigation).count() == 1


def test_a_ledger_that_is_lost_does_not_repeat_an_investigation(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    run_stage(world)
    world.ledger.unlink()

    again = run_stage(world)

    assert again["anomalies"]["investigations"] == []  # the registry remembers it too


def test_changed_flags_are_a_reason_to_investigate_again(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    run_stage(world)
    with db.session_scope() as session:  # the run is scanned again and another flag fires
        row = session.query(AnomalyScan).filter_by(idea_id="alpha").one()
        row.fired = ["implausible_day", "venue_divergence"]

    again = run_stage(world)

    assert [e["idea"] for e in again["anomalies"]["investigations"]] == ["alpha"]
    with db.session_scope() as session:
        assert session.query(AnomalyInvestigation).filter_by(idea_id="alpha").count() == 2


def test_the_code_part_alone_calls_no_agent(world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    args_log = fake_claude(stream(GOOD))

    state = run_stage(world, agent=False)

    assert not args_log.exists() and state["anomalies"]["investigations"] == []


def test_a_stop_of_the_guard_is_saved_and_the_night_goes_on_tomorrow(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"], "gamma": ["implausible_day"]})
    args_log = fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])

    def late_week(session):  # the weekly window is already past the 70% line
        return NightBudget.open(session, _reading(0.75), now=NOW)

    state = run_stage(world, budget_factory=late_week)

    stopped = state["anomalies"]["investigations"]
    assert len(stopped) == 1 and "stopped" in stopped[0]
    assert not args_log.exists()  # the agent was never called
    assert json.loads(world.ledger.read_text()) == {}  # and nothing was recorded as done
    with db.session_scope() as session:
        assert session.query(AnomalyInvestigation).count() == 0
    tomorrow = run_stage(world)  # the window has room again
    assert len(tomorrow["anomalies"]["investigations"]) == 2


def test_a_night_that_cannot_open_its_budget_says_so(world, monkeypatch) -> None:
    from qlab.budget import NightExhausted

    fire(monkeypatch, world, {"alpha": ["implausible_day"]})

    def no_reading(session):
        raise NightExhausted("the account is not the one named in night/account.yaml")

    state = run_stage(world, budget_factory=no_reading)

    assert any("разбор агентом не начат" in n for n in state["anomalies"]["notes"])


def test_a_failing_agent_is_retried_up_to_the_bound_then_waits_for_the_flags_to_change(
        world, monkeypatch) -> None:
    fire(monkeypatch, world, {"alpha": ["implausible_day"]})
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(world.tmp / "no-such-binary"))

    first = run_stage(world)
    second = run_stage(world)
    third = run_stage(world)

    assert "error" in first["anomalies"]["investigations"][0]
    assert "error" in second["anomalies"]["investigations"][0]
    assert third["anomalies"]["investigations"] == []  # max_attempts = 2 reached
    assert json.loads(world.ledger.read_text())[str(world.ids["alpha"])]["attempts"] == 2


def test_only_names_the_ideas_to_handle(world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day"], "gamma": ["implausible_day"]})
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])

    state = run_stage(world, only={"gamma"})

    assert [e["idea"] for e in state["anomalies"]["scans"]] == ["gamma"]
    assert [e["idea"] for e in state["anomalies"]["investigations"]] == ["gamma"]


# --- the morning report -------------------------------------------------------------------


def test_the_report_says_what_was_found_and_what_the_owner_may_do(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day", "venue_divergence"]})
    fake_claude(stream({**GOOD, "caught_by": [], "new_check": "Сверять список монет с биржей."}),
                calls=[CALL_ONE, CALL_TWO])
    state = run_stage(world)

    with db.session_scope() as session:
        lines = report.report_lines(state["anomalies"], session)

    text = "\n".join(lines)
    assert "Признак ничего не решает" in text
    assert "**alpha**" in text and "сработало 2 из 10" in text
    assert "Разбор агентом-следователем" in text and "результат был бы недостижим" in text
    assert "ни один признак причину не поймал" in text
    assert "предложена проверка: Сверять список монет с биржей." in text
    assert "заявки программисту:" in text and "uv run qlab anomalies show" in text
    assert "ждут решения владельца" in text and "anomalies check" in text


def test_a_quiet_night_has_no_section(world, monkeypatch) -> None:
    assert report.report_lines(None) == [] and report.report_lines({}) == []
    assert report.report_lines({"scans": [], "investigations": [], "notes": []}) == []


def test_the_owner_settles_a_proposed_check_and_the_loop_is_counted(
        world, monkeypatch, fake_claude) -> None:  # noqa: F811
    fire(monkeypatch, world, {"alpha": ["implausible_day", "venue_divergence"],
                              "gamma": ["implausible_day"]})
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    run_stage(world)
    fake_claude(stream({**GOOD, "caught_by": [], "new_check": "Сверять."}),
                calls=[CALL_ONE, CALL_TWO])

    with db.session_scope() as session:
        learned = store.learning(session)
        assert learned["confirmed"] == 2 and learned["caught_by_none"] == 1  # gamma: no flag fired
        assert learned["caught_by"] == {"venue_divergence": 1}
        assert store.pending_checks(session) == []
        row = session.query(AnomalyInvestigation).first()
        row.new_check = "Сверять списки монет."
        row.new_check_status = "proposed"
        session.flush()
        assert [r.id for r in store.pending_checks(session)] == [row.id]
        assert store.set_check_status(session, row.id, "adopted").new_check_status == "adopted"
        assert store.pending_checks(session) == []
        with pytest.raises(ValueError, match="status"):
            store.set_check_status(session, row.id, "maybe")
        with pytest.raises(ValueError, match="no investigation"):
            store.set_check_status(session, 999, "adopted")
    assert NOW < RESET and timedelta(days=1) > timedelta(0)  # keep the shared constants used
    assert datetime(2026, 10, 28, tzinfo=UTC) == NOW
    assert Path(world.config).exists()
