"""Tests for the account check (`qlab.budget.account`, `qlab.budget.agent.check_account`):
the agents run under the account of night/account.yaml or not at all. The real CLI is never
called: `conftest.py` puts a logged-in test account over `agent._read_status`, and a test of
the spawning itself uses a fake `claude` script."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import qlab.budget.account as account_module
import qlab.budget.agent as agent
from qlab.budget import ExplicitNoBudget, NightBudget, NightExhausted, WrongAccount, run_agent
from qlab.budget.agent import CLAUDE_BIN_ENV, check_account, open_candidate
from qlab.budget.guard import PRIOR_UTIL_PER_TOKEN, Stage, calibrate, last_reading
from qlab.budget.usage import UsageReading, Window
from qlab.registry.models import Base, TokenSpend

MINE = "moroz.denis@gmail.com"
WORK = "denis.moroz@toptal.com"
NOW = datetime(2026, 10, 28, 0, 0, tzinfo=UTC)
RESET = NOW + timedelta(days=2)


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


@pytest.fixture()
def config_dir(tmp_path, monkeypatch) -> Path:
    """A night/account.yaml of the test, with a config directory inside the temp folder."""
    path = tmp_path / "account.yaml"
    path.write_text(f"email: {MINE}\nconfig_dir: {tmp_path}/claude-qlab\n")
    monkeypatch.setattr(account_module, "ACCOUNT_FILE", path)
    return tmp_path / "claude-qlab"


def _logged_in_as(monkeypatch, email: str | None) -> list[object]:
    """Make `claude auth status` answer for `email` (None: nobody is logged in)."""
    calls: list[object] = []

    def status(account) -> dict:
        calls.append(account)
        if email is None:
            return {"loggedIn": False}
        return {"loggedIn": True, "email": email, "orgName": "x", "subscriptionType": "pro"}

    monkeypatch.setattr(agent, "_read_status", status)
    monkeypatch.setattr(agent, "_checked", {})
    return calls


def _stream(week: float = 0.2) -> str:
    rate = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed",
            "unifiedWindows": {
                "five_hour": {"utilization": 0.1,
                              "resetsAt": int((NOW + timedelta(hours=3)).timestamp())},
                "seven_day": {"utilization": week, "resetsAt": int(RESET.timestamp())}}}}
    result = {"type": "result", "is_error": False, "result": "OK", "total_cost_usd": 0.08,
              "usage": {"input_tokens": 10, "output_tokens": 40,
                        "cache_creation_input_tokens": 39_950, "cache_read_input_tokens": 0},
              "modelUsage": {"claude-haiku-4-5": {}}}
    return json.dumps(rate) + "\n" + json.dumps(result) + "\n"


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    """A `claude` that logs its arguments and the environment it was started in, and prints
    the stream-json in $FAKE_OUT (or, for `auth status`, the JSON in $FAKE_STATUS)."""
    script = tmp_path / "claude"
    script.write_text(
        '#!/bin/sh\n'
        'echo "$@" >> "$FAKE_LOG"\n'
        'echo "config=$CLAUDE_CONFIG_DIR kept=$QLAB_TEST_KEPT" >> "$FAKE_LOG"\n'
        'if [ "$1" = auth ]; then cat "$FAKE_STATUS"; else cat "$FAKE_OUT"; fi\n')
    script.chmod(0o755)
    log = tmp_path / "calls.log"
    (tmp_path / "out.jsonl").write_text(_stream())
    (tmp_path / "status.json").write_text(json.dumps({"loggedIn": True, "email": MINE}))
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out.jsonl"))
    monkeypatch.setenv("FAKE_STATUS", str(tmp_path / "status.json"))
    monkeypatch.setenv("QLAB_TEST_KEPT", "yes")
    return log


def _ledger_row(session, account: str | None, week: float, *, minutes: int = 0,
                tokens: int = 40_000) -> None:
    session.add(TokenSpend(
        at=NOW + timedelta(minutes=minutes), stage="x", agent="a", tokens_in=10, tokens_out=40,
        tokens_cache_write=tokens - 50, tokens_cache_read=0, usd_est=0.08, model="claude-haiku-4-5",
        five_hour_util=0.1, seven_day_util=week, seven_day_resets_at=RESET, status="ok",
        account=account))
    session.flush()


def test_the_right_account_passes_and_the_call_records_it(session, config_dir, monkeypatch,
                                                           fake_claude) -> None:
    _logged_in_as(monkeypatch, MINE)
    result = run_agent(ExplicitNoBudget("test"), "x", session=session, agent="a", model="haiku",
                       max_turns=1)
    assert result.account == MINE
    assert session.query(TokenSpend).one().account == MINE


def test_a_wrong_account_refuses_before_any_spend(session, config_dir, monkeypatch,
                                                   fake_claude) -> None:
    _logged_in_as(monkeypatch, WORK)
    with pytest.raises(WrongAccount) as refusal:
        run_agent(ExplicitNoBudget("test"), "x", session=session, agent="a", model="haiku",
                  max_turns=1)
    assert not fake_claude.exists()  # the agent's binary was not started
    assert session.query(TokenSpend).count() == 0
    message = str(refusal.value)
    assert WORK in message and MINE in message
    assert "CLAUDE_CONFIG_DIR=" in message and "claude auth login" in message
    assert (refusal.value.found, refusal.value.expected) == (WORK, MINE)


def test_the_login_command_is_the_one_the_config_names(session, config_dir, monkeypatch) -> None:
    _logged_in_as(monkeypatch, WORK)
    with pytest.raises(WrongAccount, match=f"CLAUDE_CONFIG_DIR={config_dir} claude auth login"):
        check_account()


def test_the_shipped_account_file_names_the_owners_own_account() -> None:
    loaded = account_module.load_account(Path(__file__).resolve().parents[3] / "night"
                                         / "account.yaml")
    assert loaded.email == MINE
    assert loaded.config_dir == Path("~/.claude-qlab").expanduser()
    assert loaded.login_command == "CLAUDE_CONFIG_DIR=~/.claude-qlab claude auth login"


def test_nobody_logged_in_refuses(session, config_dir, monkeypatch, fake_claude) -> None:
    _logged_in_as(monkeypatch, None)
    with pytest.raises(WrongAccount, match="никто не вошёл") as refusal:
        run_agent(ExplicitNoBudget("test"), "x", session=session, agent="a", model="haiku",
                  max_turns=1)
    assert refusal.value.found is None
    assert session.query(TokenSpend).count() == 0 and not fake_claude.exists()


def test_the_email_is_compared_without_regard_to_case(config_dir, monkeypatch) -> None:
    _logged_in_as(monkeypatch, MINE.upper())
    assert check_account().email == MINE


def test_a_refusal_stops_the_night_like_its_budget_does(session, config_dir, monkeypatch) -> None:
    # Every agent stage catches NightExhausted and saves its state; WrongAccount rides on it.
    assert issubclass(WrongAccount, NightExhausted)
    _logged_in_as(monkeypatch, WORK)
    with pytest.raises(NightExhausted):
        agent.probe(session)
    with pytest.raises(NightExhausted):
        NightBudget.open(session, UsageReading("allowed", None, Window(0.2, RESET)), now=NOW)


def test_a_success_is_kept_for_ten_minutes_and_a_refusal_is_not(config_dir, monkeypatch) -> None:
    calls = _logged_in_as(monkeypatch, MINE)
    check_account()
    check_account()
    assert len(calls) == 1
    (key,) = agent._checked
    stamp, account = agent._checked[key]
    agent._checked[key] = (stamp - agent.ACCOUNT_CACHE_SECONDS - 1, account)  # ten minutes on
    check_account()
    assert len(calls) == 2

    calls = _logged_in_as(monkeypatch, None)
    for _ in range(2):
        with pytest.raises(WrongAccount):
            check_account()
    assert len(calls) == 2  # asked again each time: the owner's login takes effect at once


def test_the_config_directory_reaches_the_agent_and_the_status_call(
        session, config_dir, monkeypatch, fake_claude) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/the/owners/work/config")  # must be overridden
    monkeypatch.setattr(agent, "_read_status", agent._read_status_cli)  # the real spawning
    run_agent(ExplicitNoBudget("test"), "x", session=session, agent="a", model="haiku",
              max_turns=1)
    lines = fake_claude.read_text().splitlines()
    status_call, status_env, agent_call, agent_env = lines
    assert status_call == "auth status"
    assert agent_call.startswith("-p x")
    for env in (status_env, agent_env):
        assert env == f"config={config_dir} kept=yes"  # the rest of the environment stays


def test_unreadable_status_is_a_refusal_not_a_crash(config_dir, monkeypatch, fake_claude,
                                                     tmp_path) -> None:
    (tmp_path / "status.json").write_text("Error: not json")
    monkeypatch.setattr(agent, "_read_status", agent._read_status_cli)
    with pytest.raises(WrongAccount, match="не удалось прочитать"):
        check_account()


def test_a_logged_out_cli_that_exits_non_zero_is_still_read(config_dir, monkeypatch, tmp_path
                                                            ) -> None:
    script = tmp_path / "claude"
    script.write_text('#!/bin/sh\necho \'{"loggedIn": false}\'\nexit 1\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    monkeypatch.setattr(agent, "_read_status", agent._read_status_cli)
    with pytest.raises(WrongAccount, match="никто не вошёл"):
        check_account()


def test_calibration_ignores_other_accounts_rows(session, config_dir) -> None:
    # The work account's Team windows: a percent there is many tokens. Mine has no rows.
    _ledger_row(session, WORK, 0.20)
    _ledger_row(session, WORK, 0.20, minutes=5)
    assert calibrate(session, since=None, account=WORK).source == "ledger (1 pairs)"
    mine = calibrate(session, since=None)  # the current account by default
    assert mine.source == "prior" and mine.util_per_token == PRIOR_UTIL_PER_TOKEN
    _ledger_row(session, MINE, 0.30, minutes=10)
    _ledger_row(session, MINE, 0.30, minutes=15)
    assert calibrate(session, since=None).source == "ledger (1 pairs)"  # its own pair only


def test_an_account_with_no_rows_gets_the_conservative_default(session, config_dir) -> None:
    for minutes in (0, 5, 10):
        _ledger_row(session, WORK, 0.20, minutes=minutes)  # a rich Team ledger
    night = NightBudget.open(session, UsageReading("allowed", None, Window(0.20, RESET)), now=NOW)
    assert night.calibration.source == "prior"
    assert night.calibration.util_per_token == PRIOR_UTIL_PER_TOKEN
    # about 470k tokens per +4 pp of the week, not the Team ledger's ~1 pp per million
    assert night.tokens_left() == int((night.target_util - 0.20) / PRIOR_UTIL_PER_TOKEN)


def test_the_last_reading_ignores_other_accounts_rows(session, config_dir) -> None:
    _ledger_row(session, MINE, 0.10)
    _ledger_row(session, WORK, 0.90, minutes=30)  # newer, but another subscription's window
    _ledger_row(session, None, 0.95, minutes=40)
    assert last_reading(session).seven_day_util == 0.10
    assert last_reading(session, WORK).seven_day_util == 0.90


def test_no_reading_of_this_account_means_a_probe_never_a_guess(
        session, config_dir, monkeypatch, fake_claude) -> None:
    # A fresh Team reading is in the ledger; for the current account there is none, so the
    # run asks the CLI (the probe) instead of taking the other account's windows.
    _ledger_row(session, WORK, 0.90, minutes=0)
    session.query(TokenSpend).update({"at": datetime.now(UTC)})
    session.flush()
    candidate = open_candidate(session, None, Stage.SEARCH)
    probe_rows = session.query(TokenSpend).filter(TokenSpend.agent == "probe").all()
    assert len(probe_rows) == 1 and probe_rows[0].account == MINE
    assert candidate.night.start_util == pytest.approx(0.2)  # the probe's reading, not 0.90
    # and with no reading at all the guard refuses outright
    with pytest.raises(NightExhausted, match="does not guess"):
        NightBudget.open(session, UsageReading("allowed", None, None), now=NOW)


def test_the_night_state_and_report_name_the_account(tmp_path, monkeypatch, config_dir) -> None:
    import qlab.night as night

    _logged_in_as(monkeypatch, WORK)
    state = night.account_state()
    assert state["ok"] is False and (state["found"], state["expected"]) == (WORK, MINE)
    assert "claude auth login" in state["message"]
    _logged_in_as(monkeypatch, MINE)
    assert night.account_state() == {"ok": True, "email": MINE}


def test_the_morning_report_shows_a_wrong_account(tmp_path, monkeypatch, config_dir) -> None:
    import qlab.night as night

    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_token_lines", lambda day: "Токены за эту ночь не тратились.")
    _logged_in_as(monkeypatch, WORK)
    state = {"results": [], "done": {}, "account": night.account_state()}
    text = night.write_report(datetime(2026, 10, 28).date(), state).read_text()
    assert "не под тем аккаунтом" in text and WORK in text and MINE in text
    assert "CLAUDE_CONFIG_DIR=" in text
    assert "Аккаунт Claude CLI не тот" in text.split("## Токены")[1]


def test_the_ledger_section_of_the_report_names_the_account(tmp_path, monkeypatch) -> None:
    import qlab.night as night
    import qlab.registry.db as db_module
    from qlab.registry.db import session_scope

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "q.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    with session_scope() as s:
        s.add(TokenSpend(at=datetime(2026, 10, 28, 3, tzinfo=UTC), stage="search", agent="a",
                         tokens_in=100, tokens_out=0, usd_est=0.0, account=MINE))
    assert night._token_lines(datetime(2026, 10, 28).date()).startswith(f"Аккаунт: {MINE}.")


# -- the migration -------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _keep_loggers_enabled():
    """Alembic's `fileConfig` disables the loggers that exist; the other tests need theirs."""
    before = {name: logger.disabled for name, logger in logging.Logger.manager.loggerDict.items()
              if isinstance(logger, logging.Logger)}
    yield
    for name, disabled in before.items():
        logging.getLogger(name).disabled = disabled


def _alembic(tmp_path: Path, monkeypatch, target: str) -> Path:
    from alembic import command
    from alembic.config import Config

    db = tmp_path / "migrate.db"
    monkeypatch.setenv("QLAB_DB", str(db))
    config = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    command.upgrade(config, target)
    return db


def test_the_migration_applies_to_a_fresh_database(tmp_path, monkeypatch) -> None:
    db = _alembic(tmp_path, monkeypatch, "head")
    with sqlite3.connect(db) as con:
        columns = {row[1] for row in con.execute("pragma table_info(token_spend)")}
        assert "account" in columns
        assert con.execute("select count(*) from token_spend").fetchone() == (0,)


def test_the_migration_marks_the_existing_rows_as_the_work_account(tmp_path, monkeypatch) -> None:
    from alembic import command
    from alembic.config import Config

    db = _alembic(tmp_path, monkeypatch, "e3f7c1d9a204")  # the revision before
    with sqlite3.connect(db) as con:
        assert "account" not in {row[1] for row in con.execute("pragma table_info(token_spend)")}
        con.execute(
            "insert into token_spend (at, stage, agent, tokens_in, tokens_out, usd_est) "
            "values ('2026-10-08 03:00:00', 'implement', 'researcher', 10, 40, 0.08)")
        con.execute(
            "insert into token_spend (at, stage, agent, tokens_in, tokens_out, usd_est) "
            "values ('2026-10-27 03:00:00', 'search', 'scout', 10, 40, 0.08)")
    command.upgrade(Config(str(Path(__file__).resolve().parents[3] / "alembic.ini")), "head")
    with sqlite3.connect(db) as con:
        assert con.execute("select account, count(*) from token_spend group by 1").fetchall() == [
            (WORK, 2)]
    command.downgrade(Config(str(Path(__file__).resolve().parents[3] / "alembic.ini")),
                      "e3f7c1d9a204")
    with sqlite3.connect(db) as con:
        assert "account" not in {row[1] for row in con.execute("pragma table_info(token_spend)")}
        assert con.execute("select count(*) from token_spend").fetchone() == (2,)


def test_no_other_module_spawns_the_claude_binary_without_the_config_directory() -> None:
    """`subprocess` in qlab.budget.agent is used by `run_claude` alone, and it always passes
    the environment with `CLAUDE_CONFIG_DIR`."""
    source = Path(agent.__file__).read_text()
    assert source.count("subprocess.run(") == 1
    assert "env=claude_env(account)" in source
