"""The ONE way q-lab calls an agent (docs/BUDGET.md).

CLAUDE.md: «Вызов агента вне бюджетного гарда невозможен по конструкции — как и
прогон без accrual: функция без переданного бюджета падает, а не тратит молча.»

`run_agent` takes a `CandidateBudget` or an `ExplicitNoBudget` and nothing
else; there is no default. Before the call it checks that the cheapest call
still fits; it hands the CLI its own hard stop (`--max-budget-usd`, the
candidate's remaining tokens at the ledger's price) so one runaway agent loop
cannot spend past the cap; after the call it writes the spend to
`token_spend` -- also when the call failed -- and lets the budget read the
windows the call reported. `qlab.budget.test_guard` checks that no other
module in q-lab starts the `claude` CLI.

**The account.** Every start of the CLI goes through `run_claude`, which puts the q-lab
config directory (`night/account.yaml`) into the environment as `CLAUDE_CONFIG_DIR`: an
independent login, apart from the owner's work CLI. Before any agent call `check_account`
asks `claude auth status` (it costs no tokens, and its answer is kept for ten minutes) and
raises `WrongAccount` -- nothing spent -- when nobody is logged in or the email is not the
expected one. The ledger row of a call records the checked email.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.budget.account import (
    CONFIG_DIR_ENV,
    Account,
    load_account,
    same_email,
    wrong_account,
)
from qlab.budget.guard import CandidateBudget, ExplicitNoBudget, Stage, last_reading
from qlab.budget.usage import CallReport, parse_stream
from qlab.registry.models import TokenSpend

CLAUDE_BIN_ENV = "QLAB_CLAUDE_BIN"


@dataclass(frozen=True)
class AgentResult:
    text: str | None
    report: CallReport
    exit_code: int
    account: str | None = None  # the checked email the call ran under


ACCOUNT_CACHE_SECONDS = 600
STATUS_TIMEOUT_SECONDS = 60


def _claude_bin() -> str:
    return os.environ.get(CLAUDE_BIN_ENV, "claude")


def claude_env(account: Account | None = None) -> dict[str, str]:
    """The environment of every `claude` start: the caller's own, with the q-lab config
    directory over `CLAUDE_CONFIG_DIR`."""
    account = account or load_account()
    return {**os.environ, CONFIG_DIR_ENV: str(account.config_dir)}


def run_claude(args: list[str], *, account: Account | None = None, **kwargs
               ) -> subprocess.CompletedProcess:
    """The ONE place the `claude` CLI is started: `args` after the binary, under q-lab's own
    config directory. Anything else in `kwargs` goes to `subprocess.run`."""
    return subprocess.run([_claude_bin(), *args], env=claude_env(account), check=False,
                          **kwargs)


def _read_status_cli(account: Account) -> dict:
    """`claude auth status` under the q-lab config directory: the CLI's own JSON (`loggedIn`,
    `email`, `orgName`, `subscriptionType`). A logged-out CLI prints `"loggedIn": false`, and
    may exit non-zero doing so, so the output is read whatever the exit code."""
    done = run_claude(["auth", "status"], account=account, stdin=subprocess.DEVNULL,
                      capture_output=True, text=True, timeout=STATUS_TIMEOUT_SECONDS)
    status = json.loads(done.stdout)
    if not isinstance(status, dict):
        raise ValueError("not an object")
    return status


def _read_status(account: Account) -> dict:
    return _read_status_cli(account)


_checked: dict[tuple, tuple[float, Account]] = {}


def check_account() -> Account:
    """The account of `night/account.yaml`, after `claude auth status` confirmed the CLI is
    logged in to it. Costs no tokens; a success is kept for `ACCOUNT_CACHE_SECONDS` (a refusal
    never is, so the owner's login takes effect at once). Raises `WrongAccount`."""
    expected = load_account()
    key = (_claude_bin(), str(expected.config_dir), expected.email)
    now = time.monotonic()
    hit = _checked.get(key)
    if hit is not None and now - hit[0] < ACCOUNT_CACHE_SECONDS:
        return hit[1]
    try:
        status = _read_status(expected)
    except (subprocess.TimeoutExpired, ValueError) as exc:  # a bad answer is not a login
        raise wrong_account(expected, None, detail=(
            f"не удалось прочитать `claude auth status` ({type(exc).__name__})")) from exc
    found = status.get("email") if status.get("loggedIn") else None
    if not same_email(found, expected.email):
        raise wrong_account(expected, found)
    _checked[key] = (now, expected)
    return expected


def open_candidate(session, idea_id: str | None, stage: Stage) -> CandidateBudget:
    """A one-candidate budget for a run started by hand: tonight's ceiling
    from the latest reading of the shared windows, refreshed by a probe when
    that reading is older than the five-hour window's own length."""
    from datetime import timedelta

    from qlab.budget.guard import NightBudget
    from qlab.budget.usage import UsageReading, Window

    # Only this account's rows: another account's windows are another subscription's.
    last = last_reading(session, check_account().email)
    now = datetime.now(UTC)
    if last is not None and last.at.replace(tzinfo=UTC) > now - timedelta(hours=5):
        reading = UsageReading("allowed", None, Window(
            last.seven_day_util, last.seven_day_resets_at.replace(tzinfo=UTC)))
    else:
        reading = probe(session).report.reading
    night = NightBudget.open(session, reading, now=now)
    return night.stage(stage, candidates=1).candidate(idea_id)


def run_agent(
    budget: CandidateBudget | ExplicitNoBudget,
    prompt: str,
    *,
    session: Session,
    agent: str,
    model: str,
    max_turns: int,
    cwd: Path | None = None,
    extra_args: tuple[str, ...] = (),
) -> AgentResult:
    if not isinstance(budget, CandidateBudget | ExplicitNoBudget):
        raise TypeError(
            "run_agent needs a CandidateBudget, or ExplicitNoBudget(reason) to refuse one "
            f"explicitly; got {type(budget).__name__}"
        )
    account = check_account()  # before anything else: a wrong account spends nothing
    # No MCP servers: no q-lab agent uses one, and the user's servers' tool
    # definitions otherwise ride along in every call -- ~140k tokens measured
    # 2026-10-24 on a one-word probe, ~1.3k without them.
    args = ["-p", prompt, "--output-format", "stream-json", "--verbose",
            "--model", model, "--max-turns", str(max_turns), "--strict-mcp-config",
            *extra_args]
    if isinstance(budget, CandidateBudget):
        budget.check()
        from qlab.budget.guard import usd_per_token

        if not budget.night.limits_off:  # the owner turned the guard's caps off
            usd_cap = budget.remaining() * usd_per_token(session, model)
            args += ["--max-budget-usd", f"{usd_cap:.4f}"]
        stage, idea_id = budget.stage.name.lower(), budget.idea_id
    else:
        stage, idea_id = "no-budget", None

    # A call takes minutes. The caller's session may hold the registry's one write lock (a row it
    # flushed, the previous call's token row); steps running together (docs/PIPELINE.md) would wait
    # out their busy timeout and fail, so what is pending is committed before the call starts.
    session.commit()
    done = run_claude(args, account=account, cwd=cwd, stdin=subprocess.DEVNULL,
                      capture_output=True, text=True)
    report = parse_stream(done.stdout.splitlines())
    usage, reading = report.usage, report.reading
    status = "ok" if done.returncode == 0 and not report.is_error else f"failed ({done.returncode})"
    if isinstance(budget, ExplicitNoBudget):
        status += f"; no budget: {budget.reason}"
    session.add(TokenSpend(
        at=datetime.now(UTC),
        stage=stage,
        idea_id=idea_id,
        agent=agent,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
        usd_est=usage.cost_usd if usage else 0.0,
        model=usage.model if usage else model,
        tokens_cache_write=usage.cache_write if usage else None,
        tokens_cache_read=usage.cache_read if usage else None,
        five_hour_util=reading.five_hour.utilization if reading and reading.five_hour else None,
        seven_day_util=reading.seven_day.utilization if reading and reading.seven_day else None,
        seven_day_resets_at=reading.seven_day.resets_at if reading and reading.seven_day else None,
        status=status,
        account=account.email,
    ))
    session.commit()  # the token row is kept whatever the caller does next, and the lock let go
    if isinstance(budget, CandidateBudget):
        budget.record(usage.total if usage else 0, reading)
    return AgentResult(text=report.text, report=report, exit_code=done.returncode,
                       account=account.email)


def probe(session: Session) -> AgentResult:
    """The cheapest call there is, to read the shared windows. It is spent
    like any other call: explicitly outside a night budget, and recorded."""
    # Its own one-line system prompt and no tools: the call then costs a few
    # hundred tokens instead of the CLI's own ~40k-token prompt.
    return run_agent(ExplicitNoBudget("reading the subscription windows"),
                     "Reply with exactly: OK", session=session, agent="probe",
                     model="haiku", max_turns=1,
                     extra_args=("--system-prompt", "Answer in one word.", "--tools", ""))


__all__ = ["ACCOUNT_CACHE_SECONDS", "CLAUDE_BIN_ENV", "AgentResult", "check_account",
           "claude_env", "open_candidate", "probe", "run_agent", "run_claude"]
