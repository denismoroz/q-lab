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
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.budget.guard import CandidateBudget, ExplicitNoBudget
from qlab.budget.usage import CallReport, parse_stream
from qlab.registry.models import TokenSpend

CLAUDE_BIN_ENV = "QLAB_CLAUDE_BIN"


@dataclass(frozen=True)
class AgentResult:
    text: str | None
    report: CallReport
    exit_code: int


def _claude_bin() -> str:
    return os.environ.get(CLAUDE_BIN_ENV, "claude")


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
    args = [_claude_bin(), "-p", prompt, "--output-format", "stream-json", "--verbose",
            "--model", model, "--max-turns", str(max_turns), *extra_args]
    if isinstance(budget, CandidateBudget):
        budget.check()
        usd_cap = budget.remaining() * budget.night.calibration.usd_per_token
        args += ["--max-budget-usd", f"{usd_cap:.4f}"]
        stage, idea_id = budget.stage.name.lower(), budget.idea_id
    else:
        stage, idea_id = "no-budget", None

    done = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, check=False)
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
    ))
    session.flush()
    if isinstance(budget, CandidateBudget):
        budget.record(usage.total if usage else 0, reading)
    return AgentResult(text=report.text, report=report, exit_code=done.returncode)


def probe(session: Session) -> AgentResult:
    """The cheapest call there is, to read the shared windows. It is spent
    like any other call: explicitly outside a night budget, and recorded."""
    return run_agent(ExplicitNoBudget("reading the subscription windows"),
                     "Reply with exactly: OK", session=session, agent="probe",
                     model="haiku", max_turns=1)


__all__ = ["CLAUDE_BIN_ENV", "AgentResult", "probe", "run_agent"]
