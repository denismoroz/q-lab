"""Tests for the budget guard (`qlab.budget`), with a fake `claude` binary."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.budget import (
    CandidateExhausted,
    ExplicitNoBudget,
    NightBudget,
    NightExhausted,
    Stage,
    calibrate,
    run_agent,
)
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.budget.guard import PRIOR_UTIL_PER_TOKEN, WEEKLY_CEILING
from qlab.budget.usage import UsageReading, Window, parse_stream
from qlab.registry.models import Base, TokenSpend

NOW = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)
RESET = NOW + timedelta(days=5)


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _reading(week: float, five: float = 0.1, status: str = "allowed") -> UsageReading:
    return UsageReading(status, Window(five, NOW + timedelta(hours=3)), Window(week, RESET))


def _events(week: float, tokens: int, status: str = "allowed") -> str:
    rate = {"type": "rate_limit_event", "rate_limit_info": {"status": status, "unifiedWindows": {
        "five_hour": {"utilization": 0.1, "resetsAt": int((NOW + timedelta(hours=3)).timestamp())},
        "seven_day": {"utilization": week, "resetsAt": int(RESET.timestamp())}}}}
    result = {"type": "result", "is_error": False, "result": "OK", "total_cost_usd": tokens * 2e-6,
              "usage": {"input_tokens": 10, "output_tokens": 40,
                        "cache_creation_input_tokens": tokens - 50, "cache_read_input_tokens": 0},
              "modelUsage": {"claude-haiku-4-5": {}}}
    return json.dumps(rate) + "\n" + json.dumps(result) + "\n"


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    """A `claude` that prints the stream-json in $FAKE_OUT and logs its args."""
    script = tmp_path / "claude"
    script.write_text('#!/bin/sh\necho "$@" >> "$FAKE_ARGS"\ncat "$FAKE_OUT"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    monkeypatch.setenv("FAKE_ARGS", str(tmp_path / "args.log"))

    def set_output(text: str) -> Path:
        out = tmp_path / "out.jsonl"
        out.write_text(text)
        monkeypatch.setenv("FAKE_OUT", str(out))
        return tmp_path / "args.log"

    return set_output


def test_parse_stream_reads_windows_and_usage() -> None:
    report = parse_stream(_events(0.17, 40_000).splitlines())
    assert report.reading.seven_day.utilization == 0.17
    assert report.reading.status == "allowed"
    assert report.usage.total == 40_000 and report.text == "OK"


def test_night_target_follows_the_formula(session) -> None:
    night = NightBudget.open(session, _reading(0.20), now=NOW)
    assert night.target_util == pytest.approx(0.20 + (WEEKLY_CEILING - 0.20) / 5)
    assert night.calibration.source == "prior"
    late = NightBudget.open(session, _reading(0.75), now=NOW)  # already past 70%
    assert late.target_util == pytest.approx(0.75)
    with pytest.raises(NightExhausted):
        late.check()


def test_no_reading_no_night(session) -> None:
    with pytest.raises(NightExhausted, match="does not guess"):
        NightBudget.open(session, UsageReading("allowed", None, None), now=NOW)


def test_stage_shares_what_is_left_among_waiting_candidates(session) -> None:
    night = NightBudget.open(session, _reading(0.20), now=NOW)
    stage = night.stage(Stage.IMPLEMENT, candidates=4)
    first = stage.candidate("a")
    assert first.cap_tokens == night.tokens_left() // 4
    first.record(first.cap_tokens // 2, _reading(0.20))  # spends half, unused share flows on
    second = stage.candidate("b")
    assert second.cap_tokens == night.tokens_left() // 3


def test_run_agent_requires_a_budget(session, fake_claude) -> None:
    fake_claude(_events(0.2, 40_000))
    with pytest.raises(TypeError, match="ExplicitNoBudget"):
        run_agent(None, "hi", session=session, agent="x", model="haiku", max_turns=1)
    with pytest.raises(ValueError, match="reason"):
        ExplicitNoBudget("  ")


def test_call_is_recorded_capped_and_the_night_stops_at_its_target(session, fake_claude) -> None:
    night = NightBudget.open(session, _reading(0.20), now=NOW)
    candidate = night.stage(Stage.IMPLEMENT, candidates=1).candidate("idea")
    args_log = fake_claude(_events(0.21, 50_000))
    run_agent(candidate, "work", session=session, agent="implementer", model="haiku", max_turns=3)
    row = session.query(TokenSpend).one()
    assert (row.stage, row.idea_id, row.seven_day_util, row.status) == (
        "implement", "idea", 0.21, "ok")
    assert candidate.spent_tokens == 50_000
    assert "--max-budget-usd" in args_log.read_text()

    fake_claude(_events(night.target_util + 0.01, 50_000))  # the call crosses tonight's target
    run_agent(candidate, "more", session=session, agent="implementer", model="haiku",
              max_turns=3)
    with pytest.raises(NightExhausted, match="target"):
        candidate.check()


def test_a_candidate_stops_at_its_cap_without_stopping_the_night(session, fake_claude) -> None:
    night = NightBudget.open(session, _reading(0.20), now=NOW)
    stage = night.stage(Stage.IMPLEMENT, candidates=2)
    greedy = stage.candidate("greedy")
    fake_claude(_events(0.20, greedy.cap_tokens))
    run_agent(greedy, "x", session=session, agent="a", model="haiku", max_turns=1)
    with pytest.raises(CandidateExhausted):
        greedy.check()
    stage.candidate("next").check()  # the night goes on


def test_a_refused_window_stops_the_night(session, fake_claude) -> None:
    night = NightBudget.open(session, _reading(0.20), now=NOW)
    candidate = night.stage(Stage.SEARCH, candidates=1).candidate(None)
    fake_claude(_events(0.20, 40_000, status="rejected"))
    run_agent(candidate, "x", session=session, agent="scout", model="haiku", max_turns=1)
    with pytest.raises(NightExhausted, match="rejected"):
        night.check()


def test_calibration_moves_from_prior_to_ledger_and_starts_cautious(session, fake_claude) -> None:
    assert calibrate(session).util_per_token == PRIOR_UTIL_PER_TOKEN
    for week in (0.20, 0.20):
        fake_claude(_events(week, 40_000))
        run_agent(ExplicitNoBudget("test"), "x", session=session, agent="a", model="haiku",
                  max_turns=1)
    cal = calibrate(session)
    assert cal.source == "ledger (1 pairs)"
    assert cal.util_per_token == pytest.approx(0.01 / 40_000)  # resolution over tokens: cautious
    assert cal.min_call_tokens == 40_000


def _code_strings(source: str) -> list[str]:
    """String literals of a module, docstrings left out."""
    import ast

    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant):
            docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_no_other_module_starts_the_claude_cli() -> None:
    """By construction (CLAUDE.md): only qlab.budget.agent may run `claude`."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if path.name.startswith("test_") or path == root / "budget" / "agent.py":
            continue
        for text in _code_strings(path.read_text()):
            if text.strip() == "claude" or text.startswith("claude ") or "claude -p" in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_the_price_per_token_is_the_calling_models_own(session) -> None:
    from qlab.budget.guard import usd_per_token

    session.add_all([
        TokenSpend(at=NOW, stage="x", agent="a", tokens_in=1000, tokens_out=0, usd_est=0.001,
                   model="claude-haiku-4-5"),
        TokenSpend(at=NOW, stage="x", agent="b", tokens_in=1000, tokens_out=0, usd_est=0.010,
                   model="claude-opus-5"),
    ])
    session.flush()
    assert usd_per_token(session, "opus") == pytest.approx(1e-5)
    assert usd_per_token(session, "haiku") == pytest.approx(1e-6)
    assert usd_per_token(session, "sonnet") == pytest.approx(1e-5)  # unknown: the highest seen
