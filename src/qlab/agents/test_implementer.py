"""Tests for the implementer agent's guards, with a fake `claude` that writes
files the way the real agent would."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import qlab.agents.implementer as impl
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.registry.models import Base

SPEC = """idea_id: fake-card
title: "Тест"
code_ref: "qlab.strategies.trend:TrendTSMOMEnsemble"
params: {lookbacks_days: [30]}
data: {source: hyperliquid, interval: 1d, start: 2025-01-01, end: 2026-09-20}
costs: {taker_fee_bps: 3.5, slippage_bps: 0.9}
min_leg_notional: 10.0
unexpressed_mechanisms: []
sources:
  lookbacks_days: "owner 2026-10-03: test fixture"
  costs: "owner 2026-10-03: test fixture"
  min_leg_notional: "owner 2026-10-03: test fixture"
"""


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=tmp_path, check=True)
    monkeypatch.setattr(impl, "QLAB_ROOT", tmp_path)
    (tmp_path / "card.yaml").write_text("idea_id: fake-card\ntitle: x\n")
    return tmp_path


def _fake_claude(tmp_path, monkeypatch, write_spec: bool) -> None:
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    result = {"type": "result", "is_error": False, "result": "FakeStrategy",
              "total_cost_usd": 0.01, "usage": {"input_tokens": 100, "output_tokens": 10},
              "modelUsage": {"claude-sonnet": {}}}
    rate = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
        "seven_day": {"utilization": 0.2, "resetsAt": reset}}}}
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps(rate) + "\n" + json.dumps(result) + "\n")
    (tmp_path / "spec.txt").write_text(SPEC)
    spec_cmd = ('mkdir -p specs/agent && cp spec.txt specs/agent/fake-card.yaml\n'
                if write_spec else "")
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        "mkdir -p src/qlab/strategies/agent\n"
        "echo 'x = 1' > src/qlab/strategies/agent/fake_card.py\n"
        f"{spec_cmd}"
        "echo 'stray' > STRAY.txt\n"
        f'cat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def test_writes_outside_its_two_files_are_reverted_and_fail(repo, session, monkeypatch) -> None:
    _fake_claude(repo, monkeypatch, write_spec=True)
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"),
                             session=session)
    assert not (repo / "STRAY.txt").exists()
    assert "STRAY.txt" in outcome.reverted
    assert any("outside" in p for p in outcome.problems)
    assert (repo / "specs/agent/fake-card.yaml").exists()  # its own files stay


def test_a_missing_spec_is_a_failed_attempt(repo, session, monkeypatch) -> None:
    _fake_claude(repo, monkeypatch, write_spec=False)
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"),
                             session=session)
    assert "the agent did not write both files" in outcome.problems
