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


# ---- a card from the ideas inbox (cards/SCHEMA.md, «Карточка из входящих идей») -----------

INBOX_CARD = """idea_id: toy-idea
title: Игрушка
plain_summary: Держим монеты, которые выросли.
idea_text: купить то, что выросло за неделю
baseline_spec: base.yaml
grid:
  params:
    signal.window_days: [7, 14, 30]
new_block:
  name: weekly-momentum
  params:
    - {name: window_days, meaning: сколько суток считать рост}
  rule: держим монеты, которые сильнее всего выросли за окно
data: {venue: binance, interval: 1d, coins: [BTC, ETH], honest_from: 2022-01-01}
"""
BLOCK_SPEC = """idea_id: toy-idea
title: "Игрушка"
code_ref: "qlab.strategies.agent.weekly_momentum:WeeklyMomentum"
params: {signal: {window_days: %(window)s}}
data: {source: binance, interval: 1d, start: 2022-01-01, end: 2026-09-20}
costs: {taker_fee_bps: 3.5, slippage_bps: 0.9}
min_leg_notional: 10.0
unexpressed_mechanisms: []
sources:
  signal.window_days: "card.yaml"
  costs: "card.yaml"
  min_leg_notional: "card.yaml"
"""
BLOCK_CODE = "window = params['signal']['window_days']\n"


def _args(repo):
    """Where the fake agent keeps what it was called with: outside the repo, so that guard 1
    does not see (and revert) it."""
    return repo.parent / f"{repo.name}-args.txt"


@pytest.fixture()
def inbox(repo, session, monkeypatch):
    """A card from the inbox in the temporary repo and a fake agent that writes the block, the
    spec and keeps the arguments it was called with."""
    (repo / "base.yaml").write_text("idea_id: base\n")
    (repo / "card.yaml").write_text(INBOX_CARD)

    def agent(spec_text: str = BLOCK_SPEC % {"window": 7}, code: str = BLOCK_CODE) -> None:
        _fake_claude(repo, monkeypatch, write_spec=False)
        (repo / "spec.txt").write_text(spec_text)
        (repo / "code.txt").write_text(code)
        script = repo / "claude"
        script.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$@" > "{_args(repo)}"\n'
            "mkdir -p src/qlab/strategies/agent specs/agent\n"
            "cp code.txt src/qlab/strategies/agent/weekly_momentum.py\n"
            "cp spec.txt specs/agent/toy-idea.yaml\n"
            f'cat "{repo}/out.jsonl"\n')
        script.chmod(0o755)

    return agent


def test_an_inbox_card_briefs_the_agent_on_the_block_the_grid_and_the_data(
        repo, session, inbox) -> None:
    inbox()
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    prompt = _args(repo).read_text()
    assert "src/qlab/strategies/agent/weekly_momentum.py" in prompt  # one module per block
    assert "window_days: сколько суток считать рост" in prompt  # every param with its meaning
    assert "REUSABLE block" in prompt and "FIRST variant" in prompt
    assert "signal.window_days = 7" in prompt  # the first variant the spec carries
    assert "Retune" in prompt  # the agent is told NOT to write one
    assert "BASELINE" in prompt and "base.yaml" in prompt
    assert 'data.source "binance"' in prompt and "2022-01-01" in prompt
    assert outcome.fields is not None and outcome.fields.new_block.name == "weekly-momentum"
    assert outcome.code_path.name == "weekly_momentum.py"
    assert not outcome.block_reused
    assert not any("grid:" in p for p in outcome.problems), outcome.problems


def test_a_hand_written_card_gets_no_inbox_brief(repo, session, monkeypatch) -> None:
    _fake_claude(repo, monkeypatch, write_spec=True)
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    assert outcome.fields is None
    assert impl.card_brief(impl.read_card_fields(repo / "card.yaml", repo),
                           outcome.code_path, block_exists=False) == ""


def test_a_broken_card_costs_no_tokens(repo, session) -> None:
    (repo / "card.yaml").write_text(INBOX_CARD.replace("honest_from: 2022-01-01",
                                                      "honest_from: soon"))
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    assert outcome.agent_reply is None  # the agent was never called
    assert outcome.problems and "the card's fields are wrong" in outcome.problems[0]


def test_the_spec_must_carry_the_first_variant_and_the_code_must_name_every_grid_param(
        repo, session, inbox) -> None:
    inbox(spec_text=BLOCK_SPEC % {"window": 14}, code="window = 7  # a constant\n")
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    assert any("the card's first variant is 7" in p for p in outcome.problems), outcome.problems
    assert any("the code never names 'window_days'" in p for p in outcome.problems)
    assert not outcome.ok


def test_the_spec_must_read_the_cards_data(repo, session, inbox) -> None:
    inbox(spec_text=(BLOCK_SPEC % {"window": 7}).replace("source: binance",
                                                        "source: hyperliquid"))
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    assert any("the card says venue binance, the spec reads hyperliquid" in p
               for p in outcome.problems)


def test_a_block_that_already_exists_is_reused_and_never_rewritten(
        repo, session, inbox) -> None:
    module = repo / "src/qlab/strategies/agent/weekly_momentum.py"
    module.parent.mkdir(parents=True)
    module.write_text("window = params['signal']['window_days']  # the earlier card's block\n")
    inbox(code="window = 1  # a rewrite\n")
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    assert "the earlier card's block" in module.read_text()  # the change was undone
    assert outcome.block_reused and str(outcome.code_path) in outcome.reverted
    assert any("allowed files (reverted)" in p for p in outcome.problems)
    prompt = _args(repo).read_text()
    assert "ALREADY EXISTS" in prompt and "write only the spec" in prompt


def test_the_block_is_registered_by_code_after_the_guards(repo, session, inbox,
                                                           tmp_path) -> None:
    inbox()
    outcome = impl.implement(repo / "card.yaml", budget=ExplicitNoBudget("test"), session=session)
    register = tmp_path / "register.yaml"
    assert impl.register_block(outcome, path=register)
    (entry,) = impl.agent_blocks.load(register)
    assert entry.name == "weekly-momentum" and entry.specs == ["specs/agent/toy-idea.yaml"]
    assert entry.code_ref == "qlab.strategies.agent.weekly_momentum:WeeklyMomentum"
    assert entry.params == [{"name": "window_days", "meaning": "сколько суток считать рост"}]
    # the next card's brief lists it, so the agent reuses a block before writing another
    assert "weekly-momentum" in impl.card_brief(
        outcome.fields, outcome.code_path, block_exists=False,
        blocks=impl.agent_blocks.block_lines(register))
    outcome.fields = None
    assert not impl.register_block(outcome, path=register)  # a card without a block: none
