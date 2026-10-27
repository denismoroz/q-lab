"""Tests for the researcher agent's guards, with a fake `claude`."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import qlab.agents.researcher as res
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.registry.models import Base


def _repo_and_agent(tmp_path, monkeypatch, body: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=tmp_path, check=True)
    monkeypatch.setattr(res, "QLAB_ROOT", tmp_path)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", tmp_path)
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.2,
                                                              "resetsAt": reset}}}}) + "\n"
        + json.dumps({"type": "result", "is_error": False, "result": "done",
                      "total_cost_usd": 0.0, "usage": {"input_tokens": 1, "output_tokens": 1},
                      "modelUsage": {"claude-opus": {}}}) + "\n")
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\n{body}cat \"{out}\"\n")
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))


def test_experiments_over_the_limit_never_reach_the_stand(tmp_path, monkeypatch) -> None:
    # A night rehearsal of 2026-10-27: a strategy with no free variant slot (limit 0) had its
    # experiment run anyway, the excess only noted.
    today = datetime.now(UTC).date().isoformat()
    _repo_and_agent(tmp_path, monkeypatch,
                    f"echo '# memo' > docs/research/t-{today}.md\n"
                    "echo 'idea_id: a' > specs/research/t/a.yaml\n"
                    "echo 'idea_id: b' > specs/research/t/b.yaml\n")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    outcome = res.research("t", idea_ids=["nothing"], specs=[], docs=[],
                           budget=ExplicitNoBudget("test"), session=session, max_experiments=0)
    assert outcome.specs == []
    assert not list((tmp_path / "specs/research/t").glob("*.yaml"))
    assert any("at most 0: not run and removed" in p for p in outcome.problems)


def test_only_the_memo_and_experiment_specs_are_kept(tmp_path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=tmp_path, check=True)
    monkeypatch.setattr(res, "QLAB_ROOT", tmp_path)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", tmp_path)
    today = datetime.now(UTC).date().isoformat()
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.2,
                                                              "resetsAt": reset}}}}) + "\n"
        + json.dumps({"type": "result", "is_error": False, "result": "done",
                      "total_cost_usd": 0.0, "usage": {"input_tokens": 1, "output_tokens": 1},
                      "modelUsage": {"claude-opus": {}}}) + "\n")
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\n"
                      f"echo '# memo' > docs/research/t-{today}.md\n"
                      "echo 'stray' > src_stray.py\n"
                      f'cat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    outcome = res.research("t", idea_ids=["nothing"], specs=[], docs=[],
                           budget=ExplicitNoBudget("test"), session=session)
    assert outcome.memo is not None and not (tmp_path / "src_stray.py").exists()
    assert "src_stray.py" in outcome.reverted
    assert (tmp_path / f"data/research/t-{today}.md").read_text().startswith("# Evidence")
