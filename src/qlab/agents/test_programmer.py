"""The programmer agent (qlab.agents.programmer): a request becomes a tested change on its own
branch, never on main without the owner. The agent is faked; git, the path guard, the tests and
the merge are real, on a throwaway repository."""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import qlab.agents.programmer as prog
from qlab.budget import ExplicitNoBudget
from qlab.directions import parse_block


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    work = tmp_path / "repo"
    (work / "pkg" / "strategy" / "tests").mkdir(parents=True)
    (work / "pkg" / "__init__.py").write_text("")
    (work / "pkg" / "strategy" / "__init__.py").write_text("")
    (work / "pkg" / "strategy" / "tests" / "__init__.py").write_text("")
    (work / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\n")
    (work / "pkg" / "strategy" / "tests" / "test_signal.py").write_text(
        "from pkg.strategy.signal import WINDOW\n\n\ndef test_window():\n    assert WINDOW == 14\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "t")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "start")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "-u", "origin", "main")
    return work


def _config(repo: Path) -> dict:
    return {"model": "sonnet", "max_turns": 5, "repos": {"demo": {
        "path": str(repo), "base": "main", "allowed": ["pkg/strategy/*"],
        "test_command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "pkg"],
        "test_timeout_s": 120}}}


def _agent(monkeypatch, write, review="VERDICT: OK"):
    calls = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        calls.append(agent)
        if agent == "programmer":
            write(Path(cwd))
            return SimpleNamespace(text="made the window a setting", exit_code=0)
        return SimpleNamespace(text=review, exit_code=0)

    monkeypatch.setattr(prog, "run_agent", fake)
    return calls


def _request(queue: list) -> prog.Request:
    return prog.enqueue(queue, repo="demo", what="make the signal window a setting",
                        why="the owner asks for shorter windows", source="owner",
                        day=date(2026, 10, 26))


def _good_change(wt: Path) -> None:
    (wt / "pkg" / "strategy" / "signal.py").write_text(
        "WINDOW = 14\n\n\ndef window(days: int | None = None) -> int:\n"
        "    return WINDOW if days is None else days\n")
    (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text(
        "from pkg.strategy.signal import window\n\n\ndef test_default():\n"
        "    assert window() == 14 and window(7) == 7\n")


def test_a_passing_change_waits_on_its_branch_and_merges_only_on_the_owners_command(
        repo, tmp_path, monkeypatch):
    calls = _agent(monkeypatch, _good_change)
    queue: list = []
    req = _request(queue)
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "ready", req.problems
    assert calls == ["programmer", "programmer-review"]
    assert req.branch == f"agent/{req.id}" and req.commit
    assert "window" not in (repo / "pkg" / "strategy" / "signal.py").read_text()  # main untouched
    head = prog.merge(req, _config(repo))
    assert req.status == "merged" and "def window" in (repo / "pkg/strategy/signal.py").read_text()
    assert _git(tmp_path / "origin.git", "rev-parse", "main").strip() == head  # pushed
    assert not Path(req.worktree).exists()


def test_a_change_outside_the_allowed_paths_fails(repo, tmp_path, monkeypatch):
    def outside(wt: Path) -> None:
        _good_change(wt)
        (wt / "README.md").write_text("changed\n")

    _agent(monkeypatch, outside)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed"
    assert any("README.md is outside" in p for p in req.problems)


def test_a_change_without_a_test_or_with_a_failing_one_fails(repo, tmp_path, monkeypatch):
    def no_test(wt: Path) -> None:
        (wt / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\nOTHER = 1\n")

    _agent(monkeypatch, no_test)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed" and any("no test" in p for p in req.problems)

    def breaks_default(wt: Path) -> None:
        (wt / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 7\n")
        (wt / "pkg" / "strategy" / "tests" / "test_new.py").write_text("def test_x():\n    pass\n")

    _agent(monkeypatch, breaks_default)
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed" and any("the tests failed" in p for p in req.problems)
    assert req.attempts == 2


def test_a_gap_the_reviewer_names_blocks_the_request(repo, tmp_path, monkeypatch):
    _agent(monkeypatch, _good_change, review="GAP: the default path changed\nVERDICT: GAP")
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "blocked" and req.review == ["the default path changed"]
    assert req.commit is None
    prog.reject(req, _config(repo), "not now")
    assert req.status == "rejected" and req.branch not in _git(repo, "branch")


def test_merge_refuses_a_dirty_base(repo, tmp_path, monkeypatch):
    _agent(monkeypatch, _good_change)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    (repo / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\n# local edit\n")
    with pytest.raises(ValueError, match="uncommitted"):
        prog.merge(req, _config(repo))
    assert req.status == "ready"


def test_the_queue_takes_a_request_once(tmp_path):
    queue: list = []
    assert _request(queue) is not None and _request(queue) is None
    prog.save_queue(queue, tmp_path / "q.json")
    assert [r.id for r in prog.load_queue(tmp_path / "q.json")] == ["2026-10-26-1"]


def test_review_without_a_verdict_fails():
    assert prog.review_lines("VERDICT: OK") == (True, [])
    assert prog.review_lines("looks fine")[0] is False


def test_the_director_asks_for_capabilities_not_values():
    block = """
code_requests:
  - {repo: frab, what: "make the hedge signal's windows settings", why: "shorter windows"}
  - {repo: frab, what: "set the window to seven days", why: "x"}
  - {repo: web, what: "a page", why: "y"}
  - {repo: frab, what: "set the window to 7 days", why: "z"}
"""
    result = parse_block(block, mechanism_ids=set(), strategy_names={"bv2"}, family_names=set(),
                         day=date(2026, 10, 26))
    requests = result.directions.code_requests
    assert [r.what for r in requests] == ["make the hedge signal's windows settings",
                                          "set the window to seven days"]
    assert any("has a digit" in p for p in result.problems)
    assert any("needs repo" in p for p in result.problems)


def test_a_skipped_parity_test_is_named_and_ignored_data_is_linked(tmp_path):
    out = ("SKIPPED [1] src/frab/strategy/b2/tests/test_book_parity.py:36: research data not "
           "available\n1 passed, 1 skipped")
    assert prog.skipped_required(out, ["test_book_parity.py", "test_other.py"]) == [
        "test_book_parity.py"]
    main = tmp_path / "main"
    (main / "research" / "data").mkdir(parents=True)
    (main / "research" / "data" / "BTC_1h.csv").write_text("x")
    wt = tmp_path / "wt"
    subprocess.run(["git", "init", "-q", str(wt)], check=True)
    prog.link_ignored(main, wt, ["research/data", "missing/dir"])
    assert (wt / "research" / "data" / "BTC_1h.csv").read_text() == "x"
    assert not (wt / "missing").exists()
    assert prog.changed_paths(wt) == []  # the link is not a change


def test_a_retry_is_told_why_the_last_attempt_failed(repo, tmp_path, monkeypatch):
    prompts = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        prompts.append(prompt)
        if agent == "programmer":
            (Path(cwd) / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\nX = 1\n")
        return SimpleNamespace(text="VERDICT: OK", exit_code=0)

    monkeypatch.setattr(prog, "run_agent", fake)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert "previous attempt" not in prompts[0]
    assert "previous attempt" in prompts[1] and "no test among the changes" in prompts[1]
