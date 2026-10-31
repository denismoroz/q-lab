"""The programmer agent (qlab.agents.programmer): a request becomes a tested change on its own
branch, never on main without the owner. The agent is faked; git, the path guard, the tests and
the merge are real, on a throwaway repository."""

from __future__ import annotations

import inspect
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


def _config(repo: Path, **extra) -> dict:
    return {"model": "sonnet", "max_turns": 5, **extra, "repos": {"demo": {
        "path": str(repo), "base": "main", "allowed": ["pkg/strategy/*"],
        "test_command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "pkg"],
        "test_timeout_s": 120}}}


class _Calls(list):
    """The agent names called, in order; `.prompts`, `.args` and `.cwds` of the programmer's."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []
        self.args: list[tuple[str, ...]] = []
        self.cwds: list[Path] = []


def _agent(monkeypatch, write, review="VERDICT: OK"):
    """A fake run_agent. `write(worktree)` is the programmer's work; when it takes a second
    argument that is the number of the programmer's call (0 for the first)."""
    calls = _Calls()
    takes_n = len(inspect.signature(write).parameters) > 1

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        calls.append(agent)
        if agent == "programmer":
            n = len(calls.prompts)
            calls.prompts.append(prompt)
            calls.args.append(tuple(extra_args))
            calls.cwds.append(Path(cwd))
            if takes_n:
                write(Path(cwd), n)
            else:
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
    assert req.iterations == 1 and req.history == [{"n": 1, "result": "accepted", "remarks": 0}]
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
    prog.run_request(req, _config(repo, max_iterations=1), budget=ExplicitNoBudget("test"),
                     session=None, worktrees=tmp_path / "wt")
    assert req.status == "failed"
    assert any("README.md is outside" in p for p in req.problems)


def test_a_change_without_a_test_or_with_a_failing_one_fails(repo, tmp_path, monkeypatch):
    def no_test(wt: Path) -> None:
        (wt / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\nOTHER = 1\n")

    _agent(monkeypatch, no_test)
    req = _request([])
    one = _config(repo, max_iterations=1)
    prog.run_request(req, one, budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed" and any("no test" in p for p in req.problems)
    prog.retry(req)

    def breaks_default(wt: Path) -> None:
        (wt / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 7\n")
        (wt / "pkg" / "strategy" / "tests" / "test_new.py").write_text("def test_x():\n    pass\n")

    _agent(monkeypatch, breaks_default)
    prog.run_request(req, one, budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed" and any("the tests failed" in p for p in req.problems)
    assert req.attempts == 2


def test_a_gap_the_reviewer_names_goes_back_to_the_writer_not_to_the_owner(
        repo, tmp_path, monkeypatch):
    """Before 2026-10-31 a remark ended the request as `blocked`; now it is the next round's
    input. The same remark three times in a row is the loop's early stop."""
    calls = _agent(monkeypatch, _good_change, review="GAP: the default path changed\nVERDICT: GAP")
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("test"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "blocked" and req.review == ["the default path changed"]
    assert req.iterations == 3 and calls.count("programmer") == 3  # not 7: the same remark
    assert "the same failure three times" in req.problems[-1]
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
    def no_test(wt: Path) -> None:
        (wt / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 14\nX = 1\n")

    calls = _agent(monkeypatch, no_test)
    cfg = _config(repo, max_iterations=1)
    req = _request([])
    prog.run_request(req, cfg, budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    prog.retry(req)
    prog.run_request(req, cfg, budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert "This is round" not in calls.prompts[0]
    assert "This is round 1 of at most 1" in calls.prompts[1]
    assert "no test among the changes" in calls.prompts[1]


def test_the_agent_may_run_the_tests_and_no_other_shell_command(repo, tmp_path, monkeypatch):
    calls = _agent(monkeypatch, _good_change)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    args = calls.args[0]
    assert args[0] == "--allowedTools" and "--disallowedTools" not in args
    tools = args[1].split(",")
    assert tools == ["Read", "Glob", "Grep", "Write", "Edit",
                     f"Bash({sys.executable} -m pytest:*)"]
    assert "Bash" not in tools  # never an unrestricted shell
    assert "KEEP GOING until ALL tests pass" in calls.prompts[0]
    assert "You cannot run anything" not in calls.prompts[0]


def test_the_shell_pattern_is_the_test_commands_prefix():
    assert prog.tests_prefix(["uv", "run", "pytest", "-q", "-rs", "src/frab/strategy"]) == (
        "uv run pytest")
    assert prog.tools_pattern(["uv", "run", "pytest", "-q"]).endswith(",Bash(uv run pytest:*)")
    assert prog.tests_prefix(["python", "-m", "pytest", "-q", "pkg"]) == "python -m pytest"


def test_the_repositories_config_allows_seven_iterations_of_the_loop():
    config = prog.load_config()
    assert config["max_iterations"] == 7
    for name, repo_cfg in config["repos"].items():
        assert prog.tests_prefix(repo_cfg["test_command"]) == "uv run pytest", name
    queue: list = []
    req = _request(queue)
    req.iterations = 6
    assert prog.next_request(queue, config["max_iterations"]) is req  # the seventh is allowed
    req.iterations = 7
    assert prog.next_request(queue, config["max_iterations"]) is None


def _wrong_then_right(wt: Path, n: int) -> None:
    """The first pass writes a test whose expected value is wrong (the code is right); the next
    ones fix the test."""
    _good_change(wt)
    expected = "15" if n == 0 else "14"
    (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text(
        "from pkg.strategy.signal import window\n\n\ndef test_default():\n"
        f"    assert window() == {expected} and window(7) == 7\n")


def test_a_failing_code_run_sends_the_same_worktree_back_for_a_fix_pass(
        repo, tmp_path, monkeypatch):
    calls = _agent(monkeypatch, _wrong_then_right)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "ready", req.problems
    assert calls == ["programmer", "programmer", "programmer-review"]
    assert req.attempts == 1 and req.fix_passes == 1 and req.problems == []
    assert req.iterations == 2 and [h["result"] for h in req.history] == ["checks", "accepted"]
    assert calls.cwds[0] == calls.cwds[1]  # one worktree for both passes
    fix = calls.prompts[1]
    assert "round 2 of at most 7" in fix and "the tests failed" in fix and "14 == 15" in fix
    assert "1. " in fix  # the remarks are numbered
    written = Path(req.worktree) / "pkg/strategy/tests/test_window_setting.py"
    assert "15" not in written.read_text()


def test_the_iterations_are_bounded_and_then_the_request_fails_with_the_last_remarks(
        repo, tmp_path, monkeypatch):
    def always_wrong(wt: Path) -> None:
        _good_change(wt)
        (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text(
            "from pkg.strategy.signal import window\n\n\ndef test_default():\n"
            "    assert window() == 15\n")

    calls = _agent(monkeypatch, always_wrong)
    req = _request([])
    prog.run_request(req, _config(repo, max_iterations=2), budget=ExplicitNoBudget("t"),
                     session=None, worktrees=tmp_path / "wt")
    assert req.status == "failed" and req.attempts == 1 and req.iterations == 2
    assert calls == ["programmer"] * 2  # no review while the checks fail
    assert "2 iterations without" in req.problems[0] and f"qlab code retry {req.id}" in \
        req.problems[0]
    assert any("the tests failed" in p for p in req.problems[1:])  # the last remarks


def test_a_retry_edits_the_earlier_worktree_instead_of_recreating_it(
        repo, tmp_path, monkeypatch):
    def first_wrong_then_right(wt: Path, n: int) -> None:
        if n == 1:
            assert (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").is_file()
        _wrong_then_right(wt, n)

    calls = _agent(monkeypatch, first_wrong_then_right)
    cfg = _config(repo, max_iterations=1)
    req = _request([])
    prog.run_request(req, cfg, budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "failed"
    first_worktree, first_branch = req.worktree, req.branch
    marker = Path(first_worktree) / "pkg" / "strategy" / "marker.py"
    marker.write_text("KEPT = True\n")  # work of the earlier attempt a fresh start would lose
    prog.retry(req)
    prog.run_request(req, cfg, budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "ready", req.problems
    assert (req.worktree, req.branch) == (first_worktree, first_branch)
    assert calls.cwds[0] == calls.cwds[1]
    assert marker.is_file()
    retry = calls.prompts[1]
    assert "Do NOT write it again from scratch" in retry and "marker.py" in retry
    assert "the tests failed" in retry


def test_a_retry_without_an_earlier_worktree_starts_fresh(repo, tmp_path, monkeypatch):
    calls = _agent(monkeypatch, _good_change)
    req = _request([])
    req.attempts = 1  # an earlier attempt that left no worktree
    req.problems = ["something"]
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "ready", req.problems
    assert "files changed so far: none" in calls.prompts[0]  # nothing of the earlier work is there
    assert "something" in calls.prompts[0]


def test_a_stray_file_is_undone_by_code_and_reported(repo, tmp_path, monkeypatch):
    def stray_then_fine(wt: Path, n: int) -> None:
        _good_change(wt)
        if n == 0:
            (wt / "README.md").write_text("changed\n")

    calls = _agent(monkeypatch, stray_then_fine)
    req = _request([])
    prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert req.status == "ready", req.problems
    assert "README.md is outside" in calls.prompts[1] and "deleted or restored" in calls.prompts[1]
    assert not (Path(req.worktree) / "README.md").exists()


def test_a_budget_stop_mid_attempt_leaves_the_worktree_for_the_next_night(
        repo, tmp_path, monkeypatch):
    from qlab.budget import BudgetExhausted

    seen = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        seen.append(agent)
        if len(seen) == 2:
            raise BudgetExhausted("ceiling")
        _wrong_then_right(Path(cwd), 0)
        return SimpleNamespace(text="done", exit_code=0)

    monkeypatch.setattr(prog, "run_agent", fake)
    req = _request([])
    with pytest.raises(BudgetExhausted):
        prog.run_request(req, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                         worktrees=tmp_path / "wt")
    # the first iteration finished (its checks failed), the second was stopped before it began
    assert req.status == "queued" and req.attempts == 1 and req.iterations == 1
    assert any("the tests failed" in p for p in req.problems)  # saved for the next night
    assert any("the tests failed" in r for r in req.remarks)  # ... as the next round's input
    assert (Path(req.worktree) / "pkg/strategy/tests/test_window_setting.py").is_file()


# ---- the two levels: a planner on opus, a writer on sonnet (owner, 2026-10-30) ----------------

PLAN = ("1. Files and functions: pkg/strategy/signal.py -- add window(days=None) returning WINDOW "
        "when days is None, else days.\n2. Settings and defaults: none; the default stays 14.\n"
        "3. Tests to add: pkg/strategy/tests/test_window_setting.py::test_default asserts "
        "window() == 14 and window(7) == 7.\n4. What must not change: WINDOW and test_signal.py.\n"
        "5. How it is checked: python -m pytest pkg, narrow first.\n6. Open questions: none.\n"
        "VERDICT: PROCEED")


class _Levels(_Calls):
    """The planner's and the writer's calls of a fake that answers both."""

    def __init__(self):
        super().__init__()
        self.plan_prompts: list[str] = []
        self.plan_args: list[dict] = []


def _two_levels(monkeypatch, write, plans=(PLAN,), review="VERDICT: OK"):
    """A fake run_agent for a planner (`programmer-plan`; replies are taken from `plans` in turn,
    the last one repeats), the writer and the reviewer."""
    calls = _Levels()
    takes_n = len(inspect.signature(write).parameters) > 1

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        calls.append(agent)
        if agent == "programmer-plan":
            calls.plan_prompts.append(prompt)
            calls.plan_args.append({"model": model, "args": tuple(extra_args), "cwd": cwd})
            return SimpleNamespace(text=plans[min(len(calls.plan_prompts) - 1, len(plans) - 1)],
                                   exit_code=0)
        if agent == "programmer":
            n = len(calls.prompts)
            calls.prompts.append(prompt)
            calls.cwds.append(Path(cwd))
            write(Path(cwd), n) if takes_n else write(Path(cwd))
            return SimpleNamespace(text="made the window a setting", exit_code=0)
        return SimpleNamespace(text=review, exit_code=0)

    monkeypatch.setattr(prog, "run_agent", fake)
    return calls


def _two_level_config(repo: Path, **extra) -> dict:
    return _config(repo, planner_model="opus", **extra)


def _run(req, config, tmp_path):
    return prog.run_request(req, config, budget=ExplicitNoBudget("t"), session=None,
                            worktrees=tmp_path / "wt", plans=tmp_path / "plans")


def test_the_planner_reads_first_read_only_and_the_writer_follows_its_plan(
        repo, tmp_path, monkeypatch):
    calls = _two_levels(monkeypatch, _good_change)
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "ready", req.problems
    assert calls == ["programmer-plan", "programmer", "programmer-review"]
    assert calls.plan_args[0]["model"] == "opus"
    assert "Bash" in calls.plan_args[0]["args"][-1] and "Write" in calls.plan_args[0]["args"][-1]
    assert "--allowedTools" in calls.plan_args[0]["args"] and "Write" not in \
        calls.plan_args[0]["args"][1]  # the planner has Read, Glob, Grep only
    assert calls.plan_args[0]["cwd"] == calls.cwds[0]  # it reads the same checkout the writer edits
    assert "The PLAN" in calls.prompts[0] and "add window(days=None)" in calls.prompts[0]
    assert req.plan == PLAN and req.plan_hash and req.questions == []
    assert Path(req.plan_file).read_text().strip() == PLAN  # kept on disk, one file per attempt
    assert Path(req.plan_file).name == f"{req.id}-1.md"


def test_without_a_planner_model_the_programmer_is_one_level_as_before(
        repo, tmp_path, monkeypatch):
    calls = _two_levels(monkeypatch, _good_change)
    req = _request([])
    _run(req, _config(repo), tmp_path)
    assert calls == ["programmer", "programmer-review"] and req.plan is None
    assert "The PLAN" not in calls.prompts[0]


def test_an_open_behaviour_question_stops_the_request_before_a_writing_token(
        repo, tmp_path, monkeypatch):
    asked = ("1. Files: signal.py.\n" + "x" * 200 + "\nOPEN: should one window below the "
             "threshold open a partial hedge, or only both windows?\nVERDICT: BLOCKED")
    calls = _two_levels(monkeypatch, _good_change, plans=(asked,))
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert calls == ["programmer-plan"]  # no writer, no reviewer
    assert req.status == "blocked" and req.attempts == 0  # nothing was written: not an attempt
    assert req.questions == ["should one window below the threshold open a partial hedge, or "
                             "only both windows?"]
    assert "needs the owner's answer" in req.problems[0]
    assert req.worktree is None and not (tmp_path / "wt" / f"demo-{req.id}").exists()
    # the owner answers: the request is queued again for both levels, with a clean count
    prog.answer(req, "one window is enough: hedge half of it")
    assert req.status == "queued" and req.plan is None and req.questions == []
    calls2 = _two_levels(monkeypatch, _good_change)
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "ready", req.problems
    assert "one window is enough" in calls2.plan_prompts[0] and "binding" in calls2.plan_prompts[0]
    assert "one window is enough" in calls2.prompts[0]
    with pytest.raises(ValueError, match="nothing is waiting"):
        prog.answer(req, "again")


def test_a_verdict_blocked_without_a_question_still_stops_and_says_so(
        repo, tmp_path, monkeypatch):
    calls = _two_levels(monkeypatch, _good_change, plans=("VERDICT: BLOCKED",))
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert calls == ["programmer-plan"] and req.status == "blocked"
    assert "named no question" in req.questions[0]


def test_a_retry_makes_the_planner_read_why_the_last_attempt_failed(repo, tmp_path, monkeypatch):
    revised = PLAN.replace("add window(days=None)", "REVISED add window(days=None)")
    calls = _two_levels(monkeypatch, _wrong_then_right, plans=(PLAN, revised))
    cfg = _two_level_config(repo, max_iterations=1)
    req = _request([])
    _run(req, cfg, tmp_path)
    assert req.status == "failed" and len(req.failures) == 1
    prog.retry(req)  # the owner asks for another try: the planner revises its plan
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert calls == ["programmer-plan", "programmer", "programmer-plan", "programmer",
                     "programmer-review"]
    retry = calls.plan_prompts[1]
    assert "RETRY" in retry and "the tests failed" in retry and PLAN[:60] in retry
    assert "REVISED" in calls.prompts[1] and req.plan == revised
    assert {Path(req.plan_file).name} == {f"{req.id}-2.md"}
    assert (tmp_path / "plans" / f"{req.id}-1.md").is_file()  # the first plan stays on disk


def test_the_same_plan_after_the_same_failure_is_not_tried_again(repo, tmp_path, monkeypatch):
    def always_wrong(wt: Path) -> None:
        _good_change(wt)
        (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text(
            "from pkg.strategy.signal import window\n\n\ndef test_default():\n"
            "    assert window() == 15\n")

    calls = _two_levels(monkeypatch, always_wrong)  # the planner repeats itself
    cfg = _two_level_config(repo, max_iterations=1)
    req = _request([])
    _run(req, cfg, tmp_path)
    assert req.status == "failed"
    prog.retry(req)
    _run(req, cfg, tmp_path)
    assert req.status == "blocked" and calls.count("programmer") == 1  # no second writing
    assert "same plan after the same failure" in req.problems[0]
    assert req.attempts == 1  # the blocked re-attempt wrote nothing: it does not count


def test_a_third_identical_failure_ends_the_loop_early(repo, tmp_path, monkeypatch):
    def always_wrong(wt: Path) -> None:
        _good_change(wt)
        (wt / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text(
            "from pkg.strategy.signal import window\n\n\ndef test_default():\n"
            "    assert window() == 15\n")

    calls = _two_levels(monkeypatch, always_wrong)
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)  # up to 7 iterations, but not these three times
    assert calls.count("programmer") == 3 and calls.count("programmer-plan") == 1
    assert len({f["sig"] for f in req.failures}) == 1 and len(req.failures) == 3
    assert req.status == "blocked" and "same failure three times" in req.problems[-1]
    assert req.iterations == 3


def test_failures_with_different_tests_are_not_identical(repo):
    a = prog.failure_signature(["the tests failed:\nFAILED pkg/test_a.py::test_x - assert 1\n"])
    b = prog.failure_signature(["the tests failed:\nFAILED pkg/test_b.py::test_y - assert 1\n"])
    c = prog.failure_signature(["the tests failed:\nFAILED pkg/test_a.py::test_x - assert 2\n"])
    assert a != b and a == c
    assert prog.failure_signature(["took 12 s", "x"]) == prog.failure_signature(["took 99 s", "x"])


def test_the_planner_reply_is_read_by_its_question_lines():
    text, questions, blocked = prog.parse_plan("plan " * 60 + "\nOPEN: one?\nOPEN: two?\n")
    assert questions == ["one?", "two?"] and blocked and text.startswith("plan")
    _, questions, blocked = prog.parse_plan("plan " * 60 + "\nVERDICT: PROCEED")
    assert questions == [] and not blocked
    with pytest.raises(prog.EnvironmentFailure, match="no plan"):
        prog.parse_plan("ok")


# ---- the environment (2026-10-28-1: 34 tests failed in its worktree, none by its work) ---------


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def test_a_test_that_fails_on_the_untouched_base_too_is_not_the_agents(
        repo, tmp_path, monkeypatch):
    (repo / "pkg" / "strategy" / "tests" / "test_environment.py").write_text(
        "def test_needs_something_a_worktree_lacks():\n    assert False\n")
    _commit(repo, "a test that fails everywhere")
    calls = _two_levels(monkeypatch, _good_change)
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "ready", req.problems
    assert req.fix_passes == 0 and calls.count("programmer") == 1  # no fix pass was spent on it
    assert req.env_failures == ["pkg/strategy/tests/test_environment.py::"
                                "test_needs_something_a_worktree_lacks"]


def test_a_failure_of_the_agents_own_is_still_a_failure_next_to_the_environments(
        repo, tmp_path, monkeypatch):
    (repo / "pkg" / "strategy" / "tests" / "test_environment.py").write_text(
        "def test_needs_something_a_worktree_lacks():\n    assert False\n")
    _commit(repo, "a test that fails everywhere")
    calls = _two_levels(monkeypatch, _wrong_then_right)
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "ready", req.problems
    assert calls.count("programmer") == 2 and req.fix_passes == 1  # its own wrong test: a pass
    fix = calls.prompts[1]
    assert "test_default" in fix and "14 == 15" in fix


def test_a_failure_that_cannot_be_read_is_never_excused(repo, tmp_path, monkeypatch):
    def break_collection(wt: Path) -> None:
        _good_change(wt)
        (wt / "pkg" / "strategy" / "signal.py").write_text("def broken(:\n")

    _two_levels(monkeypatch, break_collection)
    req = _request([])
    _run(req, _two_level_config(repo, max_iterations=1), tmp_path)
    assert req.status == "failed" and req.env_failures == []


def test_the_sibling_repository_and_the_prepare_commands_are_made_ready(
        repo, tmp_path, monkeypatch):
    sibling = tmp_path / "other-repo"
    sibling.mkdir()
    (sibling / "notes.md").write_text("cited by a test\n")
    (repo / "pkg" / "strategy" / "tests" / "test_sibling.py").write_text(
        "from pathlib import Path\n\n\ndef test_the_sibling_is_next_to_the_checkout():\n"
        "    assert (Path(__file__).resolve().parents[4] / 'other-repo' / 'notes.md').is_file()\n"
        "\n\ndef test_prepared():\n"
        "    assert Path('prepared.txt').is_file()\n")
    (repo / ".gitignore").write_text("prepared.txt\n")  # like data/qlab.db: not the agent's change
    _commit(repo, "tests that read what a worktree lacks")
    cfg = _two_level_config(repo, siblings={"other-repo": str(sibling)})
    cfg["repos"]["demo"]["prepare"] = [[sys.executable, "-c",
                                        "open('prepared.txt', 'w').write('x')"]]
    _two_levels(monkeypatch, _good_change)
    req = _request([])
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert req.env_failures == []  # they PASSED in the worktree: the environment was made ready
    assert (tmp_path / "wt" / "other-repo").is_symlink()


def test_a_failing_prepare_command_is_the_environments_and_does_not_count(
        repo, tmp_path, monkeypatch):
    cfg = _two_level_config(repo)
    cfg["repos"]["demo"]["prepare"] = [[sys.executable, "-c", "raise SystemExit(3)"]]
    calls = _two_levels(monkeypatch, _good_change)
    req = _request([])
    for stops in (1, 2):
        _run(req, cfg, tmp_path)
        assert req.status == "queued" and req.attempts == 0 and req.env_stops == stops
        assert req.problems[0].startswith("environment:")
    _run(req, cfg, tmp_path)
    assert req.status == "blocked" and "environment" in req.problems[-1]
    assert calls == []  # no agent was called for any of them


def test_a_worktree_behind_the_base_is_brought_onto_its_tip_with_the_work_kept(
        repo, tmp_path, monkeypatch):
    cfg = _two_level_config(repo, max_iterations=1)
    calls = _two_levels(monkeypatch, _wrong_then_right, plans=(PLAN, PLAN + "\nrevised"))
    req = _request([])
    _run(req, cfg, tmp_path)
    assert req.status == "failed"
    prog.retry(req)
    work = Path(req.worktree)
    before = _git(work, "rev-parse", "HEAD").strip()
    (repo / "pkg" / "strategy" / "data.txt").write_text("committed after the worktree was cut\n")
    _commit(repo, "the base moves on")  # the next night's main
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert _git(work, "rev-parse", "HEAD").strip() != before
    assert (work / "pkg" / "strategy" / "data.txt").is_file()  # now on the new tip ...
    assert (work / "pkg/strategy/tests/test_window_setting.py").is_file()  # ... and the work kept
    assert _git(work, "merge-base", "--is-ancestor", "main", "HEAD") == ""
    assert calls.count("programmer-plan") == 2


def test_work_that_does_not_apply_to_the_new_base_starts_fresh_and_keeps_a_patch(
        repo, tmp_path, monkeypatch):
    def edits_signal(wt: Path, n: int) -> None:
        _wrong_then_right(wt, n)

    cfg = _two_level_config(repo, max_iterations=1)
    _two_levels(monkeypatch, edits_signal, plans=(PLAN, PLAN + "\nrevised"))
    req = _request([])
    _run(req, cfg, tmp_path)
    assert req.status == "failed"
    prog.retry(req)
    (repo / "pkg" / "strategy" / "signal.py").write_text("WINDOW = 21\n\n\ndef window(d=None):\n"
                                                         "    return d or WINDOW\n")
    (repo / "pkg" / "strategy" / "tests" / "test_signal.py").write_text(
        "from pkg.strategy.signal import WINDOW\n\n\ndef test_window():\n    assert WINDOW == 21\n")
    (repo / "pkg" / "strategy" / "tests" / "test_window_setting.py").write_text("# main's own\n")
    _commit(repo, "main changes the same files")
    _run(req, cfg, tmp_path)
    patch = tmp_path / "wt" / f"demo-{req.id}.conflict.patch"
    assert patch.is_file() and "window" in patch.read_text()


# ---- the order of the queue (owner, 2026-10-30) ------------------------------------------------


def test_requests_that_give_the_generator_a_new_block_go_first():
    day = date(2026, 10, 30)
    q: list = []
    texts = {
        "anomaly": ("Add an anomaly flag that fires when the counterfactual turns it", "x"),
        "report": ("Показывать в файле доказательств число вызовов генератора", "x"),
        "mask": ("Make the liquidity mask at decision bar t use the 24h of volume", "x"),
        "block": ("Блок сигнала: положение текущей ставки среди её собственной истории", "x"),
        "pair": ("Способ собрать связку монета против контракта с открытыми числами", "x"),
        "panel": ("Панель из ставок и цен двух площадок", "x"),
        "newblock": ("Add a selection block that picks the coins with the lowest turnover", "x"),
    }
    reqs = {k: prog.enqueue(q, repo="demo", what=w, why=y, source="owner", day=day)
            for k, (w, y) in texts.items()}
    kinds = {k: prog.request_kind(r) for k, r in reqs.items()}
    assert {k for k, v in kinds.items() if v == "mechanism"} == {"block", "pair", "panel",
                                                                 "newblock"}
    order = [r.id for r in prog.runnable(q, 3)]
    first = {reqs[k].id for k in ("block", "pair", "panel", "newblock")}
    assert set(order[:4]) == first
    assert order[:4] == [r.id for r in q if r.id in first]  # queue order inside a kind
    assert prog.next_request(q, 3, skip={order[0]}).id == order[1]


def test_a_new_block_request_with_iterations_behind_it_still_goes_before_older_ones():
    q: list = []
    other = prog.enqueue(q, repo="demo", what="Show a count in the director's file", why="y",
                         source="owner", day=date(2026, 10, 26))
    block = prog.enqueue(q, repo="demo", what="Блок сигнала: положение ставки", why="y",
                         source="owner", day=date(2026, 10, 28))
    block.iterations = 2
    assert prog.next_request(q, 7) is block and prog.runnable(q, 7)[1] is other


def test_the_report_shows_the_plan_the_question_and_what_waits(repo, tmp_path, monkeypatch):
    asked = ("Plan text about the signal window. " * 12 + "\nOPEN: one window or both?\n"
             "VERDICT: BLOCKED")
    _two_levels(monkeypatch, _good_change, plans=(asked,))
    queue: list = []
    req = _request(queue)
    _run(req, _two_level_config(repo), tmp_path)
    from dataclasses import asdict

    text = "\n".join(prog.report_lines([asdict(req)], queue))
    assert "планировщик" in text and "план: `" in text and "суть плана: Plan text" in text
    assert "вопрос владельцу: one window or both?" in text
    assert f"qlab code answer {req.id}" in text  # how the owner answers
    prog.enqueue(queue, repo="demo", what="Блок сигнала: x", why="y", source="owner",
                 day=date(2026, 10, 27))  # another, queued
    assert "из них даёт генератору новый блок: 1" in "\n".join(prog.report_lines([], queue))


# ---- the loop: writer, checks, reviewer, back to the writer (owner, 2026-10-31) ---------------


def _reviewer_says(monkeypatch, write, verdicts, plans=(PLAN,)):
    """A fake whose reviewer answers from `verdicts` in turn (the last one repeats): a string is
    its whole reply."""
    calls = _two_levels(monkeypatch, write, plans=plans)
    inner = prog.run_agent
    turn = {"n": 0}

    def fake(budget, prompt, **kw):
        if kw["agent"] == "programmer-review":
            calls.append("programmer-review")
            reply = verdicts[min(turn["n"], len(verdicts) - 1)]
            turn["n"] += 1
            return SimpleNamespace(text=reply, exit_code=0)
        return inner(budget, prompt, **kw)

    monkeypatch.setattr(prog, "run_agent", fake)
    return calls


def test_the_writer_is_told_its_goal_is_working_code_and_which_round_it_is(
        repo, tmp_path, monkeypatch):
    calls = _agent(monkeypatch, _good_change)
    _run(_request([]), _config(repo), tmp_path)
    first = calls.prompts[0]
    assert "YOUR GOAL is code that WORKS" in first and "not code that merely gets past" in first
    assert "round 1 of at most 7" in first


def test_a_remark_goes_back_to_the_writer_and_the_second_round_is_accepted(
        repo, tmp_path, monkeypatch):
    def with_leftover(wt: Path, n: int) -> None:
        _good_change(wt)
        if n == 0:  # the trifle that stopped two requests: an empty leftover file
            (wt / "pkg" / "strategy" / "tests" / "test_band.py").write_text('"""leftover"""\n')
        else:
            (wt / "pkg" / "strategy" / "tests" / "test_band.py").unlink()

    calls = _reviewer_says(monkeypatch, with_leftover, [
        "GAP: pkg/strategy/tests/test_band.py is an empty leftover: remove it\nVERDICT: GAP",
        "VERDICT: OK"])
    req = _request([])
    done = []
    prog.run_request(req, _two_level_config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt", plans=tmp_path / "plans",
                     checkpoint=lambda r: done.append((r.iterations, r.status)))
    assert req.status == "ready", req.problems
    assert calls == ["programmer-plan", "programmer", "programmer-review", "programmer",
                     "programmer-review"]  # one plan, two rounds, two reviews
    assert req.iterations == 2 and [h["result"] for h in req.history] == ["review", "accepted"]
    assert calls.cwds[0] == calls.cwds[1]  # the same worktree
    second = calls.prompts[1]
    assert "round 2 of at most 7" in second
    assert "reviewer: pkg/strategy/tests/test_band.py" in second
    assert "every remark, the small ones too" in second
    assert not (Path(req.worktree) / "pkg/strategy/tests/test_band.py").exists()
    assert req.commit and req.review == [] and req.remarks == []
    assert done == [(1, "queued"), (2, "ready")]  # saved after every finished iteration


def test_seven_refusals_end_the_loop_as_failed_with_the_last_remarks(repo, tmp_path, monkeypatch):
    # distinct remarks: each differs in words, not just in digits, or the early stop would fire
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    replies = [f"GAP: the {w} list misses a family\nVERDICT: GAP" for w in words]
    calls = _reviewer_says(monkeypatch, _good_change, replies)
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "failed" and req.iterations == 7
    assert calls.count("programmer") == 7 and calls.count("programmer-review") == 7
    assert calls.count("programmer-plan") == 1  # the plan is written once
    assert req.review == ["the golf list misses a family"]  # the last remark is the one shown
    assert req.problems[0].startswith("7 iterations without")
    assert "qlab code retry" in req.problems[0]
    assert "round 7 of at most 7" in calls.prompts[6]
    assert req.commit is None
    # the owner sees it in the report, with the last remark and the way to another try
    from dataclasses import asdict

    text = "\n".join(prog.report_lines([asdict(req)], [req]))
    assert "Ждут вашего решения" in text and "the golf list misses a family" in text
    assert f"qlab code retry {req.id}" in text and "итерации (писатель" in text


def test_the_same_remark_three_times_stops_the_loop_early(repo, tmp_path, monkeypatch):
    calls = _reviewer_says(monkeypatch, _good_change, ["GAP: the same thing\nVERDICT: GAP"])
    req = _request([])
    _run(req, _two_level_config(repo), tmp_path)
    assert req.status == "blocked" and req.iterations == 3 and calls.count("programmer") == 3
    assert "same failure three times" in req.problems[-1]


def test_an_iteration_the_environment_ended_does_not_count_and_the_loop_goes_on(
        repo, tmp_path, monkeypatch):
    """The second round's tests cannot be judged (the base checkout could not be made ready):
    that round is not one of the seven, and the next session goes on with the same remarks."""
    calls = _reviewer_says(monkeypatch, _good_change, [
        "GAP: add the second assertion\nVERDICT: GAP", "VERDICT: OK"])
    real = prog.judge_tests
    seen = {"n": 0}

    def judge(*a, **kw):
        seen["n"] += 1
        if seen["n"] == 2:
            raise prog.EnvironmentFailure("git worktree add failed in _base-repo")
        return real(*a, **kw)

    monkeypatch.setattr(prog, "judge_tests", judge)
    req = _request([])
    cfg = _two_level_config(repo)
    _run(req, cfg, tmp_path)
    assert req.status == "queued" and req.iterations == 1  # round 2 did not count
    assert req.env_stops == 1 and req.attempts == 0 and req.problems[0].startswith("environment:")
    assert req.remarks == ["reviewer: add the second assertion"]  # kept for the next session
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert req.iterations == 2 and req.env_stops == 0
    assert "add the second assertion" in calls.prompts[-1]
    assert "environment" not in calls.prompts[-1]


def test_the_budget_stops_the_loop_where_it_is_and_the_next_night_resumes_it(
        repo, tmp_path, monkeypatch):
    from qlab.budget import CandidateExhausted

    calls = _reviewer_says(monkeypatch, _good_change, [
        "GAP: wire the setting into the list\nVERDICT: GAP", "VERDICT: OK"])
    inner = prog.run_agent
    writes = {"n": 0}

    def limited(budget, prompt, **kw):
        if kw["agent"] == "programmer":
            writes["n"] += 1
            if writes["n"] == 2:  # the request's share is spent before the second round begins
                raise CandidateExhausted("used 250,000 of 250,000 tokens")
        return inner(budget, prompt, **kw)

    monkeypatch.setattr(prog, "run_agent", limited)
    req = _request([])
    cfg = _two_level_config(repo)
    with pytest.raises(CandidateExhausted):
        _run(req, cfg, tmp_path)
    assert req.status == "queued" and req.iterations == 1
    assert req.remarks == ["reviewer: wire the setting into the list"]
    worktree = Path(req.worktree)
    assert (worktree / "pkg/strategy/tests/test_window_setting.py").is_file()  # work kept
    # the next night: no new plan, the same worktree, the remark is the first thing said
    plans_before = calls.count("programmer-plan")
    monkeypatch.setattr(prog, "run_agent", inner)
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert calls.count("programmer-plan") == plans_before == 1
    assert Path(req.worktree) == worktree and req.iterations == 2
    assert "wire the setting into the list" in calls.prompts[-1]
    assert "round 2 of at most 7" in calls.prompts[-1]


def test_a_request_the_old_way_ended_on_reviewer_remarks_continues_from_its_worktree(
        repo, tmp_path, monkeypatch):
    """2026-10-28-1 and -2 were `blocked` with the reviewer's gaps in `review`: reset to queued,
    their plan and code kept, the gaps are the first round's input."""
    calls = _agent(monkeypatch, _good_change)
    first = _request([])
    prog.run_request(first, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert first.status == "ready"
    legacy = prog.Request(**{**first.__dict__, "status": "queued", "iterations": 0,
                             "history": [], "commit": None, "remarks": [],
                             "review": ["`pkg/strategy/signal.py` reflows an import: revert it"]})
    marker = Path(legacy.worktree) / "pkg" / "strategy" / "tests" / "test_kept.py"
    marker.write_text("def test_kept():\n    assert True\n")  # the code the request already has
    assert prog.runnable([prog.Request(id="x", repo="demo", what="Блок сигнала: x", why="y",
                                       source="owner", created="t"), legacy], 7)[0] is legacy
    prog.run_request(legacy, _config(repo), budget=ExplicitNoBudget("t"), session=None,
                     worktrees=tmp_path / "wt")
    assert legacy.status == "ready", legacy.problems
    assert "reflows an import" in calls.prompts[-1] and "test_kept.py" in calls.prompts[-1]
    assert marker.is_file() and legacy.iterations == 1


def test_the_owner_can_ask_for_another_try_at_a_request_that_used_its_iterations(
        repo, tmp_path, monkeypatch):
    calls = _reviewer_says(
        monkeypatch, _good_change, [f"GAP: the {w} list\nVERDICT: GAP" for w in "abcdefg"]
        + ["VERDICT: OK"], plans=(PLAN, PLAN + "\nrevised by the last remarks"))
    req = _request([])
    cfg = _two_level_config(repo)
    _run(req, cfg, tmp_path)
    assert req.status == "failed" and req.iterations == 7
    worktree = req.worktree
    prog.retry(req, "also list the new family in docs/BLOCKS.md")
    assert req.status == "queued" and req.iterations == 0 and req.replan
    assert req.remarks[0] == "the owner: also list the new family in docs/BLOCKS.md"
    _run(req, cfg, tmp_path)
    assert req.status == "ready", req.problems
    assert req.worktree == worktree and req.iterations == 1 and not req.replan
    assert calls.count("programmer-plan") == 2  # revised once, by the last remarks
    assert "also list the new family" in calls.prompts[-1]
    with pytest.raises(ValueError, match="nothing to retry"):
        prog.retry(req)  # a ready request has nothing to retry
    asking = _request([])
    asking.status, asking.questions = "blocked", ["one window or both?"]
    with pytest.raises(ValueError, match="waits for an answer"):
        prog.retry(asking)


def test_a_begun_request_goes_before_a_new_block_request():
    q: list = []
    block = prog.enqueue(q, repo="demo", what="Блок сигнала: положение ставки", why="y",
                         source="owner", day=date(2026, 10, 30))
    begun = prog.enqueue(q, repo="demo", what="Make the liquidity mask use 24h", why="y",
                         source="owner", day=date(2026, 10, 28))
    begun.worktree, begun.remarks, begun.iterations = "/w", ["reviewer: x"], 1
    assert [r.id for r in prog.runnable(q, 7)] == [begun.id, block.id]
