"""The programmer agent: a code request -> a tested change on a branch, never on main
(docs/PROGRAMMER.md).

Owner, 2026-10-26: «так нам нужен еще один агент - который будет дописывать код если нужно»;
«будет дописывать код если руководитель посчитает что направление - следует проверить»;
«да, делай агента-программиста».

A request says which repository (`frab` -- the live strategies' code, or `qlab`), what to
make and why. It comes from the research director's `code_requests` (`qlab.directions`, applied
once the owner has reviewed the directions) or from the owner (`qlab code request`). One request
a night:

1. a separate worktree of the repository on a new branch `agent/<id>`, from its base branch;
2. one agent call in that worktree: it may read anything and write files (Read, Glob, Grep,
   Write, Edit), never run anything;
3. code checks what it wrote: every changed path matches the repository's `allowed` list
   (`night/programmer.yaml`) -- for frab only the pure strategy modules q-lab's stand runs,
   never an engine, the exchange or execution; a test file must be among the changes;
4. code runs the repository's tests in the worktree; every test passes, the parity tests with
   them (the default behaviour must not change);
5. a reviewer agent reads the request and the diff; a gap it names blocks the request;
6. a change that passes is committed on its branch and waits: `qlab code merge <id>` -- the
   owner's command -- merges it into the base branch and pushes; `qlab code reject <id>` drops
   it. Nothing here merges or pushes.

The agent's spending goes through the budget guard (`qlab.budget.run_agent`)."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

CONFIG = Path("night/programmer.yaml")
QUEUE = Path("data/programmer/queue.json")
WORKTREES = Path("data/programmer/worktrees")
TOOLS = "Read,Glob,Grep,Write,Edit"
REVIEW_TOOLS = "Read,Glob,Grep"
BRANCH_PREFIX = "agent/"
STATUSES = ("queued", "ready", "failed", "blocked", "merged", "rejected")
"""queued: waits for a night; ready: passed, waits for the owner's merge; failed: a guard or a
test failed (tried again until max_attempts); blocked: the reviewer named a gap, or the attempts
ran out -- the owner decides; merged / rejected: the owner's decision."""
_TEST = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$)")


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else QLAB_ROOT / path


def load_config(path: Path = CONFIG) -> dict:
    config = yaml.safe_load(_abs(path).read_text(encoding="utf-8")) or {}
    for name, repo in (config.get("repos") or {}).items():
        if not repo.get("allowed") or not repo.get("test_command") or not repo.get("path"):
            raise ValueError(f"{path}: repo {name} needs path, allowed and test_command")
    return config


# --------------------------------------------------------------------------
# The queue


@dataclass
class Request:
    id: str
    repo: str
    what: str
    why: str
    source: str
    created: str
    strategy: str | None = None
    status: str = "queued"
    attempts: int = 0
    branch: str | None = None
    worktree: str | None = None
    commit: str | None = None
    diffstat: str | None = None
    reply: str | None = None
    review: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    tokens: int = 0
    decided: str | None = None  # the owner's merge or reject, with its reason


def load_queue(path: Path = QUEUE) -> list[Request]:
    target = _abs(path)
    if not target.is_file():
        return []
    return [Request(**r) for r in json.loads(target.read_text(encoding="utf-8"))]


def save_queue(queue: list[Request], path: Path = QUEUE) -> None:
    target = _abs(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in queue], indent=1, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(target)


def _key(repo: str, what: str) -> tuple[str, str]:
    return repo, re.sub(r"\s+", " ", what.strip().lower())


def enqueue(queue: list[Request], *, repo: str, what: str, why: str, source: str, day: date,
            strategy: str | None = None) -> Request | None:
    """Add a request unless the same one (repo, what) is already queued, waiting or merged.
    Returns the new request, or None for a duplicate."""
    taken = {_key(r.repo, r.what) for r in queue if r.status != "rejected"}
    if _key(repo, what) in taken:
        return None
    n = 1 + sum(1 for r in queue if r.id.startswith(day.isoformat()))
    req = Request(id=f"{day.isoformat()}-{n}", repo=repo, what=what.strip(), why=why.strip(),
                  source=source, created=datetime.now(UTC).isoformat(), strategy=strategy)
    queue.append(req)
    return req


def next_request(queue: list[Request], max_attempts: int) -> Request | None:
    for req in queue:
        if req.status in ("queued", "failed") and req.attempts < max_attempts:
            return req
    return None


# --------------------------------------------------------------------------
# Git and the checks (code, no agent)


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {cwd}: {out.stderr.strip() or out.stdout}")
    return out.stdout


def make_worktree(repo_path: Path, base: str, req: Request, *,
                  root: Path = WORKTREES) -> Path:
    """A fresh worktree on `agent/<id>` from `base`; an earlier attempt's is removed first."""
    branch = BRANCH_PREFIX + req.id
    path = _abs(root) / f"{req.repo}-{req.id}"
    if path.exists():
        _git(repo_path, "worktree", "remove", "--force", str(path), check=False)
        shutil.rmtree(path, ignore_errors=True)
    _git(repo_path, "worktree", "prune")
    _git(repo_path, "branch", "-D", branch, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_path, "worktree", "add", "-b", branch, str(path), base)
    req.branch, req.worktree = branch, str(path)
    return path


def changed_paths(worktree: Path) -> list[str]:
    """Every path changed or added in the worktree, relative to it."""
    out = _git(worktree, "status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in out.splitlines():
        name = line[3:].strip()
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        paths.append(name.strip('"'))
    return sorted(paths)


def path_problems(paths: list[str], allowed: list[str]) -> list[str]:
    """What is wrong with the set of changed paths: one outside `allowed`, none at all, no test."""
    if not paths:
        return ["nothing was written"]
    problems = [f"{p} is outside what the agent may change ({', '.join(allowed)})"
                for p in paths if not any(fnmatch.fnmatch(p, pattern) for pattern in allowed)]
    if not any(_TEST.search(p) for p in paths):
        problems.append("no test among the changes: a new behaviour needs its test")
    return problems


def run_tests(worktree: Path, command: list[str], timeout_s: int) -> tuple[bool, str]:
    """The repository's tests in the worktree; (passed, the output's tail)."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    try:
        out = subprocess.run(command, cwd=worktree, capture_output=True, text=True,
                             timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return False, f"the tests did not finish in {timeout_s} s"
    tail = "\n".join((out.stdout + out.stderr).strip().splitlines()[-15:])
    return out.returncode == 0, tail


def diff_text(worktree: Path) -> tuple[str, str]:
    """(the full diff of everything changed, its --stat), staged so new files are in it."""
    _git(worktree, "add", "-A")
    return _git(worktree, "diff", "--cached"), _git(worktree, "diff", "--cached", "--stat")


# --------------------------------------------------------------------------
# The agents


INSTRUCTIONS = """You are q-lab's programmer agent. q-lab is a research lab that finds and filters
trading strategies; its verdicts are made by code. You write the code a research direction needs
and that does not exist yet. You work in a separate copy of the repository `{repo}` at {worktree}
(your working directory), on the branch {branch}; nothing you write reaches the main branch
unless the owner merges it.

The request ({source}):
  what: {what}
  why: {why}{strategy}

Rules -- code checks every one of them after you finish, and a broken rule fails the request:
- Write ONLY files matching these patterns (relative to the repository root): {allowed}. A change
  anywhere else fails the request.{repo_rule}
- The behaviour with the default settings must not change AT ALL. Add the new capability behind a
  setting whose default reproduces today's behaviour exactly (bit for bit), the way earlier
  settings were added (look at how the code already handles optional settings: stored only when
  set, validated, documented in a comment). Never change a default.
- Every existing test must keep passing unchanged, the parity tests among them. The one edit to
  an existing test that is allowed: registering the new setting where a test pins the list of
  settings or fields.
- Add tests for the new behaviour, in the repository's test folders: that the default is
  today's behaviour, that the new setting does what the request asks, that bad values are refused.
- You cannot run anything (no shell): code runs the tests ({tests}) after you finish.
- Code is in English (names, comments, docstrings, test names); match the style of the code
  around it. No new dependency.

Reply in a few sentences: what you changed, the names of the new settings and their defaults,
and anything the reviewer should look at."""

FRAB_RULE = """
- This repository trades real money. Touch only the pure strategy modules allowed above: their
  signals, books and parameters. Never an engine, the exchange, order placement or the database."""

REVIEW = """You review a change the programmer agent made in `{repo}` (working directory: the
copy with the change, branch {branch}). The request was:
  what: {what}
  why: {why}

The diff:

{diff}

Check, reading the code around the change as needed:
1. With the default settings, is the behaviour EXACTLY what it was (no default changed, no code
   path the defaults take altered in effect)?
2. Does the new setting do what the request asks, and nothing beyond it?
3. Do the new tests test it, and does no existing test lose an assertion?
4. Anything that could reach order placement, an engine, the exchange or stored state in a way
   the old code would read differently?

Reply with lines only: `GAP: <what is wrong, where>` for every problem you are sure of, then a
last line `VERDICT: OK` if there is none or `VERDICT: GAP` if there is any."""


def review_lines(text: str) -> tuple[bool, list[str]]:
    """(passed, the gaps) from the reviewer's reply. No verdict line is a failed review."""
    gaps = [m.group(1).strip() for m in re.finditer(r"^GAP:\s*(.+)$", text, re.MULTILINE)]
    verdict = re.findall(r"^VERDICT:\s*(OK|GAP)\s*$", text, re.MULTILINE)
    if not verdict:
        return False, [*gaps, "the reviewer gave no verdict"]
    return verdict[-1] == "OK" and not gaps, gaps


def run_request(req: Request, config: dict, *, budget: CandidateBudget | ExplicitNoBudget,
                session, worktrees: Path = WORKTREES) -> Request:
    """One attempt at one request: worktree, agent, path check, tests, review, commit."""
    repo_cfg = config["repos"][req.repo]
    repo_path = _abs(Path(repo_cfg["path"]))
    req.attempts += 1
    req.problems, req.review, req.reply = [], [], None
    worktree = make_worktree(repo_path, repo_cfg.get("base", "main"), req, root=worktrees)
    allowed = list(repo_cfg["allowed"])
    prompt = INSTRUCTIONS.format(
        repo=req.repo, worktree=worktree, branch=req.branch, source=req.source, what=req.what,
        why=req.why, strategy=f"\n  strategy: {req.strategy}" if req.strategy else "",
        allowed=", ".join(allowed), repo_rule=FRAB_RULE if req.repo == "frab" else "",
        tests=" ".join(repo_cfg["test_command"]))
    result = run_agent(budget, prompt, session=session, agent="programmer",
                       model=config.get("model", "sonnet"),
                       max_turns=int(config.get("max_turns", 60)), cwd=worktree,
                       extra_args=("--allowedTools", TOOLS, "--disallowedTools", "Bash"))
    req.reply = (result.text or "").strip()[:1500]
    req.tokens = getattr(budget, "spent_tokens", 0)
    if result.exit_code != 0:
        req.problems.append(f"the agent stopped before finishing (exit {result.exit_code})")
    paths = changed_paths(worktree)
    req.problems += path_problems(paths, allowed)
    if req.problems:
        req.status = "failed"
        return req
    passed, tail = run_tests(worktree, list(repo_cfg["test_command"]),
                             int(repo_cfg.get("test_timeout_s", 1800)))
    if not passed:
        req.problems.append(f"the tests failed:\n{tail}")
        req.status = "failed"
        return req
    diff, req.diffstat = diff_text(worktree)
    review = run_agent(budget, REVIEW.format(repo=req.repo, branch=req.branch, what=req.what,
                                             why=req.why, diff=diff[:60_000]),
                       session=session, agent="programmer-review",
                       model=config.get("reviewer_model", "sonnet"), max_turns=20, cwd=worktree,
                       extra_args=("--allowedTools", REVIEW_TOOLS,
                                   "--disallowedTools", "Bash,Write,Edit"))
    req.tokens = getattr(budget, "spent_tokens", 0)
    ok, req.review = review_lines(review.text or "")
    if not ok:
        req.status = "blocked"
        return req
    _git(worktree, "commit", "-q", "-m",
         f"{req.what[:60]} (programmer agent, request {req.id})\n\n"
         f"Why: {req.why}\nSource: {req.source}\n\n{req.reply[:800]}")
    req.commit = _git(worktree, "rev-parse", "HEAD").strip()
    req.status = "ready"
    return req


# --------------------------------------------------------------------------
# The owner's decisions


def merge(req: Request, config: dict) -> str:
    """Merge a ready request's branch into its base and push (the owner's command). Refuses a
    base checkout with uncommitted changes or on another branch, and runs the tests first."""
    if req.status != "ready":
        raise ValueError(f"request {req.id} is {req.status}, not ready")
    repo_cfg = config["repos"][req.repo]
    repo_path = _abs(Path(repo_cfg["path"]))
    base = repo_cfg.get("base", "main")
    current = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if current != base:
        raise ValueError(f"{repo_path} is on {current}, not {base}: merge refused")
    dirty = [line for line in _git(repo_path, "status", "--porcelain").splitlines()
             if line.strip()]
    if dirty:
        raise ValueError(f"{repo_path} has uncommitted changes ({len(dirty)}): merge refused")
    passed, tail = run_tests(Path(req.worktree), list(repo_cfg["test_command"]),
                             int(repo_cfg.get("test_timeout_s", 1800)))
    if not passed:
        raise ValueError(f"the tests fail on {req.branch} now:\n{tail}")
    _git(repo_path, "merge", "--no-ff", "-q", "-m",
         f"Merge {req.branch}: {req.what[:60]} (owner's merge of programmer request {req.id})",
         req.branch)
    _git(repo_path, "push", "-q")
    drop_worktree(req, config)
    req.status = "merged"
    req.decided = f"merged {datetime.now(UTC).date().isoformat()}"
    return _git(repo_path, "rev-parse", "HEAD").strip()


def drop_worktree(req: Request, config: dict, *, delete_branch: bool = False) -> None:
    repo_path = _abs(Path(config["repos"][req.repo]["path"]))
    if req.worktree:
        _git(repo_path, "worktree", "remove", "--force", req.worktree, check=False)
        shutil.rmtree(req.worktree, ignore_errors=True)
    if delete_branch and req.branch:
        _git(repo_path, "branch", "-D", req.branch, check=False)


def reject(req: Request, config: dict, reason: str) -> None:
    drop_worktree(req, config, delete_branch=True)
    req.status = "rejected"
    req.decided = f"rejected {datetime.now(UTC).date().isoformat()}: {reason}"


# --------------------------------------------------------------------------
# The morning report


_STATUS_RU = {"queued": "в очереди", "ready": "готово, ждёт вашего слияния",
              "failed": "не вышло", "blocked": "остановлено проверяющим, решает владелец",
              "merged": "слито", "rejected": "отклонено"}


def report_lines(entries: list[dict], queue: list[Request]) -> list[str]:
    """The night's attempts and what waits for the owner, in plain Russian."""
    lines = ["Пишет код, которого не хватает для проверки направления, — в отдельной ветке; в "
             "основную ветку ничего не попадает без владельца (`qlab code merge <id>`).", ""]
    for e in entries:
        lines.append(f"- **{e['id']}** ({e['repo']}): {e['what']} — "
                     f"{_STATUS_RU.get(e['status'], e['status'])}; {e.get('tokens', 0):,} "
                     "токенов")
        lines += [f"    - {p}" for p in e.get("problems") or []]
        lines += [f"    - проверяющий: {g}" for g in e.get("review") or []]
        if e.get("status") == "ready":
            lines.append(f"    - ветка `{e['branch']}`; изменения: "
                         f"{(e.get('diffstat') or '').strip().splitlines()[-1:]}")
    waiting = [r for r in queue if r.status in ("ready", "blocked")]
    if waiting:
        lines += ["", "Ждут вашего решения:"]
        lines += [f"- {r.id} ({r.repo}): {r.what} — {_STATUS_RU[r.status]}" for r in waiting]
    queued = [r for r in queue if r.status in ("queued", "failed")]
    if queued:
        lines.append(f"В очереди: {len(queued)}.")
    return lines


__all__ = ["CONFIG", "QUEUE", "Request", "changed_paths", "diff_text", "drop_worktree",
           "enqueue", "load_config", "load_queue", "make_worktree", "merge", "next_request",
           "path_problems", "reject", "report_lines", "review_lines", "run_request", "run_tests",
           "save_queue"]
