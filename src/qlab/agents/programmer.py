"""The programmer agent: a code request -> a tested change on a branch, never on main
(docs/PROGRAMMER.md).

Owner, 2026-10-26: «так нам нужен еще один агент - который будет дописывать код если нужно»;
«будет дописывать код если руководитель посчитает что направление - следует проверить»;
«да, делай агента-программиста».

A request says which repository (`frab` -- the live strategies' code, or `qlab`), what to
make and why. It comes from the research director's `code_requests` (`qlab.directions`, applied
once the owner has reviewed the directions) or from the owner (`qlab code request`). Up to
`per_night` requests a night, the ones that give the generator a new block first:

Two levels (owner, 2026-10-30: «может нужно сделать программиста 2х уровневым - опус делает план
а sonnet - пишет. так как сейчас какаято- чехарда»): a planner on `planner_model` (read-only)
reads the request and the code and writes the plan; a question about a strategy's behaviour that
nobody answered blocks the request BEFORE a writing token is spent; the writer follows the plan.

1. a separate worktree of the repository on a new branch `agent/<id>`, from its base branch (a
   retry's worktree is first brought onto the base branch's tip and its environment made ready:
   the sibling repository the tests cite, a migrated database);
1a. the planner call (`programmer-plan`): the plan, or the open questions that block the request;
2. one agent call in that worktree: it may read anything, write files (Read, Glob, Grep, Write,
   Edit) and run ONE shell command, the repository's own test command (`Bash(<its prefix>:*)`,
   `tests_pattern`) -- it sees its tests fail while it works and goes on until they pass
   (owner, 2026-10-28: «добавь в инструкцию доводить тесты до конца - если ошибка в них и в
   коде - не останавливаться после пары попыток»);
3. code checks what it wrote: every changed path matches the repository's `allowed` list
   (`night/programmer.yaml`) -- for frab only the pure strategy modules q-lab's stand runs,
   never an engine, the exchange or execution; a test file must be among the changes;
4. code runs the repository's tests in the worktree; every test passes, the parity tests with
   them (the default behaviour must not change). The agent's own test run is never trusted:
   this one is the gate. A test that fails here and ALSO fails on the untouched base (the same
   command on a clean checkout of the base branch's tip) is the environment's, not the agent's:
   it is named in the report and costs no fix pass (2026-10-28-1 burned 2.6M tokens on 34 such
   failures). A failed check does not end the attempt: the same worktree goes back
   to the agent for a fix pass with the numbered problems and the failing output's tail, up to
   `fix_rounds` passes, and only then is the attempt failed. A retry (the next attempt) EDITS
   the earlier attempt's worktree and branch, like the carder's retry, instead of starting
   from scratch -- a fresh start would break what was right;
5. a reviewer agent reads the request and the diff; what it names goes back to the writer;
6. a change that passes both is committed on its branch and waits: `qlab code merge <id>` -- the
   owner's command -- merges it into the base branch and pushes; `qlab code reject <id>` drops
   it. Nothing here merges or pushes.

The loop (owner, 2026-10-31: «цикл программист/проверяющий - продолжать пока не заработает
задача. но не более 7 итераций. программист должен иметь цель сделать работающий код»): steps 2
to 5 repeat in the SAME worktree -- the writer's pass, code's checks, the reviewer -- until the
checks pass and the reviewer accepts, at most `max_iterations` (7) times for a request, over as
many nights as its share of the stage's allowance takes. Before, a reviewer's remark ended the
request as `blocked` for the owner: two requests that had written working code stopped on an empty
leftover file, a reformatted import and a family missing from a list, at 5.3M tokens. One
iteration is one writer pass plus its checks and, when they pass, its review. What does not count:
an iteration the environment ended (a setup or git step failed), and the planner's stop for an
unanswered behaviour question, which comes before any writing. The same failure three times in a
row stops the loop early (`blocked`); seven iterations without acceptance end it as `failed`, with
the last remarks, for the owner. A stop of the budget in the middle keeps the worktree and the
remarks, and the next night goes on from there.

The agent's spending goes through the budget guard (`qlab.budget.run_agent`)."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

CONFIG = Path("night/programmer.yaml")
QUEUE = Path("data/programmer/queue.json")
WORKTREES = Path("data/programmer/worktrees")
PLANS = Path("data/programmer/plans")
TOOLS = "Read,Glob,Grep,Write,Edit"
"""The agent's tools without a shell; `tools_pattern` adds the one allowed shell command."""
REVIEW_TOOLS = "Read,Glob,Grep"
BRANCH_PREFIX = "agent/"
STATUSES = ("queued", "ready", "failed", "blocked", "merged", "rejected")
"""queued: waits for a night (also one the budget or the environment stopped in the middle of its
loop); ready: the checks passed and the reviewer accepted, waits for the owner's merge; failed:
`max_iterations` iterations without acceptance (the last remarks are kept; `qlab code retry`);
blocked: an open behaviour question, the same failure three times, or three stops by the
environment -- the owner decides; merged / rejected: the owner's decision."""
DEFAULT_MAX_ITERATIONS = 7
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
    fix_passes: int = 0  # fix passes the last attempt used after code's checks failed
    review: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    tokens: int = 0
    decided: str | None = None  # the owner's merge or reject, with its reason
    kind: str | None = None  # "mechanism": gives the generator a new block (taken first)
    plan: str | None = None  # the planner's plan of the last attempt (docs/PROGRAMMER.md)
    plan_file: str | None = None  # the same plan on disk, one file per attempt
    plan_hash: str | None = None
    plan_tokens: int = 0  # what the planner spent in the last attempt
    questions: list[str] = field(default_factory=list)  # open behaviour questions: the owner's
    answers: list[str] = field(default_factory=list)  # the owner's answers, given to both levels
    failures: list[dict] = field(default_factory=list)  # {"attempt", "plan", "sig"} per failure
    env_failures: list[str] = field(default_factory=list)  # tests that fail on the base too
    env_stops: int = 0  # attempts the environment (not the work) ended
    iterations: int = 0  # writer passes that went through checks and review (the loop's count)
    remarks: list[str] = field(default_factory=list)  # what the next writer pass must address
    history: list[dict] = field(default_factory=list)  # {"n", "result", "remarks"} per iteration
    replan: bool = False  # the owner asked for another try: the planner revises its plan first
    stopped: str | None = None  # why the last session stopped before the loop ended (budget)


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
    req.kind = request_kind(req)
    queue.append(req)
    return req


_MECHANISM = re.compile(r"\b(blocks?|panels?)\b|блок|панел|связк|способ собрать", re.IGNORECASE)
_NOT_A_MECHANISM = re.compile(r"anomal|counterfactual|flag\b|аномал", re.IGNORECASE)


def request_kind(req: Request) -> str:
    """`mechanism` when the request gives the generator a new building block (a signal, a
    selection, a panel, a way to put legs together) -- owner, 2026-10-30: new strategies wait
    for such blocks, so these are taken first; `other` for a fix, a check or a report line."""
    if req.kind:
        return req.kind
    text = req.what
    if req.source.startswith("anomaly investigator") or _NOT_A_MECHANISM.search(text):
        return "other"
    return "mechanism" if _MECHANISM.search(text) else "other"


def _started(req: Request) -> bool:
    """Work in progress: a worktree with remarks still to address."""
    return bool(req.worktree) and bool(req.remarks or req.review or req.problems)


def runnable(queue: list[Request], max_iterations: int, skip: set[str] = frozenset()
             ) -> list[Request]:
    """The requests a night may take, in the order it takes them: the ones already begun (their
    loop goes on in the worktree they have), then the ones that give the generator a new block,
    then queue order, which is age."""
    todo = [r for r in queue if r.status == "queued" and r.iterations < max_iterations
            and r.id not in skip]
    return sorted(todo, key=lambda r: (not _started(r), request_kind(r) != "mechanism"))


def next_request(queue: list[Request], max_iterations: int, skip: set[str] = frozenset()
                 ) -> Request | None:
    todo = runnable(queue, max_iterations, skip)
    return todo[0] if todo else None


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


def reuse_worktree(repo_path: Path, req: Request, *, root: Path = WORKTREES) -> Path | None:
    """The earlier attempt's worktree, when it is still there on its branch -- a retry edits it
    (what was right in it stays) instead of starting from scratch. None: no such worktree."""
    branch = BRANCH_PREFIX + req.id
    path = _abs(root) / f"{req.repo}-{req.id}"
    if not (path / ".git").exists():
        return None
    if _git(path, "rev-parse", "--abbrev-ref", "HEAD", check=False).strip() != branch:
        return None
    req.branch, req.worktree = branch, str(path)
    return path


def revert_paths(worktree: Path, paths: list[str]) -> None:
    """Undo changes the agent made outside what it may change: a tracked file is restored, a new
    one is deleted (the agent has no way to delete a file itself)."""
    for rel in paths:
        if _git(worktree, "ls-files", "--error-unmatch", "--", rel, check=False).strip():
            _git(worktree, "checkout", "--", rel, check=False)
        else:
            (worktree / rel).unlink(missing_ok=True)


def tests_prefix(test_command: list[str]) -> str:
    """The words of the test command before its first option (`uv run pytest`; for
    `python -m pytest` the `-m pytest` stays): the shell command the agent may run, with any
    arguments after it."""
    words: list[str] = []
    it = iter(test_command)
    for word in it:
        if word == "-m":
            words += [word, next(it, "")]
        elif word.startswith("-"):
            break
        else:
            words.append(word)
    return " ".join(w for w in words if w)


def tools_pattern(test_command: list[str]) -> str:
    """`--allowedTools` value: the file tools and Bash restricted to the test command's prefix --
    no other shell command (a command outside the list is refused in a non-interactive run)."""
    return f"{TOOLS},Bash({tests_prefix(test_command)}:*)"


def link_ignored(repo_path: Path, worktree: Path, paths: list[str]) -> None:
    """Link untracked data the tests read (e.g. frab's research/data) from the main checkout, and
    keep the links out of git: a symlink does not match an ignore rule written for a directory
    ("data/"), so each link is named in the repository's info/exclude (shared by its worktrees;
    the path is ignored in the main checkout already)."""
    if paths:
        common = Path(_git(worktree, "rev-parse", "--git-common-dir").strip())
        exclude = (common if common.is_absolute() else worktree / common) / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        have = exclude.read_text(encoding="utf-8").splitlines() if exclude.is_file() else []
        missing = [f"/{rel}" for rel in paths if f"/{rel}" not in have]
        if missing:
            with exclude.open("a", encoding="utf-8") as fh:
                fh.write("".join(f"{line}\n" for line in missing))
    for rel in paths:
        source, target = repo_path / rel, worktree / rel
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)


GIT_IDENTITY = ("-c", "user.name=qlab-programmer", "-c", "user.email=programmer@qlab.invalid")


class EnvironmentFailure(Exception):  # noqa: N818 - the environment's, not the work's
    """The attempt could not be judged because of the environment (a git or a setup step failed),
    not because of the agent's work: the night does not count it as an attempt."""


def sync_worktree(repo_path: Path, base: str, worktree: Path) -> str:
    """Bring the worktree's branch onto the tip of `base` before its tests are judged: a retry's
    worktree is a night or two old and the base has moved (2026-10-30: the files the tests read
    were committed after the worktree was cut). Uncommitted work is carried over. Returns
    `current` (nothing to do), `rebased` or `conflict` (the work did not apply to the new tip:
    the worktree is left as it was)."""
    tip = _git(repo_path, "rev-parse", base).strip()
    behind = subprocess.run(["git", "-C", str(worktree), "merge-base", "--is-ancestor", tip,
                             "HEAD"], capture_output=True).returncode != 0
    if not behind:
        return "current"
    dirty = bool(_git(worktree, "status", "--porcelain").strip())
    if dirty:
        _git(worktree, "add", "-A")
        _git(worktree, *GIT_IDENTITY, "commit", "-q", "--no-verify", "-m", "WIP (temporary)")
    done = subprocess.run(["git", "-C", str(worktree), *GIT_IDENTITY, "rebase", tip],
                          capture_output=True, text=True)
    if done.returncode != 0:
        _git(worktree, "rebase", "--abort", check=False)
    if dirty:
        _git(worktree, "reset", "-q", "--mixed", "HEAD~1")  # the work is back, unstaged
    return "rebased" if done.returncode == 0 else "conflict"


def link_siblings(root: Path, siblings: dict) -> None:
    """Sibling repositories the tests cite by a path next to the checkout
    (`qlab.pipeline.sources.FRAB_ROOT` is `<checkout>/../funding-rate-arbitrage`): a worktree sits
    in `data/programmer/worktrees`, so a link of that name in the worktrees' folder is where the
    tests look for it. Read only: nothing is written to the sibling."""
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (siblings or {}).items():
        link, source = root / name, Path(target).expanduser()
        if source.exists() and not link.exists() and not link.is_symlink():
            link.symlink_to(source)


def prepare_environment(worktree: Path, repo_cfg: dict, config: dict, root: Path) -> None:
    """What a fresh worktree lacks that the tests read: the sibling repositories (`siblings`),
    the ignored data (`link_ignored`) and whatever `prepare` commands make (for q-lab a migrated
    database: a worktree has none, and tests open `data/qlab.db` of their working directory --
    2026-10-28-1 saw an empty file with no tables). A failing command is the environment's."""
    link_siblings(root, config.get("siblings") or {})
    link_ignored(_abs(Path(repo_cfg["path"])), worktree, list(repo_cfg.get("link_ignored") or []))
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    for command in repo_cfg.get("prepare") or []:
        done = subprocess.run(list(command), cwd=worktree, capture_output=True, text=True, env=env)
        if done.returncode != 0:
            raise EnvironmentFailure(
                f"{' '.join(command)} failed in {worktree}: {(done.stderr or done.stdout)[-400:]}")


_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+?\.py(?:::\S.*?)?|\S+?\.py)(?: - .*)?$",
                          re.MULTILINE)


def failed_tests(output: str) -> list[str]:
    """The test ids pytest's short summary names as failed or errored."""
    return sorted(set(_FAILED_LINE.findall(output)))


def files_command(test_command: list[str], worktree: Path, files: list[str]) -> list[str]:
    """The test command with its path arguments replaced by `files`: the failing tests' files
    are what the untouched base is asked to run."""
    split = next((i + 1 for i, w in enumerate(test_command) if w == "pytest"), len(test_command))
    words, skip_next = list(test_command[:split]), False
    for word in test_command[split:]:
        if skip_next:
            words.append(word)
            skip_next = False
        elif word in ("-p", "-k", "-m", "-c", "-o", "--rootdir", "--ignore"):
            words.append(word)
            skip_next = True
        elif not word.startswith("-") and (worktree / word).exists():
            continue  # a path argument
        else:
            words.append(word)
    return [*words, *files]


def base_worktree(repo_path: Path, base: str, root: Path) -> Path:
    """A clean checkout of the base branch's tip, detached, kept between runs (`_base-<repo>`)."""
    tip = _git(repo_path, "rev-parse", base).strip()
    path = _abs(root) / f"_base-{repo_path.name}"
    if not (path / ".git").exists():
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        _git(repo_path, "worktree", "prune")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo_path, "worktree", "add", "--detach", str(path), tip)
    elif _git(path, "rev-parse", "HEAD").strip() != tip:
        _git(path, "checkout", "-q", "--detach", "--force", tip)
        _git(path, "clean", "-fdq")
    return path


@dataclass
class Judgement:
    """The repository's tests, judged: `passed` once only the environment's failures remain;
    `env_failed` are the tests that fail on the untouched base too."""
    passed: bool
    output: str
    env_failed: list[str] = field(default_factory=list)


def judge_tests(worktree: Path, repo_cfg: dict, config: dict, root: Path) -> Judgement:
    """Run the repository's tests in the worktree. A failure is then run again, by file, on a
    clean checkout of the base branch's tip with the same command: a test that fails there too is
    not the agent's (2026-10-28-1: 34 failures of a worktree with no sibling repository and no
    database cost three fix passes and 2.6M tokens). Failures that cannot be read from the output
    (a crash, a timeout) are never excused."""
    command = list(repo_cfg["test_command"])
    timeout = int(repo_cfg.get("test_timeout_s", 1800))
    passed, output = run_tests(worktree, command, timeout)
    if passed:
        return Judgement(True, output)
    failed = failed_tests(output)
    if not failed:
        return Judgement(False, output)
    repo_path = _abs(Path(repo_cfg["path"]))
    base_path = base_worktree(repo_path, repo_cfg.get("base", "main"), root)
    prepare_environment(base_path, repo_cfg, config, root)
    files = sorted({f.split("::")[0] for f in failed if (base_path / f.split("::")[0]).is_file()})
    on_base: set[str] = set()
    if files:
        _, base_output = run_tests(base_path, files_command(command, base_path, files), timeout)
        on_base = set(failed_tests(base_output))
    env_failed = [f for f in failed if f in on_base]
    if len(env_failed) < len(failed):
        return Judgement(False, output, env_failed)
    return Judgement(True, output, env_failed)  # every failure is the base's own


def skipped_required(output: str, must_run: list[str]) -> list[str]:
    """The required tests (`must_run`) the run skipped -- pytest -rs names each skip."""
    return [name for name in must_run
            if re.search(rf"^SKIPPED \[\d+\] \S*{re.escape(name)}", output, re.MULTILINE)]


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
    text = (out.stdout + out.stderr).strip()
    return out.returncode == 0, text


def _tail(output: str, lines: int = 15) -> str:
    return "\n".join(output.splitlines()[-lines:])


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

YOUR GOAL is code that WORKS: it does what the request asks, and your own tests prove it by
exercising the new behaviour with numbers you worked out by hand -- not code that merely gets past
the checks. After you finish, code runs the whole test command and a reviewer reads the diff. If
either has a remark, it comes back to you in this same working directory and you go on from what
is there, until the checks pass and the reviewer accepts ({rounds}). So do not leave a remark for
the next round that you can see now: a leftover or empty file, a line reformatted without a
reason, a name missing from a list that names every family are all things to fix before you
finish, and they are quick.

The request ({source}):
  what: {what}
  why: {why}{strategy}{answers}
{plan}{previous}
Rules -- code checks every one of them after you finish, and a broken rule fails the request:
- Write ONLY files matching these patterns (relative to the repository root): {allowed}. A change
  anywhere else fails the request.{repo_rule}
- The behaviour with the default settings must not change AT ALL. Add the new capability behind a
  setting whose default reproduces today's behaviour exactly (bit for bit), the way earlier
  settings were added (look at how the code already handles optional settings: stored only when
  set, validated, documented in a comment). Never change a default.
- Every existing test must keep passing unchanged, the parity tests among them. The one edit to
  an existing test that is allowed: registering the new setting where a test pins the list of
  settings or fields. Never weaken, delete or skip an existing test.
- Add tests for the new behaviour, in the repository's test folders: that the default is
  today's behaviour, that the new setting does what the request asks, that bad values are refused.
- Code is in English (names, comments, docstrings, test names); match the style of the code
  around it. No new dependency.

Running the tests -- you have ONE shell command, the repository's test command, and no other:
`{tests}`. Run it from your working directory, alone (no pipe, no `cd`, no redirect, no `&&`); you
may shorten it to the part you work on (`{prefix} <path>`, `{prefix} -k <name>`) and run the whole
at the end. Write the code and its tests, then run them, and KEEP GOING until ALL tests pass, the
new ones and the existing ones. Code runs the same command after you finish and its result is the
only one that counts: your word that the tests pass is never trusted, and a failing run fails the
request.
When a test fails, do not stop after a couple of tries and do not hand the request back: find the
cause. Decide first whether the TEST or the CODE is wrong, and fix whichever is wrong. A test you
wrote can be wrong as easily as the code: check the test's own numbers against what it claims --
work out by hand what the inputs it builds (prices, returns, windows, dates) give, and compare that
with what its name and its comments say it builds. Example: a test whose comment says "the 14-day
return is +5.6% and the 30-day one is -5%" while the prices it builds give a 30-day return of 0%
is a wrong test; the code was right to return 0, and the test's numbers or its expected value are
what must change. NEVER weaken, delete or skip an existing test to make a run pass (the one edit
allowed above stays the only one); a test you wrote yourself you may fix freely.

Reply in a few sentences: what you changed, the names of the new settings and their defaults,
whether you ran the tests and how that ended, and anything the reviewer should look at."""


PLAN_BLOCK = """
The PLAN (written for this request by the planner, who read the code; follow it: the files, the
functions, the tests, what must not change). If you find that a fact in it is wrong, do the
closest right thing and say so in your reply. The plan:

{plan}
"""

PLANNER = """You are the planner of q-lab's programmer agent. q-lab is a research lab that finds and
filters trading strategies; its verdicts are made by code. You do NOT write code: you read the
request and the code of the repository `{repo}` (your working directory, branch {branch}; you have
read-only tools) and write the PLAN that a second agent, who writes the code, will follow. A plan
that is concrete saves that agent from guessing; a guess about how a strategy should behave costs
a whole attempt.

The request ({source}):
  what: {what}
  why: {why}{strategy}{answers}
{previous}
The writer may change ONLY files matching: {allowed}. The repository's test command is
`{tests}`.{repo_rule}

Write the plan with these headings:
1. Files and functions: every file to change or create (path), and in it the functions, classes and
   settings to add or change, with names and signatures, after reading how the neighbouring code
   does the same kind of thing.
2. Settings and defaults: each new setting, its default (the behaviour with the defaults must stay
   today's, bit for bit), where it is validated and stored.
3. Tests to add: file, test names, what each asserts -- with the expected numbers worked out by hand
   from the inputs the test builds.
4. What must not change: the existing behaviour, tests and files that are not to be touched.
5. How it is checked: the exact test commands, narrow first, the whole last.
6. Open questions.

Open questions are ONLY decisions about how a STRATEGY should behave (when it trades, how much, what
counts as a signal) that change its results, that the request does not answer, and that two sensible
people would answer differently. Every such question goes on a line of its own starting `OPEN: `.
Decisions about code structure, naming, file layout are yours: take them in the plan, do not ask.
Never choose a strategy's behaviour for the owner to avoid asking. If there is any `OPEN:` line the
request is stopped and the owner is asked, before anything is written.

The last line is `VERDICT: PROCEED` (no open question) or `VERDICT: BLOCKED` (at least one).
"""

PLAN_RETRY = """
This is a RETRY. The previous attempt (plan below) failed and the writer's work is still in the
working directory ({files}). {repeats}Read why it failed, look at the code as it now is, and REVISE
the plan: say what was wrong in the earlier plan and what the writer must do differently. Do not
repeat a plan that has failed; if you cannot see a different way that would work, or the failure
shows a decision about a strategy's behaviour that is unanswered, put it on an `OPEN:` line.

Why the previous attempt failed (numbered):
{problems}

The previous plan:
{previous_plan}
"""

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

Your remarks go straight back to the writer, who fixes them in this same working directory and
sends the work to you again, so say exactly what to change and where. Name a gap only when
something is wrong or missing (a wrong behaviour, a missing test, a leftover file, an unrelated
change, an entry missing from a list that must name every member); a taste in style is not one.

Reply with lines only: `GAP: <what is wrong, where, and what to do>` for every problem you are
sure of, then a last line `VERDICT: OK` if there is none or `VERDICT: GAP` if there is any."""


RETRY_NOTE = """
This is a CONTINUATION, not a fresh start. The earlier attempt's work is in your working directory
now; the files it changed: {files}. Do NOT write it again from scratch: read what is there, keep
everything that is right, and fix exactly what is named below.
{header}
{problems}
"""

ROUND_NOTE = """
This is round {n} of at most {limit}. The work of the earlier rounds is in your working directory;
the files changed so far: {files}. Do NOT write it again from scratch: read what is there, keep
everything that is right, and fix exactly what is named below -- every remark, the small ones too,
because the work is accepted only when the checks pass and the reviewer has no remark left.
{reverted}
Remarks, numbered:
{remarks}
"""

FIX_NOTE = """
Code has just checked your work and it did NOT pass. Your work is in your working directory; the
files changed so far: {files}. Fix exactly what is named below -- the cause, not the symptom --
keep everything that is right, and run the tests again until they all pass.
{reverted}
Problems, numbered:
{problems}
"""


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"  {n}. " + line.replace("\n", "\n     ") for n, line in enumerate(lines, 1))


def continuation(kind: str, problems: list[str], files: list[str], *,
                 reverted: tuple[str, ...] | list[str] = ()) -> str:
    """The prompt's note for a worktree that already holds work: a retry (`kind` "retry", the
    earlier attempt's problems) or a fix pass (`kind` "fix", code's checks of this attempt)."""
    shown = ", ".join(files) if files else "none"
    if kind == "fix":
        note = ("Files outside what you may change were found and deleted or restored by code: "
                + ", ".join(reverted) + ".") if reverted else ""
        return FIX_NOTE.format(files=shown, reverted=note, problems=_numbered(problems))
    header = ("The previous attempt failed on (numbered):" if problems else
              "The previous attempt was cut short before it finished, with no problem recorded: "
              "finish it, run the tests, and make them pass.")
    return RETRY_NOTE.format(files=shown, header=header, problems=_numbered(problems))


def round_note(n: int, limit: int, remarks: list[str], files: list[str], *,
               reverted: tuple[str, ...] | list[str] = ()) -> str:
    """The prompt's note for round `n` of the loop, when earlier rounds left remarks."""
    note = ("Files outside what you may change were found and deleted or restored by code: "
            + ", ".join(reverted) + ".") if reverted else ""
    return ROUND_NOTE.format(n=n, limit=limit, files=", ".join(files) if files else "none",
                             reverted=note, remarks=_numbered(remarks))


def review_lines(text: str) -> tuple[bool, list[str]]:
    """(passed, the gaps) from the reviewer's reply. No verdict line is a failed review."""
    gaps = [m.group(1).strip() for m in re.finditer(r"^GAP:\s*(.+)$", text, re.MULTILINE)]
    verdict = re.findall(r"^VERDICT:\s*(OK|GAP)\s*$", text, re.MULTILINE)
    if not verdict:
        return False, [*gaps, "the reviewer gave no verdict"]
    return verdict[-1] == "OK" and not gaps, gaps


@dataclass
class Check:
    """Code's check of the worktree: `problems` are what the request record keeps, `detail` the
    same with the long output tail for the agent; `reverted` are paths code undid."""
    problems: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)
    env_failed: list[str] = field(default_factory=list)  # fail on the untouched base too

    @property
    def passed(self) -> bool:
        return not self.problems


def check_worktree(worktree: Path, repo_cfg: dict, config: dict | None = None,
                   root: Path = WORKTREES) -> Check:
    """The gate, run by code: the changed paths, then the repository's tests (the agent's own run
    is never trusted). A path outside `allowed` is undone on the spot, so it cannot stay in the
    worktree for a later pass or attempt; the problem is still reported."""
    allowed = list(repo_cfg["allowed"])
    paths = changed_paths(worktree)
    out = Check(problems=path_problems(paths, allowed))
    outside = [p for p in paths if not any(fnmatch.fnmatch(p, pat) for pat in allowed)]
    if outside:
        revert_paths(worktree, outside)
        out.reverted = outside
    out.detail = list(out.problems)
    if out.problems:
        return out
    judged = judge_tests(worktree, repo_cfg, config or {}, root)
    out.env_failed = judged.env_failed
    skipped = skipped_required(judged.output, list(repo_cfg.get("must_run") or []))
    if not judged.passed or skipped:
        head = (f"tests that prove the default unchanged were skipped: {skipped}\n" if skipped
                else "") + "the tests failed:\n" * (not judged.passed)
        own = [t for t in failed_tests(judged.output) if t not in judged.env_failed]
        shown = ("\n".join(f"FAILED {t}" for t in own[:15]) if judged.env_failed and own
                 else _tail(judged.output))
        out.problems.append(head + shown)
        out.detail.append(head + _tail(judged.output, 60))
    return out


def plan_hash(text: str) -> str:
    """A fingerprint of a plan that survives rewording of whitespace and case only."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()[:12]


def parse_plan(text: str) -> tuple[str, list[str], bool]:
    """(the plan, the open questions, blocked) from the planner's reply. A question on an `OPEN:`
    line, or a `VERDICT: BLOCKED`, blocks the request; a reply with no plan in it raises."""
    questions = [m.group(1).strip() for m in re.finditer(r"^OPEN:\s*(.+)$", text, re.MULTILINE)]
    verdicts = re.findall(r"^VERDICT:\s*(PROCEED|BLOCKED)\s*$", text, re.MULTILINE)
    blocked = bool(questions) or verdicts[-1:] == ["BLOCKED"]
    if blocked and not questions:
        questions = ["the planner stopped the request and named no question: read its plan"]
    if len(text.strip()) < 200 and not blocked:
        raise EnvironmentFailure("the planner gave no plan")
    return text.strip(), questions, blocked


def failure_signature(problems: list[str]) -> str:
    """What a failed attempt failed ON: the failing tests when it has them, else the first lines
    of each problem with the numbers taken out. Two attempts with the same signature failed the
    same way."""
    text = "\n".join(problems)
    names = failed_tests(text)
    basis = "\n".join(names) if names else re.sub(
        r"\d+", "#", "\n".join(p.splitlines()[0] for p in problems if p.strip()))
    return hashlib.sha1(basis.encode()).hexdigest()[:12]


def identical_failures(req: Request) -> int:
    """How many of the latest failures have the same signature. The owner's `retry` leaves a
    marker (no signature) that ends the count: an owner's second wish starts it again."""
    if not req.failures or req.failures[-1]["sig"] is None:
        return 0
    last, n = req.failures[-1]["sig"], 0
    for entry in reversed(req.failures):
        if entry["sig"] != last:
            break
        n += 1
    return n


def _answers(req: Request) -> str:
    if not req.answers:
        return ""
    return "\n  the owner's answers to open questions (binding):\n" + "\n".join(
        f"    - {a}" for a in req.answers)


def plan_request(req: Request, config: dict, worktree: Path, *, budget, session,
                 earlier: list[str], files: list[str], plans: Path = PLANS) -> bool:
    """The planner level: read-only agent call, the plan stored on the request and on disk.
    Returns False when the request is stopped before anything is written -- an open behaviour
    question (status `blocked`, the question for the owner), or the same plan again after the
    same failure."""
    repo_cfg = config["repos"][req.repo]
    retry = ""
    if earlier:
        repeats = identical_failures(req)
        retry = PLAN_RETRY.format(
            files=", ".join(files) or "none", problems=_numbered(earlier),
            previous_plan=req.plan or "(none was kept)",
            repeats=(f"The same failure has now happened {repeats} times. " if repeats > 1
                     else ""))
    spent = getattr(budget, "spent_tokens", 0)
    result = run_agent(
        budget, PLANNER.format(
            repo=req.repo, branch=req.branch, source=req.source, what=req.what, why=req.why,
            strategy=f"\n  strategy: {req.strategy}" if req.strategy else "",
            answers=_answers(req), previous=retry, allowed=", ".join(repo_cfg["allowed"]),
            tests=" ".join(repo_cfg["test_command"]),
            repo_rule=FRAB_RULE if req.repo == "frab" else ""),
        session=session, agent="programmer-plan", model=config["planner_model"],
        max_turns=int(config.get("planner_max_turns", 30)), cwd=worktree,
        extra_args=("--allowedTools", REVIEW_TOOLS, "--disallowedTools", "Bash,Write,Edit"))
    req.plan_tokens = getattr(budget, "spent_tokens", 0) - spent
    req.tokens = getattr(budget, "spent_tokens", 0)
    plan, questions, blocked = parse_plan(result.text or "")
    fresh_hash = plan_hash(plan)
    target = _abs(plans) / f"{req.id}-{req.attempts}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plan + "\n", encoding="utf-8")
    same_plan = bool(earlier) and req.plan_hash == fresh_hash and bool(req.failures)
    req.plan, req.plan_file, req.plan_hash = plan, str(target), fresh_hash
    if blocked:
        req.questions = questions
        req.status = "blocked"
        req.problems = [f"needs the owner's answer: {q}" for q in questions]
        return False
    if same_plan:
        req.status = "blocked"
        req.problems = ["the planner returned the same plan after the same failure: a re-attempt "
                        "on it would fail the same way; the owner decides",
                        *(e.splitlines()[0] for e in earlier if e.strip())]
        return False
    req.questions = []
    return True


def environment_stop(req: Request, exc: Exception, limit: int = 3) -> Request:
    """An attempt the environment ended does not count; three in a row wait for the owner."""
    req.attempts = max(0, req.attempts - 1)
    req.env_stops += 1
    req.problems = [f"environment: {type(exc).__name__}: {exc}"]
    req.status = "blocked" if req.env_stops >= limit else "queued"
    if req.status == "blocked":
        req.problems.append(f"{limit} attempts ended by the environment: the owner decides")
    return req


def run_request(req: Request, config: dict, *, budget: CandidateBudget | ExplicitNoBudget,
                session, worktrees: Path = WORKTREES, plans: Path = PLANS,
                checkpoint: Callable[[Request], None] | None = None) -> Request:
    """One session at one request: the loop of the writer, code's checks and the reviewer in one
    worktree until the checks pass and the reviewer accepts, `max_iterations` times at most.

    The worktree of an earlier session is edited, not recreated (brought onto the base's tip and
    made ready first); the planner (when `planner_model` is set) writes the plan once -- or an
    open question stops the request before a writing token is spent -- and revises it only when
    the owner asked for another try (`retry`). Every agent call goes through the budget guard; a
    stop (BudgetExhausted) in the middle leaves the worktree, the remarks and the count of
    finished iterations as they are, and the next night continues in them. A session the
    environment ended (EnvironmentFailure) does not count. `checkpoint(req)` is called after every
    finished iteration, so a crash cannot lose them."""
    try:
        return _attempt(req, config, budget=budget, session=session, worktrees=worktrees,
                        plans=plans, checkpoint=checkpoint)
    except EnvironmentFailure as exc:
        return environment_stop(req, exc)


def iteration_signature(problems: list[str], gaps: list[str]) -> str:
    """What an unaccepted iteration failed ON: the checks' problems, else the reviewer's gaps."""
    if problems:
        return failure_signature(problems)
    return failure_signature([f"reviewer: {g}" for g in gaps])


def _attempt(req: Request, config: dict, *, budget, session, worktrees: Path, plans: Path,
             checkpoint: Callable[[Request], None] | None) -> Request:
    repo_cfg = config["repos"][req.repo]
    repo_path = _abs(Path(repo_cfg["path"]))
    base = repo_cfg.get("base", "main")
    limit = int(config.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    # What the first writer pass of this session must address: the remarks of the last finished
    # iteration (kept in `remarks`, so a stop in the middle of an iteration loses none), or, for
    # a request that was ended the old way, its problems and the reviewer's gaps.
    earlier = list(req.remarks) or [
        *(p for p in req.problems if not p.startswith("environment:")),
        *(f"reviewer: {g}" for g in req.review)]
    req.attempts += 1
    req.reply, req.fix_passes, req.stopped = None, 0, None
    req.env_failures, req.plan_tokens = [], 0
    # An earlier session's worktree is edited, not recreated -- also when the count was reset
    # (the owner's answer, or a session the environment ended): the work in it is kept.
    worktree = reuse_worktree(repo_path, req, root=worktrees)
    continuing = worktree is not None
    if worktree is not None:
        try:
            moved = sync_worktree(repo_path, base, worktree)
        except RuntimeError as exc:
            raise EnvironmentFailure(f"bringing {worktree} onto {base}: {exc}") from exc
        if moved == "conflict":  # the earlier work does not apply to the new base: keep a patch
            _git(worktree, "add", "-A")
            lost = _abs(worktrees) / f"{req.repo}-{req.id}.conflict.patch"
            lost.write_text(_git(worktree, "diff", "--cached", "HEAD"), encoding="utf-8")
            worktree, continuing = None, False
            earlier = [*earlier, f"the earlier work did not apply to the new {base}; it is kept "
                       f"as {lost.name}, this is a fresh start"]
    if worktree is None:
        try:
            worktree = make_worktree(repo_path, base, req, root=worktrees)
        except RuntimeError as exc:
            raise EnvironmentFailure(f"making the worktree: {exc}") from exc
    prepare_environment(worktree, repo_cfg, config, worktrees)
    allowed = list(repo_cfg["allowed"])
    test_command = list(repo_cfg["test_command"])
    tools = tools_pattern(test_command)
    model, max_turns = config.get("model", "sonnet"), int(config.get("max_turns", 60))
    if config.get("planner_model") and (not req.plan or req.replan):
        if not plan_request(
                req, config, worktree, budget=budget, session=session,
                earlier=earlier if req.replan else [],
                files=changed_paths(worktree) if continuing else [], plans=plans):
            if req.status == "blocked":
                req.attempts = max(0, req.attempts - 1)  # nothing was written: not an attempt
                if not continuing:  # no work in it worth keeping
                    drop_worktree(req, config, delete_branch=True)
                    req.worktree = req.branch = None
            return req
        req.replan = False
    req.remarks = earlier
    plan_block = PLAN_BLOCK.format(plan=req.plan) if req.plan else ""

    def call(note: str, n: int) -> int:
        """One writer pass in the worktree; the agent's reply and the spend are recorded."""
        prompt = INSTRUCTIONS.format(
            repo=req.repo, worktree=worktree, branch=req.branch, source=req.source,
            what=req.what, why=req.why,
            strategy=f"\n  strategy: {req.strategy}" if req.strategy else "",
            answers=_answers(req), plan=plan_block, rounds=f"round {n} of at most {limit}",
            allowed=", ".join(allowed), repo_rule=FRAB_RULE if req.repo == "frab" else "",
            tests=" ".join(test_command), prefix=tests_prefix(test_command), previous=note)
        result = run_agent(budget, prompt, session=session, agent="programmer", model=model,
                           max_turns=max_turns, cwd=worktree, extra_args=("--allowedTools", tools))
        req.reply = (result.text or "").strip()[:1500]
        req.tokens = getattr(budget, "spent_tokens", 0)
        return result.exit_code

    reverted: list[str] = []
    while True:
        n = req.iterations + 1
        note = (round_note(n, limit, req.remarks, changed_paths(worktree), reverted=reverted)
                if req.remarks else "")
        exit_code = call(note, n)
        check = check_worktree(worktree, repo_cfg, config, worktrees)
        req.env_failures = check.env_failed
        reverted = check.reverted
        stopped = ([f"the agent stopped before finishing (exit {exit_code})"]
                   if exit_code != 0 else [])
        problems = stopped + check.problems
        gaps: list[str] = []
        if not problems:
            diff, req.diffstat = diff_text(worktree)
            review = run_agent(
                budget, REVIEW.format(repo=req.repo, branch=req.branch, what=req.what,
                                      why=req.why, diff=diff[:60_000]),
                session=session, agent="programmer-review",
                model=config.get("reviewer_model", "sonnet"), max_turns=20, cwd=worktree,
                extra_args=("--allowedTools", REVIEW_TOOLS, "--disallowedTools", "Bash,Write,Edit"))
            req.tokens = getattr(budget, "spent_tokens", 0)
            accepted, gaps = review_lines(review.text or "")
            if accepted:
                req.iterations, req.env_stops = n, 0
                req.history.append({"n": n, "result": "accepted", "remarks": 0})
                req.problems, req.review, req.remarks = [], [], []
                _git(worktree, "commit", "-q", "-m",
                     f"{req.what[:60]} (programmer agent, request {req.id})\n\n"
                     f"Why: {req.why}\nSource: {req.source}\n\n{req.reply[:800]}")
                req.commit = _git(worktree, "rev-parse", "HEAD").strip()
                req.status = "ready"
                if checkpoint is not None:
                    checkpoint(req)
                return req
        # Not accepted: the remarks go back to the writer for the next round, in the same worktree.
        req.iterations, req.env_stops = n, 0
        req.fix_passes += 1 if problems else 0
        req.problems, req.review = problems, gaps
        req.remarks = (stopped + check.detail) if problems else [f"reviewer: {g}" for g in gaps]
        req.history.append({"n": n, "result": "checks" if problems else "review",
                            "remarks": len(problems) or len(gaps)})
        req.failures.append({"attempt": req.attempts, "iteration": n, "plan": req.plan_hash,
                             "sig": iteration_signature(problems, gaps)})
        if checkpoint is not None:
            checkpoint(req)
        if identical_failures(req) >= 3:
            req.status = "blocked"
            req.problems.append("the same failure three times: the owner decides")
            return req
        if req.iterations >= limit:
            req.status = "failed"
            req.problems = [f"{limit} iterations without the checks passing and the reviewer "
                            "accepting: the last remarks follow; the owner decides "
                            f"(`qlab code retry {req.id}`)", *req.problems]
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
    worktree = Path(req.worktree)
    if sync_worktree(repo_path, base, worktree) == "conflict":
        raise ValueError(f"{req.branch} does not apply to the tip of {base} now: rebase it by hand")
    root = worktree.parent
    prepare_environment(worktree, repo_cfg, config, root)
    judged = judge_tests(worktree, repo_cfg, config, root)
    skipped = skipped_required(judged.output, list(repo_cfg.get("must_run") or []))
    if not judged.passed or skipped:
        raise ValueError(f"the tests fail or skip {skipped} on {req.branch} now:\n"
                         f"{_tail(judged.output)}")
    req.commit = _git(worktree, "rev-parse", "HEAD").strip()
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
              "failed": "не вышло за предел итераций, решает владелец",
              "blocked": "остановлено, решает владелец",
              "merged": "слито", "rejected": "отклонено"}


def _plan_gist(plan: str, limit: int = 300) -> str:
    """The first lines of a plan that are not headings, for a one-line sight of it."""
    body = " ".join(line.strip() for line in plan.splitlines()
                    if line.strip() and not line.lstrip().startswith(("#", "VERDICT", "OPEN")))
    return body[:limit] + ("…" if len(body) > limit else "")


def answer(req: Request, text: str) -> None:
    """THE OWNER'S ANSWER to the open questions of a request the planner stopped: it joins the
    request for both levels, the old plan goes, and the request is queued with a clean count (the
    attempts that were spent on a wrong guess do not count against it)."""
    if req.status != "blocked":
        raise ValueError(f"request {req.id} is {req.status}: nothing is waiting for an answer")
    req.answers.append(f"{datetime.now(UTC).date().isoformat()}: {text.strip()}")
    req.questions, req.plan, req.plan_file, req.plan_hash = [], None, None, None
    req.failures, req.problems, req.attempts, req.env_stops = [], [], 0, 0
    req.iterations, req.remarks, req.replan = 0, [], False
    req.status = "queued"


def retry(req: Request, text: str | None = None) -> None:
    """THE OWNER'S WISH for another try at a request the loop gave up on (`failed`: the iterations
    ran out; `blocked` by the same failure three times or by the environment): a clean count of
    iterations, the worktree and the work in it kept, the planner revises its plan by the last
    remarks, and the owner's text (when given) is the first thing the writer is told. A request
    waiting for an answer to a question is answered with `answer`, not retried."""
    if req.status not in ("failed", "blocked") or req.questions:
        raise ValueError(f"request {req.id} is {req.status}"
                         + (" and waits for an answer" if req.questions else "")
                         + ": there is nothing to retry")
    last = [*(p for p in req.problems if not p.startswith("environment:")),
            *(f"reviewer: {g}" for g in req.review)]
    req.remarks = [*([f"the owner: {text.strip()}"] if text and text.strip() else []),
                   *(req.remarks or last)]
    req.iterations, req.env_stops, req.replan = 0, 0, True
    req.failures.append({"retry": True, "sig": None})
    req.status = "queued"


_RESULT_RU = {"checks": "проверки кодом не прошли", "review": "замечания проверяющего",
              "accepted": "принято"}


def _iterations_ru(history: list[dict]) -> str:
    """`1 проверки кодом не прошли (2) → 2 принято`: how each finished iteration ended."""
    return " → ".join(
        f"{h['n']} {_RESULT_RU.get(h['result'], h['result'])}"
        + (f" ({h['remarks']})" if h.get("remarks") else "") for h in history)


def report_lines(entries: list[dict], queue: list[Request]) -> list[str]:
    """The night's attempts and what waits for the owner, in plain Russian."""
    lines = ["Пишет код, которого не хватает для проверки направления, — в отдельной ветке; в "
             "основную ветку ничего не попадает без владельца (`qlab code merge <id>`). Два "
             "уровня: планировщик (Opus) читает запрос и код и пишет план, исполнитель (Sonnet) "
             "пишет по плану; вопрос о поведении стратегии, на который никто не ответил, "
             "останавливает запрос до того, как потрачены токены на код. Дальше цикл: писатель, "
             "проверка кодом, проверяющий; замечания возвращаются писателю в ту же рабочую "
             "копию, пока проверки не пройдут и проверяющий не примет, но не больше 7 итераций "
             "на запрос.", ""]
    for e in entries:
        lines.append(f"- **{e['id']}** ({e['repo']}): {e['what']} — "
                     f"{_STATUS_RU.get(e['status'], e['status'])}; {e.get('tokens', 0):,} "
                     "токенов")
        if e.get("plan_file"):
            lines.append(f"    - план: `{e['plan_file']}`; планировщик потратил "
                         f"{e.get('plan_tokens', 0):,} токенов")
            if e.get("plan"):
                lines.append(f"    - суть плана: {_plan_gist(e['plan'])}")
        lines += [f"    - вопрос владельцу: {q}" for q in e.get("questions") or []]
        if e.get("history"):
            lines.append("    - итерации (писатель, проверка кодом, проверяющий): "
                         + _iterations_ru(e["history"]) + f"; всего {e.get('iterations', 0)}")
        if e.get("stopped"):
            lines.append(f"    - {e['stopped']}")
        if e.get("env_failures"):
            lines.append(f"    - упавших тестов, которые падают и на чистой основной ветке "
                         f"(это среда, а не работа агента; исправлений на них не тратилось): "
                         f"{len(e['env_failures'])}")
        lines += [f"    - {p}" for p in e.get("problems") or []]
        lines += [f"    - проверяющий: {g}" for g in e.get("review") or []]
        if e.get("status") == "ready":
            lines.append(f"    - ветка `{e['branch']}`; изменения: "
                         f"{(e.get('diffstat') or '').strip().splitlines()[-1:]}")
    waiting = [r for r in queue if r.status in ("ready", "blocked", "failed")]
    if waiting:
        lines += ["", "Ждут вашего решения:"]
        for r in waiting:
            lines.append(f"- {r.id} ({r.repo}): {r.what} — {_STATUS_RU[r.status]}")
            lines += [f"    - вопрос: {q} (ответ: `qlab code answer {r.id} --text \"...\"`)"
                      for q in r.questions]
            if r.status == "failed":
                lines += [f"    - последнее замечание: {p.splitlines()[0]}" for p in r.problems[1:]]
                lines += [f"    - последнее замечание проверяющего: {g}" for g in r.review]
                lines.append(f"    - ещё одна попытка: `qlab code retry {r.id}` (можно с "
                             "`--text \"...\"`: это скажут писателю первым)")
    started = [r for r in queue if r.status == "queued" and _started(r)]
    if started:
        lines += ["", "Начатые, продолжатся первыми в той же рабочей копии:"]
        lines += [f"- {r.id} ({r.repo}): {r.what}; итераций {r.iterations}" for r in started]
    queued = [r for r in queue if r.status == "queued"]
    if queued:
        first = sum(1 for r in queued if request_kind(r) == "mechanism")
        lines.append(f"В очереди: {len(queued)} (из них даёт генератору новый блок: {first}; "
                     "такие идут первыми).")
    return lines


__all__ = ["CONFIG", "DEFAULT_MAX_ITERATIONS", "PLANS", "QUEUE", "EnvironmentFailure", "Request",
           "answer", "changed_paths", "diff_text", "drop_worktree", "enqueue", "environment_stop",
           "failure_signature", "iteration_signature", "judge_tests", "load_config", "load_queue",
           "make_worktree", "merge", "next_request", "parse_plan", "path_problems", "plan_request",
           "reject", "report_lines", "request_kind", "retry", "reuse_worktree", "review_lines",
           "round_note", "run_request", "run_tests", "runnable", "save_queue", "sync_worktree",
           "tests_prefix", "tools_pattern"]
