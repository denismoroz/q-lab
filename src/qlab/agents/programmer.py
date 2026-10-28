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
   this one is the gate. A failed check does not end the attempt: the same worktree goes back
   to the agent for a fix pass with the numbered problems and the failing output's tail, up to
   `fix_rounds` passes, and only then is the attempt failed. A retry (the next attempt) EDITS
   the earlier attempt's worktree and branch, like the carder's retry, instead of starting
   from scratch -- a fresh start would break what was right;
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
"""The agent's tools without a shell; `tools_pattern` adds the one allowed shell command."""
REVIEW_TOOLS = "Read,Glob,Grep"
BRANCH_PREFIX = "agent/"
STATUSES = ("queued", "ready", "failed", "blocked", "merged", "rejected")
"""queued: waits for a night; ready: passed, waits for the owner's merge; failed: a guard or a
test failed even after the fix passes (tried again until max_attempts); blocked: the reviewer
named a gap, or the attempts ran out -- the owner decides; merged / rejected: the owner's
decision."""
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

The request ({source}):
  what: {what}
  why: {why}{strategy}
{previous}
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


RETRY_NOTE = """
This is a CONTINUATION, not a fresh start. The earlier attempt's work is in your working directory
now; the files it changed: {files}. Do NOT write it again from scratch: read what is there, keep
everything that is right, and fix exactly what is named below.
{header}
{problems}
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

    @property
    def passed(self) -> bool:
        return not self.problems


def check_worktree(worktree: Path, repo_cfg: dict) -> Check:
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
    passed, output = run_tests(worktree, list(repo_cfg["test_command"]),
                               int(repo_cfg.get("test_timeout_s", 1800)))
    skipped = skipped_required(output, list(repo_cfg.get("must_run") or []))
    if not passed or skipped:
        head = (f"tests that prove the default unchanged were skipped: {skipped}\n" if skipped
                else "") + "the tests failed:\n" * (not passed)
        out.problems.append(head + _tail(output))
        out.detail.append(head + _tail(output, 60))
    return out


def run_request(req: Request, config: dict, *, budget: CandidateBudget | ExplicitNoBudget,
                session, worktrees: Path = WORKTREES) -> Request:
    """One attempt at one request: worktree, agent (which runs the tests itself as it works),
    code's checks -- a failed check sends the same worktree back to the agent for a fix pass, up
    to `fix_rounds` times -- then review and commit. From the second attempt the earlier
    attempt's worktree is edited, not recreated. Every agent call goes through the budget guard;
    a stop (BudgetExhausted) in the middle leaves the worktree and the request as they are, and
    the next night continues in them."""
    repo_cfg = config["repos"][req.repo]
    repo_path = _abs(Path(repo_cfg["path"]))
    earlier = [*req.problems, *(f"reviewer: {g}" for g in req.review)]
    req.attempts += 1
    req.problems, req.review, req.reply, req.fix_passes = [], [], None, 0
    worktree = reuse_worktree(repo_path, req, root=worktrees) if req.attempts > 1 else None
    continuing = worktree is not None
    if worktree is None:
        worktree = make_worktree(repo_path, repo_cfg.get("base", "main"), req, root=worktrees)
    link_ignored(repo_path, worktree, list(repo_cfg.get("link_ignored") or []))
    allowed = list(repo_cfg["allowed"])
    test_command = list(repo_cfg["test_command"])
    tools = tools_pattern(test_command)
    model, max_turns = config.get("model", "sonnet"), int(config.get("max_turns", 60))

    def call(note: str) -> int:
        """One agent call in the worktree; the agent's reply and the spend are recorded."""
        prompt = INSTRUCTIONS.format(
            repo=req.repo, worktree=worktree, branch=req.branch, source=req.source,
            what=req.what, why=req.why,
            strategy=f"\n  strategy: {req.strategy}" if req.strategy else "",
            allowed=", ".join(allowed), repo_rule=FRAB_RULE if req.repo == "frab" else "",
            tests=" ".join(test_command), prefix=tests_prefix(test_command), previous=note)
        result = run_agent(budget, prompt, session=session, agent="programmer", model=model,
                           max_turns=max_turns, cwd=worktree, extra_args=("--allowedTools", tools))
        req.reply = (result.text or "").strip()[:1500]
        req.tokens = getattr(budget, "spent_tokens", 0)
        return result.exit_code

    note = (continuation("retry", ["\n".join(e.splitlines()[:20]) for e in earlier],
                         changed_paths(worktree)) if continuing else "")
    exit_code = call(note)
    while True:
        check = check_worktree(worktree, repo_cfg)
        stopped = ([f"the agent stopped before finishing (exit {exit_code})"]
                   if exit_code != 0 else [])
        req.problems = stopped + check.problems
        if not req.problems:
            break
        if req.fix_passes >= int(config.get("fix_rounds", 3)):
            req.status = "failed"
            return req
        req.fix_passes += 1
        exit_code = call(continuation("fix", stopped + check.detail, changed_paths(worktree),
                                      reverted=check.reverted))
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
    link_ignored(repo_path, Path(req.worktree), list(repo_cfg.get("link_ignored") or []))
    passed, output = run_tests(Path(req.worktree), list(repo_cfg["test_command"]),
                               int(repo_cfg.get("test_timeout_s", 1800)))
    skipped = skipped_required(output, list(repo_cfg.get("must_run") or []))
    if not passed or skipped:
        raise ValueError(f"the tests fail or skip {skipped} on {req.branch} now:\n"
                         f"{_tail(output)}")
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
        if e.get("fix_passes"):
            lines.append(f"    - исправлений по ходу работы (после проверки кодом): "
                         f"{e['fix_passes']}")
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
           "path_problems", "reject", "report_lines", "reuse_worktree", "review_lines",
           "run_request", "run_tests", "save_queue", "tests_prefix", "tools_pattern"]
