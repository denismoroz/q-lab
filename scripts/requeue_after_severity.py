"""One-off: put the two programmer requests that failed on the night of 2026-11-04 back in the
queue under the severity rules (owner, 2026-11-04: «делай все три по порядку разделяя находки на
Critical, hight, medium, low. цикл останавливать когда все critical and hight пофикшены»).

Requests 2026-10-30-3 and 2026-11-04-1 each ran all seven iterations and ended `failed`, the
reviewer finding a new, ever smaller remark every round. Their worktrees and plans are kept; this
script makes each one `queued` again with a clean count of iterations, and re-grades the last
remarks it holds, which are plain lines (the old reviewer gave no severity):

* a remark about a missing documentation line, a plain-language phrase, a prompt that does not name
  a new block, or about wording or naming is MEDIUM or LOW by the definitions in
  `qlab.agents.programmer.REVIEW`; anything else stays HIGH (the rule for a line without a
  severity); a failing run of the checks is CRITICAL, as the loop treats it;
* the graded remarks are kept as the first thing the writer is told, the MEDIUM and LOW ones also as
  the request's follow-up list, so the owner sees them whether or not they get fixed;
* the plan is kept (no planner call: `replan` stays off, the planner cost about a million tokens on
  each request), the failure marker of the early stop is reset, the worktree is untouched.

The queue file is backed up next to itself (`queue.json.before-severity`) before it is written.

    uv run python scripts/requeue_after_severity.py [--apply] [--queue PATH]

Without --apply it only says what it would do.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from qlab.agents import programmer as prog

IDS = ("2026-10-30-3", "2026-11-04-1")

_MEDIUM = re.compile(
    r"plain.language|no.jargon|blocks\.yaml|says nothing about|does not name|doesn't name|"
    r"add a bullet|docs?/|documentation|doc line|mention", re.IGNORECASE)
_LOW = re.compile(r"\b(style|wording|rename|naming|typo|comment)\b", re.IGNORECASE)
_TESTS_FAILED = re.compile(r"^the tests failed", re.IGNORECASE)
_FAILED_LINE = re.compile(r"^FAILED (\S+)", re.MULTILINE)


def grade(remark: str) -> tuple[dict, str]:
    """(the finding, why) for one old plain remark."""
    text = re.sub(r"^reviewer:\s*", "", remark.strip())
    if _TESTS_FAILED.match(text):
        failed = _FAILED_LINE.findall(text)
        named = ", ".join(failed) or "the whole test command"
        timing = "timing assertion" if "span(" in text and "<" in text else "failure"
        shorter = (f"the last run of the whole test command failed on {named} -- a {timing} "
                   "(a clock comparison with a 0.05 s margin) in a file this request does not "
                   "touch, while 3490 other tests passed. Run the test command again; if it "
                   "fails again, see whether your change can affect that test, and if it "
                   "cannot, leave the test alone and say so")
        return ({"severity": "CRITICAL", "text": shorter, "unparsed": False},
                "a failing run of the checks counts as CRITICAL")
    if _MEDIUM.search(text):
        return ({"severity": "MEDIUM", "text": text, "unparsed": False},
                "a missing documentation line, plain phrase or prompt mention: MEDIUM")
    if _LOW.search(text):
        return ({"severity": "LOW", "text": text, "unparsed": False}, "wording or naming: LOW")
    return ({"severity": "HIGH", "text": text, "unparsed": True},
            "no severity and no sign of a smaller kind: HIGH, as for any unparsed line")


def regrade(req: prog.Request) -> list[tuple[dict, str]]:
    """The graded remarks the request holds: its open remarks, else the reviewer's last gaps."""
    old = list(req.remarks) or [*(f"reviewer: {g}" for g in req.review)]
    return [grade(r) for r in old]


def requeue(req: prog.Request, graded: list[tuple[dict, str]]) -> None:
    findings = [f for f, _ in graded]
    req.findings = [f for f in findings if f["severity"] != "CRITICAL"]
    req.followups = [{**f, "round": req.iterations, "outside": False} for f in findings
                     if f["severity"] in ("MEDIUM", "LOW")]
    req.raised_files = sorted(set(req.raised_files)
                              | set(prog.remark_files([f["text"] for f in findings])))
    req.remarks = [f"reviewer [{f['severity']}]: {f['text']}" for f in findings]
    req.review = []
    req.problems = []
    req.iterations, req.env_stops, req.replan, req.stopped = 0, 0, False, None
    req.failures.append({"retry": True, "sig": None})
    req.status = "queued"


def main(apply: bool, path: Path = prog.QUEUE) -> None:
    queue = prog.load_queue(path)
    for request_id in IDS:
        req = next((r for r in queue if r.id == request_id), None)
        if req is None:
            print(f"{request_id}: not in the queue: skipped")
            continue
        if req.status != "failed" or not req.worktree:
            print(f"{request_id}: {req.status}, worktree {req.worktree}: not in the state the "
                  "failed night left: skipped")
            continue
        graded = regrade(req)
        print(f"{request_id}: failed after {req.iterations} iterations; worktree {req.worktree}")
        for finding, why in graded:
            print(f"    {finding['severity']:8s} {finding['text'][:110]}  ({why})")
        if request_id == "2026-11-04-1":
            print("    the reviewer's remarks of iterations 2-6 (four a round) were not kept, only "
                  "the last failing run of the checks; the next review finds what is still open "
                  "and grades it")
        print("    -> queued, iterations 0, the plan kept (no planner call), the graded remarks "
              "first, the MEDIUM and LOW ones also in the follow-up list")
        if apply:
            requeue(req, graded)
    if apply:
        target = prog._abs(path)
        shutil.copy2(target, target.with_name(target.name + ".before-severity"))
        prog.save_queue(queue, path)
        print(f"written {target} (backup: {target.name}.before-severity)")
    else:
        print("dry run: nothing written (use --apply)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main("--apply" in argv,
         Path(argv[argv.index("--queue") + 1]) if "--queue" in argv else prog.QUEUE)
