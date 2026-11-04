"""The one-off script that puts the two requests of the failed night of 2026-11-04 back in the
queue under the severity rules (scripts/requeue_after_severity.py). No agent is called."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import qlab.agents.programmer as prog

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "requeue_after_severity.py"


def _script():
    spec = importlib.util.spec_from_file_location("requeue_after_severity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPT = ("reviewer: `src/qlab/agents/generator.py:584` — the generator's prompt lists the extra "
          "blocks the catalogue may name. It names `spot_perp_carry` but says nothing about "
          "`spot_perp_carry_positive`. Add a bullet there.")
PHRASE = ("reviewer: `strategies/plain/blocks.yaml:28` — the file has a plain-language phrase for "
          "every portfolio block, but none for `spot_perp_carry_positive`. Add a no-jargon phrase.")
TESTS = ("the tests failed:\n...\nFAILED src/qlab/nightflow/test_lanes.py::test_the_agents_lane_"
         "is_never_idle\n>       assert clock.span(\"plan\")[1] < scan_end + 0.05\n"
         "1 failed, 3490 passed, 6 skipped")


def _failed(request_id: str, remarks: list[str], review: list[str]) -> prog.Request:
    req = prog.Request(id=request_id, repo="qlab", what="x", why="y", source="owner",
                       created="t", status="failed", worktree="/w/" + request_id,
                       branch="agent/" + request_id, iterations=7, remarks=remarks, review=review,
                       problems=["7 iterations without the checks passing"])
    req.plan = "the plan"
    return req


def test_old_plain_remarks_are_graded_by_the_rules_and_the_two_requests_are_queued_again(
        tmp_path):
    script = _script()
    script.IDS = (*script.IDS, "2026-11-04-2")  # a third, to see the rule for a plain line
    first = _failed("2026-10-30-3", [PROMPT, PHRASE], [PROMPT[10:], PHRASE[10:]])
    second = _failed("2026-11-04-1", [TESTS], [])
    odd = _failed("2026-11-04-2", ["reviewer: window(0) returns a wrong result"], [])
    queue = tmp_path / "queue.json"
    prog.save_queue([first, second, odd], queue)

    script.main(False, queue)  # a dry run writes nothing
    assert [r.status for r in prog.load_queue(queue)] == ["failed"] * 3
    assert not (tmp_path / "queue.json.before-severity").exists()

    script.main(True, queue)
    a, b, c = prog.load_queue(queue)
    assert (tmp_path / "queue.json.before-severity").is_file()  # the old queue is kept
    # a missing prompt mention and a missing plain phrase are MEDIUM: nothing blocks, so the
    # request is accepted as soon as the checks pass; both stay as the follow-up list
    assert a.status == "queued" and a.iterations == 0 and a.worktree == "/w/2026-10-30-3"
    assert [f["severity"] for f in a.findings] == ["MEDIUM", "MEDIUM"]
    assert [f["severity"] for f in a.followups] == ["MEDIUM", "MEDIUM"]
    assert a.remarks[0].startswith("reviewer [MEDIUM]: `src/qlab/agents/generator.py:584`")
    assert a.review == [] and a.problems == [] and a.plan == "the plan" and not a.replan
    assert {"generator.py", "blocks.yaml"} <= set(a.raised_files)
    assert a.failures[-1] == {"retry": True, "sig": None}  # the early-stop count starts again
    # a failing run of the checks is CRITICAL, shortened to what the writer must do
    assert b.status == "queued" and b.remarks[0].startswith("reviewer [CRITICAL]:")
    assert "test_lanes.py::test_the_agents_lane_is_never_idle" in b.remarks[0]
    assert "Run the test command again" in b.remarks[0] and b.followups == []
    # anything else keeps the rule for a line without a severity: HIGH, unparsed
    assert c.findings[0]["severity"] == "HIGH" and c.findings[0]["unparsed"] is True
    assert c.followups == [] and c.status == "queued"


def test_a_request_that_is_not_in_the_failed_state_is_left_alone(tmp_path):
    script = _script()
    ready = _failed("2026-10-30-3", [PROMPT], [])
    ready.status = "ready"
    queue = tmp_path / "queue.json"
    prog.save_queue([ready], queue)
    script.main(True, queue)
    assert prog.load_queue(queue)[0].status == "ready"
