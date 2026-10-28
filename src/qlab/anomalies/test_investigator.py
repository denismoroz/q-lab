"""The investigator agent: its answer is checked by code against the log of the toolkit calls it
really made, it has no free shell, it goes through the budget guard, and it changes no verdict
(docs/ANOMALIES.md). The `claude` binary is a fake script."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.agents import programmer as prog
from qlab.anomalies import flags as fl
from qlab.anomalies import investigator as inv
from qlab.anomalies import store
from qlab.anomalies.config import parse_config
from qlab.anomalies.scan import ScanResult
from qlab.anomalies.test_flags import RAW
from qlab.anomalies.test_scan import _idea, _trial
from qlab.budget import CandidateExhausted, ExplicitNoBudget, NightBudget, Stage, WrongAccount
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.budget.guard import USD_PER_TOKEN_PRIOR
from qlab.budget.usage import UsageReading, Window
from qlab.pipeline.sources import QLAB_ROOT
from qlab.registry.models import AnomalyInvestigation, Idea, TokenSpend, Trial, Verdict

NOW = datetime(2026, 10, 28, 0, 0, tzinfo=UTC)
RESET = NOW + timedelta(days=5)
CALL_ONE = ("[call 1] qlab anomaly contributions 5 --days 2026-01-21\n"
            "2026-01-21  PUMP  +0.1000  long  +200.0%  +20.00%  $200.0k")
CALL_TWO = ("[call 2] qlab anomaly counterfactual 5 --only-hl-listed\n"
            "yearly return  +210.9%  +10.9%")


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from qlab.registry.models import Base

    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def cfg_with(tmp_path: Path, **investigator):
    raw = {**RAW, "investigator": {**RAW["investigator"], "calls_dir": str(tmp_path / "calls"),
                                   **investigator}}
    return parse_config(raw)


def _reading(week: float = 0.20) -> UsageReading:
    return UsageReading("allowed", Window(0.1, NOW + timedelta(hours=3)), Window(week, RESET))


def candidate(session, idea: str = "xs"):
    night = NightBudget.open(session, _reading(), now=NOW)
    return night.stage(Stage.RECHECK, candidates=1).candidate(idea)


def flagged_scan(session, idea_id: str = "xs"):
    _idea(session, idea_id)
    trial_id = _trial(session, idea_id, metrics={"ann_return_net": 0.36, "total_return_net": 7.0})
    flags = [fl.FlagResult("implausible_day", fl.FIRES, 1.8, 1.0,
                           detail={"top": [{"day": "2026-01-21", "return": 0.2}]}),
             fl.FlagResult("venue_divergence", fl.FIRES, 1.7, 0.5, detail={"days": 600}),
             fl.FlagResult("stand_vs_paper", fl.UNDEFINED, reason="у идеи нет бумажной книги")]
    return store.save_scan(session, ScanResult(trial_id, idea_id, "paper", flags), "test.1")


def stream(answer: dict | str, tokens: int = 60_000, week: float = 0.21) -> str:
    text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    rate = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
        "five_hour": {"utilization": 0.1, "resetsAt": int((NOW + timedelta(hours=3)).timestamp())},
        "seven_day": {"utilization": week, "resetsAt": int(RESET.timestamp())}}}}
    result = {"type": "result", "is_error": False, "result": text, "total_cost_usd": tokens * 2e-6,
              "usage": {"input_tokens": 10, "output_tokens": 40,
                        "cache_creation_input_tokens": tokens - 50, "cache_read_input_tokens": 0},
              "modelUsage": {"claude-opus-4-8": {}}}
    return json.dumps(rate) + "\n" + json.dumps(result) + "\n"


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    """A `claude` that logs its args, writes the toolkit calls the agent would have made into the
    investigation's log (`$QLAB_ANOMALY_LOG`, which the investigator puts in the environment) and
    prints `$FAKE_OUT`."""
    script = tmp_path / "claude"
    script.write_text(
        '#!/bin/sh\necho "$@" >> "$FAKE_ARGS"\n'
        'if [ -n "$FAKE_CALLS" ]; then cat "$FAKE_CALLS" >> "$QLAB_ANOMALY_LOG"; fi\n'
        'echo "LOG=$QLAB_ANOMALY_LOG" >> "$FAKE_ARGS"\ncat "$FAKE_OUT"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    monkeypatch.setenv("FAKE_ARGS", str(tmp_path / "args.log"))

    def run(output: str, calls: list[str] | None = ()) -> Path:
        out = tmp_path / "out.jsonl"
        out.write_text(output)
        monkeypatch.setenv("FAKE_OUT", str(out))
        if calls:
            made = tmp_path / "calls.jsonl"
            made.write_text("".join(json.dumps({"cmd": "x", "output": c}) + "\n" for c in calls))
            monkeypatch.setenv("FAKE_CALLS", str(made))
        else:
            monkeypatch.delenv("FAKE_CALLS", raising=False)
        return tmp_path / "args.log"

    return run


GOOD = {
    "cause": "unrealistic-tradability",
    "summary": "Большая часть прибыли получена на монетах, которых нет на Hyperliquid, "
               "поэтому заработать так там было бы нельзя.",
    "evidence": [
        {"text": "День 21 января даёт почти всю прибыль месяца.", "call": 1,
         "quote": "2026-01-21  PUMP  +0.1000  long  +200.0%"},
        {"text": "Если оставить только монеты Hyperliquid, доходность падает с 211% до 11%.",
         "call": 2, "quote": "yearly return  +210.9%  +10.9%"},
        {"text": "Выдуманная цитата.", "call": 2, "quote": "yearly return  +999.0%"},
    ],
    "proposals": [
        {"kind": "data_filter", "text": "Не считать монеты, которых нет на второй бирже."},
        {"kind": "new_check", "text": "Сверять список монет с биржей исполнения.",
         "code_request": {"repo": "qlab",
                          "what": "Add a check that every held coin is listed on the execution "
                                  "venue",
                          "why": "XSMOM earned on coins Hyperliquid does not list"}},
    ],
    "caught_by": ["venue_divergence", "stand_vs_paper", "no_such_flag"],
    "new_check": None,
}


# --- the checks on the answer -------------------------------------------------------------


CALLS = {1: CALL_ONE, 2: CALL_TWO}


def test_evidence_stands_only_with_a_logged_call_and_a_verbatim_quote(tmp_path) -> None:
    answer = {**GOOD, "evidence": [
        *GOOD["evidence"],
        {"text": "Вызова с таким номером не было.", "call": 9, "quote": "anything"},
        {"text": "Цитата без вызова.", "quote": "PUMP"},
        {"text": "Without Russian letters.", "call": 1, "quote": "PUMP"},
        {"text": "Без цитаты.", "call": 1},
        "not an object"]}

    out = inv.verify(answer, CALLS, ["implausible_day", "venue_divergence"],
                     cfg_with(tmp_path), {"qlab"})

    assert [e["call"] for e in out.evidence] == [1, 2]
    reasons = " | ".join(r["why"] for r in out.rejected)
    assert "not verbatim in the output of call 2" in reasons
    assert "call 9 is not in the log" in reasons and "not Russian" in reasons
    assert "no quote" in reasons and "not an object" in reasons
    assert out.cause == "unrealistic-tradability"


def test_a_cause_without_verified_evidence_is_only_unclear(tmp_path) -> None:
    answer = {**GOOD, "evidence": [GOOD["evidence"][2]], "new_check": "Сверять списки монет."}

    out = inv.verify(answer, CALLS, ["venue_divergence"], cfg_with(tmp_path), {"qlab"})

    assert out.cause == "unclear" and out.evidence == []
    assert any("stood without one verified piece of evidence" in n for n in out.notes)
    assert out.new_check is None  # nothing is confirmed, so there is nothing to learn from


def test_an_unknown_cause_is_unclear_and_a_non_russian_summary_is_replaced(tmp_path) -> None:
    out = inv.verify({"cause": "aliens", "summary": "english only", "evidence": []}, CALLS, [],
                     cfg_with(tmp_path), {"qlab"})

    assert out.cause == "unclear"
    assert "Агент не дал связного пересказа" in out.summary
    assert any("not one of the five" in n for n in out.notes)


def test_the_learning_loop_names_only_flags_that_fired(tmp_path) -> None:
    out = inv.verify(GOOD, CALLS, ["implausible_day", "venue_divergence"], cfg_with(tmp_path),
                     {"qlab"})

    # stand_vs_paper did not fire and the third name is no flag at all
    assert out.caught_by == ["venue_divergence"]
    assert any("did not fire" in n for n in out.notes)


def test_a_new_check_is_kept_only_in_russian_and_only_for_a_confirmed_cause(tmp_path) -> None:
    ok = inv.verify({**GOOD, "caught_by": [], "new_check": "Сверять список монет."}, CALLS,
                    ["venue_divergence"], cfg_with(tmp_path), {"qlab"})
    english = inv.verify({**GOOD, "caught_by": [], "new_check": "compare coin lists"}, CALLS,
                         ["venue_divergence"], cfg_with(tmp_path), {"qlab"})
    genuine = inv.verify({**GOOD, "cause": "genuine", "new_check": "Сверять."}, CALLS,
                         ["venue_divergence"], cfg_with(tmp_path), {"qlab"})

    assert ok.new_check == "Сверять список монет."
    assert english.new_check is None and genuine.new_check is None


def test_a_code_request_survives_only_for_a_known_repository_and_a_code_proposal(tmp_path) -> None:
    out = inv.verify(GOOD, CALLS, ["venue_divergence"], cfg_with(tmp_path), {"qlab"})
    request = out.proposals[1]["code_request"]
    assert request == {"repo": "qlab", "what": GOOD["proposals"][1]["code_request"]["what"],
                       "why": GOOD["proposals"][1]["code_request"]["why"]}
    assert out.proposals[0]["code_request"] is None

    unknown_repo = inv.verify(GOOD, CALLS, [], cfg_with(tmp_path), {"frab"})
    assert unknown_repo.proposals[1]["code_request"] is None
    assert any("unknown repository" in n for n in unknown_repo.notes)


def test_the_answer_json_is_found_inside_surrounding_text() -> None:
    assert inv.parse_answer('Вот ответ: {"cause": "genuine"} Готово.') == {"cause": "genuine"}
    assert inv.parse_answer("no json") is None and inv.parse_answer(None) is None
    assert inv.parse_answer("[1, 2]") is None


def test_the_calls_log_is_numbered_as_the_tools_print_it(tmp_path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"output": "a"}) + "\n" + json.dumps({"output": "b"}) + "\n")

    assert inv.read_calls(log) == {1: "a", 2: "b"}
    assert inv.read_calls(tmp_path / "missing.jsonl") == {}


# --- the call -----------------------------------------------------------------------------


def test_the_agent_has_no_free_shell() -> None:
    assert inv.ALLOWED == "Read,Glob,Grep,Bash(uv run qlab anomaly:*)"
    assert set(inv.TOOLS.split(",")) == {"Read", "Glob", "Grep", "Bash"}
    assert {"Write", "Edit", "NotebookEdit"} <= set(inv.DISALLOWED.split(","))
    assert "Write" not in inv.TOOLS and "Edit" not in inv.TOOLS
    assert inv.ALLOWED.count("Bash") == 1  # Bash only ever with the toolkit's prefix


def test_an_investigation_goes_through_the_guard_with_restricted_tools(
        session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    monkeypatch.setattr(inv, "programmer_repos", lambda: {"qlab"})
    args_log = fake_claude(stream(GOOD, tokens=70_000), calls=[CALL_ONE, CALL_TWO])
    budget = candidate(session)
    before = _state(session)

    new = inv.investigate(session, scan, cfg_with(tmp_path), budget=budget, day=date(2026, 10, 28),
                          queue_path=tmp_path / "queue.json")

    args = args_log.read_text()
    assert "--model opus" in args and "--tools Read,Glob,Grep,Bash" in args
    assert "--allowedTools Read,Glob,Grep,Bash(uv run qlab anomaly:*)" in args
    assert "--disallowedTools Write,Edit,NotebookEdit" in args
    assert "--max-budget-usd" in args and "--strict-mcp-config" in args
    log_path = Path(args.split("LOG=")[1].split()[0])
    assert log_path.parent == tmp_path / "calls" and log_path.exists()
    import os

    assert "QLAB_ANOMALY_LOG" not in os.environ  # put back as it was

    spend = session.query(TokenSpend).one()
    assert (spend.agent, spend.stage, spend.idea_id) == ("anomaly-investigator", "recheck", "xs")
    assert budget.spent_tokens == 70_000 and new.tokens == 70_000

    row = session.query(AnomalyInvestigation).one()
    assert (row.cause, row.model, row.trial_id) == ("unrealistic-tradability", "opus",
                                                    scan.trial_id)
    assert len(row.evidence) == 2 and len(row.rejected) == 1
    assert row.caught_by == ["venue_divergence"] and row.calls == 2
    assert row.flags_fingerprint == store.fingerprint(scan.fired, "test.1")
    assert "# Разбор аномалии: xs" in row.dossier and "## Коротко" in row.dossier
    assert "Причина" in row.dossier and "| Что установлено | Вызов |" in row.dossier
    assert "Агент ничего не менял" in row.dossier or "ничего не менял" in row.dossier
    assert _state(session) == before  # no verdict, status or route moved


def test_a_code_proposal_goes_to_the_programmers_queue_as_a_request_never_applied(
        session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    monkeypatch.setattr(inv, "programmer_repos", lambda: {"qlab"})
    fake_claude(stream(GOOD), calls=[CALL_ONE, CALL_TWO])
    queue = tmp_path / "queue.json"

    new = inv.investigate(session, scan, cfg_with(tmp_path), budget=candidate(session),
                          day=date(2026, 10, 28), queue_path=queue)

    requests = prog.load_queue(queue)
    assert len(requests) == 1 and new.programmer_requests == [requests[0].id]
    request = requests[0]
    assert request.repo == "qlab" and request.status == "queued" and request.branch is None
    assert "listed on the execution venue" in request.what
    assert request.source == f"anomaly investigator, trial {scan.trial_id}"
    assert f"trial {scan.trial_id}" in request.why and "xs" in request.why
    assert str(request.id) in session.query(AnomalyInvestigation).one().dossier


def test_requests_are_capped_and_a_duplicate_is_not_queued_twice(
        session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    queue = tmp_path / "queue.json"
    dossier = inv.Dossier(cause="stand-bug", summary="x", proposals=[
        {"kind": "stand_fix", "text": "t", "code_request": {"repo": "qlab", "what": f"w{i}",
                                                           "why": "y"}} for i in range(3)])

    first = inv.enqueue_requests(dossier, scan, date(2026, 10, 28), 2, queue_path=queue)
    again = inv.enqueue_requests(dossier, scan, date(2026, 10, 28), 2, queue_path=queue)

    assert len(first) == 2  # the cap holds back the third
    assert len(again) == 1  # the first two are already queued; only the held-back one is new
    assert [r.what for r in prog.load_queue(queue)] == ["w0", "w1", "w2"]


def test_the_ceiling_of_one_investigation_is_respected(
        session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    args_log = fake_claude(stream(GOOD, tokens=50_000), calls=[CALL_ONE, CALL_TWO])
    monkeypatch.setattr(inv, "programmer_repos", lambda: set())
    budget = candidate(session)
    assert budget.cap_tokens > 200_000

    inv.investigate(session, scan, cfg_with(tmp_path, max_tokens_per_investigation=200_000),
                    budget=budget, day=date(2026, 10, 28), queue_path=tmp_path / "q.json")

    assert budget.cap_tokens == 200_000  # the smaller of the guard's share and the ceiling
    cap_usd = float(args_log.read_text().split("--max-budget-usd ")[1].split()[0])
    assert cap_usd <= 200_000 * USD_PER_TOKEN_PRIOR * 1.001


def test_a_ceiling_below_the_cheapest_call_stops_before_anything_is_spent(
        session, tmp_path, fake_claude) -> None:
    scan = flagged_scan(session)
    args_log = fake_claude(stream(GOOD))

    with pytest.raises(CandidateExhausted):
        inv.investigate(session, scan, cfg_with(tmp_path, max_tokens_per_investigation=10_000),
                        budget=candidate(session), day=date(2026, 10, 28))

    assert not args_log.exists() and session.query(TokenSpend).count() == 0
    assert session.query(AnomalyInvestigation).count() == 0


def test_there_is_no_investigation_without_a_budget(session, tmp_path, fake_claude) -> None:
    scan = flagged_scan(session)
    fake_claude(stream(GOOD))

    with pytest.raises(TypeError, match="ExplicitNoBudget"):
        inv.investigate(session, scan, cfg_with(tmp_path), budget=None, day=date(2026, 10, 28))
    # Refusing the budget is possible, but it has to say so (CLAUDE.md).
    with pytest.raises(ValueError, match="reason"):
        ExplicitNoBudget("  ")


def test_the_wrong_account_spends_nothing(session, tmp_path, fake_claude, monkeypatch) -> None:
    import qlab.budget.agent as agent

    scan = flagged_scan(session)
    args_log = fake_claude(stream(GOOD))
    monkeypatch.setattr(agent, "_read_status", lambda account: {
        "loggedIn": True, "email": "someone@work.example", "orgName": "x"})

    with pytest.raises(WrongAccount):
        inv.investigate(session, scan, cfg_with(tmp_path), budget=candidate(session),
                        day=date(2026, 10, 28))

    assert not args_log.exists() and session.query(TokenSpend).count() == 0


def test_an_answer_that_is_not_json_is_recorded_as_unclear(
        session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    fake_claude(stream("I could not do it."))
    monkeypatch.setattr(inv, "programmer_repos", lambda: {"qlab"})

    new = inv.investigate(session, scan, cfg_with(tmp_path), budget=candidate(session),
                          day=date(2026, 10, 28), queue_path=tmp_path / "q.json")

    assert new.cause == "unclear" and "не вернул ответ" in new.summary
    assert new.programmer_requests == [] and new.evidence == []


def test_the_prompt_carries_the_fired_flags_the_toolkit_and_the_rules_of_the_answer(
        session, tmp_path) -> None:
    scan = flagged_scan(session)

    prompt = inv.build_prompt(session, scan, cfg_with(tmp_path))

    assert "idea xs" in prompt and f"trial {scan.trial_id}" in prompt
    assert "- implausible_day: fires" in prompt and "- venue_divergence: fires" in prompt
    assert "stand_vs_paper: not defined" in prompt and "нет бумажной книги" in prompt
    for command in ("contributions", "counterfactual", "compare-venues", "coin COIN", "flags"):
        assert f"uv run qlab anomaly {command}" in prompt or command in prompt
    for cause in store.CAUSES:
        assert cause in prompt
    assert "VERBATIM" in prompt and "plain Russian" in prompt and "REQUEST" in prompt
    assert "never change code yourself" in prompt


def test_the_agent_runs_in_the_repository_root(session, tmp_path, fake_claude, monkeypatch) -> None:
    scan = flagged_scan(session)
    seen = {}

    def spy(budget, prompt, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(inv, "run_agent", spy)

    with pytest.raises(RuntimeError):
        inv.investigate(session, scan, cfg_with(tmp_path), budget=candidate(session),
                        day=date(2026, 10, 28))

    assert seen["cwd"] == QLAB_ROOT and seen["agent"] == "anomaly-investigator"
    assert seen["model"] == "opus" and seen["max_turns"] == 10


def _state(session) -> tuple:
    """Everything an investigation must not touch: ideas' statuses, trials' routes, verdicts."""
    return (sorted((i.id, i.status.value) for i in session.query(Idea)),
            sorted((t.id, t.route.value) for t in session.query(Trial)),
            session.query(Verdict).count())
