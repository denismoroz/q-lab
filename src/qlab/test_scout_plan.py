"""Tests for what the scout may do (`qlab.scout_plan`): whether it is called, which tools it is
handed, how its use of the network is audited, which sources a draft may cite."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from qlab import scout_plan
from qlab.budget.usage import ToolCall, parse_stream

REAL = Path(__file__).resolve().parents[2] / "night" / "scout.yaml"
DAY = date(2026, 11, 3)


def _cfg(**over) -> dict:
    return scout_plan.load_config(REAL, root=REAL.parents[1]) | over


def _measure(**kw) -> dict:
    return {"id": "short_perpetual@30d", "position": "short", "horizon_days": 30, "n": 45,
            "months": 17, "mean_pct": 27.5, "median_pct": 41.8, "t": 2.6, "first_half_pct": 25.9,
            "second_half_pct": 29.0, **kw}


def _evidence(*patterns) -> dict:
    return {"patterns": list(patterns)}


def _pattern(key="listing/hyperliquid", tradeable=True, **measure) -> dict:
    return {"key": key, "tradeable": tradeable, "measures": [_measure(**measure)]}


# --------------------------------------------------------------------------
# The real file


def test_the_real_config_loads_and_keeps_the_web_inside_declared_domains() -> None:
    cfg = _cfg()
    assert cfg["web"]["enabled"] is True and cfg["papers"]["mode"] == "from_cataloguer"
    domains = {e["domain"] for e in scout_plan.domain_entries(cfg)}
    assert {"hyperliquid.gitbook.io", "arxiv.org", "ssrn.com", "doi.org",
            "developers.binance.com"} <= domains
    # Every pattern that may be carded names a holder that exists as a card or a catalogue id.
    root = REAL.parents[1]
    known = {p.stem for p in (root / "cards").glob("*.yaml")} | set(
        re.findall(r"^### (\S+)", (root / "docs/MECHANISMS.md").read_text(), 8))
    for holders in cfg["wake"]["already_carded"].values():
        assert set(holders) <= known


def test_without_a_config_file_the_scout_is_what_it_was_before() -> None:
    cfg = scout_plan.load_config(Path("night/missing.yaml"), root=Path("/nonexistent"))
    assert cfg["wake"]["on_evidence"] is False and not scout_plan.web_enabled(cfg)


# --------------------------------------------------------------------------
# Tools


def test_network_tools_are_allowed_only_for_declared_domains() -> None:
    args = scout_plan.tool_args(_cfg())
    allowed = args[args.index("--allowedTools") + 1].split(",")
    denied = args[args.index("--disallowedTools") + 1].split(",")
    assert "WebFetch(domain:hyperliquid.gitbook.io)" in allowed and "WebSearch" in allowed
    assert "WebFetch" not in allowed  # never the bare tool: it would allow every domain
    assert {"Read", "Glob", "Grep", "Write"} <= set(allowed)
    assert denied == ["Bash", "Edit"]


def test_one_flag_switches_the_network_off() -> None:
    cfg = _cfg()
    cfg["web"] = {**cfg["web"], "enabled": False}
    args = scout_plan.tool_args(cfg)
    assert "WebFetch(" not in args[1] and "WebSearch" not in args[1]
    assert {"WebFetch", "WebSearch"} <= set(args[3].split(","))
    assert not scout_plan.web_enabled(cfg)


def test_search_can_be_switched_off_alone() -> None:
    cfg = _cfg()
    cfg["web"] = {**cfg["web"], "search": False}
    args = scout_plan.tool_args(cfg)
    assert "WebSearch" not in args[1].split(",") and "WebSearch" in args[3].split(",")


def test_a_url_is_allowed_by_domain_and_by_path_when_the_domain_lists_paths() -> None:
    cfg = _cfg()
    ok = scout_plan.url_allowed
    assert ok("https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees", cfg)
    assert ok("https://arxiv.org/abs/2401.00001", cfg)
    assert ok("https://www.binance.com/en/support/faq/x", cfg)
    assert not ok("https://www.binance.com/en/activity/x", cfg)   # a domain with listed paths
    assert not ok("https://evil.example.com/arxiv.org", cfg)
    assert not ok("https://arxiv.org.evil.com/abs/1", cfg)
    assert not ok("ftp://arxiv.org/abs/1", cfg)


# --------------------------------------------------------------------------
# The audit of the network


def test_every_fetched_url_is_logged_and_those_outside_the_list_are_problems() -> None:
    cfg = _cfg()
    log = scout_plan.audit_web([
        ToolCall("WebFetch", {"url": "https://hyperliquid.gitbook.io/hyperliquid-docs/x",
                              "prompt": "p"}),
        ToolCall("WebFetch", {"url": "https://example.com/x", "prompt": "p"}, refused=True),
        ToolCall("WebFetch", {"url": "https://news.example.org/y", "prompt": "p"}),
        ToolCall("WebSearch", {"query": "binance delisting rules",
                               "allowed_domains": ["developers.binance.com"]}),
        ToolCall("WebSearch", {"query": "anything"}),
        ToolCall("Read", {"file_path": "cards/SCHEMA.md"}),
    ], cfg, day=DAY)
    assert [f.get("url") or f.get("query") for f in log.fetched] == [
        "https://hyperliquid.gitbook.io/hyperliquid-docs/x", "https://example.com/x",
        "https://news.example.org/y", "binance delisting rules", "anything"]
    assert log.fetched[1]["refused"] is True
    assert len(log.problems) == 2
    assert "news.example.org" in log.problems[0] and "allowed_domains" in log.problems[1]
    assert log.urls() == {"https://hyperliquid.gitbook.io/hyperliquid-docs/x",
                          "https://news.example.org/y"}


def test_too_many_fetches_in_one_call_is_a_problem() -> None:
    cfg = _cfg()
    calls = [ToolCall("WebFetch", {"url": f"https://arxiv.org/abs/{i}"}) for i in range(13)]
    assert any("13 fetches" in p for p in scout_plan.audit_web(calls, cfg, day=DAY).problems)


def test_the_tool_calls_of_a_stream_are_read_with_their_refusals() -> None:
    lines = [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "reading"},
            {"type": "tool_use", "id": "t1", "name": "WebFetch",
             "input": {"url": "https://arxiv.org/abs/1", "prompt": "p"}},
            {"type": "tool_use", "id": "t2", "name": "WebFetch",
             "input": {"url": "https://example.com", "prompt": "p"}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
             "content": "permission denied"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "page"}]}}),
        "not json",
        json.dumps({"type": "result", "result": "NO CANDIDATE: x", "usage": {}}),
    ]
    report = parse_stream(lines)
    assert [(c.name, c.input["url"], c.refused) for c in report.tool_calls] == [
        ("WebFetch", "https://arxiv.org/abs/1", False),
        ("WebFetch", "https://example.com", True)]
    assert report.text == "NO CANDIDATE: x"


# --------------------------------------------------------------------------
# Sources of a draft card


def test_a_draft_may_cite_the_evidence_file_a_fetched_page_or_a_handed_paper(tmp_path) -> None:
    (tmp_path / "data/events/evidence").mkdir(parents=True)
    (tmp_path / "data/events/evidence/2026-11-03.json").write_text("{}")
    web = scout_plan.WebLog(fetched=[{"tool": "WebFetch", "url": "https://arxiv.org/abs/9",
                                      "refused": False}])
    card = {"sources": [
        {"url": "data/events/evidence/2026-11-03.json", "title": "history", "published": "2026-11"},
        {"url": "https://arxiv.org/abs/9", "title": "paper", "retrieved": "2026-11-03"},
        {"url": "https://arxiv.org/abs/7", "title": "handed"}]}
    problems = scout_plan.source_problems(
        card, evidence_file="data/events/evidence/2026-11-03.json", web=web,
        handed_links={"https://arxiv.org/abs/7"}, root=tmp_path)
    assert problems == []


def test_a_draft_that_cites_what_was_not_read_is_refused(tmp_path) -> None:
    web = scout_plan.WebLog(fetched=[{"tool": "WebFetch", "url": "https://arxiv.org/abs/9",
                                      "refused": False}])
    card = {"sources": [
        {"url": "https://example.com/made-up", "title": "x"},
        {"url": "https://arxiv.org/abs/9", "title": "no date"},
        {"url": "data/events/evidence/2026-01-01.json", "title": "gone"},
        {"url": "data/events/evidence/2026-11-02.json", "title": "yesterday's"}]}
    (tmp_path / "data/events/evidence").mkdir(parents=True)
    (tmp_path / "data/events/evidence/2026-11-02.json").write_text("{}")
    problems = scout_plan.source_problems(
        card, evidence_file="data/events/evidence/2026-11-03.json", web=web, handed_links=set(),
        root=tmp_path)
    assert len(problems) == 4
    assert "not fetched" in problems[0] and "retrieved" in problems[1]
    assert "not a file" in problems[2] and "tonight's evidence" in problems[3]
    assert scout_plan.source_problems({"sources": []}, evidence_file=None, web=web,
                                      handed_links=set(), root=tmp_path)


# --------------------------------------------------------------------------
# Whether the scout is called


def test_a_structural_event_calls_the_scout_as_before() -> None:
    cfg = _cfg()
    wake = scout_plan.decide([{"kind": "listed", "source": "binance", "instrument": "X"}], None,
                             cfg)
    assert wake.call and "structural" in wake.reason


def test_nothing_structural_and_nothing_measured_is_no_call() -> None:
    cfg = _cfg()
    wake = scout_plan.decide([{"kind": "funding-extreme"}], _evidence(), cfg)
    assert not wake.call and "no listing" in wake.reason


def test_a_measured_pattern_calls_the_scout_on_a_night_without_listings() -> None:
    wake = scout_plan.decide([], _evidence(_pattern()), _cfg())
    assert wake.call and [s["id"] for s in wake.signals] == [
        "listing/hyperliquid::short_perpetual@30d"]


def test_every_line_of_the_rule_is_needed() -> None:
    cfg = _cfg()
    for change in ({"n": 29}, {"months": 9}, {"mean_pct": 0.9}, {"t": 2.4},
                   {"first_half_pct": -1.0}, {"second_half_pct": None}, {"t": None}):
        assert not scout_plan.decide([], _evidence(_pattern(**change)), cfg).call, change
    assert not scout_plan.decide([], _evidence(_pattern(tradeable=False)), cfg).call


def test_a_pattern_whose_mechanism_a_card_holds_does_not_call_the_scout() -> None:
    wake = scout_plan.decide([], _evidence(_pattern(key="funding-extreme/binance/low/all")),
                             _cfg())
    assert not wake.call
    assert wake.held_back and "perp-funding-extreme-rebound" in wake.held_back[0]["carded_by"]


def test_a_pattern_the_scout_looked_at_wakes_it_again_only_when_its_sample_has_grown() -> None:
    cfg = _cfg()
    evidence = _evidence(_pattern(n=45))
    looked = {"listing/hyperliquid::short_perpetual@30d": {"day": "2026-11-01", "n": 40}}
    assert not scout_plan.decide([], evidence, cfg, looked_at=looked).call
    looked["listing/hyperliquid::short_perpetual@30d"]["n"] = 30   # 45 >= 30 x 1.25
    assert scout_plan.decide([], evidence, cfg, looked_at=looked).call


def test_looking_is_remembered(tmp_path) -> None:
    signal = {"id": "a::b", "n": 50}
    scout_plan.mark_looked_at([signal], DAY, "NO CANDIDATE: no payer", root=tmp_path)
    assert scout_plan.load_looked_at(tmp_path)["a::b"] == {
        "day": "2026-11-03", "n": 50, "reply": "NO CANDIDATE: no payer"}


def test_the_scouts_own_nightly_token_cap_lowers_the_candidates_cap() -> None:
    class Budget:
        cap_tokens = 900_000
        spent_tokens = 0

    scout_plan.cap_budget(Budget, _cfg(), spent_tonight=100_000)
    assert Budget.cap_tokens == 300_000   # 400k a night, 100k already spent
