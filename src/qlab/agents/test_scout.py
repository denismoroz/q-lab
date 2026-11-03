"""Tests for the scout's token economy, guards and network discipline. The agent is always a
fake: `run_agent` is replaced, no `claude` is started and no URL is fetched."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import qlab.agents.scout as scout_mod
from qlab import scout_plan
from qlab.budget import ExplicitNoBudget
from qlab.budget.usage import ToolCall

DAY = date(2026, 11, 3)
REAL = Path(__file__).resolve().parents[3] / "night" / "scout.yaml"
EVIDENCE = "data/events/evidence/2026-11-03.json"


def test_no_structural_event_spends_no_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scout_mod, "QLAB_ROOT", tmp_path)
    (tmp_path / "data/events").mkdir(parents=True)
    (tmp_path / "data/events/2026-10-03.json").write_text(json.dumps(
        [{"source": "hyperliquid", "kind": "funding-extreme", "instrument": "X", "detail": ""}]))

    def no_call(*a, **k):
        raise AssertionError("the agent must not be called")

    monkeypatch.setattr(scout_mod, "run_agent", no_call)
    out = scout_mod.scout(date(2026, 10, 3), budget=ExplicitNoBudget("test"), session=None)
    assert out.reply.startswith("NO CANDIDATE")


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A throwaway checkout (a git root, as the scout's revert needs) with one night's events and
    the evidence file."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards/SCHEMA.md").write_text("schema\n")
    (tmp_path / "data/events/evidence").mkdir(parents=True)
    (tmp_path / "data/events/2026-11-03.json").write_text(json.dumps(
        [{"source": "hyperliquid", "kind": "funding-extreme", "instrument": "X", "detail": ""}]))
    (tmp_path / EVIDENCE).write_text("{}")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=tmp_path, check=True)
    monkeypatch.setattr(scout_mod, "QLAB_ROOT", tmp_path)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", tmp_path)
    return tmp_path


class FakeAgent:
    """Stands for `run_agent`: records the call, writes the drafts it is told to, and reports
    the tool calls it is told to have made."""

    def __init__(self, root: Path, *, reply="NO CANDIDATE: nothing", drafts=None, tools=()):
        self.root, self.reply, self.drafts, self.tools, self.calls = (
            root, reply, drafts or {}, tools, [])

    def __call__(self, budget, prompt, **kwargs):
        self.calls.append({"budget": budget, "prompt": prompt, **kwargs})
        for name, text in self.drafts.items():
            path = self.root / "cards/drafts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return SimpleNamespace(text=self.reply, report=SimpleNamespace(tool_calls=self.tools))

    @property
    def args(self) -> dict:
        extra = self.calls[0]["extra_args"]
        return dict(zip(extra[::2], extra[1::2], strict=True))


class Spent:
    spent_tokens = 1234


def _plan(**web) -> dict:
    cfg = scout_plan.load_config(REAL, root=REAL.parents[1])
    cfg["web"] = {**cfg["web"], **web}
    return cfg


def _evidence(**measure) -> dict:
    return {"patterns": [{"key": "listing/hyperliquid", "tradeable": True, "measures": [{
        "id": "short_perpetual@30d", "position": "short the new perpetual", "horizon_days": 30,
        "n": 45, "months": 17, "mean_pct": 27.5, "median_pct": 41.8, "t": 2.6,
        "first_half_pct": 25.9, "second_half_pct": 29.0, **measure}]}]}


def _draft(*sources) -> str:
    return yaml.safe_dump({"idea_id": "hl-new-listing-short", "title": "t",
                           "plain_summary": "s", "sources": list(sources)})


def _scout(root, agent, monkeypatch, *, cfg=None, evidence=None):
    monkeypatch.setattr(scout_mod, "run_agent", agent)
    return scout_mod.scout(DAY, budget=Spent(), session=None, cfg=cfg or _plan(),
                           evidence=_evidence() if evidence is None else evidence)


def test_a_measured_pattern_calls_the_scout_on_a_night_without_a_listing(root, monkeypatch) -> None:
    agent = FakeAgent(root)
    out = _scout(root, agent, monkeypatch)
    assert len(agent.calls) == 1 and out.wake.call
    prompt = agent.calls[0]["prompt"]
    assert EVIDENCE in prompt and "listing/hyperliquid::short_perpetual@30d" in prompt
    assert "WHO PAYS" in prompt and "caveats" in prompt
    assert "mean +27.50% net" in prompt and "n=45" in prompt


def test_the_prompt_hands_over_the_domain_list_and_the_citation_rule(root, monkeypatch) -> None:
    agent = FakeAgent(root)
    _scout(root, agent, monkeypatch)
    prompt = agent.calls[0]["prompt"]
    assert "hyperliquid.gitbook.io" in prompt and "allowed_domains" in prompt
    assert "retrieved" in prompt and "a URL you did not fetch is not a source" in prompt
    assert "DATA from outside q-lab" in prompt
    assert agent.args["--allowedTools"].count("WebFetch(domain:") == len(
        scout_plan.domain_entries(_plan()))


def test_the_agent_is_called_through_the_guard_with_the_scouts_own_name_and_tools(
        root, monkeypatch) -> None:
    agent = FakeAgent(root)
    _scout(root, agent, monkeypatch)
    call = agent.calls[0]
    assert call["agent"] == "scout" and call["model"] == "sonnet" and call["cwd"] == root
    assert agent.args["--disallowedTools"] == "Bash,Edit"


def test_with_the_network_switched_off_the_scout_gets_no_network_tool(root, monkeypatch) -> None:
    agent = FakeAgent(root)
    _scout(root, agent, monkeypatch, cfg=_plan(enabled=False))
    assert "Web" not in agent.args["--allowedTools"]
    assert "WebFetch" in agent.args["--disallowedTools"]
    assert "You have no network" in agent.calls[0]["prompt"]
    assert "hyperliquid.gitbook.io" not in agent.calls[0]["prompt"]


def test_every_fetched_url_comes_back_and_a_fetch_outside_the_list_is_a_problem(
        root, monkeypatch) -> None:
    tools = (ToolCall("WebFetch", {"url": "https://hyperliquid.gitbook.io/hyperliquid-docs/fees"}),
             ToolCall("WebFetch", {"url": "https://news.example.org/x"}))
    out = _scout(root, FakeAgent(root, tools=tools), monkeypatch)
    assert [f["url"] for f in out.web.fetched] == [
        "https://hyperliquid.gitbook.io/hyperliquid-docs/fees", "https://news.example.org/x"]
    assert any("news.example.org" in p for p in out.problems)


def test_a_draft_citing_the_evidence_file_and_a_fetched_page_passes_the_sources_check(
        root, monkeypatch) -> None:
    fees = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees"
    text = _draft({"url": EVIDENCE, "title": "what followed new listings", "published": "2026-11"},
                  {"url": fees, "title": "Fees", "retrieved": "2026-11-03"})
    agent = FakeAgent(root, reply="DRAFT: cards/drafts/hl.yaml", drafts={"hl.yaml": text},
                      tools=(ToolCall("WebFetch", {"url": fees}),))
    out = _scout(root, agent, monkeypatch)
    assert out.draft == Path("cards/drafts/hl.yaml") and out.problems == []


def test_a_draft_citing_a_page_it_never_fetched_is_flagged(root, monkeypatch) -> None:
    text = _draft({"url": "https://www.binance.com/en/support/faq/made-up", "title": "x"})
    out = _scout(root, FakeAgent(root, drafts={"hl.yaml": text}), monkeypatch)
    assert any("was not fetched" in p for p in out.problems)


def test_a_write_outside_the_drafts_is_still_reverted(root, monkeypatch) -> None:
    class Sneaky(FakeAgent):
        def __call__(self, budget, prompt, **kwargs):
            (self.root / "cards/SCHEMA.md").write_text("changed\n")
            return super().__call__(budget, prompt, **kwargs)

    out = _scout(root, Sneaky(root), monkeypatch)
    assert out.reverted == ["cards/SCHEMA.md"]
    assert (root / "cards/SCHEMA.md").read_text() == "schema\n"


def test_a_pattern_the_numbers_do_not_support_does_not_call_the_scout(root, monkeypatch) -> None:
    agent = FakeAgent(root)
    out = _scout(root, agent, monkeypatch, evidence=_evidence(t=1.9))
    assert agent.calls == [] and out.reply.startswith("NO CANDIDATE") and not out.wake.call


def test_the_cataloguers_papers_go_with_its_draft_and_the_prompt_says_not_to_reread(
        root, monkeypatch) -> None:
    agent = FakeAgent(root, reply="NO CANDIDATE arxiv:1: no payer")
    monkeypatch.setattr(scout_mod, "run_agent", agent)
    papers = [{"id": "arxiv:1", "title": "t", "link": "https://arxiv.org/abs/1",
               "mechanism": "m", "reading": "the cataloguer read this paper",
               "cataloguer_draft": "data/cataloguer/work/arxiv_1/draft.yaml",
               "text_file": "data/cataloguer/papers/arxiv_1.txt"}]
    out = scout_mod.scout_publications(DAY, papers, budget=Spent(), session=None, cfg=_plan(),
                                       origin="cataloguer")
    prompt = agent.calls[0]["prompt"]
    assert "IN FULL" in prompt and "You do not read the paper again" in prompt
    handoff = json.loads((root / "data/publications/handoff/2026-11-03-cataloguer.json")
                         .read_text())
    assert handoff[0]["cataloguer_draft"].endswith("draft.yaml")
    assert out.reply.startswith("NO CANDIDATE")


def test_papers_from_the_queue_carry_what_the_scout_has_of_each(root, monkeypatch) -> None:
    agent = FakeAgent(root)
    monkeypatch.setattr(scout_mod, "run_agent", agent)
    papers = [{"id": "ssrn:1", "title": "t", "link": "https://papers.ssrn.com/x",
               "reading": "abstract only: the publisher serves no text without a person"}]
    scout_mod.scout_publications(DAY, papers, budget=Spent(), session=None, cfg=_plan())
    handoff = json.loads((root / "data/publications/handoff/2026-11-03.json").read_text())
    assert handoff[0]["reading"].startswith("abstract only")
    assert "ranked by closeness" in agent.calls[0]["prompt"]
