"""The night's publication steps: the tail, the cap, the budget stops. The agent is
always a fake: `run_agent` is replaced, no `claude` is started."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

import qlab.agents.scout as scout_mod
import qlab.night as night
import qlab.publications.stage as stage
import qlab.registry.db as db_module
from qlab.budget import ExplicitNoBudget, NightExhausted
from qlab.publications import store
from qlab.publications.filter import FILTER_PATH
from qlab.registry.db import session_scope
from qlab.registry.models import Base, Publication

DAY = date(2026, 10, 25)
NOW = datetime(2026, 10, 25, 5, 30, tzinfo=UTC)
ABSTRACT = ("We study the funding rate of perpetual futures: who pays it, when it turns "
            "negative and how the basis closes. ") * 3


OFF_TOPIC = "Walks on graphs, quantum and classical, and their spectra. " * 5


def atom(entries: list[tuple[str, datetime, str]]) -> str:
    body = "".join(
        f"<entry><id>http://arxiv.org/abs/{i}v1</id><title>{t}</title>"
        f"<summary>{OFF_TOPIC if 'Quantum' in t else ABSTRACT}</summary>"
        f"<published>{p:%Y-%m-%dT%H:%M:%SZ}</published>"
        f'<category term="q-fin.TR" scheme="x"/><author><name>An Author</name></author></entry>'
        for i, p, t in entries)
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A throwaway project: a git root for the scout, a registry, the night's
    state folder and a feed config that reads the real topic rule."""
    root = tmp_path / "repo"
    (root / "cards").mkdir(parents=True)
    (root / "cards" / "SCHEMA.md").write_text("schema\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=root, check=True)
    monkeypatch.setattr(scout_mod, "QLAB_ROOT", root)
    monkeypatch.setattr(stage, "QLAB_ROOT", root)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", root)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    cfg = {
        "lookback_days": 90, "overlap_days": 3, "max_abstracts_per_night": 2,
        "max_abstract_chars": 3000, "min_abstract_chars": 200, "contact": None,
        "filter": str(Path(FILTER_PATH).resolve()),
        "arxiv": {"enabled": True, "endpoint": "https://export.arxiv.org/api/query",
                  "min_interval_seconds": 0.0, "page_size": 100, "max_pages_per_query": 2,
                  "queries": [{"name": "TR", "search_query": "cat:q-fin.TR"}]},
        "crossref": {"enabled": True, "endpoint": "https://api.crossref.org/works",
                     "min_interval_seconds": 0.0, "rows": 100, "max_pages_per_query": 1,
                     "prefix": "10.2139", "source_name": "ssrn", "queries": ["perpetual futures"]},
    }
    config = tmp_path / "publications.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    yield SimpleNamespace(root=root, config=config, cfg=cfg, tmp=tmp_path)
    db_module._engine = None
    db_module._SessionLocal = None


SSRN = [{"DOI": "10.2139/ssrn.5", "title": ["SSRN paper on perpetual futures"],
         "created": {"date-time": "2026-10-24T08:00:00Z"}, "abstract": ABSTRACT,
         "URL": "https://doi.org/10.2139/ssrn.5", "author": [{"given": "A", "family": "B"}]}]


def transport(arxiv_entries, crossref_params: list | None = None, fail: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("offline")
        if "arxiv" in request.url.host:
            return httpx.Response(200, text=atom(arxiv_entries))
        if crossref_params is not None:
            crossref_params.append(dict(request.url.params))
        return httpx.Response(200, json={"message": {"items": SSRN}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def seed(titles: list[str], *, abstract: str = ABSTRACT) -> None:
    """Candidates already in the table, newest last."""
    from qlab.publications.filter import load_rules
    from qlab.publications.items import Item

    items = [Item(id=f"arxiv:9{i}", source="arxiv", title=t, authors=["A"],
                  published=NOW - timedelta(days=len(titles) - i), abstract=abstract,
                  link=f"https://arxiv.org/abs/9{i}", categories=[])
             for i, t in enumerate(titles)]
    with session_scope() as session:
        store.add_new(session, items, now=NOW)
        store.apply_filter(session, load_rules())


class FakeAgent:
    """Stands for `run_agent`: records the call and writes the drafts it is told to."""

    def __init__(self, root: Path, drafts: dict[str, str] | None = None, reply: str = "") -> None:
        self.root, self.drafts, self.reply, self.calls = root, drafts or {}, reply, []

    def __call__(self, budget, prompt, **kwargs):
        self.calls.append({"budget": budget, "prompt": prompt, **kwargs})
        for name, text in self.drafts.items():
            path = self.root / "cards" / "drafts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return SimpleNamespace(text=self.reply)


def draft_text(idea: str, link: str) -> str:
    return yaml.safe_dump({"idea_id": idea, "title": "t", "plain_summary": "s",
                           "sources": [{"url": link, "title": "paper"}]})


class Budget:
    spent_tokens = 4321


# ---- step 1: fetch and filter, token-free ----------------------------------------------------


def test_a_night_stores_the_papers_marks_candidates_and_a_second_night_reads_only_the_tail(
        env) -> None:
    first = [("2610.2", NOW - timedelta(days=1), "Funding rates of perpetual futures"),
             ("2610.1", NOW - timedelta(days=30), "Quantum walks on graphs")]
    params: list = []
    state = {"done": {}}
    stage.fetch_stage(DAY, state, config=env.config, client=transport(first, params), now=NOW)
    summary = state["publications"]["fetch"]
    assert (summary["fetched"], summary["new"], summary["errors"]) == (3, 3, [])
    assert sorted(c["id"] for c in summary["new_candidates"]) == [
        "arxiv:2610.2", "doi:10.2139/ssrn.5"]  # not "Quantum walks on graphs"
    assert "perpetual futures" in summary["new_candidates"][0]["why"]
    assert params[0]["filter"].endswith("from-created-date:2026-07-27")  # 90 days back

    # The same night resumed: nothing is read again.
    stage.fetch_stage(DAY, state, config=env.config, client=transport([], params), now=NOW)
    assert len(params) == 1

    # The next night: the APIs send the old papers again plus one new; only the tail is asked
    # for (from the newest stored paper less the overlap) and only the new paper is added.
    later = NOW + timedelta(days=1)
    again = first + [("2610.3", later - timedelta(hours=2), "Perpetual swap funding payment")]
    state2 = {"done": {}}
    stage.fetch_stage(DAY + timedelta(days=1), state2, config=env.config,
                      client=transport(again, params), now=later)
    assert params[1]["filter"].endswith("from-created-date:2026-10-21")  # 2026-10-24 - 3 days
    summary2 = state2["publications"]["fetch"]
    assert (summary2["new"], [c["id"] for c in summary2["new_candidates"]]) == (
        1, ["arxiv:2610.3"])
    with session_scope() as session:
        assert session.query(Publication).count() == 4


def test_a_dead_api_is_reported_and_the_night_goes_on(env) -> None:
    state = {"done": {}}
    stage.fetch_stage(DAY, state, config=env.config, client=transport([], fail=True), now=NOW,
                      sleep=lambda s: None)
    fetch = state["publications"]["fetch"]
    assert fetch["new"] == 0 and fetch["errors"] and "ConnectError" in fetch["errors"][0]


def test_a_broken_config_is_reported_not_raised(env) -> None:
    env.config.write_text("lookback_days: [", encoding="utf-8")
    state = {"done": {}}
    stage.fetch_stage(DAY, state, config=env.config, client=transport([]), now=NOW)
    assert state["publications"]["fetch"]["errors"]


def test_no_config_no_stage(env) -> None:
    state = {"done": {}}
    stage.fetch_stage(DAY, state, config=env.tmp / "missing.yaml", now=NOW)
    stage.scout_stage(DAY, state, config=env.tmp / "missing.yaml")
    assert "publications" not in state or "fetch" not in state["publications"]


# ---- step 2: the scout's batch ---------------------------------------------------------------

TITLES = ["Funding rates of perpetual futures", "Perpetual swap funding payment design",
          "Perpetual contract funding rate arbitrage", "Funding rate asymmetry in perpetuals"]


def run_scout(env, monkeypatch, agent, state=None, budget=None):
    state = state if state is not None else {"done": {}, "publications": {"fetch": {}}}
    monkeypatch.setattr(scout_mod, "run_agent", agent)
    monkeypatch.setattr(stage, "_open_budget", lambda session: budget or Budget())
    stage.scout_stage(DAY, state, config=env.config)
    return state


def test_at_most_the_configured_number_of_abstracts_go_to_the_scout_in_one_call(
        env, monkeypatch) -> None:
    seed(TITLES)
    agent = FakeAgent(env.root, reply="\n".join(
        f"NO CANDIDATE arxiv:9{i}: nothing to test" for i in range(4)))
    state = run_scout(env, monkeypatch, agent)
    assert len(agent.calls) == 1
    handoff = json.loads((env.root / "data/publications/handoff/2026-10-25.json").read_text())
    assert len(handoff) == env.cfg["max_abstracts_per_night"] == 2
    assert {p["id"] for p in handoff} == set(state["publications"]["handoff"]["ids"])
    assert all(p["abstract"] and p["link"] and p["matched"] for p in handoff)
    assert "data/publications/handoff/2026-10-25.json" in agent.calls[0]["prompt"]
    assert "DATA from outside" in agent.calls[0]["prompt"]
    assert agent.calls[0]["agent"] == "scout" and agent.calls[0]["budget"].spent_tokens == 4321
    assert state["publications"]["handoff"]["waiting_after"] == 2

    with session_scope() as session:
        handed = session.query(Publication).filter(Publication.handed_at.is_not(None)).all()
        assert len(handed) == 2
        assert all(h.scout_note.startswith("NO CANDIDATE") for h in handed)

    # The next night the other two go, and nobody goes twice.
    state2 = run_scout(env, monkeypatch, agent, state={"done": {}, "publications": {"fetch": {}}})
    second = json.loads((env.root / "data/publications/handoff/2026-10-25.json").read_text())
    assert {p["id"] for p in second}.isdisjoint({p["id"] for p in handoff})
    assert state2["publications"]["handoff"]["waiting_after"] == 0


def test_a_draft_is_tied_to_its_paper(env, monkeypatch) -> None:
    seed(TITLES[:2])
    with session_scope() as session:
        rows = {r.title: r for r in session.query(Publication)}
    paper = rows[TITLES[0]]
    other = rows[TITLES[1]]
    agent = FakeAgent(
        env.root, drafts={"funding-deadband.yaml": draft_text("funding-deadband", paper.link)},
        reply=f"DRAFT {paper.id}: cards/drafts/funding-deadband.yaml\n"
              f"NO CANDIDATE {other.id}: only a pricing model")
    state = run_scout(env, monkeypatch, agent)
    hand = state["publications"]["handoff"]
    assert hand["drafts"] == {paper.id: "cards/drafts/funding-deadband.yaml"}
    assert hand["problems"] == []
    by_id = {p["id"]: p["result"] for p in hand["papers"]}
    assert by_id[paper.id].startswith("черновик") and "only a pricing model" in by_id[other.id]
    with session_scope() as session:
        assert session.get(Publication, paper.id).draft == "cards/drafts/funding-deadband.yaml"
        assert session.get(Publication, other.id).draft is None


def test_a_draft_the_reply_does_not_name_is_found_by_the_link_it_cites(env, monkeypatch) -> None:
    seed(TITLES[:1])
    with session_scope() as session:
        paper = session.query(Publication).one()
    agent = FakeAgent(env.root, drafts={"x.yaml": draft_text("x", paper.link)}, reply="done")
    state = run_scout(env, monkeypatch, agent)
    assert state["publications"]["handoff"]["drafts"] == {paper.id: "cards/drafts/x.yaml"}


def test_what_the_scout_writes_outside_the_drafts_is_reverted(env, monkeypatch) -> None:
    seed(TITLES[:1])
    agent = FakeAgent(env.root, drafts={"../elsewhere.yaml": "x: 1\n"}, reply="NO CANDIDATE a: b")
    state = run_scout(env, monkeypatch, agent)
    assert not (env.root / "cards" / "elsewhere.yaml").exists()
    assert any("outside cards/drafts" in p for p in state["publications"]["handoff"]["problems"])


def test_a_night_stop_leaves_every_paper_waiting_and_is_tried_again(env, monkeypatch) -> None:
    seed(TITLES)
    agent = FakeAgent(env.root, reply="")

    def exhausted(session):
        raise NightExhausted("weekly utilization reached tonight's target")

    monkeypatch.setattr(stage, "_open_budget", exhausted)
    monkeypatch.setattr(scout_mod, "run_agent", agent)
    state = {"done": {}, "publications": {"fetch": {}}}
    stage.scout_stage(DAY, state, config=env.config)
    assert agent.calls == []
    assert "tonight's target" in state["publications"]["handoff"]["stopped"]
    with session_scope() as session:
        assert session.query(Publication).filter(Publication.handed_at.is_not(None)).count() == 0

    # Resumed with budget: the stop was not final.
    monkeypatch.setattr(stage, "_open_budget", lambda session: Budget())
    stage.scout_stage(DAY, state, config=env.config)
    assert len(agent.calls) == 1 and "stopped" not in state["publications"]["handoff"]
    # And once done it is not repeated.
    stage.scout_stage(DAY, state, config=env.config)
    assert len(agent.calls) == 1


def test_a_candidates_cap_reached_inside_the_call_is_a_stop_too(env, monkeypatch) -> None:
    from qlab.budget import CandidateExhausted

    seed(TITLES)

    def capped(*a, **k):
        raise CandidateExhausted("candidate None used 5 of 5 tokens")

    state = run_scout(env, monkeypatch, capped)
    assert "used 5 of 5" in state["publications"]["handoff"]["stopped"]
    with session_scope() as session:
        assert session.query(Publication).filter(Publication.handed_at.is_not(None)).count() == 0


def test_when_the_venue_scout_was_stopped_by_the_budget_no_paper_is_handed_over(
        env, monkeypatch) -> None:
    seed(TITLES)
    agent = FakeAgent(env.root)
    state = {"done": {}, "publications": {"fetch": {}},
             "scout": {"reply": None, "stopped": "the night ended"}}
    state = run_scout(env, monkeypatch, agent, state=state)
    assert agent.calls == [] and "ended in the search stage" in state["publications"]["handoff"][
        "stopped"]


def test_with_nothing_waiting_the_budget_is_not_even_opened(env, monkeypatch) -> None:
    seed(["Quantum walks on graphs"], abstract=OFF_TOPIC)
    agent = FakeAgent(env.root)
    monkeypatch.setattr(scout_mod, "run_agent", agent)

    def never(session):
        raise AssertionError("no budget should be opened for an empty queue")

    monkeypatch.setattr(stage, "_open_budget", never)
    state = {"done": {}, "publications": {"fetch": {}}}
    stage.scout_stage(DAY, state, config=env.config)
    assert agent.calls == [] and state["publications"]["handoff"]["ids"] == []


def test_the_owner_can_switch_the_hand_off_off(env, monkeypatch) -> None:
    seed(TITLES)
    env.config.write_text(yaml.safe_dump(env.cfg | {"scout_handoff": False}), encoding="utf-8")
    agent = FakeAgent(env.root)
    state = run_scout(env, monkeypatch, agent)
    assert agent.calls == [] and "handoff" not in state["publications"]


def test_a_paper_without_a_readable_abstract_is_never_handed_over(env, monkeypatch) -> None:
    from qlab.publications.filter import load_rules
    from qlab.publications.items import Item

    with session_scope() as session:
        store.add_new(session, [Item(
            id="doi:10.2139/ssrn.1", source="ssrn", title="Perpetual futures funding rate",
            authors=[], published=NOW, abstract="", link="https://doi.org/10.2139/ssrn.1")],
            now=NOW)
        assert store.apply_filter(session, load_rules())  # it is a candidate by topic
    agent = FakeAgent(env.root)
    state = run_scout(env, monkeypatch, agent)
    assert agent.calls == [] and state["publications"]["handoff"]["ids"] == []


def test_the_scout_cannot_read_papers_outside_the_budget_guard(env) -> None:
    with pytest.raises(TypeError, match="run_agent needs a CandidateBudget"):
        scout_mod.scout_publications(DAY, [{"id": "a", "title": "t", "link": "l", "abstract": "x"}],
                                     budget=None, session=None)
    # The only way around the guard is to say so, with a reason.
    with pytest.raises(ValueError):
        ExplicitNoBudget("")


# ---- the report and the place in the night ---------------------------------------------------


def test_the_report_lists_new_candidates_and_what_became_of_the_handed_ones(
        env, monkeypatch) -> None:
    seed(TITLES[:1])
    with session_scope() as session:
        paper = session.query(Publication).one()
    agent = FakeAgent(env.root, drafts={"d.yaml": draft_text("d", paper.link)},
                      reply=f"DRAFT {paper.id}: cards/drafts/d.yaml")
    state = run_scout(env, monkeypatch, agent)
    state["publications"]["fetch"] = {
        "fetched": 10, "new": 4, "requests": 3, "errors": ["arxiv TR: ConnectError: offline"],
        "filter_version": "v1", "new_candidates": [
            {"id": paper.id, "title": paper.title, "link": paper.link, "source": "arxiv",
             "published": "2026-10-24", "why": "strong: perpetual futures"}]}
    text = "\n".join(stage.report_lines(state))
    assert "## Новые публикации" in text
    assert f"[{paper.title}]({paper.link})" in text and "strong: perpetual futures" in text
    assert "сбой чтения: arxiv TR" in text
    assert "черновик cards/drafts/d.yaml" in text
    assert "доходность" in text  # the report says the rule looks at topic only
    assert stage.report_lines({}) == []


def test_the_search_stage_fetches_first_and_hands_papers_over_after_the_venue_scout(
        env, monkeypatch) -> None:
    order = []
    monkeypatch.setattr(stage, "fetch_stage", lambda day, state: order.append("fetch"))
    monkeypatch.setattr(stage, "scout_stage", lambda day, state: order.append("papers"))
    monkeypatch.setattr("qlab.venue_watch.watch", lambda day: order.append("events") or [])
    monkeypatch.setattr("qlab.agents.scout.scout",
                        lambda day, **kw: order.append("venue scout") or SimpleNamespace(
                            reply="NO CANDIDATE", draft=None, problems=[]))

    class _Stage:
        def candidate(self, _idea):
            return Budget()

    class _Night:
        @staticmethod
        def open(_session, _reading):
            return _Night()

        def stage(self, _stage, candidates):
            return _Stage()

    monkeypatch.setattr("qlab.budget.NightBudget", _Night)
    monkeypatch.setattr("qlab.budget.probe",
                        lambda session: SimpleNamespace(report=SimpleNamespace(reading=None)))
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert order == ["fetch", "events", "venue scout", "papers"]
    # A resumed night whose venue scout already ran still gets its papers step.
    order.clear()
    night.search_stage(DAY, state)
    assert order == ["fetch", "papers"]
