"""The night's scout steps (`qlab.night.search_stage`, `scout_cards_stage`, the cataloguer's
hand-off): when the scout is called, what it is given, and that every agent is a fake."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
import yaml

import qlab.night as night
import qlab.registry.db as db_module
from qlab import scout_plan
from qlab.agents import cataloguer as cat
from qlab.budget import BudgetExhausted
from qlab.registry.models import Base
from qlab.venue_watch import Event

DAY = date(2026, 11, 3)
MEASURE = {"id": "short_perpetual@30d", "position": "short", "horizon_days": 30, "n": 45,
           "months": 17, "mean_pct": 27.5, "median_pct": 41.8, "t": 2.6,
           "first_half_pct": 25.9, "second_half_pct": 29.0}
EVIDENCE = {"patterns": [{"key": "listing/hyperliquid", "tradeable": True, "measures": [MEASURE]}]}


class Budget:
    def __init__(self) -> None:
        self.cap_tokens, self.spent_tokens = 900_000, 0


class Night:
    opened_with: list = []

    @staticmethod
    def open(_session, _reading):
        Night.opened_with.append("open")
        return Night()

    def stage(self, _stage, candidates):
        Night.candidates = candidates
        return SimpleNamespace(candidate=lambda _idea: Night.budget)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A registry in a temp file, a scout config in the scout's root, a fake budget guard and the
    two neighbours of the scout (the papers' fetch and hand-off) stubbed out."""
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    root = tmp_path / "scout-root"
    (root / "night").mkdir(parents=True)
    monkeypatch.setattr(scout_plan, "QLAB_ROOT", root)
    Night.opened_with, Night.budget = [], Budget()
    monkeypatch.setattr("qlab.budget.NightBudget", Night)
    monkeypatch.setattr("qlab.budget.probe",
                        lambda session: SimpleNamespace(report=SimpleNamespace(reading=None)))
    monkeypatch.setattr("qlab.publications.stage.fetch_stage", lambda day, state: None)
    monkeypatch.setattr("qlab.publications.stage.scout_stage", lambda day, state: None)

    def config(**sections) -> None:
        (root / "night/scout.yaml").write_text(yaml.safe_dump(sections), encoding="utf-8")

    yield SimpleNamespace(root=root, config=config, tmp=tmp_path)
    db_module._engine = None
    db_module._SessionLocal = None


def _fake_scout(monkeypatch, calls: list, *, reply="NO CANDIDATE: no payer named"):
    def scout(day, **kwargs):
        calls.append(kwargs)
        Night.budget.spent_tokens = 4321
        return SimpleNamespace(
            reply=reply, draft=None, problems=[], evidence_file="data/events/evidence/x.json",
            web=scout_plan.WebLog(fetched=[{"tool": "WebFetch", "url": "https://arxiv.org/abs/1",
                                            "day": "2026-11-03", "refused": False}]))

    monkeypatch.setattr("qlab.agents.scout.scout", scout)


def _evidence(monkeypatch, evidence=EVIDENCE):
    monkeypatch.setattr("qlab.event_evidence.ensure",
                        lambda day, root, cfg=None, **kw: (root / "x.json", evidence))


def _watch(monkeypatch, events):
    monkeypatch.setattr("qlab.venue_watch.watch", lambda day: events)


EVIDENCE_ON = {"wake": {"on_evidence": True}, "budget": {"share_divisor": 4,
                                                          "max_tokens_per_night": 400_000}}


# --------------------------------------------------------------------------
# Whether the scout is called


def test_a_measured_pattern_calls_the_scout_once_and_the_same_sample_does_not_again(
        env, monkeypatch) -> None:
    env.config(**EVIDENCE_ON)
    _watch(monkeypatch, [])
    _evidence(monkeypatch)
    calls: list = []
    _fake_scout(monkeypatch, calls)
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert len(calls) == 1 and calls[0]["wake"].call
    scout = state["scout"]
    assert scout["woken_by"].startswith("1 measured pattern")
    assert scout["signals"] == ["listing/hyperliquid::short_perpetual@30d"]
    assert scout["web"]["fetched"][0]["url"] == "https://arxiv.org/abs/1"
    assert state["scout_wake"]["call"] and state["evidence"]["patterns"] == 1
    # The share and the nightly cap of night/scout.yaml shape the scout's own budget.
    assert Night.candidates == 4 and Night.budget.cap_tokens == 400_000

    again: dict = {"done": {}}
    night.search_stage(date(2026, 11, 4), again)
    assert len(calls) == 1                                  # not called: looked at, same sample
    assert again["scout"]["reply"].startswith("NO CANDIDATE") and again["scout"]["tokens"] == 0
    assert again["scout_wake"]["seen"] == ["listing/hyperliquid::short_perpetual@30d"]


def test_a_night_with_nothing_to_look_at_opens_no_budget_and_spends_nothing(
        env, monkeypatch) -> None:
    env.config(**EVIDENCE_ON)
    _watch(monkeypatch, [Event("hyperliquid", "funding-extreme", "X", "+10% a year")])
    _evidence(monkeypatch, {"patterns": []})
    calls: list = []
    _fake_scout(monkeypatch, calls)
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert calls == [] and Night.opened_with == []     # not even the probe that opens the budget
    assert state["scout"]["reply"].startswith("NO CANDIDATE: no listing")


def test_a_structural_event_still_calls_the_scout_without_evidence(env, monkeypatch) -> None:
    env.config()                                       # the defaults: no evidence rule
    _watch(monkeypatch, [Event("binance", "listed", "NEWUSDT", "")])
    calls: list = []
    _fake_scout(monkeypatch, calls)
    night.search_stage(DAY, {"done": {}})
    assert len(calls) == 1 and calls[0]["evidence"] is None


def test_a_failed_evidence_build_is_reported_and_the_scout_goes_on_with_the_events(
        env, monkeypatch) -> None:
    env.config(**EVIDENCE_ON)
    _watch(monkeypatch, [Event("binance", "listed", "NEWUSDT", "")])

    def broken(*a, **k):
        raise FileNotFoundError("no cached daily history for binance")

    monkeypatch.setattr("qlab.event_evidence.ensure", broken)
    calls: list = []
    _fake_scout(monkeypatch, calls)
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert len(calls) == 1 and "no cached daily history" in state["evidence"]["error"]
    assert "История событий не посчитана" in "\n".join(night._scout_lines(state))


def test_the_scout_can_be_switched_off(env, monkeypatch) -> None:
    env.config(enabled=False)
    _watch(monkeypatch, [Event("binance", "listed", "NEWUSDT", "")])
    calls: list = []
    _fake_scout(monkeypatch, calls)
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert calls == [] and "выключен" in state["scout"]["reply"]


def test_a_budget_stop_is_a_normal_stop_and_the_pattern_is_not_marked_as_looked_at(
        env, monkeypatch) -> None:
    env.config(**EVIDENCE_ON)
    _watch(monkeypatch, [])
    _evidence(monkeypatch)

    def stopped(day, **kwargs):
        raise BudgetExhausted("night budget")

    monkeypatch.setattr("qlab.agents.scout.scout", stopped)
    state = {"done": {}}
    night.search_stage(DAY, state)
    assert state["scout"]["stopped"] == "night budget"
    assert scout_plan.load_looked_at() == {}


# --------------------------------------------------------------------------
# The report


def test_the_report_says_why_the_scout_was_called_what_it_read_and_what_it_answered(
        env, monkeypatch) -> None:
    env.config(**EVIDENCE_ON)
    _watch(monkeypatch, [])
    _evidence(monkeypatch)
    _fake_scout(monkeypatch, [])
    state = {"done": {}}
    night.search_stage(DAY, state)
    text = "\n".join(night._scout_lines(state))
    assert "## Разведчик" in text and "звали" in text
    assert "https://arxiv.org/abs/1" in text and "NO CANDIDATE: no payer named" in text


# --------------------------------------------------------------------------
# The boundary with the cataloguer


def _flagged_state(**extra) -> dict:
    entry = {"id": "arxiv:2601.00001", "title": "A paper", "link": "https://arxiv.org/abs/2601.1",
             "status": "catalogued", "verdict": "mechanism", "mechanism": "new-mech",
             "card_candidate": True}
    return {"done": {}, "catalogue": {"entries": [entry]}, **extra}


def test_a_new_mechanism_of_the_cataloguer_goes_to_the_scout_with_its_draft_once(
        env, monkeypatch) -> None:
    env.config(papers={"mode": "from_cataloguer"}, wake=EVIDENCE_ON["wake"],
               budget={**EVIDENCE_ON["budget"], "max_calls_per_night": 2})
    ledger = [cat.Paper(id="arxiv:2601.00001", link="https://arxiv.org/abs/2601.1",
                        title="A paper", status="catalogued", mechanism="new-mech")]
    monkeypatch.setattr(cat, "load_ledger", lambda: ledger)
    monkeypatch.setattr(cat, "save_ledger", lambda held: None)
    seen: list = []

    def hand(day, papers, **kwargs):
        seen.append((papers, kwargs))
        Night.budget.spent_tokens = 4321
        return SimpleNamespace(
            reply="NO CANDIDATE arxiv:2601.00001: no payer", drafts=[], problems=[],
            web=scout_plan.WebLog())

    monkeypatch.setattr("qlab.agents.scout.scout_publications", hand)
    state = _flagged_state(scout={"calls": 1, "tokens": 100_000})
    night.scout_cards_stage(DAY, state)
    [(papers, kwargs)] = seen
    assert kwargs["origin"] == "cataloguer" and papers[0]["mechanism"] == "new-mech"
    assert papers[0]["cataloguer_draft"].endswith("draft.yaml")
    assert papers[0]["text_file"].endswith("arxiv_2601.00001.txt")
    assert state["scout_cards"]["ids"] == ["arxiv:2601.00001"]
    assert state["scout_cards"]["tokens"] == 4321
    assert Night.budget.cap_tokens == 300_000          # 400k a night, 100k spent by the scout
    assert ledger[0].scout.startswith("кандидата нет: no payer")
    night.scout_cards_stage(DAY, state)                # a paper is handed once
    assert len(seen) == 1
    text = "\n".join(night._scout_lines(state))
    assert "Новые механизмы от разборщика каталога переданы разведчику: 1" in text


def test_the_scouts_nightly_call_cap_holds_back_the_cataloguers_papers(env, monkeypatch) -> None:
    env.config(papers={"mode": "from_cataloguer"}, budget={"max_calls_per_night": 1})
    monkeypatch.setattr(cat, "load_ledger", lambda: [])

    def never(*a, **k):
        raise AssertionError("no call is left")

    monkeypatch.setattr("qlab.agents.scout.scout_publications", never)
    state = _flagged_state(scout={"calls": 1})
    night.scout_cards_stage(DAY, state)
    assert "calls of the night" in state["scout_cards"]["skipped"]


def test_in_queue_mode_the_cataloguers_papers_are_not_handed_over(env, monkeypatch) -> None:
    env.config(papers={"mode": "queue"})

    def never(*a, **k):
        raise AssertionError("the scout reads the queue itself in this mode")

    monkeypatch.setattr("qlab.agents.scout.scout_publications", never)
    state = _flagged_state()
    night.scout_cards_stage(DAY, state)
    assert "scout_cards" not in state


def test_the_cataloguer_marks_only_a_new_mechanism_and_skips_what_the_scout_read(
        env, monkeypatch, tmp_path) -> None:
    config = tmp_path / "cataloguer.yaml"
    config.write_text(yaml.safe_dump({"enabled": True, "per_night": 2}), encoding="utf-8")
    env.config(papers={"mode": "from_cataloguer", "rank": {"strong_term": 3}})
    ledger: list = []
    monkeypatch.setattr(cat, "load_ledger", lambda: ledger)
    monkeypatch.setattr(cat, "save_ledger", lambda held: None)
    asked: list = []
    queue = [cat.Paper(id="arxiv:new", link="l1"), cat.Paper(id="arxiv:ext", link="l2")]

    def next_paper(held, session, **kwargs):
        asked.append(kwargs)
        return queue.pop(0) if queue else None

    def run_paper(paper, cfg, **kwargs):
        paper.status, paper.verdict = (("catalogued", "mechanism") if paper.id == "arxiv:new"
                                       else ("extended", "extends"))
        return paper

    monkeypatch.setattr(cat, "next_paper", next_paper)
    monkeypatch.setattr(cat, "run_paper", run_paper)
    monkeypatch.setattr("qlab.agents.scout.scout_publications",
                        lambda *a, **k: SimpleNamespace(reply="", drafts=[], problems=[],
                                                        web=scout_plan.WebLog()))
    state = {"done": {}, "papers_read": {"arxiv:scouted": "scout", "arxiv:mine": "cataloguer"}}
    night._catalogue_papers(DAY, state, config=config)
    flags = {e["id"]: e["card_candidate"] for e in state["catalogue"]["entries"]}
    assert flags == {"arxiv:new": True, "arxiv:ext": False}
    assert state["papers_read"]["arxiv:new"] == "cataloguer"
    # The scout's papers are skipped, the cataloguer's own are not, and the feed is ranked.
    assert asked[0]["skip_ids"] == {"arxiv:scouted"}
    assert asked[0]["rank_weights"] == {"strong_term": 3}
