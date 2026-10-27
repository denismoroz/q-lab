"""The carder's place in the night (docs/IDEAS.md): after the implement stage and before the
generator, behind --cards, one idea a night, one card waiting at a time, resumable, a line in the
morning report; and the ideas that researcher and director memos carry reach the inbox. The agents
are faked; the stage, the inbox and the report are real."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import qlab.agents.carder as carder
import qlab.agents.director as director
import qlab.agents.researcher as researcher
import qlab.budget as budget_module
import qlab.night as night
import qlab.registry.db as registry_db
from qlab import ideas
from qlab.budget import NightExhausted
from qlab.night import cards_stage as real_cards_stage  # the conftest stubs night.cards_stage
from qlab.test_night_director import _director_env
from qlab.test_night_research import _stage_env

DAY = date(2026, 10, 27)
FLAGS = dict(graveyard=False, paper=False, detectors=False, assess=False, research=False,
             waiting=False, programmer=False)

MEMO = """# Записка

```ideas
- "Страховка Bv2 по объёму торгов: включать, когда оборот растёт, а цена стоит"
```
"""


def _night(tmp_path, monkeypatch) -> list[str]:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "unwatched_bench", lambda watch: [])  # no registry needed
    order: list[str] = []
    for name in ("implement_stage", "search_stage", "catalogue_stage", "cards_stage",
                 "generate_stage"):
        monkeypatch.setattr(night, name, lambda *a, _n=name, **kw: order.append(_n))
    return order


def test_the_cards_stage_runs_after_implement_and_before_the_generator(
        tmp_path, monkeypatch) -> None:
    order = _night(tmp_path, monkeypatch)
    night.run(DAY, search=True, implement=True, **FLAGS)
    # implement first (the top of the cut order); a card written tonight is built tomorrow
    assert order == ["implement_stage", "search_stage", "catalogue_stage", "cards_stage",
                     "generate_stage"]
    order.clear()
    night.run(DAY, search=True, implement=False, cards=False, **FLAGS)
    assert order == ["search_stage", "catalogue_stage", "generate_stage"]
    order.clear()
    night.run(DAY, search=False, implement=False, **FLAGS)
    assert order == []  # the search stage's budget pays for the card: no search, no card


def test_a_night_stopped_before_the_cards_stage_names_it(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    order = _night(tmp_path, monkeypatch)
    assert night.STOP_STAGES["cards"] == "карточки для входящих идей"
    past = datetime.now(UTC) - timedelta(minutes=1)
    night.run(DAY, search=True, implement=False, stop_at=past, **FLAGS)
    assert order == []  # nothing new starts after the stop time
    assert night.load_state(DAY)["stopped_at"]["before"]


def test_a_failing_cards_stage_is_a_line_of_the_report_not_the_end_of_the_night(
        tmp_path, monkeypatch) -> None:
    order = _night(tmp_path, monkeypatch)

    def broken(*a, **kw):
        raise RuntimeError("no inbox")

    monkeypatch.setattr(night, "cards_stage", broken)
    report = night.run(DAY, search=True, implement=False, **FLAGS)
    assert order == ["search_stage", "catalogue_stage", "generate_stage"]
    assert "карточки для входящих идей" in report.read_text(encoding="utf-8")
    assert "no inbox" in report.read_text(encoding="utf-8")


# ---- the stage ---------------------------------------------------------------------------


class _Budget:
    spent_tokens = 11


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    """cards_stage with the budget, the session and the agent faked."""
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    config = tmp_path / "carder.yaml"
    config.write_text("enabled: true\nmodel: sonnet\nmax_turns: 5\nper_night: 1\n"
                      "max_attempts: 2\nnearest: 3\n", encoding="utf-8")
    asked: list = []
    ran: list[str] = []

    class _Stage:
        def candidate(self, name):
            return _Budget()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, stage, candidates):
            asked.append((stage, candidates))
            return _Stage()

    @contextmanager
    def fake_session():
        yield None

    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    monkeypatch.setattr(budget_module, "probe",
                        lambda _s: SimpleNamespace(report=SimpleNamespace(reading=None)))
    monkeypatch.setattr(registry_db, "session_scope", fake_session)
    monkeypatch.setattr(carder, "card_implemented", lambda *a, **kw: False)

    def fake_run(idea, inbox, cfg, *, budget, session, day, registry=None, **kw):
        ran.append(idea["id"])
        idea["attempts"] = int(idea.get("attempts") or 0) + 1
        ideas.set_status(idea, "carded", "карточка написана")
        idea["card"] = f"cards/{idea['id']}.yaml"
        return {"id": idea["id"], "source": idea["source"], "text": idea["text"],
                "status": "carded", "card": idea["card"], "new_block": "b", "grid": 4,
                "tried": ["bv2-flow-taker-72h"], "tokens": 11, "problems": []}

    monkeypatch.setattr(carder, "run_idea", fake_run)
    inbox: list[dict] = []
    for text in ("the first idea, long enough to be one", "the second idea, long enough"):
        ideas.add(inbox, text, source="owner", day=DAY)
    ideas.save(inbox)
    return SimpleNamespace(config=config, asked=asked, ran=ran)


def test_one_idea_a_night_in_the_search_stages_budget(stage_env) -> None:
    from qlab.budget import Stage

    state = {"done": {}}
    real_cards_stage(DAY, state, config=stage_env.config)
    assert stage_env.ran == ["idea-0001"] and stage_env.asked == [(Stage.SEARCH, 1)]
    assert [e["id"] for e in state["cards"]["entries"]] == ["idea-0001"]
    assert [e["id"] for e in state["cards"]["incoming"]] == ["idea-0001", "idea-0002"]
    inbox = ideas.load()
    assert [e["status"] for e in inbox] == ["carded", "new"]
    # a second call on the same night does not take another idea (per_night is 1)
    real_cards_stage(DAY, state, config=stage_env.config)
    assert stage_env.ran == ["idea-0001"]


def test_while_a_card_waits_for_the_implementer_no_idea_is_taken(stage_env, monkeypatch) -> None:
    state = {"done": {}}
    real_cards_stage(DAY, state, config=stage_env.config)
    next_night = {"done": {}}
    real_cards_stage(date(2026, 10, 28), next_night, config=stage_env.config)
    assert stage_env.ran == ["idea-0001"]  # the second waits: no tokens spent
    assert next_night["cards"]["waiting_for"] == "cards/idea-0001.yaml"
    assert next_night["cards"]["entries"] == []
    # the implementer takes the card; the idea moves on and the next night takes the next one
    monkeypatch.setattr(carder, "card_implemented", lambda *a, **kw: True)
    third = {"done": {}}
    real_cards_stage(date(2026, 10, 29), third, config=stage_env.config)
    assert third["cards"]["moved"] == ["idea-0001"]
    assert stage_env.ran == ["idea-0001", "idea-0002"]
    assert ideas.find(ideas.load(), "idea-0001")["status"] == "implemented"


def test_a_budget_stop_leaves_the_idea_new_and_a_crash_counts_an_attempt(
        stage_env, monkeypatch) -> None:
    def exhausted(*a, **kw):
        raise NightExhausted("the night's budget is spent")

    monkeypatch.setattr(carder, "run_idea", exhausted)
    state = {"done": {}}
    real_cards_stage(DAY, state, config=stage_env.config)
    assert "budget is spent" in state["cards"]["stopped"] and state["cards"]["entries"] == []
    assert ideas.find(ideas.load(), "idea-0001")["status"] == "new"

    def crash(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(carder, "run_idea", crash)
    for day in (DAY, date(2026, 10, 28)):
        real_cards_stage(day, {"done": {}}, config=stage_env.config)
    first = ideas.find(ideas.load(), "idea-0001")
    assert first["attempts"] == 2 and first["status"] == "needs-owner"
    assert "boom" in first["problems"][0]


def test_a_disabled_or_missing_config_does_nothing(stage_env, tmp_path) -> None:
    state = {"done": {}}
    real_cards_stage(DAY, state, config=tmp_path / "missing.yaml")
    stage_env.config.write_text("enabled: false\n", encoding="utf-8")
    real_cards_stage(DAY, state, config=stage_env.config)
    assert state == {"done": {}} and stage_env.ran == []


def test_the_morning_report_has_the_section_with_what_came_and_what_was_carded(
        stage_env, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    state = {"done": {}, "ideas_in": [{"id": "idea-0009", "source": "agent:researcher",
                                       "text": "an idea of the researcher about volume"}]}
    real_cards_stage(DAY, state, config=stage_env.config)
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    assert "## Входящие идеи и карточки" in text
    assert "idea-0009 (agent:researcher)" in text and "idea-0002 (owner)" in text
    assert "`cards/idea-0001.yaml`, новый блок `b`, вариантов 4" in text
    assert "Во входящих:" in text
    quiet = night.write_report(DAY, {"done": {}}).read_text(encoding="utf-8")
    assert "Входящие идеи и карточки" not in quiet


# ---- the memos ---------------------------------------------------------------------------


def test_a_researchers_ideas_block_reaches_the_inbox_once(tmp_path, monkeypatch) -> None:
    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    memo = tmp_path / "memo.md"
    memo.write_text(MEMO, encoding="utf-8")

    def fake_research(name, **kw):
        return researcher.ResearchOutcome(memo=memo, specs=[], problems=[])

    monkeypatch.setattr(researcher, "research", fake_research)
    state = {"done": {}}
    night.research_stage(DAY, state, noise_trials=1, config=config)
    night.research_stage(DAY, state, noise_trials=1, config=config)
    inbox = ideas.load()
    assert len(inbox) == 1 and inbox[0]["source"] == "agent:researcher"
    assert inbox[0]["text"].startswith("Страховка Bv2 по объёму торгов")
    assert [e["ideas_in"]["added"] for e in state["research"]] == [["idea-0001"], []]
    assert [i["id"] for i in state["ideas_in"]] == ["idea-0001"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_directors_ideas_block_reaches_the_inbox(tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)
    monkeypatch.setattr(ideas, "QLAB_ROOT", root)
    real_direct = director.direct

    def with_ideas(**kw):
        outcome = real_direct(**kw)
        memo = root / outcome.memo
        memo.write_text(MEMO.replace("Bv2", "основной стратегии") + memo.read_text(
            encoding="utf-8"), encoding="utf-8")
        return outcome

    monkeypatch.setattr(director, "direct", with_ideas)
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    inbox = ideas.load()
    assert [e["source"] for e in inbox] == ["agent:director"]
    assert "director-2026-10-26.md" in inbox[0]["notes"]
    assert state["director"]["ideas_in"]["added"] == ["idea-0001"]
    assert "directions" in state["director"]  # the directions block still applies
    db_module._engine = None
    db_module._SessionLocal = None


def test_an_unreadable_memo_never_stops_the_stage(tmp_path) -> None:
    state: dict = {}
    out = night.ingest_memo_ideas(DAY, state, tmp_path / "no-such.md", "agent:director")
    assert out["added"] == [] and "ideas_in" not in state
    bad = tmp_path / "bad.md"
    bad.write_text("```ideas\n- [unclosed\n```\n", encoding="utf-8")
    assert night.ingest_memo_ideas(DAY, state, bad, "agent:director")["added"] == []
    assert Path(str(bad)).is_file()
