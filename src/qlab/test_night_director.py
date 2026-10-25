"""Tests for the research director's place in the night: its cadence (every two
days, a stop continued), the done-markers, the order it gives the researcher, its
focus, the owner's review flag and the morning report (docs/DIRECTOR.md)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import qlab.agents.director as director
import qlab.agents.researcher as researcher
import qlab.night as night
from qlab import directions as dirs
from qlab.test_night_research import _stage_env

DAY = date(2026, 10, 26)
CONFIG = dirs.DirectorConfig(enabled=True, cadence_days=2, max_turns=5,
                             families={"momentum": ("xs-mom",)})


def _ran(day: date, state_dir: Path) -> None:
    night.save_state(day, {"done": {"director:agent": "t", "director:directions": "t"},
                           "director": {"memo": "docs/research/director-x.md"}})


# ---- cadence ------------------------------------------------------------------------------


def test_the_director_runs_every_two_days_and_never_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    assert night.director_due(DAY, CONFIG)  # never run
    _ran(date(2026, 10, 24), tmp_path)
    assert night.last_director_day(DAY) == date(2026, 10, 24)
    assert night.director_due(DAY, CONFIG)  # two days since
    assert not night.director_due(date(2026, 10, 25), CONFIG)  # the day after: not yet
    assert night.director_due(date(2026, 10, 27), CONFIG)
    assert not night.director_due(DAY, dirs.DirectorConfig(False, 2, 5, {}))
    assert night.director_due(DAY, dirs.DirectorConfig(True, 3, 5, {})) is False
    # A night the director never finished (no done-marker) does not count as a run.
    night.save_state(date(2026, 10, 25), {"done": {}, "director": {"stopped": "budget"}})
    assert night.last_director_day(DAY) == date(2026, 10, 24)


def test_a_stopped_run_is_continued_the_next_night(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    _ran(date(2026, 10, 24), tmp_path)
    night.save_state(DAY, {"done": {}, "director": {"stopped": "the night's budget ended"}})
    # 24th ran, the 26th was due and stopped; the 27th is only a day after the last FINISHED
    # run's cadence would allow -- yet the stop is continued, a stop is not a lost cycle.
    assert night.director_due(date(2026, 10, 27), CONFIG)
    night.save_state(DAY, {"done": {"director:agent": "t"}, "director": {"stopped": "x"}})
    assert night.last_director_day(date(2026, 10, 27)) == DAY
    assert not night.director_due(date(2026, 10, 27), CONFIG)


# ---- the stage ---------------------------------------------------------------------------


def _director_env(tmp_path, monkeypatch, *, exhausted: bool = False):
    """A director_stage environment: fake budget, fake agent, a real tmp root with a
    catalogue, and a research plan."""
    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    root = tmp_path / "repo"
    (root / "docs" / "research").mkdir(parents=True)
    (root / "docs" / "MECHANISMS.md").write_text(
        "# c\n\n### xs-mom\n\n**Суть:** s\n\n**Блоки:** b\n\n**Параметры:**\n\n"
        "| параметр блока | значения из источника | источник |\n|---|---|---|\n"
        "| rebalance_days | 7 | [A](https://e.org) |\n\n**Чего блоки не выражают:** (1) x\n",
        encoding="utf-8")
    monkeypatch.setattr(director, "QLAB_ROOT", root)
    config.write_text("cadence: nightly\nstrategies:\n  bv2: {ideas: [i1], specs: []}\n"
                      "  trend: {ideas: [i2], specs: []}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "night").mkdir(exist_ok=True)
    (tmp_path / "night" / "research.yaml").write_text(config.read_text(), encoding="utf-8")
    director_config = tmp_path / "director.yaml"
    director_config.write_text("enabled: true\ncadence_days: 2\nfamilies:\n"
                               "  momentum: [xs-mom]\n", encoding="utf-8")
    agent_calls: list[str] = []

    def fake_direct(*, budget, session, config, plan, **kw):
        from qlab.budget import NightExhausted

        if exhausted:
            raise NightExhausted("the night's budget is spent")
        agent_calls.append("direct")
        memo = Path("docs/research/director-2026-10-26.md")
        (root / memo).write_text(
            "# Записка\n\n```directions\ngenerator:\n  prefer_mechanisms: [xs-mom]\n"
            "researcher:\n  trend: {priority: high, note: \"искать причину просадок\"}\n"
            "  bv2: {priority: low}\npropose_close_family:\n"
            "  - {family: momentum, why: \"нет блоков\"}\n```\n", encoding="utf-8")
        return director.DirectorOutcome(memo=memo, reply="ok", problems=[])

    monkeypatch.setattr(director, "direct", fake_direct)
    return agent_calls, director_config, db_module, root


def test_the_stage_writes_the_directions_and_marks_each_step(tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    assert agent_calls == ["direct"]
    assert {"director:agent", "director:directions"} <= set(state["done"])
    entry = state["director"]
    assert entry["memo"].endswith("director-2026-10-26.md") and entry["tokens"] == 7
    # The first directions ever wait for the owner.
    assert entry["directions"]["pending_owner_review"] is True
    assert entry["directions"]["researcher"]["trend"]["priority"] == "high"
    assert dirs.read_directions(root).status == "pending"
    # A repeat in the same night does nothing: no second call, no second write.
    night.director_stage(DAY, state, config=config)
    assert agent_calls == ["direct"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_director_takes_one_share_of_the_implement_stage_not_all_of_it(
        tmp_path, monkeypatch) -> None:
    import qlab.budget as budget_module
    from qlab.budget import Stage

    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)
    asked = []

    class _Stage:
        def candidate(self, name):
            return type("B", (), {"spent_tokens": 1})()

    class _NightBudget:
        @staticmethod
        def open(_session, _reading):
            return _NightBudget()

        def stage(self, stage, candidates):
            asked.append((stage, candidates))
            return _Stage()

    monkeypatch.setattr(budget_module, "NightBudget", _NightBudget)
    night.director_stage(DAY, {"done": {}}, config=config)
    # The implement stage's budget, shared with the researcher's two strategies.
    assert asked == [(Stage.IMPLEMENT, 3)]
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_restart_after_the_agent_applies_the_memo_without_calling_it_again(
        tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    # The night died after the agent finished but before the directions were written.
    (root / dirs.DIRECTIONS).unlink()
    state["done"].pop("director:directions")
    state["director"].pop("directions")
    night.director_stage(DAY, state, config=config)
    assert agent_calls == ["direct"]  # still the one call
    assert (root / dirs.DIRECTIONS).is_file() and "director:directions" in state["done"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_short_budget_skips_the_director_and_nothing_else(tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch, exhausted=True)
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    assert agent_calls == [] and "director:agent" not in state["done"]
    assert "budget is spent" in state["director"]["stopped"]
    assert not (root / dirs.DIRECTIONS).exists()
    night.save_state(DAY, state)
    assert night.director_due(date(2026, 10, 27), CONFIG)  # continued the next night
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_failing_stage_is_a_report_line_not_a_failed_night(tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)

    def broken(**kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(director, "direct", broken)
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    assert "RuntimeError: boom" in state["director"]["error"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_off_by_config_or_by_force_and_off_without_a_config(tmp_path, monkeypatch) -> None:
    agent_calls, config, db_module, root = _director_env(tmp_path, monkeypatch)
    config.write_text("enabled: false\n", encoding="utf-8")
    state = {"done": {}}
    night.director_stage(DAY, state, config=config)
    assert agent_calls == [] and "director" not in state
    night.director_stage(DAY, state, config=config, force=False)
    assert agent_calls == []
    night.director_stage(DAY, state, config=config, force=True)  # --director overrides enabled
    assert agent_calls == ["direct"]
    night.director_stage(DAY, {"done": {}}, config=tmp_path / "missing.yaml")
    assert agent_calls == ["direct"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_director_runs_before_the_researcher_and_not_on_a_night_without_research(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    order = []
    monkeypatch.setattr(night, "director_stage", lambda *a, **kw: order.append("director"))
    monkeypatch.setattr(night, "research_stage", lambda *a, **kw: order.append("research"))
    flags = dict(graveyard=False, paper=False, implement=False, search=False, detectors=False,
                 assess=False, waiting=False)
    night.run(DAY, research=True, **flags)
    assert order == ["director", "research"]
    order.clear()
    night.run(DAY, research=False, **flags)  # no researcher tonight: nothing to steer
    assert order == []
    night.run(DAY, research=False, director=True, **flags)
    assert order == ["director"]


# ---- the researcher obeys (once reviewed) --------------------------------------------------


def _research_env(tmp_path, monkeypatch, *, pending: bool):
    calls, config, db_module = _stage_env(tmp_path, monkeypatch)
    config.write_text("cadence: nightly\nfocus: the owner's question\nstrategies:\n"
                      "  alpha: {ideas: [i1], specs: [], owner_ideas: ['an owner idea']}\n"
                      "  bravo: {ideas: [i2], specs: []}\n"
                      "  charlie: {ideas: [i3], specs: []}\n", encoding="utf-8")
    parsed = dirs.parse_block(
        "researcher:\n  charlie: {priority: high, note: \"look at the losing months\"}\n"
        "  alpha: {priority: low}\n",
        mechanism_ids=set(), strategy_names={"alpha", "bravo", "charlie"}, family_names=set(),
        day=date(2026, 10, 26), pending_owner_review=pending)
    dirs.write_directions(parsed.directions, tmp_path)  # the stage env chdir'ed to tmp_path
    seen: dict[str, str | None] = {}

    def fake_research(name, **kw):
        calls.append(name)
        seen[name] = kw.get("focus")
        return researcher.ResearchOutcome(memo=None)

    monkeypatch.setattr(researcher, "research", fake_research)
    return calls, config, seen, db_module


def test_a_high_priority_strategy_goes_first_and_gets_the_directors_note(
        tmp_path, monkeypatch) -> None:
    calls, config, seen, db_module = _research_env(tmp_path, monkeypatch, pending=False)
    night.research_stage(DAY, {"done": {}}, noise_trials=1, config=config)
    assert calls == ["charlie", "bravo", "alpha"]  # high, normal, low
    assert "priority high" in seen["charlie"] and "look at the losing months" in seen["charlie"]
    assert "the owner's question" in seen["charlie"]  # the director adds to the focus, not replaces
    assert "priority low" in seen["alpha"] and "an owner idea" in seen["alpha"]
    assert "research director" not in seen["bravo"]  # normal and no note: nothing added
    db_module._engine = None
    db_module._SessionLocal = None


def test_directions_waiting_for_review_leave_the_researcher_as_it_was(
        tmp_path, monkeypatch) -> None:
    calls, config, seen, db_module = _research_env(tmp_path, monkeypatch, pending=True)
    night.research_stage(DAY, {"done": {}}, noise_trials=1, config=config)
    assert calls == ["alpha", "bravo", "charlie"]  # by name, as before
    assert all("research director" not in (focus or "") for focus in seen.values())
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_short_budget_cuts_the_low_priority_strategy(tmp_path, monkeypatch) -> None:
    from qlab.budget import NightExhausted

    calls, config, seen, db_module = _research_env(tmp_path, monkeypatch, pending=False)

    def fake_research(name, **kw):
        if name == "alpha":
            raise NightExhausted("the night's budget is spent")
        calls.append(name)
        return researcher.ResearchOutcome(memo=None)

    monkeypatch.setattr(researcher, "research", fake_research)
    state = {"done": {}}
    night.research_stage(DAY, state, noise_trials=1, config=config)
    assert calls == ["charlie", "bravo"]  # alpha (low) was last, and is the one left over
    assert "research:alpha" not in state["done"]
    db_module._engine = None
    db_module._SessionLocal = None


# ---- the morning report --------------------------------------------------------------------


def test_the_report_has_the_directors_section_and_flags_what_the_owner_decides(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    parsed = dirs.parse_block(
        "generator:\n  prefer_mechanisms: [xs-mom]\n  pause_mechanisms:\n"
        "    - {id: xs-rev, why: \"блоков нет\"}\n"
        "researcher:\n  trend: {priority: high, note: \"смотреть на плохие месяцы\"}\n"
        "catalogue_requests:\n  - {family: momentum, look_for: \"новые работы\", why: \"пробел\"}\n"
        "data_requests:\n  - {what: \"вторая площадка\", why: \"одна площадка\"}\n"
        "propose_close_family:\n  - {family: reversal, why: \"всё отвергнуто\"}\n",
        mechanism_ids={"xs-mom", "xs-rev"}, strategy_names={"trend"}, family_names={"reversal"},
        day=DAY, pending_owner_review=True)
    state = {"done": {}, "director": {
        "memo": "docs/research/director-2026-10-26.md", "tokens": 1234, "problems": [],
        "directions": director.summary(parsed.directions),
        "rejected": ["prefer_mechanisms: 'nope' is not a mechanism of the catalogue: ignored"]}}
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    assert "## Руководитель исследований" in text
    assert "записка `docs/research/director-2026-10-26.md`, 1,234 токенов" in text
    assert "направления ждут проверки владельца" in text and "qlab director --approve" in text
    assert "генератору предпочитать: xs-mom" in text
    assert "генератору приостановить `xs-rev` — блоков нет" in text
    assert "исследователь, trend: важность high; пометка: смотреть на плохие месяцы" in text
    assert "**предлагает закрыть `reversal` — решает владелец**: всё отвергнуто" in text
    assert "просьба к каталогу (momentum): новые работы — пробел" in text
    assert "просьба о данных: вторая площадка — одна площадка" in text
    assert "отброшено кодом: prefer_mechanisms: 'nope'" in text


def test_the_report_names_a_skipped_stage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_funnel_lines", lambda day: [])
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    state = {"done": {}, "director": {"stopped": "night budget: spent"}}
    text = night.write_report(DAY, state).read_text(encoding="utf-8")
    assert "не запускался: night budget: spent; продолжит в следующую ночь" in text
    assert "Руководитель исследований" not in night.write_report(
        DAY, {"done": {}}).read_text(encoding="utf-8")


def test_a_pending_stage_is_the_only_one_that_needs_the_review_line(tmp_path, monkeypatch
                                                                    ) -> None:
    parsed = dirs.parse_block("generator: {}", mechanism_ids=set(), strategy_names=set(),
                              family_names=set(), day=DAY, pending_owner_review=False)
    lines = night._director_lines({"directions": director.summary(parsed.directions)})
    assert not any("ждут проверки" in line for line in lines)


@pytest.mark.parametrize("name", ["director.yaml"])
def test_the_shipped_config_runs_every_two_days(name) -> None:
    config = dirs.load_config(Path(__file__).resolve().parents[2] / "night" / name)
    assert config.enabled and config.cadence_days == 2
