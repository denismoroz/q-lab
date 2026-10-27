"""The «Агенты» page's data (qlab.agent_reports): each agent's runs copied from the nights'
saved state, newest first; a night the agent did not run has no row."""

from __future__ import annotations

import json

import qlab.agent_reports as reports


def _night(tmp_path, day: str, state: dict) -> None:
    (tmp_path / f"{day}.json").write_text(json.dumps(state, ensure_ascii=False))


def test_each_agent_gets_what_it_did_and_what_came_of_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reports, "_by_hand", lambda key, seen: [])
    _night(tmp_path, "2026-10-26", {
        "research": [{"strategy": "bv2", "memo": "m.md", "specs": ["a", "b"], "tokens": 1000,
                      "stand": ["", "- `bv2-x` (прогон 1): needs-forward — в год +5%",
                                "- `bv2-y` (прогон 2): reject — в год -1%"],
                      "problems": [], "drops": []}],
        "assessed": [{"idea": "frab", "trial": 3, "view": "too_harsh", "disputes": True,
                      "summary": "рано", "tokens": 50}],
        "generate": {"skipped": "в работе уже 6 стратегий при потолке 6"},
        "catalogue": {"entries": [{"id": "arxiv:1", "link": "l", "title": "Paper", "status": "none",
                                   "reason": "no rule", "tokens": 300, "problems": [],
                                   "review": []}]},
    })
    _night(tmp_path, "2026-10-27", {"assessed": [
        {"idea": "bv2", "trial": 4, "stopped": "сбой: AttributeError: x"}]})
    _night(tmp_path, "2026-10-01", {"assessed": []})  # older than the window
    out = reports.agent_reports(days=2, state_dir=tmp_path)
    assert out["nights"] == ["2026-10-27", "2026-10-26"]
    agents = {a["key"]: a for a in out["agents"]}
    research = agents["researcher"]["runs"][0]
    assert research["did"] == "1 стратегий (bv2); экспериментов 2"
    assert research["result"] == "исходы экспериментов: ждёт проверки вперёд — 1, отказ — 1"
    assert research["details"][0].startswith("bv2: bv2-x (прогон 1)")
    assessor = agents["assessor"]
    assert [r["day"] for r in assessor["runs"]] == ["2026-10-27", "2026-10-26"]
    assert assessor["runs"][0]["status"] == "stopped"
    assert assessor["runs"][1]["result"] == "спорит с вердиктом: frab"
    assert agents["generator"]["runs"][0]["status"] == "skipped"
    assert agents["cataloguer"]["runs"][0]["result"] == "механизма нет"
    assert agents["cataloguer"]["tokens"] == 300
    assert agents["programmer"]["runs"] == []  # it did not run in these nights


def test_a_night_of_an_older_format_is_a_row_that_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reports, "_by_hand", lambda key, seen: [])
    _night(tmp_path, "2026-10-20", {"research": [{"no_strategy_key": True}]})
    runs = {a["key"]: a["runs"] for a in reports.agent_reports(state_dir=tmp_path)["agents"]}
    assert runs["researcher"][0]["status"] == "failed"
    assert runs["researcher"][0]["did"] == "запись ночи не прочитана"


def test_the_carder_row_says_what_came_in_and_what_became_of_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reports, "_by_hand", lambda key, seen: [])
    _night(tmp_path, "2026-10-27", {"cards": {"entries": [
        {"id": "idea-0001", "text": "Bv3: страховка по объёму торгов, а не по цене",
         "status": "carded", "card": "cards/bv3-volume-hedge.yaml", "new_block": "volume_gated",
         "tokens": 900, "problems": []},
        {"id": "idea-0002", "text": "idea two", "status": "duplicate",
         "duplicate_of": "bv2-flow-taker-72h", "tokens": 100, "problems": ["reviewer: x"]}]}})
    _night(tmp_path, "2026-10-28", {"cards": {"entries": [],
                                              "waiting_for": "cards/bv3-volume-hedge.yaml"}})
    _night(tmp_path, "2026-10-29", {"cards": {"entries": [], "stopped": "night budget: spent"}})
    carder = {a["key"]: a for a in reports.agent_reports(state_dir=tmp_path)["agents"]}["carder"]
    third, second, first = carder["runs"]
    assert first["status"] == "ran" and carder["tokens"] == 1000
    assert first["result"] == "карточка написана; повтор уже сделанного"
    assert "idea-0002: повтор — bv2-flow-taker-72h" in first["details"]
    assert "idea-0001: карточка cards/bv3-volume-hedge.yaml, новый блок volume_gated" in \
        first["details"]
    assert second["status"] == "skipped" and "ещё не взята реализатором" in second["result"]
    assert third["status"] == "stopped" and "spent" in third["result"]
