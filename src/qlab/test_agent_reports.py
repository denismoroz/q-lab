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


def test_the_implementer_row_shows_the_block_and_the_routes_of_the_three_versions(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reports, "_by_hand", lambda key, seen: [])
    runs = {"spec": {"action": "ran", "route": "needs-forward"},
            "causal": {"action": "ran", "route": "paper"},
            "baseline": {"action": "ran", "route": "reject"},
            "stage2-causal": {"action": "error", "error": "no data"}}
    _night(tmp_path, "2026-10-28", {"implemented": [
        {"card": "cards/a.yaml", "idea_id": "a", "tokens": 900, "problems": [], "review": [],
         "block": {"name": "momentum", "reused": False}, "runs": runs},
        {"card": "cards/b.yaml", "idea_id": "b", "tokens": 100, "problems": ["source: x"],
         "attempts": 1},
        {"card": "cards/c.yaml", "idea_id": "c", "resumed": True,
         "runs": {"spec": {"action": "heavy", "reason": "hourly candles"}}}]})
    out = reports.agent_reports(days=1, state_dir=tmp_path)
    (run,) = {a["key"]: a["runs"] for a in out["agents"]}["implementer"]
    assert run["did"] == "карточки: cards/a.yaml, cards/b.yaml, cards/c.yaml"
    assert run["result"].startswith("код написан и прошёл проверки: 2 из 3")
    assert ("cards/a.yaml: написанная: ждёт проверки на новых данных; с выбором кода: прошла все "
            "правила, ждёт решения владельца; база: отвергнута") in run["result"]
    assert "cards/c.yaml: написанная: тяжёлые данные" in run["result"]
    assert "cards/a.yaml: блок `momentum` написан впервые" in run["details"]
    assert "cards/b.yaml: source: x" in run["details"]
    assert "cards/a.yaml: stage2-causal — no data" in run["details"]
    assert "cards/c.yaml: spec — hourly candles" in run["details"]
