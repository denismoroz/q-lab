"""The carder agent (qlab.agents.carder): an idea becomes a card only when code accepts the draft
(schema, tried ids, grid size, baseline, honest coin list, plain Russian) and the reviewer finds
no gap; a repeat or a question is answered the same way; one card of the carder waits for the
implementer at a time. The agent is faked; the checks, the queue and the inbox are real."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import qlab.agents.carder as carder
from qlab import ideas, tried
from qlab.budget import ExplicitNoBudget
from qlab.registry.card_importer import _problems as card_problems
from qlab.registry.models import AssetClass, Base, Idea, IdeaStatus, Profile, SourceType

DAY = date(2026, 10, 27)
ROOT = Path(__file__).resolve().parents[3]
IDEA = ("Bv3: включать страховку не по цене, а по объёму торгов: когда оборот растёт, а цена "
        "стоит, это признак скорого падения")
REGISTRY = tried.load_tried()
CONFIG = {"nearest": 3, "max_attempts": 2, "model": "sonnet"}

SUMMARY = (
    "Стратегия держит две крупные монеты, биткоин и эфир, и по каждой из них отдельно решает, "
    "нужна ли страховка. Страховка — это продажа той же монеты на контракте без срока: пока она "
    "включена, рост и падение цены гасят друг друга. Правило включения новое: смотрим не на цену, "
    "а на оборот. Если за последние дни оборот вырос вдвое, а цена почти не сдвинулась, значит "
    "крупные игроки уже продают, и страховку включаем. Пример: оборот биткоина за 7 дней вырос в "
    "два раза, цена на месте — страховка включается и держится, пока оборот не вернётся к "
    "обычному. Теряем в боковике: оборот прыгает без причины, и страховка включается зря, а "
    "каждое включение стоит комиссий. Перестанет работать, если оборот перестанет предупреждать "
    "о падениях.")


def _good(**over) -> dict:
    near = [t.id for t in tried.nearest(REGISTRY, IDEA)]
    card = {
        "verdict": "card",
        "idea_id": "bv3-volume-hedge",
        "title": "Страховка включается, когда оборот растёт, а цена стоит",
        "plain_summary": SUMMARY,
        "sources": [{"url": "ideas/inbox.yaml#idea-0001", "title": "Идея владельца",
                     "published": "2026-10"}],
        "driver": {"who_pays": "Те, кто покупает на вершине, когда крупные игроки уже продают.",
                   "why": "Крупные продавцы оставляют след в обороте раньше, чем в цене.",
                   "kill_condition": "Оборот перестаёт предшествовать падениям.",
                   "observable": "Доля падений, которым предшествовал рост оборота."},
        "prior": {"graveyard_matches": [],
                  "note": "Раньше сигнал строили по цене, потоку сделок и опционам, но не по "
                          "обороту в паре с неподвижной ценой."},
        "carder": {
            "hypothesis_ru": "Рост оборота при неподвижной цене предупреждает о падении раньше, "
                             "чем правило по цене; если страховка по обороту не даёт меньшей "
                             "просадки, гипотеза неверна.",
            "baseline": {"spec": "specs/bv2-binance.yaml",
                         "why_ru": "Это тот же книжный учёт, только правило включения другое."},
            "vs_tried": [{"id": i, "difference_ru": "Там сигнал строился по другим данным, "
                                                    "здесь по обороту рядом с ценой."}
                         for i in dict.fromkeys(["bv2-flow-taker-72h", *near])],
            "grid": [{"param": "volume_window_days", "values": [3, 7],
                      "meaning_ru": "За сколько дней сравнивать оборот.",
                      "source_ru": "Круглые значения, объявлены здесь до прогона."},
                     {"param": "volume_ratio", "values": [1.5, 2.0],
                      "meaning_ru": "Во сколько раз оборот должен вырасти.",
                      "source_ru": "Круглые значения, объявлены здесь до прогона."}],
            "data": {"venue": "binance", "interval": "1h", "start": "2019-09-08",
                     "coins": ["BTC", "ETH"]},
            "new_block": {"name": "volume_gated_hedge",
                          "params": [{"name": "volume_window_days",
                                      "meaning_ru": "Окно сравнения оборота, в сутках."},
                                     {"name": "volume_ratio",
                                      "meaning_ru": "Кратность роста оборота."}],
                          "rule_ru": "Каждый час считаем средний оборот за окно и за вдвое более "
                                     "длинное окно; если отношение выше порога, а цена за окно "
                                     "изменилась меньше чем на два процента, страховка включена."},
        },
    }
    for key, value in over.items():
        card[key] = value
    return card


def _check(draft, **over) -> list[str]:
    args = dict(registry=REGISTRY, entry_text=IDEA, graveyard={"pairs-cointegration-crypto"},
                taken_ideas={"strategy-b-v2"}, config=CONFIG, root=ROOT)
    return carder.check_draft(draft, **{**args, **over})


def _carder(draft: dict, **over) -> dict:
    out = json.loads(json.dumps(draft))
    out["carder"].update(over)
    return out


def test_a_good_card_passes_the_code_checks() -> None:
    assert tried.nearest(REGISTRY, IDEA), "the test idea must have nearest tried entries"
    assert _check(_good()) == []


def test_the_card_is_refused_for_jargon_a_short_summary_and_a_taken_name() -> None:
    jargon = _good(plain_summary=SUMMARY.replace("страховка", "хедж", 1) + " Лонг и шорт.")
    problems = _check(jargon)
    assert any("«хедж»" in p for p in problems) and any("«лонг»" in p for p in problems)
    assert any("too short" in p for p in _check(_good(plain_summary="Коротко и по-русски.")))
    assert any("is taken" in p for p in _check(_good(idea_id="strategy-b-v2")))
    assert any("lower-case" in p for p in _check(_good(idea_id="Bv3 Volume")))
    assert any("not in Russian" in p for p in _check(_good(title="Volume gated hedge for Bv2")))


def test_every_tried_id_must_exist_and_the_nearest_must_be_answered() -> None:
    draft = _good()
    ghost = _carder(draft, vs_tried=[*draft["carder"]["vs_tried"],
                                     {"id": "bv2-made-up", "difference_ru": "Другое правило."}])
    assert any("not in night/tried.yaml" in p for p in _check(ghost))
    near = [t.id for t in tried.nearest(REGISTRY, IDEA)]
    silent = _carder(draft, vs_tried=[{"id": "bv2-flow-taker-72h",
                                       "difference_ru": "Другие данные для сигнала."}])
    missed = [p for p in _check(silent) if "does not say what is new" in p]
    assert bool(missed) == bool(set(near) - {"bv2-flow-taker-72h"})
    assert any("must be a list" in p for p in _check(_carder(draft, vs_tried="none")))


def test_the_grid_is_declared_within_the_cap_and_names_real_parameters() -> None:
    draft = _good()
    big = _carder(draft, grid=[*draft["carder"]["grid"],
                               {"param": "ratchet_threshold", "values": [0.25, 0.5, 1.0],
                                "meaning_ru": "Порог.", "source_ru": "Круглые значения."}])
    assert any("12 variants, at most 8" in p for p in _check(big))
    alien = _carder(draft, grid=[{"param": "moon_phase", "values": [1, 2],
                                  "meaning_ru": "Фаза.", "source_ru": "Круглые значения."}])
    assert any("moon_phase" in p for p in _check(alien))
    single = _carder(draft, grid=[{"param": "volume_ratio", "values": [2.0],
                                   "meaning_ru": "Кратность.", "source_ru": "Круглые значения."}])
    assert any("2 or more distinct" in p for p in _check(single))
    assert any("at least one axis" in p for p in _check(_carder(draft, grid=[])))
    # a parameter of the baseline spec is a parameter too
    baseline = _carder(draft, grid=[{"param": "ratchet_threshold", "values": [0.25, 0.5],
                                     "meaning_ru": "Порог храповика.",
                                     "source_ru": "Из заранее объявленной сетки frab."}])
    assert _check(baseline) == []


def test_the_baseline_must_be_a_spec_that_loads() -> None:
    draft = _good()
    missing = _carder(draft, baseline={"spec": "specs/no-such-spec.yaml", "why_ru": "Потому."})
    assert any("does not load" in p for p in _check(missing))
    outside = _carder(draft, baseline={"spec": "docs/NIGHT.md", "why_ru": "Потому что так."})
    assert any("must be a spec file of specs/" in p for p in _check(outside))
    generated = _carder(draft, baseline={"spec": "specs/generated/x.yaml", "why_ru": "Потому."})
    assert any("not specs/generated" in p for p in _check(generated))


def test_a_hand_picked_coin_list_must_say_from_when_it_is_honest() -> None:
    draft = _good()
    data = dict(draft["carder"]["data"])
    hand = _carder(draft, data={**data, "coins": ["BTC", "ETH", "SOL", "AVAX"]})
    problems = _check(hand)
    assert any("hand-picked" in p for p in problems) and any("honest_from" in p for p in problems)
    flagged = _carder(draft, data={**data, "coins": ["BTC", "ETH", "SOL"], "hand_picked": True,
                                   "honest_from": "2025-06-01",
                                   "honest_why_ru": "Конец окна, на котором список подбирали."})
    assert _check(flagged) == []
    early = _carder(draft, data={**data, "coins": ["BTC", "SOL"], "hand_picked": True,
                                 "honest_from": "2019-01-01", "honest_why_ru": "Всегда."})
    assert any("before data.start" in p for p in _check(early))
    rule = {k: v for k, v in data.items() if k != "coins"}
    by_rule = _carder(draft, data={**rule, "coin_rule_ru": "Десять монет с самым большим оборотом "
                                                          "за прошлые тридцать дней, раз в месяц."})
    assert _check(by_rule) == []
    assert any("needs `coins`" in p for p in _check(_carder(draft, data=rule)))
    assert any("data.venue" in p for p in _check(_carder(draft, data={**data, "venue": "x"})))


def test_the_new_block_is_named_described_and_not_an_existing_one() -> None:
    draft = _good()
    block = draft["carder"]["new_block"]
    assert any("existing block" in p for p in _check(_carder(draft, new_block={
        **block, "name": "momentum"})))
    assert any("snake_case" in p for p in _check(_carder(draft, new_block={
        **block, "name": "Volume Gate"})))
    assert any("new_block.rule_ru" in p for p in _check(_carder(draft, new_block={
        **block, "rule_ru": "Rule in English only."})))
    assert any("new_block is missing" in p for p in _check(_carder(draft, new_block=None)))


def test_a_repeat_names_what_was_tried_and_a_question_is_asked_in_russian() -> None:
    ok = {"verdict": "duplicate", "duplicate_of": "bv2-flow-taker-72h",
          "reason_ru": "Это тот же поток сделок, только назван по-другому."}
    assert _check(ok) == []
    assert _check({**ok, "duplicate_of": "pairs-cointegration-crypto"}) == []  # the graveyard
    assert any("neither an id" in p for p in _check({**ok, "duplicate_of": "invented-id"}))
    assert any("reason_ru" in p for p in _check({**ok, "reason_ru": ""}))
    ask = {"verdict": "needs-owner",
           "question_ru": "Какой оборот считать: контракта или спота? От ответа зависят данные."}
    assert _check(ask) == []
    assert any("question_ru" in p for p in _check({"verdict": "needs-owner",
                                                    "question_ru": "Which volume?"}))
    assert any("verdict must be" in p for p in _check({"verdict": "maybe"}))
    assert _check("not a mapping") == ["the draft is not a mapping"]


def test_a_graveyard_match_must_be_an_idea_of_the_registry() -> None:
    draft = _good(prior={"graveyard_matches": ["no-such-idea"], "note": "Отличается вопросом."})
    assert any("no-such-idea" in p for p in _check(draft))
    ok = _good(prior={"graveyard_matches": ["pairs-cointegration-crypto"],
                      "note": "Другой вопрос, чем тот, по которому похоронили."})
    assert _check(ok) == []


def test_the_card_file_keeps_the_idea_word_for_word_and_imports_cleanly(tmp_path) -> None:
    entry = {"id": "idea-0001", "source": "owner", "date": "2026-10-27", "text": IDEA + "\n"}
    text = carder.render_card(_good(), entry, day=DAY)
    card = yaml.safe_load(text)
    assert "verdict" not in card and card["found_at"] == "2026-10-27"
    assert card["idea_source"]["text"] == IDEA + "\n"
    assert card["idea_source"]["inbox"] == "idea-0001"
    assert card_problems(card, Path("cards/bv3-volume-hedge.yaml")) == []
    assert text.startswith("# Карточка написана агентом-карточником")


def test_the_queue_gets_one_card_more_and_keeps_its_comments(tmp_path) -> None:
    queue = tmp_path / "implement_queue.yaml"
    queue.write_text("# Cards the implementer writes code for.\n# A second comment.\ncards: []\n",
                     encoding="utf-8")
    carder.append_to_queue("cards/a.yaml", queue)
    carder.append_to_queue("cards/b.yaml", queue)
    carder.append_to_queue("cards/a.yaml", queue)  # not twice
    assert carder.load_queue(queue) == ["cards/a.yaml", "cards/b.yaml"]
    assert queue.read_text(encoding="utf-8").startswith("# Cards the implementer")
    assert "# A second comment." in queue.read_text(encoding="utf-8")
    # the real file of the repository has the same shape
    real = ROOT / "night/implement_queue.yaml"
    assert isinstance(yaml.safe_load(real.read_text(encoding="utf-8"))["cards"], list)


def test_only_one_card_of_the_carder_waits_and_the_implementer_taking_it_frees_the_slot(
        tmp_path) -> None:
    inbox: list[dict] = []
    ideas.add(inbox, "the first idea, long enough to be one", source="owner", day=DAY)
    ideas.add(inbox, "the second idea, long enough to be one", source="owner", day=DAY)
    inbox[0]["card"] = "cards/first.yaml"
    ideas.set_status(inbox[0], "carded")
    assert [e["id"] for e in carder.pending_cards(inbox)] == ["idea-0001"]
    states, root = tmp_path / "night", tmp_path
    states.mkdir()
    assert carder.sync_implemented(inbox, states=states, root=root) == []
    # a night recorded the implementer's work on the card
    (states / "2026-10-28.json").write_text(
        json.dumps({"done": {"implement:cards/first.yaml": "2026-10-28T03:40:00"}}))
    assert carder.sync_implemented(inbox, states=states, root=root) == ["idea-0001"]
    assert inbox[0]["status"] == "implemented" and carder.pending_cards(inbox) == []
    # or its spec exists
    inbox[1]["card"] = "cards/second.yaml"
    ideas.set_status(inbox[1], "carded")
    (root / "specs/agent").mkdir(parents=True)
    (root / "specs/agent/second.yaml").write_text("idea_id: second\n")
    assert carder.sync_implemented(inbox, states=states, root=root) == ["idea-0002"]


# --------------------------------------------------------------------------
# One attempt, with the agent faked


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "specs").mkdir(parents=True)
    (root / "specs/bv2-binance.yaml").write_text(
        (ROOT / "specs/bv2-binance.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (root / "night").mkdir()
    queue = root / "night/implement_queue.yaml"
    queue.write_text("# queue\ncards: []\n", encoding="utf-8")
    monkeypatch.setattr(carder, "DRAFTS", Path("cards/drafts"))
    monkeypatch.setattr(carder, "CARDS", Path("cards"))
    monkeypatch.setattr(carder, "BRIEFS", Path("data/carder"))
    monkeypatch.setattr(carder, "QUEUE", queue)
    monkeypatch.setattr(ideas, "INBOX", root / "ideas/inbox.yaml")
    monkeypatch.setattr(carder, "_changed_paths", lambda: [])
    monkeypatch.setattr(carder, "_revert", lambda path: None)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Idea(id="pairs-cointegration-crypto", title="Пары монет", notes="NO-GO: спред",
                     status=IdeaStatus.REJECTED, source_type=SourceType.GRAVEYARD,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER))
    session.flush()
    inbox: list[dict] = []
    entry, _ = ideas.add(inbox, IDEA, source="owner", day=DAY)
    return SimpleNamespace(root=root, queue=queue, session=session, inbox=inbox, entry=entry)


def _agent(monkeypatch, draft: dict | None, review: str = "VERDICT: OK") -> list[dict]:
    calls: list[dict] = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        calls.append({"agent": agent, "prompt": prompt, "args": extra_args})
        if agent == "carder" and draft is not None:
            target = Path(cwd) / "cards/drafts/idea-0001.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(draft, allow_unicode=True), encoding="utf-8")
        return SimpleNamespace(text=review if agent.endswith("review") else "DRAFT: ok",
                               exit_code=0)

    monkeypatch.setattr(carder, "run_agent", fake)
    return calls


def _run(world) -> dict:
    return carder.run_idea(world.entry, world.inbox, CONFIG, budget=ExplicitNoBudget("test"),
                           session=world.session, day=DAY, registry=REGISTRY, root=world.root)


def test_the_brief_holds_the_idea_the_tried_registry_and_the_graveyard(world, monkeypatch) -> None:
    _agent(monkeypatch, _good())
    _run(world)
    brief = (world.root / "data/carder/brief-idea-0001.md").read_text(encoding="utf-8")
    assert IDEA in brief
    assert "## Nearest entries of night/tried.yaml" in brief and "bv2-flow-taker-72h" in brief
    assert "pairs-cointegration-crypto (rejected): Пары монет -- NO-GO: спред" in brief
    assert "- signal: `momentum` (lookback_days)" in brief
    assert "specs/bv2-binance.yaml" in brief and "the most candidates" in brief


def test_an_idea_that_passes_becomes_a_card_in_the_queue(world, monkeypatch) -> None:
    calls = _agent(monkeypatch, _good())
    out = _run(world)
    assert out["status"] == "carded", out
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]
    assert "Write" in calls[0]["args"][1] and "Bash" in calls[0]["args"][3]
    card = world.root / "cards/bv3-volume-hedge.yaml"
    assert yaml.safe_load(card.read_text(encoding="utf-8"))["idea_source"]["text"] == IDEA
    assert not (world.root / "cards/drafts/idea-0001.yaml").exists()  # moved, not copied
    assert carder.load_queue(world.queue) == ["cards/bv3-volume-hedge.yaml"]
    assert world.entry["card"] == "cards/bv3-volume-hedge.yaml" and world.entry["attempts"] == 1
    assert out["new_block"] == "volume_gated_hedge" and out["grid"] == 4
    assert "bv2-flow-taker-72h" in out["tried"]
    assert [e["id"] for e in carder.pending_cards(world.inbox)] == ["idea-0001"]


def test_a_repeat_closes_the_idea_naming_the_entry_and_queues_nothing(world, monkeypatch) -> None:
    _agent(monkeypatch, {"verdict": "duplicate", "duplicate_of": "bv2-flow-taker-72h",
                         "reason_ru": "Это тот же поток сделок под другим названием."})
    out = _run(world)
    assert out["status"] == "duplicate" and out["duplicate_of"] == "bv2-flow-taker-72h"
    assert "bv2-flow-taker-72h" in world.entry["notes"]
    assert carder.load_queue(world.queue) == []


def test_a_question_goes_to_the_owner(world, monkeypatch) -> None:
    _agent(monkeypatch, {"verdict": "needs-owner",
                         "question_ru": "Какой оборот считать: контракта или спота?"})
    out = _run(world)
    assert out["status"] == "needs-owner" and "контракта или спота" in out["question"]
    assert "вопрос владельцу" in world.entry["notes"]


def test_a_draft_code_refuses_is_retried_and_after_the_second_failure_waits_for_the_owner(
        world, monkeypatch) -> None:
    bad = _good(plain_summary="Слишком коротко.")
    calls = _agent(monkeypatch, bad)
    first = _run(world)
    assert first["status"] == "new" and any("too short" in p for p in first["problems"])
    assert [c["agent"] for c in calls] == ["carder"]  # no reviewer for a refused draft
    assert world.entry["problems"] == first["problems"]
    calls = _agent(monkeypatch, bad)
    second = _run(world)
    assert "too short" in calls[0]["prompt"] or "previous attempt" in calls[0]["prompt"]
    brief = (world.root / "data/carder/brief-idea-0001.md").read_text(encoding="utf-8")
    assert "Why the previous attempt at this idea was refused" in brief and "too short" in brief
    assert second["status"] == "needs-owner" and world.entry["attempts"] == 2
    assert "qlab idea retry" in world.entry["notes"]
    assert ideas.oldest_new(world.inbox, max_attempts=2) is None
    ideas.retry(world.inbox, "idea-0001")
    assert ideas.oldest_new(world.inbox, max_attempts=2) is world.entry


def test_a_reviewer_gap_blocks_the_card(world, monkeypatch) -> None:
    calls = _agent(monkeypatch, _good(),
                   review="GAP: carder.vs_tried: bv2-staged24 is the same rule\nVERDICT: GAP")
    out = _run(world)
    assert out["status"] == "new" and out["review"] == [
        "carder.vs_tried: bv2-staged24 is the same rule"]
    assert out["problems"] == ["reviewer: carder.vs_tried: bv2-staged24 is the same rule"]
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]
    assert carder.load_queue(world.queue) == [] and not (
        world.root / "cards/bv3-volume-hedge.yaml").exists()
    _agent(monkeypatch, _good(), review="no verdict at all")
    assert "the reviewer gave no verdict" in _run(world)["problems"][-1]


def test_no_draft_and_files_written_elsewhere_are_reported(world, monkeypatch) -> None:
    _agent(monkeypatch, None)
    out = _run(world)
    assert out["status"] == "new" and "no draft written" in out["problems"][0]
    world.entry["attempts"] = 0
    reverted = []
    seen = iter([[], ["src/qlab/night.py", "cards/x.yaml", "cards/drafts/idea-0001.yaml"]])
    monkeypatch.setattr(carder, "_changed_paths", lambda: next(seen))
    monkeypatch.setattr(carder, "_revert", reverted.append)
    _agent(monkeypatch, _good())
    out = _run(world)
    assert sorted(reverted) == ["cards/x.yaml", "src/qlab/night.py"]  # the draft itself stays
    assert out["status"] == "new" and any("outside its draft" in p for p in out["problems"])


def test_the_agent_cannot_change_the_inbox_or_the_queue(world, monkeypatch) -> None:
    ideas.save(world.inbox)
    original = world.queue.read_text(encoding="utf-8")

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        if agent == "carder":
            world.queue.write_text("cards: [cards/sneaky.yaml]\n", encoding="utf-8")
            Path(ideas.INBOX).write_text("ideas: []\n", encoding="utf-8")
        return SimpleNamespace(text="", exit_code=0)

    monkeypatch.setattr(carder, "run_agent", fake)
    out = _run(world)
    assert world.queue.read_text(encoding="utf-8") == original
    assert ideas.load()[0]["text"] == IDEA
    assert sum("outside its draft" in p for p in out["problems"]) >= 2


def test_the_report_names_what_came_what_was_carded_duplicates_and_questions() -> None:
    inbox: list[dict] = []
    for text in ("the first idea of the owner", "the second idea of the owner",
                 "the third idea of the owner"):
        ideas.add(inbox, text, source="owner", day=DAY)
    inbox[1]["card"] = "cards/bv3-volume-hedge.yaml"
    ideas.set_status(inbox[1], "carded")
    ideas.set_status(inbox[2], "needs-owner", "вопрос владельцу: какой оборот считать?")
    cards = {"entries": [
        {"id": "idea-0002", "status": "carded", "card": "cards/bv3-volume-hedge.yaml",
         "new_block": "volume_gated_hedge", "grid": 4, "tried": ["bv2-flow-taker-72h"],
         "tokens": 12345, "problems": []},
        {"id": "idea-0003", "status": "duplicate", "duplicate_of": "bv2-flow-taker-72h",
         "tokens": 100, "problems": ["reviewer: x"]},
        {"id": "idea-0004", "status": "needs-owner", "question": "Какой оборот считать?",
         "tokens": 50, "problems": []}],
        "incoming": [{"id": "idea-0001", "source": "owner", "text": "the first idea of the owner"}],
        "waiting_for": "cards/bv3-volume-hedge.yaml", "moved": ["idea-0009"]}
    text = "\n".join(carder.report_lines(
        cards, inbox, [{"id": "idea-0007", "source": "agent:researcher", "text": "an agent idea"}]))
    assert "Пришло за ночь:" in text and "idea-0007 (agent:researcher)" in text
    assert "`cards/bv3-volume-hedge.yaml`, новый блок `volume_gated_hedge`, вариантов 4" in text
    assert "это уже было — `bv2-flow-taker-72h`" in text
    assert "вопрос владельцу: Какой оборот считать?" in text
    assert "ещё не взята реализатором" in text and "idea-0009" in text
    assert "Ждут вашего решения:" in text and "qlab idea retry idea-0003" in text


def test_the_config_and_the_queue_of_the_repository_are_well_formed() -> None:
    config = carder.load_config()
    assert config["enabled"] is True and config["per_night"] == 1 and config["max_attempts"] == 2
    assert config["model"] == "sonnet"
    inbox = yaml.safe_load((ROOT / "ideas/inbox.yaml").read_text(encoding="utf-8"))["ideas"]
    assert isinstance(inbox, list)
    assert all(e["status"] in ideas.STATUSES and e["text"] for e in inbox)


def test_the_card_carries_the_fields_the_implementer_reads(tmp_path) -> None:
    # The carder and the implementer were built apart: the card must be readable by
    # qlab.card_fields, or it would sit in the implementer's queue for nothing.
    from qlab.card_fields import read_card_fields

    entry = {"id": "idea-0001", "source": "owner", "date": "2026-10-27", "text": IDEA}
    path = tmp_path / "cards" / "bv3-volume-hedge.yaml"
    path.parent.mkdir()
    path.write_text(carder.render_card(_good(), entry, day=DAY), encoding="utf-8")
    fields = read_card_fields(path, root=Path.cwd())  # raises CardFieldError on any problem
    assert fields.inbox and fields.new_block is not None and fields.grid is not None
    assert fields.grid.count >= 2 and fields.baseline_spec
    assert fields.idea_text == IDEA


def test_a_grid_source_is_a_citation_and_may_hold_the_repositorys_english() -> None:
    draft = _good()
    for axis in draft["carder"]["grid"]:
        axis["source_ru"] = "round values chosen here, before any run (night/own_numbers.yaml)"
    assert not any("source_ru" in p for p in _check(draft))
    draft["carder"]["grid"][0]["source_ru"] = " "
    assert any("source_ru is empty" in p for p in _check(draft))
