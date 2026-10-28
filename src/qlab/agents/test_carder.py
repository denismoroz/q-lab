"""The carder agent (qlab.agents.carder): an idea becomes a card only when code accepts the draft
(schema, tried ids, grid size, baseline, honest coin list, plain Russian) and the reviewer makes
no blocking remark (wording remarks are notes and never block); a repeat or a question is answered
the same way; one card of the carder waits for the implementer at a time. The agent is faked; the
checks, the queue and the inbox are real."""

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
CONFIG = {"nearest": 3, "max_attempts": 3, "model": "sonnet"}

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


DRAFT_FILE = "cards/drafts/idea-0001.yaml"


def _write_draft(target: Path, change) -> None:
    """What a faked writer does to the draft: a dict is written as the draft, a str as raw text
    (to break the YAML), a callable is handed the path and may do anything."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if callable(change):
        change(target)
    elif isinstance(change, str):
        target.write_text(change, encoding="utf-8")
    elif change is not None:
        target.write_text(yaml.safe_dump(change, allow_unicode=True), encoding="utf-8")


def _agent(monkeypatch, draft: dict | None, review: str = "VERDICT: OK", *, edit=None,
           notes=None) -> list[dict]:
    """Fake `run_agent`. The full writer (`carder`) writes `draft`; the edit pass (`carder-edit`)
    does `edit`, by default the same as `draft`; the notes pass (`carder-notes`) does `notes`,
    by default nothing. Each call is recorded with whether the draft was on disk when it began."""
    calls: list[dict] = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        target = Path(cwd) / DRAFT_FILE
        calls.append({"agent": agent, "prompt": prompt, "args": extra_args,
                      "max_turns": max_turns, "draft_existed": target.is_file(),
                      "draft_before": target.read_text(encoding="utf-8") if target.is_file()
                      else None})
        change = {"carder": draft, "carder-edit": draft if edit is None else edit,
                  "carder-notes": notes}.get(agent)
        _write_draft(target, change)
        return SimpleNamespace(text=review if agent.endswith("review") else "DRAFT: ok",
                               exit_code=0)

    monkeypatch.setattr(carder, "run_agent", fake)
    return calls


def _run(world, config: dict | None = None) -> dict:
    return carder.run_idea(world.entry, world.inbox, config or CONFIG,
                           budget=ExplicitNoBudget("test"), session=world.session, day=DAY,
                           registry=REGISTRY, root=world.root)


def _retitle(title: str):
    """A faked edit that changes the title of the draft on disk and nothing else."""
    def change(target: Path) -> None:
        draft = yaml.safe_load(target.read_text(encoding="utf-8"))
        draft["title"] = title
        target.write_text(yaml.safe_dump(draft, allow_unicode=True), encoding="utf-8")
    return change


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


def test_a_draft_code_refuses_is_retried_and_after_the_third_failure_waits_for_the_owner(
        world, monkeypatch) -> None:
    bad = _good(plain_summary="Слишком коротко.")
    calls = _agent(monkeypatch, bad)
    first = _run(world)
    assert first["status"] == "new" and any("too short" in p for p in first["problems"])
    assert [c["agent"] for c in calls] == ["carder"]  # no reviewer for a refused draft
    assert world.entry["problems"] == first["problems"]
    calls = _agent(monkeypatch, bad)
    second = _run(world)
    assert [c["agent"] for c in calls] == ["carder-edit"]
    assert "too short" in calls[0]["prompt"]
    brief = (world.root / "data/carder/brief-idea-0001.md").read_text(encoding="utf-8")
    assert "Why the previous attempt at this idea was refused" in brief and "too short" in brief
    assert second["status"] == "new" and world.entry["attempts"] == 2  # a third attempt is left
    assert ideas.oldest_new(world.inbox, max_attempts=3) is world.entry
    calls = _agent(monkeypatch, bad)
    third = _run(world)
    assert [c["agent"] for c in calls] == ["carder-edit"] and world.entry["attempts"] == 3
    assert third["status"] == "needs-owner" and "3 попыток" in world.entry["notes"]
    assert "qlab idea retry" in world.entry["notes"]
    assert ideas.oldest_new(world.inbox, max_attempts=3) is None
    ideas.retry(world.inbox, "idea-0001")
    assert ideas.oldest_new(world.inbox, max_attempts=3) is world.entry


def test_a_retry_edits_the_draft_of_the_earlier_attempt_instead_of_writing_it_again(
        world, monkeypatch) -> None:
    bad = _good(plain_summary="Слишком коротко.")
    _agent(monkeypatch, bad)
    first = _run(world)
    assert any("too short" in p for p in first["problems"]) and first["mode"] == "write"
    drafts = world.root / "cards/drafts"
    world.entry["review_notes"] = ["the title is clumsy"]
    calls = _agent(monkeypatch, _good(), edit=_good())
    out = _run(world, {**CONFIG, "edit_max_turns": 9})
    assert [c["agent"] for c in calls] == ["carder-edit", "carder-review"]
    edit = calls[0]
    assert out["mode"] == "edit" and out["status"] == "carded"
    # the old draft was on disk when the editor began: it was not deleted first
    assert edit["draft_existed"] and "Слишком коротко." in edit["draft_before"]
    assert edit["max_turns"] == 9
    # Edit allowed, Write and Bash not; the repository of the owner stays readable
    assert edit["args"][0:2] == ("--allowedTools", "Read,Glob,Grep,Edit")
    assert edit["args"][2:4] == ("--disallowedTools", "Bash,Write") and "--add-dir" in edit["args"]
    prompt = edit["prompt"]
    assert "cards/drafts/idea-0001.yaml" in prompt and "brief-idea-0001.md" in prompt
    assert "1. " in prompt and "plain_summary is too short" in prompt  # the numbered remark
    assert "- the title is clumsy" in prompt and "optional" in prompt
    assert "byte for byte" in prompt and "Do NOT reread the brief" in prompt
    # the rules a full writing has are the same text in the edit prompt
    assert carder.YAML_RULE in prompt and carder.PLAIN_RULE in prompt
    assert carder.READINGS_RULE.format(cap=8) in prompt and "DRAFT: <verdict>" in prompt
    assert carder.YAML_RULE in carder.INSTRUCTIONS.format(
        brief="b", draft="d", idea="i", frab="f", jargon="j", venues="v", intervals="n", cap=8,
        yaml_rule=carder.YAML_RULE, plain_rule="p", readings_rule="r", earlier="")
    assert not (drafts / "idea-0001.yaml").exists()  # accepted: moved to cards/


def test_the_first_attempt_and_a_missing_draft_still_write_the_draft_in_full(
        world, monkeypatch) -> None:
    drafts = world.root / "cards/drafts"
    drafts.mkdir(parents=True)
    (drafts / "idea-0001.yaml").write_text("stale: draft\n", encoding="utf-8")
    calls = _agent(monkeypatch, _good(plain_summary="Слишком коротко."))
    out = _run(world)
    # a stale draft before the first attempt is removed, not edited
    assert calls[0]["agent"] == "carder" and not calls[0]["draft_existed"]
    assert out["mode"] == "write"
    assert calls[0]["args"][0:2] == ("--allowedTools", "Read,Glob,Grep,Write")
    assert calls[0]["max_turns"] == 40
    # second attempt, but the draft is gone: a full write again
    (drafts / "idea-0001.yaml").unlink()
    calls = _agent(monkeypatch, _good())
    out = _run(world)
    assert world.entry["attempts"] == 2
    assert [c["agent"] for c in calls] == ["carder", "carder-review"] and out["mode"] == "write"
    assert out["status"] == "carded"


def test_an_edit_that_breaks_the_yaml_is_a_refused_attempt_without_a_rewrite(
        world, monkeypatch) -> None:
    _agent(monkeypatch, _good(plain_summary="Слишком коротко."))
    _run(world)
    calls = _agent(monkeypatch, _good(), edit="title: [unclosed\nverdict: card: broken\n")
    out = _run(world)
    assert [c["agent"] for c in calls] == ["carder-edit"]  # no fallback to a full write
    assert out["status"] == "new" and any("is not YAML" in p for p in out["problems"])
    # an edit that loses required fields is refused the same way, by the code checks
    world.entry["attempts"] = 1
    lost = _good()
    del lost["carder"]["new_block"]
    calls = _agent(monkeypatch, _good(), edit=lost)
    out = _run(world)
    assert [c["agent"] for c in calls] == ["carder-edit"]
    assert any("new_block is missing" in p for p in out["problems"])


def test_a_reviewer_block_refuses_the_card_and_keeps_the_draft_for_the_next_round(
        world, monkeypatch) -> None:
    calls = _agent(monkeypatch, _good(),
                   review="BLOCK: repeats-tried: new: carder.vs_tried: bv2-staged24 is the same "
                          "rule\nNOTE: the title is clumsy\nVERDICT: BLOCK")
    out = _run(world)
    assert out["status"] == "new" and out["review"] == [
        "carder.vs_tried: bv2-staged24 is the same rule"]
    assert out["problems"] == ["reviewer: repeats-tried: carder.vs_tried: bv2-staged24 is the "
                               "same rule"]
    block = {"kind": "repeats-tried", "ref": "new",
             "text": "carder.vs_tried: bv2-staged24 is the same rule"}
    assert world.entry["review_blocks"] == [block] and out["review_blocks"] == [block]
    assert world.entry["review_notes"] == ["the title is clumsy"] == out["notes"]
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]
    assert carder.load_queue(world.queue) == [] and not (
        world.root / "cards/bv3-volume-hedge.yaml").exists()
    drafts = world.root / "cards/drafts"
    assert (drafts / "idea-0001.prev.yaml").read_text(encoding="utf-8") == (
        drafts / "idea-0001.yaml").read_text(encoding="utf-8")
    _agent(monkeypatch, _good(), review="no verdict at all")
    assert "the reviewer gave no verdict" in _run(world)["problems"][-1]


def test_a_reply_with_notes_only_accepts_the_card_and_keeps_the_notes(world, monkeypatch) -> None:
    _agent(monkeypatch, _good(),
           review="NOTE: the word «залог» is not explained at first use\n"
                  "NOTE: the example could carry a second number\nVERDICT: OK")
    out = _run(world)
    assert out["status"] == "carded" and out["problems"] == []
    assert out["notes"] == ["the word «залог» is not explained at first use",
                            "the example could carry a second number"]
    assert world.entry["review_notes"] == out["notes"] and world.entry["review_blocks"] == []
    text = "\n".join(carder.report_lines({"entries": [{**out, "tokens": 5}]}, world.inbox, []))
    assert "замечания по словам (не блокируют): the word «залог»" in text
    assert "a second number" in text


NOTE_REVIEW = ("NOTE: the title is clumsy\nNOTE: the word «залог» is not explained at first use\n"
               "VERDICT: OK")
NEW_TITLE = "Страховка включается по обороту торгов, а не по цене"


def test_a_notes_only_accept_runs_one_edit_pass_and_accepts_the_edited_draft(
        world, monkeypatch) -> None:
    calls = _agent(monkeypatch, _good(), NOTE_REVIEW, notes=_retitle(NEW_TITLE))
    out = _run(world)
    assert [c["agent"] for c in calls] == ["carder", "carder-review", "carder-notes"]
    notes = calls[2]
    assert notes["draft_existed"] and out["notes_pass"] == "fixed" and out["status"] == "carded"
    assert notes["args"][0:2] == ("--allowedTools", "Read,Glob,Grep,Edit")
    assert notes["args"][2:4] == ("--disallowedTools", "Bash,Write")
    assert notes["max_turns"] == 15
    # only the notes are in the prompt, as the task; nothing blocking
    assert "- the title is clumsy" in notes["prompt"] and "- the word «залог»" in notes["prompt"]
    assert "- none\n\nWording notes" in notes["prompt"] and "do not touch it" in notes["prompt"]
    # the edited draft became the card; no second reviewer round
    card = yaml.safe_load((world.root / "cards/bv3-volume-hedge.yaml").read_text(encoding="utf-8"))
    assert card["title"] == NEW_TITLE and sum(c["agent"] == "carder-review" for c in calls) == 1
    assert out["notes"] == ["the title is clumsy", "the word «залог» is not explained at first use"]
    assert world.entry["attempts"] == 1 and "notes_tokens" in out
    text = "\n".join(carder.report_lines({"entries": [{**out, "tokens": 5}]}, world.inbox, []))
    assert "замечания по словам поправлены отдельной короткой правкой" in text


def test_a_notes_pass_that_breaks_a_code_check_is_reverted_and_the_original_is_accepted(
        world, monkeypatch) -> None:
    broken = _good(title=NEW_TITLE, plain_summary="Слишком коротко.")
    _agent(monkeypatch, _good(), NOTE_REVIEW, notes=broken)
    out = _run(world)
    assert out["status"] == "carded" and out["problems"] == []
    assert out["notes_pass"].startswith("reverted: ") and "too short" in out["notes_pass"]
    card = yaml.safe_load((world.root / "cards/bv3-volume-hedge.yaml").read_text(encoding="utf-8"))
    assert card["title"] == _good()["title"] and card["plain_summary"] == SUMMARY
    assert world.entry["attempts"] == 1
    text = "\n".join(carder.report_lines({"entries": [{**out, "tokens": 5}]}, world.inbox, []))
    assert "правка по замечаниям не принята и отменена" in text
    # a pass that leaves broken YAML, or changes the meaning (a grid value), is put back too
    for bad in ("title: [unclosed\n",
                _carder(_good(), grid=[{"param": "volume_window_days", "values": [3, 30],
                                        "meaning_ru": "За сколько дней сравнивать оборот.",
                                        "source_ru": "Круглые значения."},
                                       _good()["carder"]["grid"][1]])):
        world.entry["status"] = "new"
        world.entry["attempts"] = 0
        (world.root / "cards/bv3-volume-hedge.yaml").unlink()
        world.queue.write_text("# queue\ncards: []\n", encoding="utf-8")
        _agent(monkeypatch, _good(), NOTE_REVIEW, notes=bad)
        again = _run(world)
        assert again["status"] == "carded" and again["notes_pass"].startswith("reverted: ")
        assert yaml.safe_load((world.root / "cards/bv3-volume-hedge.yaml").read_text(
            encoding="utf-8"))["carder"]["grid"][0]["values"] == [3, 7]


def test_fix_notes_false_and_no_notes_skip_the_pass(world, monkeypatch) -> None:
    calls = _agent(monkeypatch, _good(), NOTE_REVIEW, notes=_retitle(NEW_TITLE))
    out = _run(world, {**CONFIG, "fix_notes": False})
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]
    assert out["status"] == "carded" and out["notes_pass"] == "skipped"
    assert out["notes"]  # the notes are still kept and shown
    world.entry["status"] = "new"
    (world.root / "cards/bv3-volume-hedge.yaml").unlink()
    world.queue.write_text("# queue\ncards: []\n", encoding="utf-8")
    calls = _agent(monkeypatch, _good(), "VERDICT: OK", notes=_retitle(NEW_TITLE))
    again = _run(world)
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]
    assert again["status"] == "carded" and again["notes_pass"] == "skipped"
    text = "\n".join(carder.report_lines({"entries": [{**out, "tokens": 5}]}, world.inbox, []))
    assert "поправлены" not in text and "не принята" not in text


def test_a_notes_pass_writing_outside_the_draft_or_failing_is_reverted(
        world, monkeypatch) -> None:
    def sneaky(target: Path) -> None:
        _retitle(NEW_TITLE)(target)
        world.queue.write_text("cards: [cards/sneaky.yaml]\n", encoding="utf-8")

    _agent(monkeypatch, _good(), NOTE_REVIEW, notes=sneaky)
    out = _run(world)
    assert out["status"] == "carded" and "outside its draft" in out["notes_pass"]
    assert carder.load_queue(world.queue) == ["cards/bv3-volume-hedge.yaml"]
    card = yaml.safe_load((world.root / "cards/bv3-volume-hedge.yaml").read_text(encoding="utf-8"))
    assert card["title"] == _good()["title"]  # the retitle done in the same call is undone too
    # a file elsewhere in the repository, seen by git status, is reverted and counts the same
    world.entry["status"] = "new"
    world.entry["attempts"] = 0
    (world.root / "cards/bv3-volume-hedge.yaml").unlink()
    reverted, seen = [], iter([[], [], [], ["src/qlab/night.py"]])
    monkeypatch.setattr(carder, "_changed_paths", lambda: next(seen))
    monkeypatch.setattr(carder, "_revert", reverted.append)
    _agent(monkeypatch, _good(), NOTE_REVIEW, notes=_retitle(NEW_TITLE))
    again = _run(world)
    assert reverted == ["src/qlab/night.py"] and "outside its draft" in again["notes_pass"]
    assert again["status"] == "carded"
    # the call itself fails (a budget stop, a crash): the draft stands, the idea is not failed
    world.entry["status"] = "new"
    world.entry["attempts"] = 0
    (world.root / "cards/bv3-volume-hedge.yaml").unlink()
    monkeypatch.setattr(carder, "_changed_paths", lambda: [])
    calls = _agent(monkeypatch, _good(), NOTE_REVIEW)
    inner = carder.run_agent

    def failing(budget, prompt, **kwargs):
        if kwargs["agent"] == "carder-notes":
            raise RuntimeError("the call failed")
        return inner(budget, prompt, **kwargs)

    monkeypatch.setattr(carder, "run_agent", failing)
    crashed = _run(world)
    assert crashed["status"] == "carded" and crashed["notes_pass"].startswith(
        "reverted: the call failed: RuntimeError")
    assert [c["agent"] for c in calls] == ["carder", "carder-review"]


def test_recheck_calls_no_writer_and_runs_no_notes_pass(world, monkeypatch) -> None:
    drafts = world.root / "cards/drafts"
    drafts.mkdir(parents=True)
    (drafts / "idea-0001.yaml").write_text(yaml.safe_dump(_good(), allow_unicode=True),
                                           encoding="utf-8")
    calls = _agent(monkeypatch, None, NOTE_REVIEW, notes=_retitle(NEW_TITLE))
    out = carder.recheck_idea(world.entry, CONFIG, budget=ExplicitNoBudget("test"),
                              session=world.session, day=DAY, root=world.root)
    assert [c["agent"] for c in calls] == ["carder-review"]
    assert out["status"] == "carded" and "notes_pass" not in out and out["recheck"]


def test_a_block_of_an_unknown_kind_is_a_note_and_says_so(world, monkeypatch) -> None:
    review = carder.parse_review("BLOCK: weird-kind: new: the colour of the title\nVERDICT: BLOCK")
    assert review.blocks == [] and len(review.notes) == 1
    assert "unknown kind 'weird-kind'" in review.notes[0] and "the colour" in review.notes[0]
    assert review.ok  # the lines decide: a note blocks nothing
    _agent(monkeypatch, _good(), review="BLOCK: weird-kind: new: wording\nVERDICT: BLOCK")
    out = _run(world)
    assert out["status"] == "carded" and "unknown kind" in out["notes"][0]
    # a BLOCK verdict that names nothing at all still blocks
    assert not carder.parse_review("VERDICT: BLOCK").ok
    assert not carder.parse_review("BLOCK: narrowed: new: x").ok  # no verdict line


def test_from_round_two_a_block_must_point_at_a_previous_remark_or_the_changed_lines() -> None:
    previous = [{"kind": "narrowed", "ref": "new", "text": "one"},
                {"kind": "rule-incomplete", "ref": "new", "text": "two"}]
    reply = "\n".join([
        "BLOCK: narrowed: prev:1: still not fixed",
        "BLOCK: unfair-grid: changed: the new axis is picked after a result",
        "BLOCK: hindsight-data: new: found in a part that did not change",
        "BLOCK: narrowed: prev:7: there is no such remark",
        "BLOCK: bad-question: no ref at all",
        "VERDICT: BLOCK"])
    review = carder.parse_review(reply, previous)
    assert review.blocks == [
        {"kind": "narrowed", "ref": "prev:1", "text": "still not fixed"},
        {"kind": "unfair-grid", "ref": "changed", "text": "the new axis is picked after a result"}]
    assert len(review.notes) == 3 and all("not blocking" in n for n in review.notes)
    assert "ref was 'new'" in review.notes[0] and "ref was 'prev:7'" in review.notes[1]
    assert "ref was 'missing'" in review.notes[2]
    # the first round marks every block `new`, ref or none; a previous remark not repeated is fixed
    first = carder.parse_review("BLOCK: narrowed: new: x\nBLOCK: unfair-grid: no ref\n"
                                "VERDICT: BLOCK")
    assert first.blocks == [{"kind": "narrowed", "ref": "new", "text": "x"},
                            {"kind": "unfair-grid", "ref": "new", "text": "no ref"}]
    assert carder.parse_review("NOTE: wording\nVERDICT: OK", previous).ok


def test_the_second_round_prompt_carries_the_numbered_blocks_and_the_diff(
        world, monkeypatch) -> None:
    first = _agent(monkeypatch, _good(),
                   review="BLOCK: narrowed: new: the second reading is dropped\n"
                          "BLOCK: rule-incomplete: new: the exit is not said\nVERDICT: BLOCK")
    _run(world)
    assert "write `new` as <ref>" in first[1]["prompt"] and "unified diff" not in first[1]["prompt"]
    drafts = world.root / "cards/drafts"
    assert (drafts / "idea-0001.prev.yaml").is_file()
    second = _agent(monkeypatch, _good(title="Страховка включается по обороту, а не по цене"),
                    review="NOTE: a nicer title\nVERDICT: OK")
    out = _run(world)
    prompt = second[1]["prompt"]
    assert "1. [narrowed] the second reading is dropped" in prompt
    assert "2. [rule-incomplete] the exit is not said" in prompt
    assert "--- previous draft" in prompt and "+++ this draft" in prompt
    assert "-title: Страховка включается, когда оборот растёт, а цена стоит" in prompt
    assert "+title: Страховка включается по обороту, а не по цене" in prompt
    assert "prev:<n>" in prompt and "changed" in prompt
    # accepted: the card stands, the working copies are gone, the card directory holds one card
    assert out["status"] == "carded" and out["notes"] == ["a nicer title"]
    assert not (drafts / "idea-0001.prev.yaml").exists()
    assert not (drafts / "idea-0001.yaml").exists()
    assert sorted(p.name for p in (world.root / "cards").glob("*.yaml")) == [
        "bv3-volume-hedge.yaml"]
    assert world.entry["review_blocks"] == []


def test_a_code_refusal_between_review_rounds_keeps_the_open_remarks(world, monkeypatch) -> None:
    _agent(monkeypatch, _good(), review="BLOCK: narrowed: new: a reading is dropped\n"
                                        "VERDICT: BLOCK")
    _run(world)
    _agent(monkeypatch, _good(), edit=_good(plain_summary="Слишком коротко."))
    out = _run(world)
    assert any("too short" in p for p in out["problems"])
    assert "reviewer: narrowed: a reading is dropped" in out["problems"]
    assert world.entry["review_blocks"][0]["kind"] == "narrowed"
    # the reviewer's next look is still held to the old remark and the old draft
    calls = _agent(monkeypatch, None, edit=_good(title="Страховка по обороту торгов, а не по цене"))
    _run(world)
    assert [c["agent"] for c in calls] == ["carder-edit", "carder-review"]
    assert "1. [narrowed] a reading is dropped" in calls[1]["prompt"]


def test_an_entry_with_plain_string_problems_from_before_still_loads(world, monkeypatch) -> None:
    world.entry["problems"] = ["reviewer: `grid`. The idea asks for two readings and one is gone.",
                               "something else, not from the reviewer"]
    blocks = carder.previous_blocks(world.entry)
    assert blocks == [{"kind": "unknown", "ref": "new",
                       "text": "`grid`. The idea asks for two readings and one is gone."}]
    calls = _agent(monkeypatch, _good(), review="NOTE: wording\nVERDICT: OK")
    out = _run(world)
    # no copy of the previous draft was kept: the whole draft counts as changed
    assert "1. [unknown] `grid`. The idea asks for two readings" in calls[1]["prompt"]
    assert "no copy of the previous draft was kept" in calls[1]["prompt"]
    assert out["status"] == "carded"
    # such a remark still blocks, and in the second round only as prev:<n>
    again = carder.parse_review("BLOCK: narrowed: prev:1: still gone\nVERDICT: BLOCK", blocks)
    assert not again.ok and again.blocks[0]["ref"] == "prev:1"


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
    assert config["enabled"] is True and config["max_attempts"] == 3
    assert config["per_night"] == 3 and config["max_waiting"] == 3  # owner, 2026-10-28
    assert config["model"] == "sonnet" and config["edit_max_turns"] == 15
    assert config["fix_notes"] is True
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


def test_the_readings_of_an_idea_are_two_or_more_plain_sentences() -> None:
    draft = _good()
    readings = ["Младший таймфрейм только поднимает страховку.",
                "Младший таймфрейм и поднимает страховку, и решает, когда её ставить."]
    assert _check(_carder(draft, readings_ru=readings)) == []
    assert _check(draft) == []  # optional
    assert any("readings_ru must be a list of 2 or more" in p
               for p in _check(_carder(draft, readings_ru=readings[:1])))
    assert any("readings_ru must be a list" in p for p in _check(_carder(draft, readings_ru="x")))
    assert any("readings_ru 2" in p for p in _check(_carder(
        draft, readings_ru=[readings[0], "The lower timeframe sets the hedge."])))
    assert any("«хедж»" in p for p in _check(_carder(
        draft, readings_ru=[readings[0], "Младший таймфрейм ставит хедж."])))


def test_the_readings_reach_the_card_and_the_implementer(tmp_path) -> None:
    from qlab.agents.implementer import card_brief
    from qlab.card_fields import read_card_fields

    readings = ["Младший таймфрейм только поднимает страховку.",
                "Младший таймфрейм и поднимает страховку, и решает, когда её ставить."]
    entry = {"id": "idea-0001", "source": "owner", "date": "2026-10-27", "text": IDEA}
    path = tmp_path / "cards" / "bv3-volume-hedge.yaml"
    path.parent.mkdir()
    text = carder.render_card(_carder(_good(), readings_ru=readings), entry, day=DAY)
    path.write_text(text, encoding="utf-8")
    card = yaml.safe_load(text)
    assert card["readings"] == readings and card["carder"]["readings_ru"] == readings
    assert card_problems(card, Path("cards/bv3-volume-hedge.yaml")) == []
    fields = read_card_fields(path, root=Path.cwd())
    assert list(fields.readings) == readings
    brief = card_brief(fields, Path("src/qlab/strategies/agent/x.py"), block_exists=False)
    assert "READINGS." in brief and readings[1] in brief
    # a card without readings reads as before
    plain = tmp_path / "cards" / "plain.yaml"
    plain.write_text(carder.render_card(_good(), entry, day=DAY), encoding="utf-8")
    assert read_card_fields(plain, root=Path.cwd()).readings == ()
