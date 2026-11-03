"""The plain-Russian top of an idea's page (docs/PLAIN_PAGE.md): the texts, the dictionary, the
resolution chain, the difference, the three ways numbers are chosen, the API field.

The fixtures are built here: a registry in memory, spec files of this repository (`specs/`) and
the texts of `strategies/plain/`. Nothing reads the real registry (`data/`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import plain
from qlab.agents.cataloguer import _russian_problems
from qlab.agents.generator import catalogue
from qlab.api import cards as cards_module
from qlab.api.app import create_app
from qlab.api.deps import get_session
from qlab.pipeline.sources import QLAB_ROOT
from qlab.pipeline.spec import load_spec
from qlab.plain import choice as choice_mod
from qlab.plain.compose import FACTS, fact_core, family_sentences
from qlab.plain.difference import difference
from qlab.plain.page import NO_TEXT, TWIN_SUFFIX, clip, plain_page
from qlab.plain.specs import SpecView, candidate_values, core_items, spec_files, view_from_raw
from qlab.plain.texts import (
    TITLE_MAX,
    dictionary_problems,
    family_problems,
    load_texts,
    placeholders,
)
from qlab.plain.words import MISSING, Words
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
)

TEXTS = load_texts(QLAB_ROOT)
WORDS = Words(TEXTS.dictionary)
OPEN = (IdeaStatus.CANDIDATE, IdeaStatus.BENCH, IdeaStatus.VALIDATED, IdeaStatus.PAPER,
        IdeaStatus.LIVE)


# --------------------------------------------------------------------------
# A registry in memory


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def add_idea(session: Session, idea_id: str, *, parent: str | None = None,
             status: IdeaStatus = IdeaStatus.CANDIDATE, title: str | None = None,
             notes: str | None = None) -> Idea:
    idea = Idea(id=idea_id, title=title or f"technical title of {idea_id}",
                source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                profile=Profile.CARRY, status=status, parent_id=parent, notes=notes,
                created_at=_now(), updated_at=_now())
    session.add(idea)
    session.flush()
    return idea


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def page_of(session: Session, idea_id: str, cards: dict | None = None) -> dict:
    return plain_page(session, session.get(Idea, idea_id), root=QLAB_ROOT, cards=cards or {})


def bv2_chain(session: Session) -> None:
    """The chain of the owner's example: the strategy, an experiment of it, the causal twin."""
    add_idea(session, "strategy-b-v2", status=IdeaStatus.PAPER)
    add_idea(session, "bv2-research-shared-pool", parent="strategy-b-v2")
    add_idea(session, "bv2-research-shared-pool-hl", parent="bv2-research-shared-pool")
    add_idea(session, "bv2-research-shared-pool-hl-causal", parent="bv2-research-shared-pool-hl",
             status=IdeaStatus.VALIDATED)


# --------------------------------------------------------------------------
# The texts


def test_every_family_and_mechanism_text_is_plain_short_and_complete():
    facts = frozenset(FACTS)
    problems = []
    for family in (*TEXTS.families, *TEXTS.mechanisms.values()):
        problems += family_problems(family, known_facts=facts)
    assert problems == []


def test_a_text_names_the_files_it_was_checked_against_and_they_exist():
    """«cite the file each text was checked against»: every path the text cites that lives in this
    repository exists (a path in the sibling repository `funding-rate-arbitrage` is not looked for:
    it is not a part of this one)."""
    missing = []
    for family in (*TEXTS.families, *TEXTS.mechanisms.values()):
        for cited in family.checked_against:
            for token in cited.replace("(", " ").replace(",", " ").split():
                if token.startswith(("src/", "docs/", "cards/", "specs/", "night/")) \
                        and "." in token and not (QLAB_ROOT / token.rstrip(".:;")).exists():
                    missing.append((family.id, token))
    assert missing == []


def test_every_mechanism_of_the_catalogue_has_a_plain_text():
    ids = {m.id for m in catalogue(QLAB_ROOT / "docs/MECHANISMS.md")}
    assert ids - set(TEXTS.mechanisms) == set(), "write strategies/plain/mechanisms/<id>.yaml"
    assert set(TEXTS.mechanisms) - ids == set(), "a plain text of a mechanism the catalogue lacks"


def test_a_family_names_a_reference_spec_that_exists_for_each_code_it_covers():
    for family in TEXTS.families:
        for path in family.base_specs.values():
            assert (QLAB_ROOT / path).is_file(), (family.id, path)
        for name in family.base_specs:
            assert name in family.code_names, (family.id, name)


def test_the_dictionary_is_plain_and_the_twin_suffix_is_the_codes():
    from qlab.twin import TWIN_SUFFIX as CODE_SUFFIX

    assert dictionary_problems(TEXTS.dictionary) == []
    assert TWIN_SUFFIX == CODE_SUFFIX
    jargon = [p for group in ("universe", "signal", "portfolio", "sizing")
              for name, phrase in TEXTS.dictionary.blocks[group].items()
              if (prose := re.sub(r"\{[a-z_]+\}", "", phrase).strip())
              for p in _russian_problems(f"blocks.{group}.{name}", prose)]
    assert jargon == []


def spec_documents() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted((QLAB_ROOT / "specs").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "idea_id" in raw and "code_ref" in raw:
            out.append((path, raw))
    return out


def test_every_parameter_of_the_specs_has_a_phrase_in_the_dictionary():
    """The test that LISTS what lacks a phrase (owner, 2026-11-03: a parameter missing from the
    dictionary still shows as «параметр X: было A, стало B», and this test names it): add the
    parameter to strategies/plain/params.yaml in the same day."""
    lacking: dict[str, set[str]] = {}
    for path, raw in spec_documents():
        view = view_from_raw(raw, text=path.read_text(encoding="utf-8"), path=str(path))
        keys = set(core_items(view.unwrapped)) | set(candidate_values(view))
        for layer in view.unwrapped.layers:
            if layer.kind in ("RegimeGate", "SplitCadence"):
                keys |= {k for k in layer.own if k in ("trade_in", "detector", "entry_every",
                                                        "exit_every")}
            if layer.kind in ("Overlaid", "Retune"):
                chains = [layer.own.get("overlays")]
                for candidates in (layer.own.get("grid") or {}).values():
                    chains += [(o or {}).get("overlays") for o in candidates.values()]
                for chain in chains:
                    for item in chain or ():
                        keys |= {f"overlay.{item['name']}.{k}" for k in item.get("params", {})}
        keys |= set((raw.get("choice") or {}).get("axes") or ())
        for key in WORDS.unknown_keys(keys):
            lacking.setdefault(key, set()).add(raw["idea_id"])
    assert lacking == {}, "parameters without a phrase (strategies/plain/params.yaml): " + \
        "; ".join(f"{k} (e.g. {sorted(v)[0]})" for k, v in sorted(lacking.items()))


def test_every_overlay_and_block_of_the_specs_has_a_phrase():
    blocks = TEXTS.dictionary.blocks
    overlays = TEXTS.dictionary.layers["overlays"]
    missing = set()
    for path, raw in spec_documents():
        view = view_from_raw(raw, path=str(path))
        core = view.unwrapped.core
        if view.unwrapped.core_name == "Composed":
            for kind, key in (("universe", "rule"), ("signal", "block"), ("portfolio", "block"),
                              ("sizing", "block")):
                name = (core.get(kind) or {}).get(key)
                if name and name not in blocks[kind]:
                    missing.add((kind, name))
        for layer in view.unwrapped.layers:
            for item in layer.own.get("overlays") or ():
                if item["name"] not in overlays:
                    missing.add(("overlay", item["name"]))
    assert missing == set()


def test_a_parameter_the_dictionary_does_not_know_still_shows_with_its_values():
    assert WORDS.change("brand_new_knob", 2, 3) == "параметр brand_new_knob: было 2, стало 3"
    assert WORDS.change("brand_new_knob", MISSING, 3).startswith(
        "параметр brand_new_knob: теперь 3")
    assert WORDS.unknown_keys(["brand_new_knob", "shared_margin"]) == ["brand_new_knob"]


# --------------------------------------------------------------------------
# The difference


def view(code_ref: str, params: dict, *, data: dict | None = None, costs: dict | None = None,
         extras: dict | None = None) -> SpecView:
    return SpecView(idea_id="x", title="", code_ref=code_ref, params=params,
                    data=data or {"source": "hyperliquid", "interval": "1h"},
                    costs=costs or {"taker_fee_bps": 5.25, "slippage_bps": 0.9},
                    extras=extras or {})


BV2 = "qlab.strategies.live.bv2:LiveBv2"


def test_a_yes_no_parameter_reads_as_the_dictionarys_clause():
    base = view(BV2, {"coins": ["BTC", "ETH"], "hedge_margin_headroom": 1.0})
    mine = view(BV2, {"coins": ["BTC", "ETH"], "hedge_margin_headroom": 1.0,
                      "shared_margin": True})
    texts = [i.text for i in difference(mine, base, WORDS)]
    assert texts == ["залог один общий на все монеты, а не у каждой свой: страховку одной монеты "
                     "можно открыть целиком за счёт свободного залога других"]


def test_a_number_reads_as_was_and_became_with_the_parameters_meaning():
    base = view(BV2, {"hedge_margin_headroom": 1.0})
    mine = view(BV2, {"hedge_margin_headroom": 1.5})
    (item,) = difference(mine, base, WORDS)
    assert item.text.startswith("запас залога под страховку")
    assert item.text.endswith("было 1, стало 1.5")
    assert item.tag == "запас залога 1.5"


def test_a_parameter_added_to_the_code_later_equals_its_default_when_absent():
    base = view(BV2, {"coins": ["BTC"]})
    explicit = view(BV2, {"coins": ["BTC"], "shared_margin": False, "short_window_days": 14})
    assert difference(explicit, base, WORDS) == []


def test_an_unknown_parameter_is_shown_not_hidden():
    base = view(BV2, {"coins": ["BTC"], "knob": 1})
    mine = view(BV2, {"coins": ["BTC"], "knob": 2})
    (item,) = difference(mine, base, WORDS)
    assert item.text == "параметр knob: было 1, стало 2"


def test_coins_and_data_and_costs_are_told():
    base = view(BV2, {"coins": ["BTC", "ETH"]})
    mine = view(BV2, {"coins": ["BTC", "ETH", "SOL"]},
                data={"source": "binance", "interval": "1h"},
                costs={"taker_fee_bps": 7.5, "slippage_bps": 0.9})
    texts = [i.text for i in difference(mine, base, WORDS)]
    assert any("добавлено: SOL" in t for t in texts)
    assert "данные: Binance вместо Hyperliquid" in texts
    assert any(t.startswith("издержки: комиссия за сделку 0.075%") for t in texts)


def test_a_variant_that_is_one_set_of_numbers_against_the_base_is_described_by_it():
    """A twin's grid holds the experiment as a candidate: that candidate IS the difference."""
    grid = {"variant": {"base": {}, "experiment": {"shared_margin": True}}}
    retune = {"inner_code_ref": BV2, "base_params": {"coins": ["BTC"]}, "grid": grid,
              "refit_every": "M", "window": "expanding", "objective": "sharpe_net"}
    mine = view("qlab.strategies.retune:Retune", retune)
    base = view(BV2, {"coins": ["BTC"]})
    (item,) = difference(mine, base, WORDS)
    assert item.text.startswith("залог один общий на все монеты") and item.tag == "общий залог"


def test_overlays_and_a_market_gate_are_told():
    overlay = {"overlays": [{"name": "gradual_entry", "params": {"tranches": 2}}]}
    inner = {"inner_code_ref": BV2, "inner_params": {"coins": ["BTC"]}, **overlay}
    mine = view("qlab.strategies.overlays:Overlaid", inner)
    gated = view("qlab.strategies.regime_gate:RegimeGate",
                 {"inner_code_ref": BV2, "inner_params": {"coins": ["BTC"]},
                  "trade_in": ["bull", "bear"]})
    base = view(BV2, {"coins": ["BTC"]})
    assert "набирает позицию частями" in difference(mine, base, WORDS)[0].text
    assert "рост, падение" in difference(gated, base, WORDS)[0].text


def test_an_open_list_of_numbers_is_told_as_a_choice_of_code():
    base = view(BV2, {"short_window_days": 14})
    mine = view(BV2, {"short_window_days": [3, 7, 14, 30]})
    (item,) = difference(mine, base, WORDS)
    assert "код выбирает из значений 3 сут., 7 сут., 14 сут., 30 сут." in item.text
    assert "у основной версии — 14 сут." in item.text


# --------------------------------------------------------------------------
# How the numbers were chosen: the three cases


def test_numbers_chosen_by_code_name_the_candidates_in_plain_words():
    grid = {"variant": {"base": {}, "experiment": {"shared_margin": True}}}
    retune = {"inner_code_ref": BV2, "base_params": {"coins": ["BTC"]}, "grid": grid,
              "refit_every": "M", "window": "expanding", "objective": "sharpe_net"}
    got = choice_mod.describe(view("qlab.strategies.retune:Retune", retune), "как на проде", WORDS)
    assert got.kind == "by_code"
    assert "каждый месяц код сам выбирает" in got.text and "ничего не видит вперёд" in got.text
    assert got.lines[0] == "вариант «как на проде»: залог у каждой монеты свой"
    assert got.lines[1].startswith("вариант «общий залог»: залог один общий на все монеты")


def test_numbers_fixed_by_hand_say_the_date_and_the_evidence():
    spec = view(BV2, {"coins": ["BTC"]}, extras={
        "params_fixed_at": "2025-06-01",
        "params_fixed_evidence": "REPORT.md fitted the numbers on 2023-06 ... 2025-05. More text."})
    got = choice_mod.describe(spec, "как на проде", WORDS)
    assert got.kind == "fixed" and "2025-06-01" in got.text
    assert "по-английски" in got.lines[0] and "REPORT.md fitted the numbers" in got.lines[0]
    assert "More text" not in got.lines[0]


def test_nothing_to_choose_gives_its_reason():
    spec = view(BV2, {"coins": ["BTC"]}, extras={
        "choice": {"nothing_to_choose": "the catalogue gives one value for each number."},
        "params_fixed_at": "2024-06-14",
        "params_fixed_evidence": "named by code: the start of the last third of the window"})
    got = choice_mod.describe(spec, "как на проде", WORDS)
    assert got.kind == "nothing_to_choose"
    assert "Выбирать не из чего" in got.text and "the catalogue gives one value" in got.text
    assert any("начало последней трети" in line for line in got.lines)


def test_a_one_check_choice_with_axes_lists_the_values():
    spec = view("qlab.strategies.blocks:Composed", {}, extras={
        "choice": {"axes": {"signal.lookback_days": [7, 14, 28]}, "source": "x"}})
    got = choice_mod.describe(spec, "как в основной версии", WORDS)
    assert got.kind == "by_code"
    assert list(got.lines) == ["за сколько суток смотрят изменение цены: 7 сут., 14 сут., 28 сут."]


# --------------------------------------------------------------------------
# The resolution chain and the page


def test_the_owners_example_idea_opens_with_what_it_does_and_how_it_differs(session):
    bv2_chain(session)
    page = page_of(session, "bv2-research-shared-pool-hl-causal")
    text = " ".join(page["what"]["paragraphs"])
    assert page["what"]["source"] == "family" and page["what"]["family"] == "bv2"
    for fact in ("BTC, ETH, SOL, AVAX", "14 суток", "30 суток", "12 часов подряд", "50%",
                 "$257", "в короткую", "бессрочный контракт"):
        assert fact in text, fact
    assert [i for i in page["difference"]["items"] if "залог один общий" in i["text"]]
    assert page["choice"]["kind"] == "by_code"
    assert "вариант «как на проде»" in " ".join(page["choice"]["lines"])
    assert page["venue"]["venue"] == "binance"
    assert "2019-09-08" in page["venue"]["lines"][0]
    assert len(page["title"]) <= TITLE_MAX and page["title"].startswith("Bv2:")
    assert page["technical_title"].startswith("technical title")


def test_the_base_strategy_is_told_to_be_the_base(session):
    add_idea(session, "strategy-b-v2", status=IdeaStatus.PAPER)
    page = page_of(session, "strategy-b-v2")
    assert page["difference"]["is_base"] and page["difference"]["items"] == []


def test_the_chain_a_spec_text_then_a_card_then_the_family_then_an_ancestors_card(session):
    add_idea(session, "strategy-b-v2", status=IdeaStatus.PAPER)
    add_idea(session, "bv2-binance", parent="strategy-b-v2")
    card = {"plain_summary": "Текст карточки.", "title": "Заголовок карточки"}
    # own card first
    own = page_of(session, "bv2-binance", {"bv2-binance": card})
    assert (own["what"]["source"], own["title"]) == ("card", "Заголовок карточки")
    # else the family of the root
    assert page_of(session, "bv2-binance")["what"]["source"] == "family"
    # a stranger code, no family: the card of an ancestor
    add_idea(session, "frab-lot-leftover")
    add_idea(session, "frab-lot-leftover-hl", parent="frab-lot-leftover")
    anc = page_of(session, "frab-lot-leftover-hl", {"frab-lot-leftover": card})
    assert anc["what"]["source"] == "ancestor_card" and "frab-lot-leftover" in anc["what"]["note"]


def test_a_generated_idea_is_read_from_its_mechanism_and_its_blocks(session):
    add_idea(session, "gen-funding-tail-rebound-memon", status=IdeaStatus.BENCH)
    add_idea(session, "gen-funding-tail-rebound-memon-hl", parent="gen-funding-tail-rebound-memon",
             notes="Second stage of gen-funding-tail-rebound-memon")
    for idea_id in ("gen-funding-tail-rebound-memon", "gen-funding-tail-rebound-memon-hl"):
        page = page_of(session, idea_id)
        assert page["what"]["source"] == "mechanism"
        assert page["what"]["family"] == "perp-funding-extreme-rebound"
        text = " ".join(page["what"]["paragraphs"])
        assert "180 суток" in text and "нижних 10%" in text and "раз в 1 суток" not in text
    second = page_of(session, "gen-funding-tail-rebound-memon-hl")
    assert second["title"].endswith("(проверка на Hyperliquid)")
    assert "Это вторая проверка той же идеи" in " ".join(second["venue"]["lines"])
    first = page_of(session, "gen-funding-tail-rebound-memon")
    assert first["choice"]["kind"] == "nothing_to_choose"


def test_an_idea_with_no_text_says_so_and_quotes_what_its_source_claimed(session):
    idea = add_idea(session, "old-rejected", status=IdeaStatus.REJECTED)
    idea.claimed_edge = "Шорт перпа на бирже с высоким funding"
    page = page_of(session, "old-rejected")
    assert page["what"]["source"] == "none" and page["what"]["paragraphs"][0] == NO_TEXT
    assert "Шорт перпа" in page["what"]["paragraphs"][1]
    assert page["title"] == "technical title of old-rejected"


def test_the_plain_title_is_at_most_eighty_characters():
    assert len(clip("слово " * 40)) <= TITLE_MAX and clip("слово " * 40).endswith("…")
    assert clip("короткий") == "короткий"


def test_every_spec_of_the_repository_gives_its_idea_a_plain_what_it_does(session):
    """«every open idea in a fixture set gets a non-empty «Что делает»»: the fixture set is every
    spec file of this repository, as an open idea each (the real registry is not read). Only a
    stage-2 copy and an idea with a card come with no text of their own; no idea falls to
    «простого описания нет»."""
    cards = {c["idea_id"]: c for c in cards_module.load_cards(QLAB_ROOT / "cards")}
    ids = {}
    for path, raw in spec_documents():
        ids.setdefault(raw["idea_id"], path)
    known = set(ids) | set(cards)  # an idea may have a card and no spec file in the repository
    for idea_id in sorted(known - set(ids)):
        add_idea(session, idea_id)
    for idea_id in ids:  # a stage-2 copy and a causal twin descend from the idea they copy
        parent = next((idea_id[: -len(s)] for s in ("-hl", TWIN_SUFFIX)
                       if idea_id.endswith(s) and idea_id[: -len(s)] in known), None)
        add_idea(session, idea_id, parent=parent)
    nothing, fillers, jargon = [], [], []
    for idea_id in ids:
        page = page_of(session, idea_id, cards)
        paragraphs = page["what"]["paragraphs"]
        if page["what"]["source"] == "none" or not any(p.strip() for p in paragraphs):
            nothing.append(idea_id)
        text = " ".join(paragraphs)
        if "{" in text or "None" in text:
            fillers.append(idea_id)
        if page["what"]["source"] in ("family", "mechanism"):
            jargon += [(idea_id, p) for p in _russian_problems("text", text)]
        assert len(page["title"]) <= TITLE_MAX, idea_id
    assert nothing == [] and fillers == [] and jargon == []


OPTIONAL_FACTS = frozenset({"book_vol_sentence"})
"""Facts a family text may lack for a spec without being wrong: Trend's book-volatility sentence
exists only for the strategy that has that layer."""


def test_a_family_text_keeps_every_sentence_for_its_base_spec():
    """A sentence that loses its fact is dropped; for the family's own base specs only a sentence
    about an optional layer may be."""
    for family in TEXTS.families:
        for name, base in family.base_specs.items():
            raw = yaml.safe_load((QLAB_ROOT / base).read_text(encoding="utf-8"))
            core = fact_core(view_from_raw(raw, path=base))
            kept = family_sentences(family, core)
            mandatory = [s for s in family.sentences
                         if not set(placeholders(s)) & OPTIONAL_FACTS]
            assert len(kept) >= len(mandatory), (family.id, name)
            assert all("{" not in s for s in kept)


# --------------------------------------------------------------------------
# What the researcher and the generator must give


GOOD = ("Стратегия покупает монеты, у которых ставка финансирования необычно низка, и держит их "
        "сутки. Ставка — это плата между участниками бессрочного контракта. Зарабатывает на "
        "отскоке цены после перекоса, рискует тем, что перекос не рассосётся. Работает на всём "
        "капитале книги.")


def test_a_variant_of_a_known_family_needs_no_summary():
    raw = yaml.safe_load((QLAB_ROOT / "specs/bv2-live.yaml").read_text(encoding="utf-8"))
    assert plain.spec_problems(raw, "") == []


def test_a_new_mechanism_needs_a_summary_that_passes_the_cards_jargon_check():
    raw = {"idea_id": "x", "code_ref": "qlab.strategies.agent.new_thing:NewThing", "params": {}}
    assert any("plain_summary is missing" in p for p in plain.spec_problems(raw, ""))
    own = {"idea_id": "gen-x", "code_ref": "qlab.strategies.blocks:Composed", "params": {}}
    assert any("plain_summary is missing" in p
               for p in plain.spec_problems(own, "# mechanism: own\nidea_id: gen-x\n"))
    assert plain.spec_problems({**own, "plain_summary": GOOD}, "# mechanism: own\n") == []
    assert any("хедж" in p or "шорт" in p for p in plain.spec_problems(
        {**raw, "plain_summary": GOOD.replace("покупает", "лонгует и шортит хеджируя")}, ""))
    assert any("at least" in p
               for p in plain.spec_problems({**raw, "plain_summary": "Коротко."}, ""))
    assert any("at most" in p for p in plain.spec_problems(
        {**raw, "plain_summary": GOOD, "plain_title": "слово " * 20}, ""))


def test_a_catalogue_mechanism_spec_needs_no_summary():
    raw = {"idea_id": "gen-x", "code_ref": "qlab.strategies.blocks:Composed", "params": {}}
    header = "# mechanism: perp-funding-extreme-rebound\nidea_id: gen-x\n"
    assert plain.spec_problems(raw, header) == []


def test_the_spec_format_accepts_the_new_fields():
    spec = load_spec(QLAB_ROOT / "specs/bv2-live.yaml")
    assert spec.plain_summary is None and spec.plain_title is None
    text = (QLAB_ROOT / "specs/bv2-live.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text) | {"plain_summary": GOOD, "plain_title": "Короткий заголовок"}
    from qlab.pipeline.spec import StrategySpec

    got = StrategySpec.model_validate(raw)
    assert got.plain_summary == GOOD and got.plain_title == "Короткий заголовок"


# --------------------------------------------------------------------------
# The API field


@pytest.fixture()
def client(session):
    engine = session.get_bind()
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    bv2_chain(session)
    session.commit()
    app = create_app()

    def _override():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as test_client:
        yield test_client


def test_the_api_carries_the_plain_field_for_a_variant_without_a_card(client):
    body = client.get("/api/ideas/bv2-research-shared-pool-hl-causal").json()
    plain_part = body["plain"]
    assert plain_part["title"].startswith("Bv2:")
    assert plain_part["what"]["paragraphs"] and plain_part["what"]["source"] == "family"
    assert plain_part["difference"]["items"] and plain_part["choice"]["kind"] == "by_code"
    assert plain_part["venue"]["lines"]
    assert body["how_it_works"] is None  # the old card block is untouched


def test_the_api_never_fails_the_page_over_the_plain_field(client, monkeypatch):
    import qlab.plain as plain_module

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(plain_module, "plain_page", broken)
    body = client.get("/api/ideas/strategy-b-v2").json()
    assert body["plain"] is None and body["id"] == "strategy-b-v2"


def test_spec_files_index_finds_the_ideas_by_their_id_line():
    index = spec_files(QLAB_ROOT)
    assert Path("specs/bv2-live.yaml") in index["strategy-b-v2"]
    assert len(index) > 100
