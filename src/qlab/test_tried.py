"""The prior-art registry (night/tried.yaml, qlab.tried): the real file is well formed and cites
files that exist; a malformed one is refused with every problem named; the entries nearest to an
idea are found by its words."""

from __future__ import annotations

import pytest
import yaml

from qlab import tried


def _write(tmp_path, entries, families=None):
    path = tmp_path / "tried.yaml"
    path.write_text(yaml.safe_dump({
        "version": "t", "families": families or {"f": "семейство для проверки"},
        "entries": entries}, allow_unicode=True), encoding="utf-8")
    return path


def _entry(**over):
    row = {"id": "one-entry", "family": "f", "what": "что пробовали одной строкой",
           "where": {"file": "docs/NIGHT.md", "section": "Этапы"},
           "result": "числа как в источнике"}
    return {**row, **over}


def test_the_real_registry_loads_and_cites_files_that_exist() -> None:
    registry = tried.load_tried()
    assert len(registry.entries) >= 45
    assert len(registry.ids()) == len(registry.entries)
    assert {"hedge-entry-signal", "hedge-staging", "parameter-refit", "regime-detection",
            "basket-coins"} <= set(registry.families)
    assert tried.missing_files(registry) == []
    # the seed names the variants frab's research tested, by the names its scripts use
    for wanted in ("bv2-flow-taker-72h", "bv2-staged24", "bv2-stop-5-10", "bv2-daily-signal",
                   "bv2-ens6-horizon-ensemble", "bv2-jump-regime-model", "bv2-macro-vix",
                   "bv2-dvol-implied-vol", "bv2-chandelier-exit", "bv2-volscale-vol-managed",
                   "bv2-refit-54-sets", "bv2-regime-switch-9", "bv2-sig-hyst"):
        assert wanted in registry.ids(), wanted
    assert registry.get("bv2-flow-taker-72h").where.startswith("funding-rate-arbitrage/")
    assert all("\n" not in e.what and "\n" not in e.result for e in registry.entries)


def test_a_well_formed_file_loads_and_unknown_ids_are_found(tmp_path) -> None:
    registry = tried.load_tried(_write(tmp_path, [_entry(), _entry(id="two-entry")]))
    assert registry.ids() == {"one-entry", "two-entry"}
    assert registry.unknown(["one-entry", "ghost"]) == ["ghost"]
    assert "one-entry [f]" in registry.get("one-entry").line


@pytest.mark.parametrize("bad, message", [
    ({"id": "Bad_Id"}, "lower-case-with-dashes"),
    ({"family": "nowhere"}, "not declared"),
    ({"what": "tried in English only"}, "not in Russian"),
    ({"result": ""}, "`result` is empty"),
    ({"what": "две\nстроки"}, "one line each"),
    ({"where": {"file": "docs/NIGHT.md"}}, "needs a `file` and a `section`"),
    ({"where": "docs/NIGHT.md"}, "`where: {file, section}`"),
])
def test_a_malformed_entry_is_refused_with_the_reason(tmp_path, bad, message) -> None:
    with pytest.raises(ValueError, match=message):
        tried.load_tried(_write(tmp_path, [_entry(**bad)]))


def test_a_repeated_id_and_an_english_family_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="the id repeats"):
        tried.load_tried(_write(tmp_path, [_entry(), _entry()]))
    with pytest.raises(ValueError, match="must be Russian"):
        tried.load_tried(_write(tmp_path, [_entry()], families={"f": "english only"}))


def test_a_file_that_is_not_there_is_reported(tmp_path) -> None:
    registry = tried.load_tried(_write(tmp_path, [_entry(where={
        "file": "docs/NO_SUCH_FILE.md", "section": "x"})]))
    assert tried.missing_files(registry) == ["one-entry: docs/NO_SUCH_FILE.md"]


def test_the_entries_nearest_to_an_idea_come_from_its_words() -> None:
    registry = tried.load_tried()
    idea = ("Bv3: включать страховку не по цене, а по объёму торгов: когда оборот растёт, а цена "
            "стоит, это сигнал для включения страховки")
    near = tried.nearest(registry, idea)
    assert 1 <= len(near) <= 3
    assert all("bv2" in e.id for e in near)
    weight = tried.rarity([{"a", "b"}, {"a", "c"}, {"a", "d"}, {"a", "b"}])
    assert weight["a"] == 0 and weight["c"] > weight["b"] > 0  # a word in every entry says nothing
    assert tried.nearest(registry, "рыба и яблоки") == []
    assert tried.stems("Страховки, страховка") == {"страх"}
    text = tried.render(registry)
    assert "## hedge-entry-signal" in text and "bv2-flow-taker-72h" in text
