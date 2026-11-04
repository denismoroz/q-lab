"""One idea, one check (`qlab.one_check`, docs/ONE_CHECK.md): the idea declares the values it leaves
open, code chooses among them once, at the cut of the window at its last third (rules
2026-11-04.1; month by month before), and the verdict is the idea's own
under its own id -- there is no `-causal` idea beside it.

The stand is replaced by a recorder; the specs are built in the test, and the lab's own lists
(`night/own_numbers.yaml`) are the real file."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from qlab import night, one_check, twin
from qlab.pipeline.sources import choice_problems, source_problems
from qlab.pipeline.spec import Choice, StrategySpec
from qlab.strategies.retune import expand_grid

OWN = "night/own_numbers.yaml"
RETUNE = "qlab.strategies.retune:Retune"


def _spec(idea_id: str = "new-idea", **over) -> StrategySpec:
    """A daily idea on Hyperliquid's data with two open numbers, declared as the lab's lists."""
    raw = {
        "idea_id": idea_id, "title": "Новая идея", "code_ref": "qlab.strategies.agent.toy:Toy",
        "params": {"signal": {"window_days": 3}, "rebalance_days": 1},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-10-26"},
        "costs": {"taker_fee_bps": 4.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [],
        "choice": {"axes": {"signal.window_days": [3, 7, 14, 30], "rebalance_days": [1, 7]},
                   "source": OWN},
        "sources": {"signal.window_days": OWN, "rebalance_days": OWN,
                    "costs": "night/costs.yaml", "min_leg_notional": "venues/hyperliquid.yaml"}}
    raw.update(over)
    return StrategySpec.model_validate(raw)


# ---- the declaration -------------------------------------------------------------------


def test_a_choice_is_exactly_one_of_lists_named_variants_or_a_reason() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Choice(source=OWN)
    with pytest.raises(ValueError, match="exactly one"):
        Choice(axes={"a": [1, 2]}, nothing_to_choose="a sentence that says why there is none",
               source=OWN)
    with pytest.raises(ValueError, match="needs `source`"):
        Choice(axes={"a": [1, 2]})
    with pytest.raises(ValueError, match="at least two"):
        Choice(axes={"a": [1]}, source=OWN)
    with pytest.raises(ValueError, match="in a sentence"):
        Choice(nothing_to_choose="none")
    ok = Choice(nothing_to_choose="the paper gives one value for every number used")
    assert not ok.lists and Choice(axes={"a": [1, 2]}, source=OWN).lists


def test_a_spec_without_a_choice_is_refused_unless_it_says_why_there_is_nothing_to_choose() -> None:
    bare = _spec(choice=None)
    assert one_check.declaration_problems(bare) == [one_check.NO_DECLARATION]
    # an idea that existed before the rule is judged as it was: only a NEW idea must declare
    assert one_check.declaration_problems(bare, require=False) == []
    why = _spec(choice={"nothing_to_choose": "the card gives one value for every number it uses"})
    assert one_check.declaration_problems(why) == []
    assert not one_check.declares_lists(why)
    assert one_check.nothing_to_choose(why).startswith("the card gives one value")


def test_the_lists_of_the_lab_are_taken_whole_or_sourced_from_another_file() -> None:
    assert one_check.declaration_problems(_spec()) == []
    subset = _spec(choice={"axes": {"signal.window_days": [3, 7], "rebalance_days": [1, 7]},
                           "source": OWN})
    found = one_check.declaration_problems(subset)
    assert any("signal.window_days" in p and "whole list" in p for p in found)
    # a list nobody declared, cited to the lab's file only
    other = _spec(choice={"axes": {"signal.window_days": [3, 7, 14, 30], "rebalance_days": [1, 7],
                                   "min_history_days": [5, 9]}, "source": OWN},
                  params={"signal": {"window_days": 3}, "rebalance_days": 1,
                          "min_history_days": 5})
    assert any("min_history_days" in p and "declares no list" in p
               for p in one_check.declaration_problems(other))
    # the same list sourced from a file the citation names is accepted
    sourced = _spec(choice={"axes": {"signal.window_days": [3, 7, 14, 30],
                                     "min_history_days": [5, 9]},
                            "source": f"{OWN}, docs/RETUNE.md"},
                    params={"signal": {"window_days": 3}, "rebalance_days": 1,
                            "min_history_days": 5})
    assert one_check.declaration_problems(sourced) == []


def test_the_product_of_the_lists_stays_within_the_cap_and_params_carry_the_first_value() -> None:
    too_many = _spec(choice={"axes": {"signal.window_days": [3, 7, 14, 30],
                                      "rebalance_days": [1, 7], "portfolio.fraction": [0.1, 0.2]},
                             "source": OWN},
                     params={"signal": {"window_days": 3}, "rebalance_days": 1,
                             "portfolio": {"fraction": 0.1}})
    assert any("at most 8" in p for p in one_check.declaration_problems(too_many))
    wrong_first = _spec(params={"signal": {"window_days": 7}, "rebalance_days": 1})
    assert any("first" in p for p in one_check.declaration_problems(wrong_first))


def test_a_list_inside_params_is_the_old_way_and_is_refused_for_a_new_idea() -> None:
    old = _spec(choice=None, params={"signal": {"window_days": [3, 7, 14, 30]},
                                     "rebalance_days": 1},
                sources={"signal.window_days": OWN, "rebalance_days": "owner 2026-10-26: test",
                         "costs": "night/costs.yaml", "min_leg_notional": "x.yaml"})
    assert one_check.declaration_problems(old) == [one_check.LEGACY_OPEN]


def test_the_codes_own_fields_are_not_the_agents_to_set() -> None:
    found = one_check.declaration_problems(_spec(selects_causally=True,
                                                 params_fixed_at="2026-01-01",
                                                 params_fixed_evidence="by hand"))
    assert any("selects_causally" in p for p in found)


def test_every_value_of_a_list_must_stand_in_the_cited_file() -> None:
    assert choice_problems(_spec()) == []
    bad = _spec(choice={"axes": {"signal.window_days": [3, 7, 14, 17], "rebalance_days": [1, 7]},
                        "source": OWN})
    assert any("17" in p and "does not appear" in p for p in choice_problems(bad))
    assert any("17" in p for p in source_problems(bad))
    nothing = _spec(choice={"nothing_to_choose": "the card gives one value for every number"})
    assert choice_problems(nothing) == []


# ---- the check -------------------------------------------------------------------------


def test_the_check_is_the_same_idea_with_its_numbers_chosen_by_code_on_binance() -> None:
    spec = _spec()
    result = one_check.check_spec(spec)
    assert result.spec is not None and result.reason is None
    checked = result.spec
    assert checked.idea_id == "new-idea"  # no `-causal` idea, no second page
    assert checked.code_ref == RETUNE and checked.selects_causally
    assert checked.data.source == "binance" and checked.data.start == date(2019, 9, 8)
    assert checked.choice is None  # the check form declares nothing more
    assert len(result.candidates) == 8
    # rules 2026-11-04.1: chosen once, at the stand's cut, on the first 2/3 of the window
    assert checked.params["refit_every"] == "once" and checked.params["choose_at"] == "split"
    assert checked.params["window"] == "expanding"
    assert checked.params["grid"] and checked.params["inner_code_ref"] == spec.code_ref
    # the date is the cut of the spec's own window (what older rules split at), not a day
    # anybody picked: 2019-09-08 + 2/3 of the days to 2026-10-26
    assert checked.params_fixed_at == date(2024, 6, 10)
    assert "choose_at" in checked.sources and "2/3" in checked.title
    assert "docs/ONE_CHECK.md" in checked.sources["grid"]
    assert "One check of new-idea" in " ".join(checked.differs_from_source)
    # the venue split: Binance's own costs, not the original venue's
    assert checked.costs.taker_fee_bps == 5.0 and checked.execution_venue is None


def test_the_check_keeps_every_candidate_the_lists_make() -> None:
    result = one_check.check_spec(_spec())
    cands = expand_grid(result.spec.params["base_params"], result.spec.params["grid"])
    assert set(cands) == set(result.candidates)
    windows = {c["signal"]["window_days"] for c in cands.values()}
    assert windows == {3, 7, 14, 30} and {c["rebalance_days"] for c in cands.values()} == {1, 7}


def test_named_candidates_are_the_other_form_of_the_same_declaration() -> None:
    spec = _spec(choice={"candidates": {"base": {"signal.window_days": 3},
                                        "fix": {"signal.window_days": 14}}, "source": OWN})
    result = one_check.check_spec(spec)
    assert list(result.candidates) == ["base", "fix"] and result.spec.idea_id == "new-idea"


def test_an_idea_with_nothing_to_choose_has_no_check_and_says_why() -> None:
    spec = _spec(choice={"nothing_to_choose": "the card gives one value for every number it uses"})
    result = one_check.check_spec(spec)
    assert result.spec is None and result.reason.startswith("the card gives")
    with pytest.raises(one_check.OneCheckRefused):
        one_check.rerun_form(spec)


def test_the_night_reruns_the_check_not_the_plain_spec() -> None:
    spec = _spec()
    assert night.rerun_spec(spec).code_ref == RETUNE
    assert night.rerun_spec(spec).idea_id == "new-idea"
    why = _spec(choice={"nothing_to_choose": "the card gives one value for every number it uses"})
    assert night.rerun_spec(why) is why  # judged after its fixation date, as always
    plain = _spec(choice=None)
    assert night.rerun_spec(plain) is plain  # an idea older than the rule


def test_an_idea_with_lists_goes_to_the_second_stage_whatever_venue_its_file_is_on() -> None:
    from qlab.twin_stage import goes_to_stage2

    assert goes_to_stage2(_spec())  # a Hyperliquid file, but its check runs on Binance
    assert not goes_to_stage2(_spec(choice=None))  # as before: Hyperliquid data has no stage 2


def test_a_fixation_day_is_not_named_for_a_spec_that_declares_lists(monkeypatch) -> None:
    from qlab import fixation

    assert fixation.check(_spec(), share=1 / 3).ok
    assert not fixation.check(_spec(choice=None), share=1 / 3).ok  # no date, so one is named
    why = _spec(choice={"nothing_to_choose": "the card gives one value for every number it uses"})
    assert not fixation.check(why, share=1 / 3).ok


# ---- trials of every value -------------------------------------------------------------


def test_every_value_is_a_trial_of_its_own_before_the_verdict(monkeypatch) -> None:
    seen: list[StrategySpec] = []

    class Out:
        trial_id = 1

    def fake(spec, **kw):
        seen.append(spec)
        assert kw["update_idea_status"] is False and kw["check_sources"] is False
        return Out()

    monkeypatch.setattr("qlab.pipeline.evaluate.evaluate_spec", fake)
    checked = one_check.check_spec(_spec()).spec
    ids = one_check.record_candidates(checked, session=None, ruleset=None)
    assert len(ids) == 8 and len(seen) == 8
    # each is the idea's own strategy with one value, on the check's panel; not a chooser
    assert all(s.code_ref == "qlab.strategies.agent.toy:Toy" and not s.selects_causally
               and s.data.source == "binance" and s.idea_id == "new-idea" for s in seen)
    assert {(s.params["signal"]["window_days"], s.params["rebalance_days"]) for s in seen} == {
        (w, r) for w in (3, 7, 14, 30) for r in (1, 7)}


def test_prepare_returns_the_check_and_records_the_candidates(monkeypatch) -> None:
    monkeypatch.setattr(one_check, "record_candidates", lambda *a, **k: [11, 12])
    got = one_check.prepare(_spec(), session=None, ruleset=None)
    assert got.declared == "lists" and got.spec.code_ref == RETUNE
    assert got.candidate_trials == (11, 12) and got.first_choice == got.spec.params_fixed_at
    quiet = one_check.prepare(_spec(), session=None, ruleset=None, record=False)
    assert quiet.candidate_trials == ()


def test_prepare_passes_an_idea_with_nothing_to_choose_as_it_is() -> None:
    spec = _spec(choice={"nothing_to_choose": "the card gives one value for every number it uses"})
    got = one_check.prepare(spec, session=None, ruleset=None)
    assert got.spec is spec and got.declared == "nothing" and got.candidate_trials == ()
    assert got.reason.startswith("the card gives")


def test_prepare_refuses_an_idea_that_declares_nothing() -> None:
    with pytest.raises(one_check.OneCheckRefused, match="no list of values"):
        one_check.prepare(_spec(choice=None), session=None, ruleset=None)


def test_a_heavy_check_is_not_recorded_or_run_unless_a_hand_asks(monkeypatch) -> None:
    monkeypatch.setattr(one_check, "_heavy", lambda spec: "too many hourly candles")
    called: list[int] = []
    monkeypatch.setattr(one_check, "record_candidates", lambda *a, **k: called.append(1) or [1])
    got = one_check.prepare(_spec(), session=None, ruleset=None)
    assert got.heavy == "too many hourly candles" and not called
    hand = one_check.prepare(_spec(), session=None, ruleset=None, allow_heavy=True)
    assert hand.heavy is None and called == [1]


def test_a_refused_check_is_written_to_the_ledger_the_ladder_reads(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    one_check.note_refusal("new-idea", "hourly candles of 90 columns", "specs/x.yaml",
                           ledger_path=ledger)
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["new-idea"]["reason"] == "hourly candles of 90 columns"
    assert saved["new-idea"]["spec"] == "specs/x.yaml"


# ---- Binance's listing and the peers ----------------------------------------------------


def test_binance_listing_is_kept_a_month_and_a_failed_fetch_falls_back_to_the_kept_one(
        tmp_path) -> None:
    fetched: list[int] = []

    def fetch():
        fetched.append(1)
        return ["BTC", "BTC-SPOT"]

    first = one_check.binance_listing(store=tmp_path, today=date(2026, 11, 1), fetch=fetch)
    again = one_check.binance_listing(store=tmp_path, today=date(2026, 11, 20), fetch=fetch)
    assert first == again == ["BTC", "BTC-SPOT"] and len(fetched) == 1

    def down():
        raise OSError("no network")

    stale = one_check.binance_listing(store=tmp_path, today=date(2026, 12, 25), fetch=down)
    assert stale == ["BTC", "BTC-SPOT"]
    with pytest.raises(OSError):
        one_check.binance_listing(store=tmp_path / "empty", today=date(2026, 12, 25), fetch=down)


def test_the_peers_are_the_other_specs_without_the_twin_files(tmp_path) -> None:
    folder = tmp_path / "specs"
    folder.mkdir()
    other = _spec("other-idea", choice=None)
    (folder / "other-idea.yaml").write_text(
        yaml.safe_dump(other.model_dump(mode="json", exclude_none=True)), encoding="utf-8")
    (folder / "new-idea-causal.yaml").write_text("not a spec", encoding="utf-8")
    (folder / "broken.yaml").write_text("not: a spec", encoding="utf-8")
    me = _spec()
    (folder / "new-idea.yaml").write_text(
        yaml.safe_dump(me.model_dump(mode="json", exclude_none=True)), encoding="utf-8")
    got = one_check.peers_of(me, folder)
    assert [p.idea_id for p, _ in got] == ["other-idea"]


# ---- what stays of the twin ------------------------------------------------------------


def test_a_twin_of_a_pair_is_still_built_by_the_same_machinery_under_its_own_name() -> None:
    plain = _spec(choice=None, params={"signal": {"window_days": [3, 7, 14, 30]},
                                       "rebalance_days": 1},
                  sources={"signal.window_days": OWN, "rebalance_days": "owner 2026-10-26: test",
                           "costs": "night/costs.yaml", "min_leg_notional": "x.yaml"})
    result = twin.causal_twin(plain)  # the declared lists of night/own_numbers.yaml
    assert result.spec is not None and result.spec.idea_id == "new-idea-causal"


def test_the_checked_form_loads_back_as_a_spec_file() -> None:
    checked = one_check.check_spec(_spec()).spec
    text = twin.twin_yaml(checked, "new-idea")
    again = StrategySpec.model_validate(yaml.safe_load(text))
    assert again.idea_id == "new-idea" and again.params_fixed_at == checked.params_fixed_at


def test_the_shipped_lists_cover_what_the_tests_assume() -> None:
    lists = yaml.safe_load(Path("night/own_numbers.yaml").read_text(encoding="utf-8"))["lists"]
    assert lists["signal.window_days"] == [3, 7, 14, 30] and lists["rebalance_days"] == [1, 7]
