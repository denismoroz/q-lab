"""The day the numbers were fixed is named by code (qlab.fixation, docs/FIXATION.md)."""

from __future__ import annotations

from datetime import date

import pytest

from qlab import fixation
from qlab.pipeline.spec import StrategySpec, load_spec

START, END = date(2019, 9, 8), date(2026, 10, 24)


def _spec(*, fixed=None, sources=None, causal=False, start=START, end=END) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "t", "title": "t", "code_ref": "a:b", "params": {},
        "data": {"source": "binance", "interval": "1d", "start": start, "end": end},
        "costs": {"taker_fee_bps": 5.0, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "params_fixed_at": fixed,
        "params_fixed_evidence": "x" if fixed else None, "selects_causally": causal,
        "sources": sources or {}})


def test_the_named_day_is_the_start_of_the_last_third() -> None:
    named = fixation.named_date(START, END, 1 / 3)
    assert named == date(2024, 6, 8)
    assert (END - named).days == pytest.approx((END - START).days / 3, abs=1)
    assert fixation.named_date(date(2025, 1, 1), date(2025, 1, 10), 1 / 3) == date(2025, 1, 7)
    with pytest.raises(ValueError):
        fixation.named_date(END, START, 1 / 3)
    with pytest.raises(ValueError):
        fixation.named_date(START, END, 1.0)


def test_the_share_is_the_rules_holdout_share() -> None:
    assert fixation.holdout_share() == pytest.approx(1 / 3)


def test_the_named_day_passes_an_earlier_or_missing_one_does_not() -> None:
    named = fixation.named_date(START, END, 1 / 3)
    assert fixation.problems(_spec(fixed=named)) == []
    assert any("missing" in p for p in fixation.problems(_spec()))
    assert any("earlier" in p for p in fixation.problems(_spec(fixed=date(2022, 1, 1))))


def test_a_later_day_needs_a_source() -> None:
    later = date(2025, 6, 1)
    assert any("needs sources['params_fixed_at']" in p
               for p in fixation.problems(_spec(fixed=later)))
    cited = _spec(fixed=later, sources={"params_fixed_at": "owner 2026-10-29: sample ends 2025-05"})
    assert fixation.problems(cited) == []


def test_a_causal_chooser_names_its_own_first_choice() -> None:
    assert fixation.problems(_spec(fixed=date(2021, 1, 1), causal=True)) == []


def test_a_file_is_corrected_to_the_named_day_and_says_so(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        "idea_id: t\ntitle: t\ncode_ref: a:b\nparams: {}\n"
        "data: {source: binance, interval: 1d, start: 2019-09-08, end: 2026-10-24}\n"
        "costs: {taker_fee_bps: 5.0, slippage_bps: 0.9}\nmin_leg_notional: 10.0\n"
        "unexpressed_mechanisms: []\nparams_fixed_at: 2019-09-08\n"
        "params_fixed_evidence: >-\n  the paper's sample ended long before\n  our history\n"
        "sources:\n  x: y\n", encoding="utf-8")
    note = fixation.normalise_file(path)
    assert note is not None and "2019-09-08 -> 2024-06-08" in note
    spec = load_spec(path)
    assert spec.params_fixed_at == date(2024, 6, 8)
    assert "named by code" in spec.params_fixed_evidence
    assert "the paper's sample ended long before our history" in spec.params_fixed_evidence
    assert spec.sources == {"x": "y"}  # nothing else moved
    assert fixation.normalise_file(path) is None  # already right: untouched


def test_a_missing_day_is_added_and_a_sourced_later_day_is_kept(tmp_path) -> None:
    base = ("idea_id: t\ntitle: t\ncode_ref: a:b\nparams: {}\n"
            "data: {source: binance, interval: 1d, start: 2019-09-08, end: 2026-10-24}\n"
            "costs: {taker_fee_bps: 5.0, slippage_bps: 0.9}\nmin_leg_notional: 10.0\n"
            "unexpressed_mechanisms: []\n")
    path = tmp_path / "s.yaml"
    path.write_text(base, encoding="utf-8")
    assert "no day ->" in fixation.normalise_file(path)
    assert load_spec(path).params_fixed_at == date(2024, 6, 8)
    kept = tmp_path / "k.yaml"
    kept.write_text(base + "params_fixed_at: 2025-06-01\nparams_fixed_evidence: sample\n"
                    "sources:\n  params_fixed_at: 'owner 2026-10-29: the sample ends 2025-05'\n",
                    encoding="utf-8")
    assert fixation.normalise_file(kept) is None
    assert load_spec(kept).params_fixed_at == date(2025, 6, 1)


def test_a_file_that_does_not_load_is_left_alone(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("idea_id: t\n", encoding="utf-8")
    assert fixation.normalise_file(path) is None
    assert path.read_text(encoding="utf-8") == "idea_id: t\n"


def test_the_prompts_of_the_generator_and_the_researcher_name_the_rule() -> None:
    from qlab.agents import generator, researcher

    for text in (generator.INSTRUCTIONS, researcher.INSTRUCTIONS):
        flat = " ".join(text.split())
        assert "LAST THIRD" in flat and "only with a source" in flat
