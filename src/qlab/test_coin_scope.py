"""No silent narrowing of a coin list (`qlab.coin_scope`, docs/COIN_SCOPE.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlab import coin_scope
from qlab.pipeline.spec import StrategySpec, load_spec

OWNER = ("ideas/inbox.yaml: «оставь BTC и ETH, остальные не нужны» -- слова владельца, "
         "2026-11-04")
PAPER = "research/strategy_b_v2/LIMITS.md: «Самый устойчивый набор — BTC+ETH»"


def _spec(idea_id: str, coins=None, instruments=None, **over) -> StrategySpec:
    params = {"capital_usd": 100.0}
    if coins is not None:
        params["coins"] = coins
    raw = {
        "idea_id": idea_id, "title": "Опыт", "code_ref": "qlab.strategies.live.bv2:LiveBv2",
        "params": params,
        "data": {"source": "binance", "interval": "1h", "start": "2023-01-01", "end": "2026-01-01",
                 **({"instruments": instruments} if instruments else {})},
        "costs": {"taker_fee_bps": 4.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []}
    raw.update(over)
    return StrategySpec.model_validate(raw)


PARENT = _spec("strategy-b-v2", coins=["BTC", "ETH", "SOL", "AVAX"])


def test_a_strict_subset_of_the_parents_coins_without_a_source_is_refused_in_russian() -> None:
    found = coin_scope.problems(_spec("bv3-x", coins=["BTC", "ETH"]), parent=PARENT)
    assert len(found) == 1
    assert found[0].startswith("список монет сужен без источника")
    assert "AVAX, BTC, ETH, SOL" in found[0] and "убраны: AVAX, SOL" in found[0]


def test_the_owners_word_in_sources_lets_the_narrowing_through() -> None:
    spec = _spec("bv3-x", coins=["BTC", "ETH"], sources={"coins": OWNER})
    assert coin_scope.problems(spec, parent=PARENT) == []


def test_a_paper_or_the_agents_own_memo_is_not_the_owners_word() -> None:
    spec = _spec("bv3-x", coins=["BTC", "ETH"], sources={"coins": PAPER})
    assert coin_scope.problems(spec, parent=PARENT)
    # a mention of the owner without a quotation is no quotation either
    spec = _spec("bv3-x", coins=["BTC", "ETH"], sources={"coins": "owner asked for two coins"})
    assert coin_scope.problems(spec, parent=PARENT)


def test_a_declared_choice_over_the_coin_list_lets_it_through() -> None:
    spec = _spec("bv3-x", coins=["BTC", "ETH"], choice={
        "candidates": {"two": {"coins": ["BTC", "ETH"]},
                       "four": {"coins": ["BTC", "ETH", "SOL", "AVAX"]}},
        "source": "cards/bv3.yaml"})
    assert coin_scope.problems(spec, parent=PARENT) == []


def test_a_rule_of_code_that_picks_the_coins_is_not_a_narrowing() -> None:
    universe = {"rule": "top_liquid", "n": 2, "rebalance": "1d", "by": "perp_dollar_volume",
                "window_days": 30, "candidates": ["BTC", "ETH"]}
    spec = _spec("bv3-x", instruments=["BTC", "BTC-SPOT", "ETH", "ETH-SPOT"])
    spec = spec.model_copy(update={"params": {"universe": universe}})
    assert coin_scope.coin_list(spec).by_code
    assert coin_scope.problems(spec, parent=PARENT) == []
    # K declared as a choice axis is a code rule too
    axed = _spec("bv3-x", coins=["BTC", "ETH"], choice={
        "axes": {"universe.n": [2, 4]}, "source": "cards/bv3.yaml"})
    assert coin_scope.problems(axed, parent=PARENT) == []


def test_the_same_or_more_coins_and_other_coins_are_not_a_narrowing() -> None:
    assert coin_scope.problems(_spec("x", coins=["BTC", "ETH", "SOL", "AVAX"]), parent=PARENT) == []
    assert coin_scope.problems(_spec("x", coins=["BTC", "ETH", "SOL", "AVAX", "XRP"]),
                               parent=PARENT) == []
    assert coin_scope.problems(_spec("x", coins=["BTC", "XRP"]), parent=PARENT) == []


def test_the_coins_of_the_data_are_compared_when_the_params_name_none() -> None:
    spec = _spec("bv3-x", instruments=["BTC", "BTC-SPOT", "ETH", "ETH-SPOT"])
    assert coin_scope.coin_list(spec).coins == frozenset({"BTC", "ETH"})
    assert coin_scope.problems(spec, parent=PARENT)


def test_a_spec_without_a_family_or_a_readable_parent_has_nothing_to_compare(
        tmp_path: Path) -> None:
    config = {"families": {"bv3": {"parent": "specs/missing.yaml", "prefixes": ["bv3-"]}}}
    assert coin_scope.problems(_spec("zzz-x", coins=["BTC"]), root=tmp_path, config=config) == []
    assert coin_scope.problems(_spec("bv3-x", coins=["BTC"]), root=tmp_path, config=config) == []


def test_the_family_is_found_by_the_id_prefix_or_the_module_the_spec_runs() -> None:
    config = {"families": {"bv3": {"parent": "p3.yaml", "prefixes": ["bv3-"]},
                           "bv2": {"parent": "p2.yaml", "prefixes": ["bv2-"]}}}
    assert coin_scope.parent_path(_spec("bv3-a", coins=["BTC"]), config) == "p3.yaml"
    assert coin_scope.parent_path(_spec("other", coins=["BTC"]), config) == "p2.yaml"  # live.bv2
    assert coin_scope.parent_path(_spec("other", coins=["BTC"], code_ref="qlab.strategies.x:Y"),
                                  config) is None


def test_a_refusal_is_written_to_the_ledger_the_console_reads(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    # the real parent file, the real config: the production Bv2 book has four coins
    spec = _spec("bv3-narrowed", coins=["BTC", "ETH"])
    found = coin_scope.refuse(spec, "specs/research/bv3/bv3-narrowed.yaml",
                              root=Path.cwd(), ledger_path=ledger)
    assert found and found[0].startswith("список монет сужен без источника")
    row = json.loads(ledger.read_text(encoding="utf-8"))["bv3-narrowed"]
    assert row["reason"] == found[0] and row["spec"].endswith("bv3-narrowed.yaml")
    assert coin_scope.refuse(_spec("bv3-fine", coins=["BTC", "ETH", "SOL", "AVAX"]), "p.yaml",
                             root=Path.cwd(), ledger_path=ledger) == []


@pytest.mark.parametrize("path", ["specs/research/bv3/bv3-research-topk-coins.yaml"])
def test_the_topk_spec_is_not_a_narrowing(path: str) -> None:
    spec = load_spec(Path(path))
    assert coin_scope.coin_list(spec).by_code
    assert coin_scope.problems(spec, root=Path.cwd()) == []


def test_the_two_existing_narrowed_bv_specs_are_found_by_the_scan() -> None:
    specs = {p: load_spec(Path(p)) for p in (
        "specs/research/bv3/bv3-research-fast-release.yaml",
        "specs/research/bv2/bv2-research-btceth.yaml")}
    found = coin_scope.narrowed_without_source(specs, root=Path.cwd())
    assert set(found) == set(specs)


def test_the_candidates_of_a_code_rule_are_judged_from_the_start_of_the_data() -> None:
    """The columns are the data of a rule of code, not a hand-picked list: no survivorship guard
    dates them (`qlab.twin.causal_twin`, `by_code`), so the check starts at the first choice."""
    from qlab import one_check

    spec = load_spec(Path("specs/research/bv3/bv3-research-topk-coins.yaml"))
    assert one_check.declaration_problems(spec) == []
    result = one_check.check_spec(spec, venue_names=[], volume_of=lambda venue: None)
    assert result.spec is not None, result.reason
    assert result.spec.judged_from is None and result.spec.instruments_chosen_at is None
    assert result.spec.coins_knowable is None
    assert result.spec.data.start.isoformat() == "2019-09-08"
    assert result.candidates and len(result.candidates) == 6
    assert any("rule of code" in line for line in result.spec.differs_from_source)
