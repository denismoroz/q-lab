"""Tests for `qlab.overlay_variants`: every single overlay and every pair, no triples, each a
causal choice between a declared family, judged on the base's forward test (docs/OVERLAYS.md)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlab import overlay_variants as ov
from qlab.harness.lookahead import lookahead_violation
from qlab.harness.panel import MarketPanel
from qlab.pipeline.evaluate import resolve_strategy
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.strategies.overlays import get

ROOT = Path(__file__).resolve().parents[2]
CAUSAL_BASE = ROOT / "specs" / "research" / "xsmom" / "xsmom-research-nobull-causal.yaml"
PLAIN_BASE = ROOT / "specs" / "trend.yaml"
ALL = ["market_gate", "vol_target", "own_results", "gradual_entry", "gradual_exit"]


def synthetic_base(**over) -> StrategySpec:
    """A base on a daily synthetic market, with the test strategy of the overlay tests."""
    raw = {
        "idea_id": "toy", "title": "Toy momentum", "code_ref":
        "qlab.strategies.overlays.test_overlays:Momentum", "params": {"lookback": 10},
        "data": {"source": "binance", "interval": "1d", "start": "2022-01-01", "end": "2023-06-30"},
        "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [],
        "sources": {"costs.taker_fee_bps": "docs/OVERLAYS.md: 3.5", "costs.slippage_bps":
                    "docs/OVERLAYS.md: 0.9", "min_leg_notional": "docs/OVERLAYS.md: 10",
                    "lookback": "docs/OVERLAYS.md: 10"},
    }
    raw.update(over)
    return StrategySpec.model_validate(raw)


# ---- which variants -------------------------------------------------------------------------


def test_every_single_and_every_pair_and_no_triple() -> None:
    made = ov.chains(ALL)
    assert len(made) == 5 + 10
    assert {len(c) for c in made} == {1, 2}
    assert sorted(set(made)) == sorted(made)            # none twice
    assert ("own_results", "gradual_entry") in made     # book level first, position level last
    assert ("gradual_entry", "own_results") not in made  # one order, not two variants
    assert ("gradual_entry", "gradual_exit") in made


def test_pairs_are_in_the_canonical_order_whatever_the_input_order() -> None:
    made = ov.chains(["gradual_exit", "own_results", "market_gate"])
    assert ("market_gate", "own_results") in made and ("own_results", "gradual_exit") in made
    assert ("gradual_exit", "own_results") not in made


def test_meta_gate_is_offered_only_where_decisions_are_declared() -> None:
    assert "meta_gate" not in ov.available_overlays()
    extra = {"meta_gate": get("meta_gate").grid_for(["trend-binance-cap"])}
    assert "meta_gate" in ov.available_overlays(extra)
    assert len(ov.chains(ov.available_overlays(extra))) == 6 + 15


def test_a_variant_id_names_its_base_and_its_overlays() -> None:
    assert ov.variant_id("xsmom-causal", ("own_results", "gradual_entry")) == \
        "xsmom-causal-ov-own-enter"
    point = ov.point_slug("vol_target:d30-max1")
    assert point == "vol-d30-max1" and ov.variant_id("b", ("vol_target",), point) == \
        "b-ov-vol-d30-max1"
    pair = ov.point_slug("own_results:w30-f0.5-smooth-up7+gradual_entry:k2-d3-plain")
    assert pair == "own-w30-f0p5-smooth-up7_enter-k2-d3-plain"
    assert ov.base_of("xsmom-causal-ov-own-enter") == "xsmom-causal"
    assert ov.base_of("xsmom-causal") is None and ov.is_variant_id("a-ov-b")


def test_a_single_offers_its_whole_family_a_pair_the_coarse_families_multiplied() -> None:
    single = ov.family(("own_results",))
    assert len(single) == len(get("own_results").grid()) == 60
    pair = ov.family(("own_results", "gradual_entry"))
    assert len(pair) == len(get("own_results").coarse_grid()) * len(
        get("gradual_entry").coarse_grid())
    one = next(iter(pair.values()))
    assert [step["name"] for step in one] == ["own_results", "gradual_entry"]
    with pytest.raises(ValueError, match="no family"):
        ov.family(("meta_gate",))


# ---- the spec -------------------------------------------------------------------------------


def test_a_causal_variant_is_a_retune_of_the_wrapper_over_the_family() -> None:
    base = load_spec(CAUSAL_BASE)
    spec = ov.build_variant(base, ("own_results",), base_cite=CAUSAL_BASE.as_posix())
    assert spec.idea_id == "xsmom-research-nobull-causal-ov-own"
    assert spec.code_ref == ov.RETUNE and spec.selects_causally
    assert spec.params["inner_code_ref"] == ov.OVERLAID
    assert spec.params["base_params"]["inner_code_ref"] == base.code_ref
    assert spec.params["base_params"]["inner_params"] == base.params
    assert len(spec.params["grid"]["overlay"]) == 60
    assert spec.params["score_costs"] == {"taker_fee_bps": base.costs.taker_fee_bps,
                                          "slippage_bps": base.costs.slippage_bps}
    assert spec.costs == base.costs and spec.data.start == base.data.start


def test_every_number_of_a_variant_has_a_checkable_source() -> None:
    for path in (CAUSAL_BASE, PLAIN_BASE):
        base = load_spec(path)
        for chain in ov.chains(ALL):
            spec = ov.build_variant(base, chain, base_cite=path.as_posix())
            assert source_problems(spec) == [], (path.name, chain)
        point = ("w30", [{"name": "own_results", "params": get("own_results").grid()[
            "w30-f0.5-smooth-up7"]}])
        fixed = ov.build_variant(base, ("own_results",), base_cite=path.as_posix(),
                                 kind=ov.KIND_FIXED, point=point)
        assert source_problems(fixed) == [], path.name


def test_no_triple_is_built() -> None:
    base = load_spec(CAUSAL_BASE)
    with pytest.raises(ValueError, match="no triples"):
        ov.build_variant(base, ("market_gate", "own_results", "gradual_entry"),
                         base_cite=CAUSAL_BASE.as_posix())


def test_the_variant_is_judged_from_its_first_choice_not_before_the_bases_forward_test() -> None:
    base = load_spec(CAUSAL_BASE)            # first causal choice 2020-01-01
    single = ov.build_variant(base, ("gradual_entry",), base_cite=CAUSAL_BASE.as_posix())
    assert single.params_fixed_at == date(2020, 3, 1)       # two whole months after the base starts
    slow = ov.build_variant(base, ("own_results",), base_cite=CAUSAL_BASE.as_posix())
    assert slow.params_fixed_at == date(2020, 5, 1)         # plus the overlay's 90-day look-back
    assert "2020-05-01" in slow.params_fixed_evidence
    plain = load_spec(PLAIN_BASE)            # numbers fixed on 2026-xx? none: data starts 2025
    assert plain.params_fixed_at is None
    spec = ov.build_variant(plain, ("gradual_entry",), base_cite=PLAIN_BASE.as_posix())
    assert spec.params_fixed_at > plain.data.start


def test_a_base_with_numbers_fixed_later_keeps_its_forward_start() -> None:
    base = synthetic_base(params_fixed_at="2023-01-15",
                          params_fixed_evidence="docs/OVERLAYS.md: the day the numbers were fixed")
    spec = ov.build_variant(base, ("gradual_entry",), base_cite="docs/OVERLAYS.md")
    assert spec.params_fixed_at == date(2023, 1, 15)
    assert "not judged before it" in spec.params_fixed_evidence


def test_a_variant_whose_first_choice_is_after_the_data_is_refused() -> None:
    base = synthetic_base(data={"source": "binance", "interval": "1d", "start": "2022-01-01",
                                "end": "2022-04-01"})
    with pytest.raises(ValueError, match="no day would be a forward test"):
        ov.build_variant(base, ("own_results",), base_cite="docs/OVERLAYS.md")


def test_the_data_window_is_extended_never_shortened() -> None:
    base = synthetic_base()
    longer = ov.build_variant(base, ("gradual_entry",), base_cite="docs/OVERLAYS.md",
                              end=date(2023, 9, 1))
    shorter = ov.build_variant(base, ("gradual_entry",), base_cite="docs/OVERLAYS.md",
                               end=date(2023, 1, 1))
    assert longer.data.end == date(2023, 9, 1) and shorter.data.end == base.data.end


def test_a_fixed_variant_is_one_point_of_the_family_on_the_bases_own_dates() -> None:
    base = synthetic_base(params_fixed_at="2023-01-15",
                          params_fixed_evidence="docs/OVERLAYS.md: the day the numbers were fixed")
    point = get("gradual_entry").grid()["k3-d7-profit"]
    spec = ov.build_variant(
        base, ("gradual_entry",), base_cite="docs/OVERLAYS.md", kind=ov.KIND_FIXED,
        point=(ov.point_slug("gradual_entry:k3-d7-profit"),
               [{"name": "gradual_entry", "params": point}]))
    assert spec.code_ref == ov.OVERLAID and not spec.selects_causally
    assert spec.params_fixed_at == base.params_fixed_at
    assert spec.idea_id == "toy-ov-enter-k3-d7-profit"
    assert spec.params["overlays"][0]["params"] == point
    with pytest.raises(ValueError, match="needs the point"):
        ov.build_variant(base, ("gradual_entry",), base_cite="docs/OVERLAYS.md",
                         kind=ov.KIND_FIXED)


def test_the_spec_file_round_trips(tmp_path) -> None:
    base = load_spec(CAUSAL_BASE)
    spec = ov.build_variant(base, ("market_gate", "gradual_exit"),
                            base_cite=CAUSAL_BASE.as_posix())
    path = ov.write_variant(tmp_path, base.idea_id, spec)
    assert path == tmp_path / "overlays" / base.idea_id / f"{spec.idea_id}.yaml"
    assert load_spec(path) == spec
    stamp = path.stat().st_mtime_ns
    ov.write_variant(tmp_path, base.idea_id, spec)       # an identical file is left alone
    assert path.stat().st_mtime_ns == stamp
    assert path.read_text(encoding="utf-8").startswith("# Overlay variant of ")


# ---- it runs, and it chooses from the past only ---------------------------------------------


def _panel(spec: StrategySpec, seed: int = 4) -> MarketPanel:
    index = pd.date_range(spec.data.start, spec.data.end, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    cols = [f"C{i}" for i in range(5)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.025, (len(index), 5)),
                                                 axis=0)), index=index, columns=cols)
    return MarketPanel(snapshot_id=f"variant-{seed}", prices=prices,
                       funding=pd.DataFrame(0.0, index=index, columns=cols),
                       tradeable=pd.DataFrame(True, index=index, columns=cols), meta={})


@pytest.fixture(autouse=True)
def _btc(monkeypatch):
    index = pd.date_range("2021-06-01", periods=1500, freq="1D", tz="UTC")
    rng = np.random.default_rng(7)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(index)))), index=index)
    monkeypatch.setattr("qlab.strategies.detectors.market_closes", lambda: closes)


@pytest.mark.parametrize("chain", [("own_results",), ("gradual_entry", "gradual_exit"),
                                   ("market_gate", "vol_target")])
def test_a_variant_runs_waits_for_its_first_choice_and_never_reads_the_future(
        chain) -> None:
    spec = ov.build_variant(synthetic_base(), chain, base_cite="docs/OVERLAYS.md")
    panel = _panel(spec)
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    # The chooser holds nothing until a candidate has traded through one whole month (the base
    # trades from 2022-01-11; months start 02-01 and 03-01); the judged part starts later or
    # at the same time, after the overlays' own warm-up.
    assert (weights.loc[weights.index < "2022-03-01"].abs().sum(axis=1) == 0).all()
    assert pd.Timestamp(spec.params_fixed_at, tz="UTC") >= pd.Timestamp("2022-03-01", tz="UTC")
    assert weights.abs().sum().sum() > 0
    assert lookahead_violation(resolve_strategy(spec.code_ref), panel, spec.params,
                               weights) is None
    assert strategy.diagnostics()["retune_candidates"] == len(spec.params["grid"]["overlay"])


def test_the_choice_changes_only_when_the_past_changes() -> None:
    """The candidate held in a month is the best of the returns realised BEFORE it: two panels
    that agree up to a date choose the same candidate for that date's month."""
    spec = ov.build_variant(synthetic_base(), ("own_results",), base_cite="docs/OVERLAYS.md")
    panel = _panel(spec)
    other = _panel(spec)
    cut = len(panel.prices) - 120
    prices = other.prices.copy()
    prices.iloc[cut:] = prices.iloc[cut:] * 1.7
    other = MarketPanel(snapshot_id="variant-other", prices=prices, funding=other.funding,
                        tradeable=other.tradeable, meta={})
    a = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    b = resolve_strategy(spec.code_ref).target_weights(other, spec.params)
    pd.testing.assert_frame_equal(a.iloc[:cut], b.iloc[:cut])
