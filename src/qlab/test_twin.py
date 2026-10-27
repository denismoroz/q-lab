"""The causal twin (qlab.twin, docs/RETUNE.md): the numbers a person or an agent chose are
replaced by a choice between declared candidates that code makes every month from the past
only, so the stand can judge the idea on Binance's long history the same night.

The real spec files are the fixtures: loading a spec needs no data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qlab.cli import app
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.strategies.retune import expand_grid
from qlab.twin import (
    BINANCE_ARCHIVE_START,
    NO_ALTERNATIVE,
    causal_twin,
    find_base,
    first_choice,
    twin_path,
    twin_yaml,
    warmup_days,
    write_twin,
)

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"
RESEARCH = SPECS / "research"
runner = CliRunner()


def _load(*parts: str) -> StrategySpec:
    return load_spec(SPECS.joinpath(*parts))


@pytest.fixture(scope="module")
def xsmom_base() -> StrategySpec:
    return _load("xsmom-honest-20legs.yaml")


@pytest.fixture(scope="module")
def xsmom_experiment() -> StrategySpec:
    return _load("research", "xsmom", "xsmom-research-nobull.yaml")


@pytest.fixture(scope="module")
def frab_base() -> StrategySpec:
    return _load("research", "frab-binance-spot", "frab-binance-spot-300.yaml")


@pytest.fixture(scope="module")
def frab_experiment() -> StrategySpec:
    return _load("research", "frab-binance-spot", "frab-binance-spot-research-sw24.yaml")


def _own_problems(spec: StrategySpec) -> list[str]:
    """The source problems that are not the sibling funding-rate-arbitrage checkout missing."""
    return [p for p in source_problems(spec) if "funding-rate-arbitrage" not in p
            and "research/STRATEGY_A_REPORT.md" not in p and "research/MARGIN_ANALYSIS.md" not in p]


# --------------------------------------------------------------------------
# The grid: a base and its experiment


def test_xsmom_experiment_and_base_make_the_two_candidate_grid_of_the_hand_made_twin(
        xsmom_base, xsmom_experiment):
    result = causal_twin(xsmom_experiment, base=xsmom_base)
    assert result.reason is None and result.kind == "wrapper"
    by_hand = _load("research", "xsmom", "xsmom-research-nobull-retune.yaml")
    twin = result.spec
    assert twin.code_ref == by_hand.code_ref
    for key in ("inner_code_ref", "base_params", "window", "objective", "score_costs"):
        assert twin.params[key] == by_hand.params[key], key
    # The same two candidates; only the experiment's label differs (skip_bull by hand).
    assert list(twin.params["grid"]) == list(by_hand.params["grid"]) == ["gate"]
    assert list(twin.params["grid"]["gate"].values()) == list(
        by_hand.params["grid"]["gate"].values())
    assert twin.params["refit_every"] == "M"
    assert (twin.data.source, twin.data.interval, twin.data.start, twin.data.end) == (
        by_hand.data.source, by_hand.data.interval, by_hand.data.start, by_hand.data.end)
    assert twin.costs == xsmom_experiment.costs
    assert twin.idea_id == "xsmom-research-nobull-causal"
    assert twin.selects_causally


def test_the_candidates_are_the_base_unchanged_and_the_experiment(xsmom_base, xsmom_experiment):
    twin = causal_twin(xsmom_experiment, base=xsmom_base).spec
    candidates = expand_grid(twin.params["base_params"], twin.params["grid"])
    assert set(candidates) == {"always", "only_flat_bear"}
    assert candidates["only_flat_bear"] == xsmom_experiment.params
    assert candidates["always"]["inner_params"] == xsmom_base.params
    assert sorted(candidates["always"]["trade_in"]) == ["bear", "bull", "flat"]


def test_two_specs_of_the_same_code_differing_in_a_few_leaves_are_two_candidates(
        frab_base, frab_experiment):
    result = causal_twin(frab_experiment, base=frab_base)
    assert result.reason is None and result.kind == "pair"
    twin = result.spec
    assert twin.params["inner_code_ref"] == frab_base.code_ref
    assert twin.params["grid"] == {"variant": {"base": {}, "experiment": {
        "signal_window_hours": 24}}}
    candidates = expand_grid(twin.params["base_params"], twin.params["grid"])
    assert candidates["base"] == frab_base.params
    assert candidates["experiment"] == frab_experiment.params
    assert result.candidates == ("base", "experiment")


def test_a_dropped_key_is_said_by_replacing_the_dict_that_held_it():
    base = _load("research", "frab", "frab-live-own-spot.yaml")
    experiment = _load("research", "frab", "frab-research-no-avax.yaml")
    result = causal_twin(experiment, base=base)
    assert result.reason is None
    overrides = result.spec.params["grid"]["variant"]["experiment"]
    assert set(overrides) == {"universe", "leverage"}
    candidates = expand_grid(result.spec.params["base_params"], result.spec.params["grid"])
    assert candidates["experiment"] == experiment.params
    assert "AVAX" not in candidates["experiment"]["leverage"]


def test_a_difference_in_the_size_of_the_money_is_not_a_choice_for_code(frab_base):
    spot = _load("frab-binance-spot.yaml")
    result = causal_twin(frab_base, base=spot)
    assert result.spec is None
    assert "budget_cap_usdc" in result.reason and "owner chooses the size" in result.reason


def test_an_experiment_on_other_data_than_its_base_is_not_a_pair():
    own = _load("research", "frab", "frab-live-own-spot.yaml")
    live = _load("frab-live.yaml")
    result = causal_twin(own, base=live)
    assert result.spec is None and "spot_prices" in result.reason


def test_a_wrapper_over_something_else_than_the_base_is_not_a_pair(xsmom_experiment, xsmom_base):
    inner = {k: v for k, v in xsmom_experiment.params["inner_params"].items()
             if k != "rebalance_days"}
    without_lists = xsmom_experiment.model_copy(update={
        "params": {**xsmom_experiment.params, "inner_params": inner}})
    other = xsmom_base.model_copy(update={"code_ref": "qlab.strategies.trend:Trend"})
    result = causal_twin(without_lists, base=other)
    assert result.spec is None
    assert "wraps" in result.reason and NO_ALTERNATIVE in result.reason


def test_the_trend_gate_is_the_same_pair_as_the_xsmom_gate():
    result = causal_twin(_load("trend-binance-cap-gated.yaml"),
                         base=_load("trend-binance-cap.yaml"))
    assert result.reason is None
    assert set(result.candidates) == {"always", "only_bull_bear"}
    assert result.spec.data.source == "binance"
    assert result.spec.data.start == BINANCE_ARCHIVE_START


# --------------------------------------------------------------------------
# The grid: the declared lists


def test_a_parameter_with_a_declared_list_gets_the_whole_list(xsmom_base):
    result = causal_twin(xsmom_base)
    assert result.reason is None and result.kind == "lists"
    assert result.candidates == ("rebalance_days=1", "rebalance_days=7")
    grid = result.spec.params["grid"]
    assert list(grid) == ["rebalance_days"]
    assert {v["rebalance_days"] for v in grid["rebalance_days"].values()} == {1, 7}
    assert result.spec.params["inner_code_ref"] == xsmom_base.code_ref
    assert _own_problems(result.spec) == []


def test_a_gate_over_a_strategy_with_a_declared_list_looks_inside_the_gate(xsmom_experiment):
    result = causal_twin(xsmom_experiment)
    assert result.reason is None and result.kind == "lists"
    assert list(result.spec.params["grid"]) == ["inner_params.rebalance_days"]


def test_more_candidates_than_the_cap_is_refused_with_the_reason(xsmom_base):
    params = {**xsmom_base.params, "signal": {"window_days": 7, "lookback_days": 7}}
    wide = xsmom_base.model_copy(update={"params": params})
    result = causal_twin(wide)
    assert result.spec is None
    assert "32 candidates" in result.reason and "at most 8" in result.reason


def test_nothing_declared_and_no_base_is_no_twin():
    result = causal_twin(_load("trend-live-top25-cap.yaml"))
    assert result.spec is None
    assert result.reason == NO_ALTERNATIVE


def test_a_pair_that_fails_falls_through_to_the_lists_and_both_reasons_show(
        xsmom_base, xsmom_experiment):
    result = causal_twin(xsmom_base, base=xsmom_experiment)
    assert result.reason is None and result.kind == "lists"
    spot = causal_twin(_load("research", "frab-binance-spot", "frab-binance-spot-300.yaml"),
                       base=_load("frab-binance-spot.yaml"))
    assert spot.spec is None
    assert "no pair with the base" in spot.reason and NO_ALTERNATIVE in spot.reason


# --------------------------------------------------------------------------
# Who gets no twin


def test_a_spec_that_selects_causally_gets_no_twin():
    spec = _load("research", "xsmom", "xsmom-research-nobull-retune.yaml")
    result = causal_twin(spec)
    assert result.spec is None and "already chooses its numbers by code" in result.reason


def test_a_retune_spec_gets_no_twin_even_without_the_flag():
    spec = _load("trend-retune-weekly.yaml")
    flagless = spec.model_copy(update={"selects_causally": False})
    assert causal_twin(flagless).spec is None


def test_the_twin_of_a_twin_is_refused(xsmom_base):
    twin = causal_twin(xsmom_base).spec
    again = causal_twin(twin)
    assert again.spec is None and "already" in again.reason
    renamed = twin.model_copy(update={"selects_causally": False, "code_ref": xsmom_base.code_ref})
    assert causal_twin(renamed).spec is None


def test_a_venue_without_a_binance_counterpart_gets_no_twin(xsmom_base):
    data = xsmom_base.data.model_copy(update={"source": "aster"})
    result = causal_twin(xsmom_base.model_copy(update={"data": data}))
    assert result.spec is None and "aster" in result.reason


def test_a_first_choice_after_the_end_of_the_data_is_refused(xsmom_base):
    data = xsmom_base.data.model_copy(update={"source": "binance", "end": date(2019, 12, 31)})
    result = causal_twin(xsmom_base.model_copy(update={"data": data}))
    assert result.spec is None and "no day would be a forward test" in result.reason


# --------------------------------------------------------------------------
# The twin as a spec


def test_the_twin_is_a_valid_spec_with_its_sources_and_survives_the_file(
        tmp_path, xsmom_base, xsmom_experiment):
    twin = causal_twin(xsmom_experiment, base=xsmom_base).spec
    assert StrategySpec.model_validate(twin.model_dump()) == twin
    path = tmp_path / "twin.yaml"
    path.write_text(twin_yaml(twin, xsmom_experiment.idea_id), encoding="utf-8")
    assert load_spec(path) == twin
    assert _own_problems(twin) == []
    assert twin.params["score_costs"] == {"taker_fee_bps": twin.costs.taker_fee_bps,
                                          "slippage_bps": twin.costs.slippage_bps}


def test_the_twin_keeps_what_the_original_declares(frab_experiment, frab_base):
    twin = causal_twin(frab_experiment, base=frab_base).spec
    assert twin.min_leg_notional == frab_experiment.min_leg_notional
    assert twin.simultaneous_legs == frab_experiment.simultaneous_legs
    assert twin.unexpressed_mechanisms == frab_experiment.unexpressed_mechanisms
    assert twin.data.interval == frab_experiment.data.interval
    assert twin.data.include_spot and twin.data.spot_prices == "own"
    assert twin.data.spot_source is None and twin.data.source == "binance"
    assert twin.data.end == frab_experiment.data.end
    assert twin.costs == frab_experiment.costs
    assert twin.title.startswith(frab_experiment.title)
    assert "числа выбирает код раз в месяц" in twin.title
    assert twin.exploratory is None and twin.judged_from is None


def test_the_first_choice_is_after_the_start_the_warm_up_and_a_whole_refit_period(
        xsmom_base, xsmom_experiment, frab_base, frab_experiment):
    pairs = [(causal_twin(xsmom_experiment, base=xsmom_base).spec, xsmom_base.params),
             (causal_twin(frab_experiment, base=frab_base).spec, frab_base.params),
             (causal_twin(xsmom_base).spec, xsmom_base.params)]
    for twin, params in pairs:
        warm, _ = warmup_days([params], twin.data.interval)
        assert twin.params_fixed_at >= twin.data.start + timedelta(days=warm + 28)
        assert twin.params_fixed_at.day == 1  # a monthly refit
        assert twin.params_fixed_at < twin.data.end


def test_the_xsmom_twin_is_no_earlier_than_arithmetic_and_no_later_than_the_hand_made_one(
        xsmom_base, xsmom_experiment):
    twin = causal_twin(xsmom_experiment, base=xsmom_base).spec
    by_hand = _load("research", "xsmom", "xsmom-research-nobull-retune.yaml")
    assert date(2020, 1, 1) <= twin.params_fixed_at <= by_hand.params_fixed_at
    evidence = twin.params_fixed_evidence
    assert "60 days" in evidence and "warm-up arithmetic" in evidence
    assert "does not prove" in evidence and "2026-06-15" in evidence


def test_a_hyperliquid_spec_that_names_instruments_needs_the_binance_listing(xsmom_base):
    data = xsmom_base.data.model_copy(update={"instruments": ["BTC", "ETH", "SOL"]})
    named = xsmom_base.model_copy(update={"data": data})
    assert "venue_names" in causal_twin(named).reason
    mapped = causal_twin(named, venue_names=["BTC", "ETH"])
    assert mapped.reason is None
    assert mapped.spec.data.instruments == ["BTC", "ETH"]
    assert any("SOL" in note for note in mapped.spec.differs_from_source)


def test_required_instruments_keep_their_names_and_say_so_without_a_listing():
    twin = causal_twin(_load("research", "bv2", "bv2-research-shared-pool-hl.yaml"),
                       base=_load("bv2-live.yaml")).spec
    assert twin.required_instruments == ["BTC-SPOT", "ETH-SPOT", "SOL-SPOT", "AVAX-SPOT",
                                         "BTC", "ETH", "SOL", "AVAX"]
    assert any("keep the original venue's names" in n for n in twin.differs_from_source)


def test_a_binance_spec_keeps_its_source_and_goes_back_to_the_archive_start():
    spec = _load("trend-binance-cap-gated.yaml")
    later = spec.model_copy(update={"data": spec.data.model_copy(
        update={"start": date(2023, 1, 1)})})
    twin = causal_twin(later, base=_load("trend-binance-cap.yaml")).spec
    assert twin.data.source == "binance" and twin.data.start == BINANCE_ARCHIVE_START
    assert "already Binance" in twin.params_fixed_evidence


def test_the_execution_venue_is_the_originals_when_the_spec_has_the_field(
        xsmom_base, xsmom_experiment):
    twin = causal_twin(xsmom_experiment, base=xsmom_base).spec
    if "execution_venue" in StrategySpec.model_fields:
        assert twin.execution_venue == "hyperliquid"


# --------------------------------------------------------------------------
# The first choice and the warm-up


def test_warmup_reads_days_hours_and_bars_and_takes_the_longest():
    assert warmup_days([{"lookbacks_days": [14, 60], "rebalance_days": 7}], "1d") == (
        60.0, "lookbacks_days = 60")
    days, label = warmup_days([{"cap_min_hold_hours": 720, "signal_window_hours": 8}], "1h")
    assert days == 30.0 and label == "cap_min_hold_hours = 720"
    days, label = warmup_days([{"lookbacks": [30, 120], "min_history_days": 151}], "1d")
    assert days == 151.0 and label == "min_history_days = 151"
    days, label = warmup_days([{"vol_window": 48}], "1h")
    assert days == 2.0 and label == "vol_window = 48"


def test_a_regime_gate_waits_for_the_detector_window():
    days, label = warmup_days([{"trade_in": ["bear"], "inner_params": {"rebalance_days": 7}}], "1d")
    assert days == 30.0 and "detector" in label


def test_first_choice_is_the_second_refit_after_the_book_can_trade():
    assert first_choice(date(2019, 9, 8), 60, "M") == (
        date(2019, 11, 7), date(2019, 12, 1), date(2020, 1, 1))
    assert first_choice(date(2019, 9, 8), 0, "M")[2] == date(2019, 11, 1)
    assert first_choice(date(2024, 1, 1), 0, "D")[2] == date(2024, 1, 3)
    # Retune's weeks start on Monday (W-SUN periods).
    active, first, choice = first_choice(date(2024, 1, 3), 0, "W")
    assert (first, choice) == (date(2024, 1, 8), date(2024, 1, 15))
    assert first_choice(date(2024, 11, 20), 20, "M")[1:] == (date(2025, 1, 1), date(2025, 2, 1))


# --------------------------------------------------------------------------
# Finding the base


def test_find_base_picks_the_spec_with_the_same_code_and_the_fewest_differences(
        frab_experiment):
    candidates = [_load("frab-binance-spot.yaml"),
                  _load("research", "frab-binance-spot", "frab-binance-spot-300.yaml"),
                  _load("research", "frab-binance-spot", "frab-binance-spot-400.yaml"),
                  _load("xsmom.yaml")]
    choice = find_base(frab_experiment, candidates)
    assert choice.reason is None and choice.base.idea_id == "frab-binance-spot-300"


def test_find_base_for_a_gate_is_the_spec_it_wraps(xsmom_experiment):
    choice = find_base(xsmom_experiment, [_load("xsmom.yaml"), _load("xsmom-honest-20legs.yaml"),
                                          _load("xsmom-frozen-universe.yaml")])
    assert choice.base.idea_id == "xsmom-honest-20legs"


def test_find_base_with_a_tie_is_no_base_and_says_which(frab_base):
    candidates = [_load("frab-binance-spot.yaml"),
                  _load("research", "frab-binance-spot", "frab-binance-spot-400.yaml")]
    choice = find_base(frab_base, candidates)
    assert choice.base is None
    assert "equally near" in choice.reason and "frab-binance-spot-400" in choice.reason


def test_find_base_without_the_same_code_is_no_base(xsmom_base):
    choice = find_base(xsmom_base, [_load("trend-live-top25-cap.yaml")])
    assert choice.base is None and "same code" in choice.reason


def test_find_base_never_offers_the_spec_itself_or_a_retune_spec(xsmom_base):
    retune = _load("research", "xsmom", "xsmom-research-nobull-retune.yaml")
    assert find_base(xsmom_base, [xsmom_base, retune]).base is None


# --------------------------------------------------------------------------
# Files and the command


def test_write_twin_writes_next_to_the_spec_and_never_overwrites_a_different_file(
        tmp_path, xsmom_base):
    spec_path = tmp_path / "xsmom-honest-20legs.yaml"
    twin = causal_twin(xsmom_base).spec
    written = write_twin(spec_path, twin, xsmom_base.idea_id)
    assert written == twin_path(spec_path) == tmp_path / "xsmom-honest-20legs-causal.yaml"
    assert load_spec(written) == twin
    assert write_twin(spec_path, twin, xsmom_base.idea_id) == written  # the same text: fine
    written.write_text("idea_id: other\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_twin(spec_path, twin, xsmom_base.idea_id)
    assert written.read_text(encoding="utf-8") == "idea_id: other\n"


def test_cli_prints_the_twin_yaml(xsmom_base):
    result = runner.invoke(app, ["twin", str(SPECS / "research" / "xsmom" /
                                             "xsmom-research-nobull.yaml"),
                                 "--base", str(SPECS / "xsmom-honest-20legs.yaml")])
    assert result.exit_code == 0, result.output
    assert "idea_id: xsmom-research-nobull-causal" in result.output
    assert "selects_causally: true" in result.output
    assert "specs/xsmom-honest-20legs.yaml" in result.output
    assert str(ROOT) not in result.output  # sources cite project paths, not this machine's


def test_cli_prints_the_reason_when_there_is_no_twin():
    result = runner.invoke(app, ["twin", str(SPECS / "trend-live-top25-cap.yaml")])
    assert result.exit_code == 0, result.output
    assert "no twin for trend-live-top25-cap" in result.output
    assert "picked by hand" in result.output and "--base" in result.output


def test_cli_write_makes_the_file_next_to_the_spec(tmp_path):
    spec = tmp_path / "x.yaml"
    spec.write_text((SPECS / "xsmom-honest-20legs.yaml").read_text(encoding="utf-8"),
                    encoding="utf-8")
    result = runner.invoke(app, ["twin", str(spec), "--write"])
    assert result.exit_code == 0, result.output
    target = tmp_path / "x-causal.yaml"
    assert target.is_file() and result.output.strip() == str(target)
    assert load_spec(target).idea_id == "xsmom-honest-20legs-causal"
    again = runner.invoke(app, ["twin", str(spec), "--write"])
    assert again.exit_code == 0
    target.write_text("idea_id: other\n", encoding="utf-8")
    refused = runner.invoke(app, ["twin", str(spec), "--write"])
    assert refused.exit_code == 1


def test_find_base_prefers_the_candidate_that_can_be_the_other_half_of_a_pair():
    """frab-research-no-avax is as near (2 numbers) to frab-live as to frab-live-own-spot, but
    only the second runs on the same data: a pair needs one data set, so it is the base."""
    experiment = _load("research", "frab", "frab-research-no-avax.yaml")
    live = _load("frab-live.yaml")
    own_spot = _load("research", "frab", "frab-live-own-spot.yaml")
    assert find_base(experiment, [live, own_spot]).base.idea_id == "frab-live-own-spot"
    assert find_base(experiment, [own_spot, live]).base.idea_id == "frab-live-own-spot"


def test_a_spec_that_leaves_numbers_open_is_twinned_on_its_declared_lists_not_as_a_pair():
    """bv2-research-partial-hedge offers partial_hedge_share as a list (night/own_numbers.yaml):
    the pair builder would carry the list into a number's place and the stand would refuse the
    twin; the whole declared list is its choice."""
    experiment = _load("research", "bv2", "bv2-research-partial-hedge.yaml")
    base = _load("bv2-binance.yaml")
    result = causal_twin(experiment, base=base)
    assert result.spec is not None and result.kind == "lists"
    assert set(result.candidates) == {"partial_hedge_share=0.5", "partial_hedge_share=1"}
    assert isinstance(result.spec.params["base_params"]["partial_hedge_share"], int | float)
