"""Every strategy family declares whether it puts its profit back into positions, and that
declaration is what makes its percent a sum or a compounding (docs/CAPITAL_BASE.md). This file
is the list the owner reads in docs/CAPITAL_BASE.md, as a test."""

from __future__ import annotations

import pytest

from qlab.harness.strategy import profit_reinvested
from qlab.pipeline.evaluate import resolve_strategy

#: code_ref -> (reinvests its profit, why)
DECLARED = {
    "qlab.strategies.live.bv2:LiveBv2": False,  # the ratchet sets cash aside for good
    "qlab.strategies.bv2:BStrategyV2": False,  # its transcription
    "qlab.strategies.live.frab:LiveFrab": False,  # a constant budget
    "qlab.strategies.live.frab_binance_spot:LiveFrabBinanceSpot": False,  # inherits
    "qlab.strategies.live.frab_binance_spot:LiveFrabAllBinance": False,  # inherits
    "qlab.strategies.frab:FrabFundingHarvest": False,  # a constant budget
    "qlab.strategies.xsmom:XsmomCrossSectionalMomentum": False,  # live: a fixed budget
    "qlab.strategies.trend:TrendTSMOMEnsemble": True,  # the paper book resizes to all its money
    "qlab.strategies.live.trend:LiveTrendTSMOMEnsemble": True,
    "qlab.strategies.bab:BettingAgainstBeta": True,  # no live engine: the stand's default
    "qlab.strategies.blocks:Composed": True,  # generated ideas: the stand's default
}


@pytest.mark.parametrize("code_ref,expected", sorted(DECLARED.items()))
def test_each_strategy_family_declares_its_choice(code_ref: str, expected: bool) -> None:
    strategy = resolve_strategy(code_ref)
    assert profit_reinvested(strategy, {}) is expected
    # an explicit declaration, not the default falling through
    assert hasattr(type(strategy), "reinvests_profit") or hasattr(strategy, "reinvests_profit")


def test_a_strategy_that_declares_nothing_keeps_the_stand_s_compounding() -> None:
    class Nothing:
        name = "nothing"

    assert profit_reinvested(Nothing(), {}) is True


def test_a_wrapper_has_the_choice_of_the_strategy_it_wraps() -> None:
    inner = {"inner_code_ref": "qlab.strategies.live.bv2:LiveBv2", "inner_params": {}}
    assert profit_reinvested(resolve_strategy("qlab.strategies.regime_gate:RegimeGate"),
                             inner) is False
    assert profit_reinvested(resolve_strategy("qlab.strategies.cadence:SplitCadence"),
                             inner) is False
    trend = {"inner_code_ref": "qlab.strategies.trend:TrendTSMOMEnsemble", "inner_params": {}}
    assert profit_reinvested(resolve_strategy("qlab.strategies.regime_gate:RegimeGate"),
                             trend) is True
    assert profit_reinvested(resolve_strategy("qlab.strategies.overlays.core:Overlaid"),
                             {**inner, "overlays": []}) is False
    retune = {"inner_code_ref": "qlab.strategies.xsmom:XsmomCrossSectionalMomentum",
              "base_params": {}}
    assert profit_reinvested(resolve_strategy("qlab.strategies.retune:Retune"), retune) is False


def test_a_noise_book_is_read_on_the_base_and_the_choice_of_the_strategy_it_imitates() -> None:
    noise = resolve_strategy("qlab.calibration.noise:NoiseStrategy")
    params = {"reference_code_ref": "qlab.strategies.live.bv2:LiveBv2", "reference_params": {}}
    assert profit_reinvested(noise, params) is False
    params = {"reference_code_ref": "qlab.strategies.trend:TrendTSMOMEnsemble",
              "reference_params": {}}
    assert profit_reinvested(noise, params) is True
