"""Tests for the `StrategySpec` YAML format: required fields, no fabricated
defaults for costs/min_leg_notional, and the `data` block's discovered-vs-
explicit-universe shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qlab.pipeline.spec import StrategySpec, load_spec

_BASE: dict[str, object] = {
    "idea_id": "toy-strategy",
    "title": "Toy strategy",
    "code_ref": "qlab.pipeline.test_evaluate:ToyStrategy",
    "params": {"weight": 0.5},
    "data": {
        "source": "hyperliquid",
        "interval": "1h",
        "start": "2026-01-01",
        "end": "2026-01-05",
    },
    "costs": {"taker_fee_bps": 3.5, "slippage_bps": 1.0},
    "min_leg_notional": 12.0,
    "unexpressed_mechanisms": [],
}


def _spec_dict(**overrides: object) -> dict:
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in _BASE.items()}
    merged.update(overrides)
    return merged


def test_valid_spec_parses() -> None:
    spec = StrategySpec.model_validate(_spec_dict())
    assert spec.idea_id == "toy-strategy"
    assert spec.data.instruments is None
    assert spec.costs.taker_fee_bps == 3.5
    assert spec.min_leg_notional == 12.0


def test_explicit_instruments_accepted() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    data["instruments"] = ["BTC", "ETH"]
    spec = StrategySpec.model_validate(_spec_dict(data=data))
    assert spec.data.instruments == ["BTC", "ETH"]


def test_min_daily_volume_usd_defaults_to_none() -> None:
    spec = StrategySpec.model_validate(_spec_dict())
    assert spec.data.min_daily_volume_usd is None


def test_min_daily_volume_usd_accepted_when_positive() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    data["min_daily_volume_usd"] = 1_000_000.0
    spec = StrategySpec.model_validate(_spec_dict(data=data))
    assert spec.data.min_daily_volume_usd == 1_000_000.0


def test_min_daily_volume_usd_rejects_non_positive() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    data["min_daily_volume_usd"] = 0.0
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(_spec_dict(data=data))


def test_spot_prices_default_to_the_perp() -> None:
    assert StrategySpec.model_validate(_spec_dict()).data.spot_prices == "perp"


def test_own_spot_prices_need_spot_columns_and_a_known_choice() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="asks for none"):
        StrategySpec.model_validate(_spec_dict(data={**data, "spot_prices": "own"}))
    own = StrategySpec.model_validate(
        _spec_dict(data={**data, "spot_prices": "own", "include_spot": True}))
    assert own.data.spot_prices == "own"
    listed = StrategySpec.model_validate(
        _spec_dict(data={**data, "spot_prices": "own", "instruments": ["BTC", "BTC-SPOT"]}))
    assert listed.data.spot_prices == "own"
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(_spec_dict(data={**data, "spot_prices": "mid"}))


def test_missing_costs_fails_validation() -> None:
    raw = _spec_dict()
    del raw["costs"]
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_missing_min_leg_notional_fails_validation() -> None:
    raw = _spec_dict()
    del raw["min_leg_notional"]
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_incomplete_costs_block_fails_validation() -> None:
    raw = _spec_dict(costs={"taker_fee_bps": 3.5})
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_negative_min_leg_notional_fails_validation() -> None:
    raw = _spec_dict(min_leg_notional=-5.0)
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_zero_min_leg_notional_fails_validation() -> None:
    raw = _spec_dict(min_leg_notional=0.0)
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_simultaneous_legs_defaults_to_one() -> None:
    spec = StrategySpec.model_validate(_spec_dict())
    assert spec.simultaneous_legs == 1


def test_simultaneous_legs_explicit_value_accepted() -> None:
    spec = StrategySpec.model_validate(_spec_dict(simultaneous_legs=2))
    assert spec.simultaneous_legs == 2


def test_simultaneous_legs_zero_fails_validation() -> None:
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(_spec_dict(simultaneous_legs=0))


def test_simultaneous_legs_negative_fails_validation() -> None:
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(_spec_dict(simultaneous_legs=-1))


def test_unknown_field_rejected() -> None:
    raw = _spec_dict(unexpected_field="nope")
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_load_spec_from_yaml_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "toy.yaml"
    spec_path.write_text(yaml.safe_dump(_spec_dict()), encoding="utf-8")
    spec = load_spec(spec_path)
    assert spec.idea_id == "toy-strategy"


def test_load_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_spec(tmp_path / "nope.yaml")


def test_load_spec_missing_costs_raises(tmp_path: Path) -> None:
    raw = _spec_dict()
    del raw["costs"]
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_spec(spec_path)


def test_unexpressed_mechanisms_is_required() -> None:
    """No default: a spec must say what it leaves out, or say it leaves out nothing."""
    raw = _spec_dict()
    del raw["unexpressed_mechanisms"]
    with pytest.raises(ValidationError, match="unexpressed_mechanisms"):
        StrategySpec.model_validate(raw)


def test_blank_unexpressed_mechanism_rejected() -> None:
    with pytest.raises(ValidationError, match="named"):
        StrategySpec.model_validate(_spec_dict(unexpressed_mechanisms=["  "]))


def test_required_instruments_defaults_to_none() -> None:
    assert StrategySpec.model_validate(_spec_dict()).required_instruments is None


def test_empty_required_instruments_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        StrategySpec.model_validate(_spec_dict(required_instruments=[]))


def test_exploratory_must_say_why() -> None:
    with pytest.raises(ValidationError, match="say why"):
        StrategySpec.model_validate(_spec_dict(exploratory="   "))


def test_spot_source_defaults_to_the_sources_own_markets() -> None:
    assert StrategySpec.model_validate(_spec_dict()).data.spot_source is None


def test_a_spot_source_is_another_venue_priced_by_its_own_closes() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    mixed = {**data, "include_spot": True, "spot_prices": "own", "spot_source": "binance"}
    assert StrategySpec.model_validate(_spec_dict(data=mixed)).data.spot_source == "binance"
    with pytest.raises(ValidationError, match="spot_prices: own"):
        StrategySpec.model_validate(_spec_dict(data={**mixed, "spot_prices": "perp"}))
    with pytest.raises(ValidationError, match="the source itself"):
        StrategySpec.model_validate(_spec_dict(data={**mixed, "spot_source": "hyperliquid"}))
    with pytest.raises(ValidationError, match="asks for none"):
        StrategySpec.model_validate(_spec_dict(data={**mixed, "include_spot": False}))


def test_execution_venue_defaults_to_the_data_source() -> None:
    assert StrategySpec.model_validate(_spec_dict()).execution_venue is None


def test_an_execution_venue_names_where_the_orders_go() -> None:
    data = {**_BASE["data"], "source": "binance"}  # type: ignore[dict-item]
    spec = StrategySpec.model_validate(_spec_dict(data=data, execution_venue="hyperliquid"))
    assert spec.execution_venue == "hyperliquid" and spec.data.source == "binance"
    with pytest.raises(ValidationError, match="must name a venue"):
        StrategySpec.model_validate(_spec_dict(execution_venue="  "))


def test_an_execution_venue_cannot_be_combined_with_a_spot_source() -> None:
    data = dict(_BASE["data"])  # type: ignore[arg-type]
    mixed = {**data, "include_spot": True, "spot_prices": "own", "spot_source": "binance"}
    with pytest.raises(ValidationError, match="already says where its orders go"):
        StrategySpec.model_validate(_spec_dict(data=mixed, execution_venue="hyperliquid"))


def test_the_two_leg_binance_data_proxies_say_their_orders_go_to_hyperliquid() -> None:
    """The specs whose comments call Binance a DATA proxy for Hyperliquid and that hold two
    legs: without the field their `atomic_execution` is read from Binance, which names no
    recovery mechanism, and they route `needs-more-data` for ever."""
    repo = Path(__file__).resolve().parents[3]
    proxies = [
        "specs/bv2-binance.yaml",
        "specs/research/bv2/bv2-research-btceth.yaml",
        "specs/research/bv2/bv2-research-headroom-retune-binance.yaml",
        "specs/research/bv2/bv2-research-partial-hedge.yaml",
        "specs/research/bv2/bv2-research-shared-pool.yaml",
        "specs/research/bv2/bv2-research-short-window.yaml",
        *(p.relative_to(repo).as_posix()
          for p in sorted((repo / "specs/bv2-owner-ideas").glob("*.yaml"))),
    ]
    for name in proxies:
        spec = load_spec(repo / name)
        assert spec.data.source == "binance" and spec.simultaneous_legs == 2, name
        assert spec.execution_venue == "hyperliquid", name


def test_the_binance_spot_strategy_runs_at_the_owners_three_budgets() -> None:
    """Owner, 2026-10-25: «если нужно 300-400-1000 - давай считать на них». The main spec
    is 1000, the variants 300 and 400, four slots each; apart from the idea and the
    budget they are one spec, and the budget's source is his quote."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    paths = {
        1000: repo / "specs/frab-binance-spot.yaml",
        300: repo / "specs/research/frab-binance-spot/frab-binance-spot-300.yaml",
        400: repo / "specs/research/frab-binance-spot/frab-binance-spot-400.yaml",
    }
    specs = {budget: load_spec(path) for budget, path in paths.items()}
    for budget, spec in specs.items():
        assert spec.params["budget_cap_usdc"] == budget and spec.params["concurrency_cap"] == 4
        assert "блин 90$" in spec.sources["budget_cap_usdc"]
        assert "300-400-1000" in (spec.params_fixed_evidence or "")
        assert str(spec.params_fixed_at) == "2026-10-10"
        assert spec.params["transfers"] == "batched"
    rest = [{k: v for k, v in s.params.items() if k != "budget_cap_usdc"} for s in specs.values()]
    assert rest[0] == rest[1] == rest[2]
    assert specs[1000].data == specs[300].data == specs[400].data


def test_the_all_binance_variant_is_one_spec_at_three_budgets() -> None:
    from pathlib import Path

    folder = Path(__file__).resolve().parents[3] / "specs/research/frab-binance-spot"
    names = ["frab-binance-spot-allbinance", "frab-binance-spot-allbinance-300",
             "frab-binance-spot-allbinance-400"]
    specs = [load_spec(folder / f"{n}.yaml") for n in names]
    assert [s.idea_id for s in specs] == names
    assert [s.params["budget_cap_usdc"] for s in specs] == [1000, 300, 400]
    for s in specs:
        assert s.data.source == "binance" and s.data.spot_source is None
        assert s.code_ref.endswith(":LiveFrabAllBinance")
        assert s.params["binance_withdrawal_fee_usdc"] == 0
        assert s.params["transfer_delay_hours"] == 0
