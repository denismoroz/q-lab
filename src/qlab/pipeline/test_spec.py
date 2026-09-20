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
