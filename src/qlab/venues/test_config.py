"""Tests for `qlab.venues.config`: YAML -> `VenueConfig`, and the
"missing file -> None, never a guess" contract `load_venue` exists to keep.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qlab.venues.config import VenueConfig, load_venue


def _write_yaml(directory: Path, venue_id: str, content: dict) -> None:
    (directory / f"{venue_id}.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")


def test_load_venue_parses_full_config(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path,
        "testvenue",
        {
            "id": "testvenue",
            "title": "Test Venue",
            "execution_adapter": True,
            "min_leg_notional_usd": 10.0,
            "free_public_data": True,
            "supports_atomic_multileg": False,
            "uncovered_leg_recovery": "some named mechanism",
        },
    )

    venue = load_venue("testvenue", venues_dir=tmp_path)

    assert venue is not None
    assert venue.id == "testvenue"
    assert venue.title == "Test Venue"
    assert venue.execution_adapter is True
    assert venue.min_leg_notional_usd == 10.0
    assert venue.free_public_data is True
    assert venue.supports_atomic_multileg is False
    assert venue.uncovered_leg_recovery == "some named mechanism"


def test_load_venue_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_venue("nope", venues_dir=tmp_path) is None


def test_load_venue_unset_fields_default_to_none(tmp_path: Path) -> None:
    """A field simply left out of the YAML must come back `None`
    ("unknown") -- never a fabricated default -- exactly like `StrategySpec`
    refuses to default costs or `min_leg_notional`."""
    _write_yaml(tmp_path, "bare", {"id": "bare", "title": "Bare Venue"})

    venue = load_venue("bare", venues_dir=tmp_path)

    assert venue is not None
    assert venue.execution_adapter is None
    assert venue.min_leg_notional_usd is None
    assert venue.free_public_data is None
    assert venue.supports_atomic_multileg is None
    assert venue.uncovered_leg_recovery is None


def test_load_venue_rejects_unknown_field(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path, "typo", {"id": "typo", "title": "Typo Venue", "executon_adapter": True}
    )

    with pytest.raises(ValidationError):
        load_venue("typo", venues_dir=tmp_path)


def test_load_venue_rejects_blank_id() -> None:
    with pytest.raises(ValidationError):
        VenueConfig.model_validate({"id": "  ", "title": "Something"})


def test_load_venue_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        VenueConfig.model_validate({"id": "x", "title": ""})


def test_load_venue_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    (tmp_path / "listy.yaml").write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_venue("listy", venues_dir=tmp_path)


def test_venue_config_is_frozen() -> None:
    venue = VenueConfig(id="x", title="X")
    with pytest.raises(ValidationError):
        venue.execution_adapter = True  # type: ignore[misc]


def test_real_hyperliquid_config_loads_from_repo_root() -> None:
    """The actual `venues/hyperliquid.yaml` this task ships must itself be
    a valid, fully-specified `VenueConfig` -- every field this task
    requires a citation for is present."""
    venue = load_venue("hyperliquid")

    assert venue is not None
    assert venue.execution_adapter is True
    assert venue.free_public_data is True
    assert venue.supports_atomic_multileg is False
    assert venue.uncovered_leg_recovery is not None
    assert venue.min_leg_notional_usd is not None


def test_real_binance_config_loads_from_repo_root() -> None:
    venue = load_venue("binance")

    assert venue is not None
    assert venue.execution_adapter is False
    # Deliberately left unset -- no execution adapter was ever built to
    # observe atomicity or recovery behaviour from (see venues/binance.yaml).
    assert venue.supports_atomic_multileg is None
    assert venue.uncovered_leg_recovery is None


def test_unconfigured_venue_returns_none_from_repo_root() -> None:
    assert load_venue("no-such-venue-at-all") is None
