"""Tests for `qlab.pipeline.variants` (docs/TASKS.md T25)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qlab.pipeline.spec import load_spec
from qlab.pipeline.variants import (
    SPLIT_CADENCE_CODE_REF,
    split_cadence_variant,
    timeframe_variants,
)

BASE = {
    "idea_id": "trend-variants",
    "title": "Trend",
    "code_ref": "qlab.strategies.trend:TrendTSMOMEnsemble",
    "params": {
        "lookbacks_days": [30, 60],
        "vol_window_days": 30,
        "vol_target_daily": 0.02,
        "leverage_cap": 3.0,
        "risk_scale": 0.2,
        "min_history_days": 61,
    },
    "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01", "end": "2025-06-01"},
    "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9},
    "min_leg_notional": 10.0,
    "unexpressed_mechanisms": [],
}


@pytest.fixture()
def base(tmp_path) -> Path:
    path = tmp_path / "trend.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    return path


def test_one_variant_per_interval_with_everything_else_unchanged(base, tmp_path) -> None:
    paths = timeframe_variants(base, ["1d", "4h", "1h"], tmp_path)

    assert [p.name for p in paths] == ["trend-tf-1d.yaml", "trend-tf-4h.yaml", "trend-tf-1h.yaml"]
    specs = [load_spec(p) for p in paths]
    assert [s.data.interval for s in specs] == ["1d", "4h", "1h"]
    assert all(s.params == BASE["params"] and s.idea_id == BASE["idea_id"] for s in specs)
    header = paths[0].read_text(encoding="utf-8")
    assert "declared before any member is run" in header and "trend-tf-1h" in header


def test_interval_the_strategy_does_not_declare_is_refused(base, tmp_path) -> None:
    with pytest.raises(ValueError, match="not valid on '15m'"):
        timeframe_variants(base, ["1d", "15m"], tmp_path)
    assert list(tmp_path.glob("trend-tf-*.yaml")) == []  # nothing half-written


def test_strategy_without_declaration_cannot_be_varied(tmp_path) -> None:
    raw = {**BASE, "code_ref": "qlab.pipeline.test_evaluate:ToyStrategy"}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="declares no valid_intervals"):
        timeframe_variants(path, ["1h"], tmp_path)


def test_split_cadence_variant_wraps_the_base_strategy(base, tmp_path) -> None:
    path = split_cadence_variant(base, fast="1h", entry_every="1D", exit_every="1h",
                                 out_dir=tmp_path)
    spec = load_spec(path)

    assert path.name == "trend-split-1D-1h.yaml"
    assert spec.code_ref == SPLIT_CADENCE_CODE_REF
    assert spec.data.interval == "1h"
    assert spec.params["inner_code_ref"] == BASE["code_ref"]
    assert spec.params["inner_params"] == BASE["params"]
    assert (spec.params["entry_every"], spec.params["exit_every"]) == ("1D", "1h")
