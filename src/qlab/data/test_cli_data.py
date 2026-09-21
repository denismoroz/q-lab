"""CLI tests for the `qlab data` survivorship-fix behavior:
`fetch --instruments` optionality/warning and `data universe`.

Lives under `qlab/data` (not `qlab/cli/test_cli.py`) per this task's scope:
only `cli/__init__.py` itself is touched for the two new commands, and the
data layer's own tests directory is where the fix for the "hand-typed
instrument list survivorship gap" is exercised end to end, CLI included.
`build_snapshot`/`describe_universe` are monkeypatched on the `qlab.cli`
module (where they were bound at import time) so no network or real
registry DB is touched.
"""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

import qlab.cli as cli_module
from qlab.data.panel import MarketPanel

runner = CliRunner()


def _fake_panel(instruments: list[str], universe_complete: bool) -> MarketPanel:
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    prices = pd.DataFrame(1.0, index=index, columns=instruments)
    funding = pd.DataFrame(0.0001, index=index, columns=instruments)
    tradeable = pd.DataFrame(True, index=index, columns=instruments)
    return MarketPanel(
        snapshot_id="f" * 64,
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta={"venue": "hyperliquid", "interval": "1h", "universe_complete": universe_complete},
    )


def test_data_fetch_without_instruments_uses_full_discovered_universe(monkeypatch):
    captured = {}

    def fake_build_snapshot(source, instruments, start, end, interval, **kwargs):
        captured["instruments"] = instruments
        return _fake_panel(["BTC", "DEADCOIN", "ETH"], universe_complete=True)

    monkeypatch.setattr(cli_module, "build_snapshot", fake_build_snapshot)

    result = runner.invoke(
        cli_module.app,
        [
            "data", "fetch", "--source", "hyperliquid",
            "--start", "2024-01-01", "--end", "2024-01-02",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["instruments"] is None  # None reaches build_snapshot, not a survivor list
    assert "universe_complete: True" in result.output
    assert "DEADCOIN" in result.output


def test_data_fetch_with_instruments_is_marked_incomplete_and_warns(monkeypatch):
    def fake_build_snapshot(source, instruments, start, end, interval, **kwargs):
        return _fake_panel(list(instruments), universe_complete=False)

    monkeypatch.setattr(cli_module, "build_snapshot", fake_build_snapshot)

    result = runner.invoke(
        cli_module.app,
        [
            "data", "fetch", "--source", "hyperliquid",
            "--instruments", "BTC,ETH",
            "--start", "2024-01-01", "--end", "2024-01-02",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "universe_complete: False" in result.output
    assert "honest_universe" in result.output  # the warning names the rule it will fail


def test_data_fetch_propagates_no_discovery_error_cleanly(monkeypatch):
    def fake_build_snapshot(*args, **kwargs):
        raise ValueError("binance does not expose a full point-in-time universe via its free API")

    monkeypatch.setattr(cli_module, "build_snapshot", fake_build_snapshot)

    result = runner.invoke(
        cli_module.app,
        ["data", "fetch", "--source", "binance", "--start", "2024-01-01", "--end", "2024-01-02"],
    )

    assert result.exit_code != 0
    assert "does not expose a full point-in-time universe" in result.output


def test_data_universe_prints_delisted_flag(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "describe_universe",
        lambda source, **kwargs: [("BTC", False), ("LUNA", True)],
    )
    result = runner.invoke(cli_module.app, ["data", "universe", "--source", "hyperliquid"])
    assert result.exit_code == 0, result.output
    assert "LUNA" in result.output
    assert "DELISTED" in result.output
    assert "total: 2 instruments (1 delisted)" in result.output


def test_data_universe_none_exits_nonzero_with_explanation(monkeypatch):
    monkeypatch.setattr(cli_module, "describe_universe", lambda source, **kwargs: None)
    result = runner.invoke(cli_module.app, ["data", "universe", "--source", "binance"])
    assert result.exit_code != 0
    assert "no survivorship-free universe" in result.output


def test_data_universe_calls_describe_universe_with_include_spot(monkeypatch):
    captured = {}

    def fake_describe_universe(source, **kwargs):
        captured.update(kwargs)
        return [("BTC", False), ("BTC-SPOT", False)]

    monkeypatch.setattr(cli_module, "describe_universe", fake_describe_universe)
    result = runner.invoke(cli_module.app, ["data", "universe", "--source", "hyperliquid"])
    assert result.exit_code == 0, result.output
    assert captured.get("include_spot") is True
    assert "BTC-SPOT" in result.output


def test_data_fetch_include_spot_flag_reaches_build_snapshot(monkeypatch):
    captured = {}

    def fake_build_snapshot(source, instruments, start, end, interval, **kwargs):
        captured.update(kwargs)
        return _fake_panel(["BTC", "BTC-SPOT"], universe_complete=True)

    monkeypatch.setattr(cli_module, "build_snapshot", fake_build_snapshot)

    result = runner.invoke(
        cli_module.app,
        [
            "data", "fetch", "--source", "hyperliquid",
            "--start", "2024-01-01", "--end", "2024-01-02", "--include-spot",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("include_spot") is True
