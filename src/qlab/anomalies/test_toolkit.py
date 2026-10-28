"""The investigator's toolkit on the toy book: what it prints is right, and it writes nothing
(docs/ANOMALIES.md). The registry is not needed: a toolkit function takes the stored run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from qlab.anomalies import toolkit, toy
from qlab.anomalies.rerun import Rule, reselect

PUMP_DAY = "2026-01-21"  # bar 20: the decision day of the +200% move


def test_contributions_put_the_pump_on_top_with_the_numbers_of_the_book() -> None:
    out = toolkit.contributions(None, 1, run=toy.toy_run(), top=5)

    first = out.data["rows"][0]
    assert (first["instrument"], first["day"], first["side"]) == ("PUMP", PUMP_DAY, "long")
    assert first["weight"] == pytest.approx(0.1)
    assert first["price_move"] == pytest.approx(2.0)
    assert first["net"] == pytest.approx(0.2 - 0.00001)
    assert first["volume_usd"] == pytest.approx(2e5)
    assert out.data["reconciles"] is True
    assert out.data["book"][PUMP_DAY] > 0.19
    assert "DECISION day" in out.text and "reconciles with the stand's run: True" in out.text


def test_contributions_can_be_narrowed_to_days_a_window_or_a_coin() -> None:
    run = toy.toy_run()

    days = toolkit.contributions(None, 1, run=run, days=toolkit.parse_days(PUMP_DAY), top=50)
    window = toolkit.contributions(None, 1, run=run, start="2026-01-30", end="2026-01-31",
                                   top=50)
    coin = toolkit.contributions(None, 1, run=run, instrument="DEAD", top=50)

    assert {r["day"] for r in days.data["rows"]} == {PUMP_DAY}
    assert {r["day"] for r in window.data["rows"]} <= {"2026-01-30", "2026-01-31"}
    assert {r["instrument"] for r in coin.data["rows"]} == {"DEAD"}
    absent = toolkit.contributions(None, 1, run=run, instrument="NOPE")
    assert "never held" in absent.text and absent.data["rows"] == []


def test_a_table_never_runs_past_the_row_limit() -> None:
    out = toolkit.contributions(None, 1, run=toy.toy_run(), top=10_000)

    assert len(out.data["rows"]) <= toolkit.MAX_ROWS


def test_coin_reports_listing_facts_and_the_other_venue(monkeypatch) -> None:
    run = toy.toy_run()
    other = toy.toy_panel()
    other = type(other)(snapshot_id="hl", prices=other.prices[["A", "PUMP"]],
                        funding=other.funding[["A", "PUMP"]],
                        tradeable=other.tradeable[["A", "PUMP"]],
                        volume=other.volume[["A", "PUMP"]], meta={})
    monkeypatch.setattr(toolkit, "listing_panel", lambda session, stored: other)

    dead = toolkit.coin(None, 1, "DEAD", run=run)
    pump = toolkit.coin(None, 1, "PUMP", start="2026-01-20", end="2026-01-22", run=run)

    assert dead.data["stopped"] is True and dead.data["last_price"] == "2026-01-31"
    assert "NOT listed on Hyperliquid" in dead.text
    assert pump.data["hyperliquid_name"] == "PUMP" and "listed on Hyperliquid" in pump.text
    # The weight set on bar 20 earns the NEXT bar's move: the table says so and shows both.
    row = next(line for line in pump.lines if line.startswith("2026-01-21"))
    assert "+200.0%" in row and "+0.1000" in row
    assert "NEXT" in pump.text
    unknown = toolkit.coin(None, 1, "PUMPX", run=run)
    assert "not in this trial's data" in unknown.text


def test_a_counterfactual_shows_the_run_beside_itself_without_the_thin_days() -> None:
    run = toy.toy_run()

    out = toolkit.counterfactual(None, 1, Rule(min_volume_usd=300_000), run=run)

    original, changed = out.data["original"], out.data["with_rule"]
    assert changed["total_return_net"] < original["total_return_net"] - 0.15
    assert out.data["removed_bars"] == toy.PUMP_BAR + 1 and out.data["unknown_bars"] == 2
    assert set(original["years"]) == {2026} == set(changed["years"])
    assert "yearly return" in out.text and "by calendar year" in out.text
    assert "$300,000" in out.text


def test_a_listing_counterfactual_is_compared_from_the_other_venues_first_day() -> None:
    run = toy.toy_run()
    listed = pd.DataFrame(True, index=run.panel.prices.index, columns=run.panel.prices.columns)
    listed.iloc[:10] = False

    out = toolkit.counterfactual(None, 1, Rule(only_listed=listed), run=run)

    assert out.data["from"] == "2026-01-11"
    assert "2026-01-11 to" in out.text


def test_a_counterfactual_leaves_the_stored_run_alone() -> None:
    run = toy.toy_run()
    before = run.weights.copy()

    toolkit.counterfactual(None, 1, Rule(delist_days=5), run=run)
    toolkit.counterfactual(None, 1, Rule(max_instrument_share=0.05), run=run)

    pd.testing.assert_frame_equal(run.weights, before)


def test_compare_venues_sets_the_two_stages_side_by_side(monkeypatch) -> None:
    from qlab import forward_series
    from qlab.anomalies import scan

    stages = [forward_series.StageRun(1, "x", 11, "paper", "binance", None),
              forward_series.StageRun(2, "x-hl", 12, "paper", "hyperliquid", None)]
    monkeypatch.setattr(forward_series, "stage_runs", lambda session, idea: (True, stages))
    index = pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC")
    series = {11: pd.Series(0.01, index=index),
              12: pd.Series(0.0, index=index[20:])}
    monkeypatch.setattr(scan, "trial_daily_capital",
                        lambda session, trial_id, **kw: (series[trial_id], True))
    runs = {11: toy.toy_run(11, "x", "binance"), 12: toy.toy_run(12, "x-hl", "hyperliquid")}
    monkeypatch.setattr(toolkit, "load_run", lambda session, trial_id, **kw: runs[trial_id])

    out = toolkit.compare_venues(None, "x")

    assert out.data["days"] == 40
    assert out.data["stage1_total"] == pytest.approx(1.01 ** 40 - 1)
    assert out.data["stage2_total"] == 0.0
    assert "by month" in out.text and "stage 1 (binance)" in out.text
    assert out.data["stage1_best"]["PUMP"] > 0.15


def test_an_idea_with_one_stage_has_nothing_to_compare(monkeypatch) -> None:
    from qlab import forward_series

    monkeypatch.setattr(forward_series, "stage_runs", lambda session, idea: (
        False, [forward_series.StageRun(1, "x", 11, "paper", "binance", None)]))

    out = toolkit.compare_venues(None, "x")

    assert "no two-stage answer" in out.text


def test_every_call_is_logged_with_its_output_when_an_investigation_asks(tmp_path) -> None:
    log = tmp_path / "calls" / "1.jsonl"
    out = toolkit.Output(["line one", "line  two"])

    toolkit.log_call("qlab anomaly   flags 1", out, str(log))
    toolkit.log_call("qlab anomaly flags 2", out, str(log))

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["cmd"] for r in rows] == ["qlab anomaly flags 1", "qlab anomaly flags 2"]
    assert rows[0]["output"] == "line one\nline  two"
    toolkit.log_call("qlab anomaly flags 3", out, None)  # no log asked for: nothing written
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_the_toolkit_cannot_write_the_registry_by_construction() -> None:
    """Read-only is structural: no session is committed, added to or flushed in the toolkit or
    in the commands under `qlab anomaly`; the commands open a session that is rolled back."""
    toolkit_source = Path(toolkit.__file__).read_text(encoding="utf-8")
    cli_source = Path(toolkit.__file__).with_name("cli.py").read_text(encoding="utf-8")
    anomaly_part = cli_source[cli_source.index("@anomaly_app.command"):
                              cli_source.index("def evaluate_lines")]

    for source in (toolkit_source, anomaly_part):
        assert not re.search(r"\.(add|add_all|commit|flush|merge|delete)\(", source)
        assert "session_scope" not in source
    assert "rollback()" in cli_source and "read_session" in anomaly_part


# --- the stress and the what-if with the strategy asked again ------------------------------


def test_one_coins_earning_can_be_clipped_to_a_share_of_the_capital() -> None:
    run = toy.toy_run()

    out = toolkit.counterfactual(None, 1, Rule(max_leg_contribution=0.05), run=run)

    original, clipped = out.data["original"], out.data["with_rule"]
    # The pump earned 0.2 on one bar; clipped to 0.05 it loses 0.15 of the capital.
    assert original["total_return_net"] - clipped["total_return_net"] == pytest.approx(0.15,
                                                                                       abs=0.01)
    assert "clipped" in out.data["mode"] and "clipped to 5.0%" in out.text


def test_the_strategy_can_be_asked_to_choose_again_without_the_banned_coin_days() -> None:
    run = toy.toy_run()

    plain = toolkit.counterfactual(None, 1, Rule(min_volume_usd=300_000), run=run)
    again = toolkit.counterfactual(None, 1, Rule(min_volume_usd=300_000), run=run,
                                   reselect_book=True)

    assert "became cash" in plain.data["mode"] and "chose again" in again.data["mode"]
    # The toy strategy holds the same coins whatever the mask says, so asking it again gives
    # the same book; what matters is that the banned coin-days were untradeable when it asked.
    assert again.data["with_rule"]["total_return_net"] == pytest.approx(
        plain.data["with_rule"]["total_return_net"], abs=1e-9)
    assert again.data["original"] == plain.data["original"]


def test_asking_again_gives_the_panel_its_own_snapshot_id_and_the_strategy_the_mask() -> None:
    run = toy.toy_run()
    blocked = pd.DataFrame(False, index=run.panel.prices.index, columns=run.panel.prices.columns)
    blocked.loc[:, "A"] = True

    again = reselect(run, blocked)

    assert again.panel.snapshot_id.startswith("toy:what-if-") and run.panel.snapshot_id == "toy"
    assert not again.panel.tradeable["A"].any() and run.panel.tradeable["A"].all()
    assert (again.weights["A"] == 0.0).all() and (run.weights["A"] == 0.1).all()


def test_the_battery_puts_the_standard_cases_in_one_table() -> None:
    run = toy.toy_run()
    listed = pd.DataFrame(True, index=run.panel.prices.index, columns=run.panel.prices.columns)
    listed["PUMP"] = False

    out = toolkit.battery(None, 1, run=run, listed=listed)

    cases = list(out.data["cases"])
    assert cases[0] == "original"
    for name, _ in toolkit.BATTERY:
        assert name in cases
    assert "only coins the other venue lists" in cases
    full = out.data["cases"]
    assert full["only coins the other venue lists"]["total_return_net"] \
        < full["original"]["total_return_net"] - 0.15
    assert "robustness battery" in out.text and "last years" in out.text
