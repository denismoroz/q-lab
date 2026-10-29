"""The protection readings P1 - P4 of the Bv2 family (docs/PROTECTION.md, rules 2026-10-29.1).

The numbers are hand-built daily series, so every figure below can be checked by eye: a strategy
that is flat in the falls, one that is no protection at all, a part too short to say anything."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

import qlab.registry.db as db_module
from qlab import protection
from qlab.cli import app
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialStatus,
)
from qlab.registry.stages import NOTES_PREFIX
from qlab.rules.loader import DEFAULT_RULES_DIR, load, load_latest

RULES = load("2026-10-29.1", DEFAULT_RULES_DIR)
START = pd.Timestamp("2026-01-05", tz="UTC")  # a Monday


def _daily(values: list[float], start: pd.Timestamp = START) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1D")
    return pd.Series(values, index=index, dtype=float)


def _weeks(week_returns: list[float]) -> pd.Series:
    """One return per Monday-to-Sunday week, spread evenly over its seven days."""
    return _daily([r / 7 for r in week_returns for _ in range(7)])


# Twelve weeks of the coins: four fall, four rise, four are flat enough not to count.
LONG_WEEKS = [-0.03, 0.02, -0.04, 0.03, -0.02, 0.04, -0.05, 0.02, -0.01, 0.03, -0.02, 0.01]
LONG = _weeks(LONG_WEEKS)
# Eight falling weeks are needed for P2: a longer run of alternating weeks.
LONG_LONG = _weeks(LONG_WEEKS * 2)


def test_a_strategy_that_stays_flat_cuts_the_whole_fall():
    metrics = protection.protection_metrics(_daily([0.0] * len(LONG_LONG)), LONG_LONG,
                                            reinvested=False)
    assert metrics["prot_weeks"] == 24
    assert metrics["prot_fall_ratio"] == pytest.approx(0.0)
    assert metrics["prot_fall_capture"] == pytest.approx(0.0)
    assert metrics["prot_rise_capture"] == pytest.approx(0.0)
    assert metrics["prot_total_return"] == pytest.approx(0.0)
    states = {r.letter: r.state for r in protection.read_protection(metrics, RULES)}
    # P3 has no line: it is a number, never a verdict
    assert states == {"П1": "holds", "П2": "holds", "П3": "number", "П4": "holds"}


def test_a_strategy_that_is_the_long_side_protects_nothing():
    metrics = protection.protection_metrics(LONG_LONG.copy(), LONG_LONG, reinvested=False)
    assert metrics["prot_fall_ratio"] == pytest.approx(1.0)
    assert metrics["prot_fall_capture"] == pytest.approx(1.0)
    assert metrics["prot_rise_capture"] == pytest.approx(1.0)
    by = {r.letter: r for r in protection.read_protection(metrics, RULES)}
    assert by["П1"].holds is False and by["П2"].holds is False
    assert by["П3"].value == pytest.approx(1.0) and by["П3"].holds is None  # a number, no line
    assert by["П3"].state == "number"
    assert by["П3"].reason is None


def test_a_strategy_that_gains_in_falling_weeks_has_a_negative_capture_and_holds():
    # the opposite of the coins every week: earns 1/2 of what the coins lose
    ours = -0.5 * LONG_LONG
    metrics = protection.protection_metrics(ours, LONG_LONG, reinvested=False)
    assert metrics["prot_fall_capture"] == pytest.approx(-0.5)
    assert metrics["prot_rise_capture"] == pytest.approx(-0.5)
    by = {r.letter: r for r in protection.read_protection(metrics, RULES)}
    assert by["П2"].holds is True


def test_the_lines_are_half_three_tenths_and_zero():
    by = {r.letter: r for r in protection.read_protection(
        protection.protection_metrics(LONG_LONG.copy(), LONG_LONG, reinvested=False), RULES)}
    assert (by["П1"].comparator, by["П1"].threshold) == ("<=", 0.5)
    assert (by["П2"].comparator, by["П2"].threshold) == ("<=", 0.3)
    assert (by["П3"].comparator, by["П3"].threshold) == (None, None)
    assert (by["П4"].comparator, by["П4"].threshold) == (">=", 0.0)


def test_exactly_on_the_line_holds_and_just_over_does_not():
    # a book that loses 30% of every falling week and 50% of the worst fall
    on_line = pd.Series(np.where(LONG_LONG < 0, 0.3 * LONG_LONG, 0.0), index=LONG_LONG.index)
    m = protection.protection_metrics(on_line, LONG_LONG, reinvested=False)
    assert m["prot_fall_capture"] == pytest.approx(0.3)
    by = {r.letter: r for r in protection.read_protection(m, RULES)}
    assert by["П2"].holds is True
    over = pd.Series(np.where(LONG_LONG < 0, 0.31 * LONG_LONG, 0.0), index=LONG_LONG.index)
    m = protection.protection_metrics(over, LONG_LONG, reinvested=False)
    assert {r.letter: r for r in protection.read_protection(m, RULES)}["П2"].holds is False


def test_a_part_shorter_than_eight_whole_weeks_defines_no_reading():
    short = _weeks([0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02])
    metrics = protection.protection_metrics(short.copy(), short, reinvested=False)
    assert metrics["prot_weeks"] == 7
    assert not {"prot_fall_ratio", "prot_fall_capture", "prot_rise_capture"} & set(metrics)
    readings = protection.read_protection(metrics, RULES)
    assert [r.state for r in readings] == ["not defined"] * 3 + ["holds"]
    assert all(r.reason == protection.FEW_WEEKS_RU for r in readings if r.value is None)
    p1 = next(r for r in readings if r.letter == "П1")
    assert p1.holds is None and p1.reason == protection.FEW_WEEKS_RU


def test_fewer_than_eight_falling_weeks_leave_p2_undefined_but_not_p1():
    # 12 weeks: 4 falling, 7 rising, 1 flat
    weeks = [-0.04, 0.02, -0.04, 0.03, -0.04, 0.04, -0.05, 0.02, 0.0, 0.03, 0.02, 0.01]
    long_side = _weeks(weeks)
    metrics = protection.protection_metrics(long_side * 0.2, long_side, reinvested=False)
    assert metrics["prot_fall_weeks"] == 4 and metrics["prot_rise_weeks"] == 7
    assert "prot_fall_ratio" in metrics and "prot_fall_capture" not in metrics
    by = {r.letter: r for r in protection.read_protection(metrics, RULES)}
    assert by["П1"].holds is True
    assert by["П2"].holds is None and by["П2"].reason == protection.FEW_FALLS_RU
    assert by["П3"].reason == protection.FEW_RISES_RU


def test_coins_that_hardly_fell_leave_p1_undefined():
    gentle = _weeks([0.01, -0.001, 0.01, -0.001] * 3)
    metrics = protection.protection_metrics(gentle * 0.5, gentle, reinvested=False)
    assert metrics["prot_long_worst_fall"] > -protection.MIN_FALL
    assert "prot_fall_ratio" not in metrics
    p1 = next(r for r in protection.read_protection(metrics, RULES) if r.letter == "П1")
    assert p1.holds is None and p1.reason == protection.SMALL_FALL_RU


def test_only_whole_calendar_weeks_count():
    # a Wednesday start: the first (partial) week and the last one are dropped
    start = pd.Timestamp("2026-01-07", tz="UTC")
    values = _daily([0.001] * 30, start)
    metrics = protection.protection_metrics(values, values, reinvested=False)
    assert metrics["prot_days"] == 30
    # Jan 7 .. Feb 5: the Mondays Jan 12, 19 and 26 start whole weeks (ending Jan 18, 25, Feb 1)
    assert metrics["prot_weeks"] == 3


def test_a_part_that_loses_money_fails_p4():
    ours = _daily([-0.001] * 90)
    metrics = protection.protection_metrics(ours, _daily([0.0] * 90), reinvested=False)
    p4 = next(r for r in protection.read_protection(metrics, RULES) if r.letter == "П4")
    assert metrics["prot_total_return"] == pytest.approx(-0.09)
    assert p4.holds is False


def test_a_book_that_does_not_reinvest_sums_and_one_that_does_compounds():
    flat_up = _daily([0.01] * 28)
    summed = protection.protection_metrics(flat_up, flat_up, reinvested=False)
    grown = protection.protection_metrics(flat_up, flat_up, reinvested=True)
    assert summed["prot_total_return"] == pytest.approx(0.28)
    assert grown["prot_total_return"] == pytest.approx(1.01 ** 28 - 1)
    # worst fall: points of the money when summed
    down = _daily([0.01] * 14 + [-0.01] * 7 + [0.01] * 7)
    assert protection.protection_metrics(down, down, reinvested=False)["prot_worst_fall"] == \
        pytest.approx(-0.07)


def test_hourly_bars_are_counted_by_the_day():
    index = pd.date_range(START, periods=24 * 56, freq="1h")
    hourly = pd.Series(0.0001, index=index)
    metrics = protection.protection_metrics(hourly, hourly * 2, reinvested=False)
    assert metrics["prot_days"] == 56 and metrics["prot_weeks"] == 8
    assert metrics["prot_total_return"] == pytest.approx(0.0001 * 24 * 56)


def test_a_series_of_one_bar_has_no_measurement():
    assert protection.protection_metrics(_daily([0.1]), _daily([0.1]), reinvested=False) == {}


# ---- the long side -------------------------------------------------------------------------


def test_the_long_side_is_the_positive_weights_times_base_times_the_coins_moves():
    index = pd.date_range(START, periods=4, freq="1D")
    prices = pd.DataFrame({"BTC": [100.0, 110.0, 99.0, 99.0],
                           "BTC_spot": [100.0, 110.0, 99.0, 99.0]}, index=index)
    # the book owns spot and is short the perpetual: only the spot half is the long side
    weights = pd.DataFrame({"BTC": [-0.4] * 4, "BTC_spot": [0.4] * 4}, index=index)
    base = pd.Series([1.0, 2.0, 2.0, 2.0], index=index)
    long_side = protection.long_side_returns(prices, weights, base, index[:-1])
    # bar t is the move over (t, t+1], on the base at t
    assert long_side.iloc[0] == pytest.approx(0.4 * 0.10 * 1.0)
    assert long_side.iloc[1] == pytest.approx(0.4 * (99 / 110 - 1) * 2.0)
    assert long_side.iloc[2] == pytest.approx(0.0)
    assert list(long_side.index) == list(index[:-1])


def test_without_a_base_the_long_side_is_on_the_weights_alone():
    index = pd.date_range(START, periods=3, freq="1D")
    prices = pd.DataFrame({"ETH_spot": [10.0, 11.0, 11.0]}, index=index)
    weights = pd.DataFrame({"ETH_spot": [0.5] * 3}, index=index)
    assert protection.long_side_returns(prices, weights, None, index[:-1]).iloc[0] == \
        pytest.approx(0.05)


# ---- which runs ----------------------------------------------------------------------------


def test_the_family_is_bv2_and_whatever_wraps_it():
    live = "qlab.strategies.live.bv2:LiveBv2"
    assert protection.is_bv2_family(live, {})
    assert protection.is_bv2_family("qlab.strategies.bv2:BStrategyV2", {})
    assert protection.is_bv2_family(
        "qlab.strategies.retune:Retune", {"inner_code_ref": live, "base_params": {}})
    assert protection.is_bv2_family(
        "qlab.strategies.regime_gate:RegimeGate",
        {"inner_code_ref": "qlab.strategies.retune:Retune",
         "inner_params": {"inner_code_ref": live, "base_params": {}}})
    assert not protection.is_bv2_family("qlab.strategies.live.trend:LiveTrendTSMOMEnsemble", {})
    assert not protection.is_bv2_family(
        "qlab.strategies.retune:Retune",
        {"inner_code_ref": "qlab.strategies.xsmom:XsmomCrossSectionalMomentum"})
    # a calibration noise book of Bv2 is an instrument, not a strategy
    assert not protection.is_bv2_family(
        "qlab.calibration.noise:NoiseStrategy",
        {"reference_code_ref": live, "reference_params": {}})


# ---- the words -----------------------------------------------------------------------------


def test_a_run_without_a_measurement_has_no_view_and_no_sentence():
    assert protection.view({"ann_return_net": 0.1}, RULES) is None
    assert protection.sentence({"ann_return_net": 0.1}, RULES) is None
    assert not protection.has_protection(None)


def test_the_sentence_and_the_view_say_holds_fails_and_not_defined():
    ours = LONG_LONG * 0.2 + 0.0005  # a fifth of the coins' moves and a small earning each day
    metrics = protection.protection_metrics(ours, LONG_LONG, reinvested=False)
    sentence = protection.sentence(metrics, RULES)
    assert sentence.startswith("защита: П1 держится (")
    assert "П2 держится (0.08)" in sentence and "П3 0.34" in sentence
    assert "П4 держится (+" in sentence
    short = protection.protection_metrics(_weeks(LONG_WEEKS[:3]), _weeks(LONG_WEEKS[:3]),
                                          reinvested=False)
    assert "П1 не определено" in protection.sentence(short, RULES)
    body = protection.view(metrics, RULES)
    assert [r["letter"] for r in body["readings"]] == ["П1", "П2", "П3", "П4"]
    assert body["numbers"]["prot_weeks"] == 24 and body["min_weeks"] == 8
    assert "не определено" in protection.reading_line(
        protection.read_protection(short, RULES)[0])


def test_an_older_ruleset_without_the_lines_gives_no_readings():
    old = load("2026-10-28.2", DEFAULT_RULES_DIR)
    assert protection.read_protection({"prot_weeks": 24.0}, old) == []


def test_the_ruleset_only_adds_readings_and_decides_nothing_new():
    before = load("2026-10-28.2", DEFAULT_RULES_DIR)
    assert RULES.based_on == "2026-10-28.2"
    assert RULES.rules == before.rules and RULES.retired == before.retired
    added = [r.id for r in RULES.informational if r.id not in {i.id for i in before.informational}]
    assert added == ["protection_fall_cut", "protection_fall_weeks_loss", "protection_rise_kept",
                     "protection_no_loss"]


# ---- the registry --------------------------------------------------------------------------

LIVE = "qlab.strategies.live.bv2:LiveBv2"
GOOD = {"prot_weeks": 24.0, "prot_days": 170.0, "prot_total_return": 0.08,
        "prot_worst_fall": -0.04, "prot_long_worst_fall": -0.13, "prot_fall_ratio": 0.31,
        "prot_fall_capture": 0.12, "prot_rise_capture": 0.5, "prot_fall_weeks": 10.0,
        "prot_rise_weeks": 11.0}


def _idea(session, idea_id, status=IdeaStatus.BENCH, *, parent=None):
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     notes=f"{NOTES_PREFIX}{parent}" if parent else None)
    session.get(Idea, idea_id).parent_id = parent


def _trial(session, idea_id, metrics, *, code_ref=LIVE, params=None, source="binance",
           route=TrialRoute.NEEDS_FORWARD):
    spec = repo.add_spec(
        session, idea_id=idea_id, version=1, params=params or {},
        data_requirements={"source": source, "end": "2026-10-20"}, rebalance="1h",
        costs_model={}, code_ref=code_ref)
    if session.get(DataSnapshot, "snap") is None:
        from datetime import date

        repo.add_data_snapshot(session, id="snap", source=source, instruments={},
                               range_start=date(2025, 1, 1), range_end=date(2026, 10, 20),
                               path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash=f"h{spec.id}", code_sha="c",
                          params={}, status=TrialStatus.OK, snapshot_id="snap", route=route,
                          route_reason="r", metrics={"ann_return_net": 0.1, **metrics})


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture()
def registry(session):
    _idea(session, "bv2-a")
    _trial(session, "bv2-a", GOOD)
    _idea(session, "bv2-a-hl", parent="bv2-a")
    _trial(session, "bv2-a-hl", {**GOOD, "prot_fall_capture": 0.55}, source="hyperliquid")
    _idea(session, "bv2-old")  # a Bv2 run made before the measurement existed
    _trial(session, "bv2-old", {})
    _idea(session, "trend-a")
    _trial(session, "trend-a", GOOD, code_ref="qlab.strategies.live.trend:LiveTrendTSMOMEnsemble")
    _idea(session, "bv2-wrapped")
    _trial(session, "bv2-wrapped", GOOD, code_ref="qlab.strategies.retune:Retune",
           params={"inner_code_ref": LIVE, "base_params": {}})
    _idea(session, "bv2-retired", IdeaStatus.RETIRED)
    _trial(session, "bv2-retired", GOOD)
    _idea(session, "noise-dollar_neutral-random_signs", IdeaStatus.CANDIDATE)
    _trial(session, "noise-dollar_neutral-random_signs", GOOD)
    session.commit()
    return session


def test_the_table_lists_the_live_bv2_family_with_their_stage(registry):
    rows = protection.collect(registry, RULES)
    assert {r.idea_id: r.stage for r in rows} == {
        "bv2-a": "Binance", "bv2-a-hl": "Hyperliquid 2", "bv2-old": "Binance",
        "bv2-wrapped": "Binance"}
    by = {r.idea_id: r for r in rows}
    assert by["bv2-a"].source == "run" and by["bv2-old"].source == "none"
    assert by["bv2-old"].readings[0].reason == protection.NO_RUN_RU
    assert "bv2-retired" in {r.idea_id for r in protection.collect(
        registry, RULES, live_only=False)}


def test_the_totals_and_the_report_lines(registry):
    rows = [r for r in protection.collect(registry, RULES) if r.source != "none"]
    totals = protection.summarize(rows)
    assert totals["П2"] == {"holds": 2, "fails": 1, "not defined": 0, "number": 0}
    assert totals["П3"]["number"] == 3
    text = "\n".join(protection.report_lines(registry, RULES))
    assert "3 идей" in text and "qlab protection" in text and "ничего не решает" in text
    assert "- П2 (в недели падения монет стратегия теряет не больше 30% их падения): " \
           "держится 2, не держится 1, не определено 0" in text


def test_the_report_names_the_runs_of_the_night(registry):
    from datetime import UTC, datetime

    lines = protection.report_lines(registry, RULES,
                                    since=datetime(2000, 1, 1, tzinfo=UTC))
    assert any(line.startswith("- **bv2-a-hl** (Hyperliquid 2") for line in lines)
    none = protection.report_lines(registry, RULES, since=datetime(2100, 1, 1, tzinfo=UTC))
    assert not any("Прогоны с" in line for line in none)


def test_no_section_without_a_single_measured_run(session):
    _idea(session, "bv2-old")
    _trial(session, "bv2-old", {})
    session.commit()
    assert protection.report_lines(session, RULES) == []


def test_a_stored_measurement_stands_in_for_a_run_made_before(registry, tmp_path):
    old = next(r for r in protection.collect(registry, RULES) if r.idea_id == "bv2-old")
    protection.save(old.trial_id, GOOD, tmp_path)
    again = next(r for r in protection.collect(registry, RULES, store=tmp_path)
                 if r.idea_id == "bv2-old")
    assert again.source == "stored" and again.readings[0].holds is True
    assert protection.load(old.trial_id, tmp_path)["prot_weeks"] == 24.0
    assert protection.load(999, tmp_path) is None


def test_the_trial_in_the_api_carries_the_view_from_the_run_or_the_store(tmp_path, monkeypatch):
    from qlab.api.serializers import trial_json

    class Trial:  # the few fields the serializer reads
        id = 77
        spec_id = 1
        config_hash = snapshot_id = code_sha = None
        started_at = finished_at = None
        metrics = GOOD
        route = None
        route_reason = None
        kept = True
        token_cost = cpu_seconds = None

        class status:  # noqa: N801
            value = "ok"

        class source:  # noqa: N801
            value = "native"

    body = trial_json(Trial)["protection"]
    assert [r["letter"] for r in body["readings"]] == ["П1", "П2", "П3", "П4"]
    assert body["readings"][1]["state"] == "holds"
    Trial.metrics = {"ann_return_net": 0.1}
    monkeypatch.setattr(protection, "STORE", tmp_path)
    assert trial_json(Trial)["protection"] is None
    protection.save(77, GOOD, tmp_path)
    assert trial_json(Trial)["protection"]["numbers"]["prot_weeks"] == 24.0


# ---- the command ---------------------------------------------------------------------------


@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    path = tmp_path / "qlab.db"
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    _idea(s, "bv2-a")
    _trial(s, "bv2-a", GOOD)
    _idea(s, "bv2-old")
    _trial(s, "bv2-old", {})
    s.commit()
    s.close()
    monkeypatch.setenv("QLAB_DB", str(path))
    monkeypatch.setattr(protection, "STORE", tmp_path / "protection")
    db_module._engine = None
    db_module._SessionLocal = None
    yield path
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_command_prints_a_row_per_idea_and_says_which_have_no_measurement(cli_db):
    result = CliRunner().invoke(app, ["protection"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert out.splitlines()[0].startswith(f"rules: {load_latest().version}")
    row = next(line for line in out.splitlines() if line.startswith("bv2-a "))
    assert "да 0.31" in row and "да 0.12" in row and "0.50" in row
    old = next(line for line in out.splitlines() if line.startswith("bv2-old"))
    assert "(нет замера)" in old
    assert "П1: худшее падение" in out and "П4:" in out


def test_the_command_json_and_an_empty_registry(cli_db, tmp_path, monkeypatch):
    body = json.loads(CliRunner().invoke(app, ["protection", "--json"]).output)
    assert body["rules_version"] == load_latest().version
    assert [r["idea_id"] for r in body["rows"]] == ["bv2-a", "bv2-old"]
    assert body["rows"][0]["readings"][0]["state"] == "holds"
    assert body["summary"]["П1"]["holds"] == 1 and body["summary"]["П1"]["not defined"] == 1

    empty = tmp_path / "empty.db"
    Base.metadata.create_all(create_engine(f"sqlite+pysqlite:///{empty}"))
    monkeypatch.setenv("QLAB_DB", str(empty))
    db_module._engine = None
    db_module._SessionLocal = None
    result = CliRunner().invoke(app, ["protection"])
    assert result.exit_code == 0 and "no run of the Bv2 family" in result.output


def test_the_night_report_has_a_section_of_its_own(cli_db):
    from qlab import night

    lines = night._protection_lines(__import__("datetime").date(2000, 1, 1))
    assert lines[0] == "## Защита Bv2: замер"
    assert "1 идей" in "\n".join(lines)
