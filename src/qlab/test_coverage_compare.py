"""The two ways to count a book whose coins were listed at different times
(`qlab.coverage_compare`, docs/COVERAGE.md): the stand's complete-book cut against each coin
entering as it appears. A reading that decides nothing.

The market is made by hand: three coins on daily bars, coin B listed on day 650 of 900, and a toy
strategy with one equal money slot per coin that holds a coin only while it is tradeable, so which
bars the book was complete on is arithmetic a reader can follow."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qlab import coverage_compare as cc
from qlab import knowable_coins as kc
from qlab.api import test_api
from qlab.cli import app
from qlab.cli.test_cli import cli_db  # noqa: F401 - fixture
from qlab.data.panel import MarketPanel
from qlab.pipeline.spec import StrategySpec
from qlab.rules.loader import load_latest

DAYS = 900
FIRST = pd.Timestamp("2020-01-01", tz="UTC")
LISTED = {"A": 0, "B": 650, "C": 100}
"""The day each coin's market opens."""
client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory


class SlotStrategy:
    """One equal slot per coin, long while the coin is tradeable, nothing otherwise."""

    name = "slots"
    reinvests_profit = False

    def target_weights(self, panel, params):
        weights = panel.tradeable.astype(float) / len(panel.prices.columns)
        return weights.where(panel.prices.notna(), 0.0)


def _spec(**over) -> StrategySpec:
    raw = {
        "idea_id": "toy-slots", "title": "Toy slots",
        "code_ref": "qlab.test_coverage_compare:SlotStrategy",
        "data": {"source": "binance", "interval": "1d", "start": "2020-01-01",
                 "end": "2022-06-18"},
        "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "required_instruments": ["A", "B", "C"],
        "strategy_capital_usd": 100.0,
        "sources": {"costs.taker_fee_bps": "t", "costs.slippage_bps": "t",
                    "min_leg_notional": "t", "strategy_capital_usd": "t"},
    }
    raw.update(over)
    return StrategySpec.model_validate(raw)


def _panel() -> MarketPanel:
    index = pd.date_range(FIRST, periods=DAYS, freq="1D")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, DAYS))) for c in LISTED}, index=index)
    for coin, day in LISTED.items():
        prices.loc[index[:day], coin] = np.nan
    return MarketPanel(snapshot_id="toy", prices=prices,
                       funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
                       tradeable=prices.notna(), meta={})


@pytest.fixture(scope="module")
def ruleset():
    return load_latest()


def test_as_is_waits_for_every_coin_and_as_listed_takes_the_whole_history(ruleset) -> None:
    panel = _panel()
    stand = cc.measure_panel(_spec(), panel, ruleset, listed=False)
    longer = cc.measure_panel(_spec(), panel, ruleset, listed=True)
    day = lambda n: (FIRST + pd.Timedelta(days=n)).date().isoformat()  # noqa: E731
    # the stand's window starts on the day the last coin is listed; the other form on the first
    assert stand["window_first"] == day(650) and longer["window_first"] == day(0)
    assert stand["window_days"] == 250 and longer["window_days"] == 900
    # the last third of each window (2/3 - 1/3 of the days, rules 2026-11-04.2)
    def near(found: str, expected: str) -> bool:
        return abs((date.fromisoformat(found) - date.fromisoformat(expected)).days) <= 1

    assert near(longer["cut_day"], day(600)) and near(stand["cut_day"], day(650 + 167))
    assert longer["judged_first"] < stand["judged_first"]


def test_a_coin_listed_late_has_a_slot_that_waits_and_the_book_is_incomplete_meanwhile(
        ruleset) -> None:
    longer = cc.measure_panel(_spec(), _panel(), ruleset, listed=True)
    # the last third is days 600..899; coin B is missing on 600..649 -- 50 of 300 bars
    assert longer["incomplete_share"] == pytest.approx(50 / 300, abs=0.01)
    # the stand's window holds only bars with every coin
    stand = cc.measure_panel(_spec(), _panel(), ruleset, listed=False)
    assert stand["incomplete_share"] == 0.0


def test_nothing_is_held_in_a_coin_before_it_is_listed() -> None:
    weights = SlotStrategy().target_weights(_panel(), {})
    assert (weights["B"].iloc[:650] == 0.0).all() and (weights["B"].iloc[650:] > 0).all()
    assert (weights["C"].iloc[:100] == 0.0).all()
    # the unlisted coin's slot stays aside: the book holds less than all its money
    assert weights.iloc[0].sum() == pytest.approx(1 / 3)


def test_a_coin_does_not_join_before_it_was_knowable(ruleset) -> None:
    gate = cc.Gate(days={"A": date(2020, 1, 1), "B": date(2022, 2, 1), "C": date(2020, 4, 10)},
                   known_from=date(2022, 2, 1), plain="x")
    panel, opened = cc.gate_panel(_panel(), gate)
    index = panel.prices.index
    # B's market opened on day 650 (2021-10-12) but it was no obvious pick before 2022-02-01
    assert not panel.tradeable.loc[index < pd.Timestamp("2022-02-01", tz="UTC"), "B"].any()
    assert panel.tradeable.loc[index >= pd.Timestamp("2022-02-01", tz="UTC"), "B"].all()
    assert opened["B"] == date(2022, 2, 1)
    longer = cc.measure_panel(_spec(), _panel(), ruleset, listed=True, gate=gate)
    assert longer["joins"]["B"] == "2022-02-01"
    # days 600..899 of the last third; B enters on day 762 (2022-02-01): 162 bars without it
    assert longer["incomplete_share"] == pytest.approx(162 / 300, abs=0.01)


def test_a_coin_never_knowable_keeps_its_listing_day() -> None:
    gate = cc.Gate(days={"A": date(2020, 1, 1), "B": None, "C": date(2020, 4, 10)},
                   known_from=None, plain="x")
    panel, opened = cc.gate_panel(_panel(), gate)
    assert panel.tradeable["B"].iloc[650:].all() and "B" not in opened


def test_the_gate_is_the_venues_own_top_coins_by_trailing_turnover(monkeypatch, tmp_path) -> None:
    """BTC and ETH are always the two biggest; SOL is listed on day 20 and only then grows into the
    top: it is knowable from the first day its trailing window put it there."""
    config = kc.KnowableConfig(top_k=2, window_days=5, raw_dir=tmp_path, interval="1d",
                               source="night/knowable_coins.yaml")
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    value = pd.DataFrame({"BTC": 1000.0, "ETH": 900.0, "F": 10.0,
                          "SOL": [np.nan] * 20 + [5.0] * 10 + [2000.0] * 30}, index=index)
    monkeypatch.setattr(kc, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(kc, "raw_volume", lambda venue, cfg=None: value)
    spec = _spec(required_instruments=["BTC", "ETH", "SOL"],
                 data={"source": "binance", "interval": "1d", "start": "2020-01-01",
                       "end": "2020-02-29", "instruments": ["BTC", "ETH", "SOL"]})
    gate = cc.knowable_gate(spec, date(2020, 1, 1), date(2020, 2, 29))
    assert gate.days["BTC"] == date(2020, 1, 6) and gate.days["ETH"] == date(2020, 1, 6)
    # SOL: 2000 a day from day 30; its 5-day trailing mean beats ETH's 900 from day 33 (the mean
    # of days 28..32 is (5+5+2000+2000+2000)/5 = 1202)
    assert gate.days["SOL"] == date(2020, 2, 3)
    assert gate.known_from == date(2020, 2, 3)


def test_a_book_that_chooses_its_own_coins_needs_no_gate(monkeypatch) -> None:
    spec = _spec(params={"universe": {"rule": "top_liquid"}},
                 data={"source": "binance", "interval": "1d", "start": "2020-01-01",
                       "end": "2022-06-18", "instruments": ["A", "B", "C"]})
    assert cc.knowable_gate(spec, date(2020, 1, 1), date(2022, 6, 18)) is None


def test_the_ledger_round_trips_and_lists_every_idea(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(cc.ENV, str(tmp_path / "ledger"))
    payload = {"idea_id": "toy-slots", "rows": {}, "differs": "x", "measured_at": "t",
               "seconds": {"total": 1.0}}
    path = cc.save(payload)
    assert path == tmp_path / "ledger" / "toy-slots.json"
    assert cc.load("toy-slots") == payload
    assert cc.load("absent") is None
    assert set(cc.load_all()) == {"toy-slots"}
    assert json.loads(path.read_text(encoding="utf-8"))["idea_id"] == "toy-slots"


def test_the_view_is_plain_russian_and_decides_nothing(ruleset) -> None:
    panel = _panel()
    payload = {
        "idea_id": "toy-slots", "data_end": "2022-06-18", "measured_at": "t", "knowable": None,
        "differs": "d", "affected": True,
        "rows": {"as_is": cc.measure_panel(_spec(), panel, ruleset, listed=False),
                 "as_listed": cc.measure_panel(_spec(), panel, ruleset, listed=True)},
    }
    shown = cc.view(payload)
    assert shown["title"] == "Два способа считать книгу, монеты которой появились в разное время"
    assert [r["id"] for r in shown["rows"]] == ["as_is", "as_listed"]
    assert shown["rows"][1]["incomplete"] == "17%"
    assert "Ничего не решает" in shown["note"]
    assert cc.view(None) is None
    assert any("доход в год" in line for line in cc.lines_ru(payload))


def test_the_idea_page_carries_the_stored_reading(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(cc.ENV, str(tmp_path))
    assert client.get("/api/ideas/alpha").json()["coverage_compare"] is None
    cc.save({"idea_id": "alpha", "data_end": "2026-11-04", "measured_at": "t", "differs": "d",
             "knowable": {"plain": "k"}, "affected": True,
             "rows": {"as_is": {"error": "boom"},
                      "as_listed": {"window_first": "2019-10-08", "window_last": "2026-11-04",
                                    "cut_day": "2024-06-26", "judged_days": 861, "p": 0.07,
                                    "ann_return": 0.17, "max_dd": -0.2,
                                    "incomplete_share": 0.0}}})
    block = client.get("/api/ideas/alpha").json()["coverage_compare"]
    assert block["title"].startswith("Два способа")
    assert block["rows"][0]["error"] == "boom"
    assert block["rows"][1]["window"] == "08.10.2019 – 04.11.2026"
    assert block["rows"][1]["ann_return"] == "+17.0%" and block["knowable"] == "k"


def test_the_cli_measures_named_ideas_and_stores_the_reading(cli_db, tmp_path, monkeypatch) -> None:  # noqa: F811
    from qlab import night, overlay_stage

    monkeypatch.setenv(cc.ENV, str(tmp_path / "ledger"))
    spec = _spec()
    seen: dict[str, object] = {}

    def fake_measure(session, given, ruleset=None, *, data_end=None):
        seen["end"] = data_end
        return {"idea_id": given.idea_id, "data_end": str(data_end), "differs": "разница",
                "measured_at": "t", "seconds": {"total": 2.0}, "affected": True, "knowable": None,
                "rows": {"as_is": {"error": "нет"}, "as_listed": {"error": "нет"}}}

    monkeypatch.setattr(cc, "spec_for_idea", lambda session, idea_id, spec_dir=None: spec)
    monkeypatch.setattr(cc, "measure_idea", fake_measure)
    monkeypatch.setattr(overlay_stage, "last_run_end", lambda session, idea_id: date(2026, 11, 4))
    monkeypatch.setattr(night, "last_closed_day", lambda: date(2026, 11, 5))
    assert CliRunner().invoke(app, ["init-db"]).exit_code == 0
    result = CliRunner().invoke(app, ["coverage-compare", "toy-slots"])
    assert result.exit_code == 0, result.output
    assert "Два способа считать книгу" in result.output and "разница" in result.output
    assert seen["end"] == date(2026, 11, 4)
    assert cc.load("toy-slots")["differs"] == "разница"
    assert CliRunner().invoke(app, ["coverage-compare"]).exit_code == 1  # nothing named


def test_candidates_are_open_ideas_cut_by_the_complete_book_rule(session_factory) -> None:
    from datetime import UTC, datetime

    from qlab.registry import repo
    from qlab.registry.models import (
        AssetClass,
        DataSnapshot,
        Idea,
        IdeaStatus,
        Profile,
        SourceType,
        TrialRoute,
        TrialStatus,
    )

    session = session_factory()
    repo.add_data_snapshot(session, id="snap-c", source="binance", instruments={},
                           range_start=date(2019, 9, 8), range_end=date(2026, 1, 1), path="p",
                           rows=1)
    assert session.get(DataSnapshot, "snap-c") is not None
    for idea_id, status, coverage in (("cut", IdeaStatus.BENCH, 0.75),
                                      ("whole", IdeaStatus.BENCH, 1.0),
                                      ("never", IdeaStatus.BENCH, 0.0),
                                      ("put-away", IdeaStatus.REJECTED, 0.5)):
        session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                         status=status))
        session.flush()
        spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                             rebalance="1d", costs_model={}, code_ref="x:Y")
        repo.add_trial(session, spec_id=spec.id, config_hash=idea_id, code_sha="x", params={},
                       status=TrialStatus.OK, snapshot_id="snap-c", route=TrialRoute.SHELF,
                       route_reason="r", started_at=datetime(2026, 11, 4, tzinfo=UTC),
                       metrics={"book_coverage": coverage})
    session.commit()
    assert cc.candidates(session) == ["cut"]
    session.close()


def test_the_ledger_default_is_under_data() -> None:
    assert cc.STORE == Path("data/coverage_compare")
