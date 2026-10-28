"""The detector's scan of one trial: the four flags that read the re-run book (on the toy book),
and the three that read the registry (the idea's other stage, the paper check, the family)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import forward_series
from qlab.anomalies import flags as fl
from qlab.anomalies import scan, toy
from qlab.anomalies.rerun import decompose
from qlab.anomalies.test_flags import CFG
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    PaperCheck,
    Profile,
    SourceType,
    TrialRoute,
    TrialStatus,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id: str, *, parent: str | None = None,
          status: IdeaStatus = IdeaStatus.PAPER) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if parent else None))
    session.flush()


def _trial(session, idea_id: str, *, source: str = "binance", code_ref: str = "qlab.s.xs:X",
           metrics: dict | None = None, route: TrialRoute = TrialRoute.PAPER, n: int = 1) -> int:
    spec = repo.add_spec(session, idea_id=idea_id, version=n, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref=code_ref)
    repo.add_data_snapshot(session, id=f"snap-{idea_id}-{n}", source=source, instruments={},
                           range_start=date(2026, 1, 1), range_end=date(2026, 2, 1), path="p",
                           rows=1)
    trial = repo.add_trial(session, spec_id=spec.id, config_hash=f"{idea_id}{n}", code_sha="x",
                           params={}, status=TrialStatus.OK, snapshot_id=f"snap-{idea_id}-{n}",
                           route=route, route_reason="r", metrics=metrics,
                           started_at=datetime(2026, 10, 28, tzinfo=UTC))
    return trial.id


# --- the four flags that read the book ----------------------------------------------------


def test_the_toy_pump_is_flagged_four_ways_with_the_numbers_of_the_book() -> None:
    flags, facts = scan.run_flags(decompose(toy.toy_run()), CFG)

    by = {f.name: f for f in flags}
    day = by["implausible_day"]
    assert day.fired and day.detail["top"][0]["day"] == "2026-01-21"
    assert day.detail["top"][0]["return"] == pytest.approx(0.2, abs=0.01)
    assert day.detail["top"][0]["positions"] == 6.0
    assert day.detail["top"][0]["gross"] == pytest.approx(0.6)
    assert day.detail["top"][0]["net"] == pytest.approx(0.2)

    coins = by["concentrated_instruments"]
    assert coins.fired and coins.detail["top"][0]["instrument"] == "PUMP"
    assert coins.detail["top1_share"] > 0.7

    coin_days = by["concentrated_coin_days"]
    assert coin_days.fired and coin_days.detail["top"][0]["instrument"] == "PUMP"

    thin = by["untradeable_coin_days"]
    assert thin.fired and thin.detail["low_volume_days"] > 0
    assert thin.detail["near_delist_days"] == 8  # DEAD's last bar and the seven before it
    assert thin.detail["unknown_volume_days"] == 2  # A's missing volume figures
    assert facts["reconciles"] is True and facts["instruments"] == 6
    assert facts["start"] == "2026-01-01" and facts["days"] == toy.BARS - 1


def test_the_same_book_without_its_pump_raises_none_of_the_four() -> None:
    run = toy.toy_run()
    run.panel.prices.loc[:, "PUMP"] = 10.0  # the coin never moves
    run.panel.volume.loc[:, "PUMP"] = 5e6 / 10.0
    # A calm book that earns a little on every coin.
    for i, name in enumerate(["A", "B", "C", "D"]):
        run.panel.prices[name] = 100.0 * (1.002 ** pd.Series(range(toy.BARS),
                                                             index=run.panel.prices.index))
        run.weights[name] = 0.1 if i % 2 == 0 else -0.1
    run.panel.volume.loc[:, "DEAD"] = 5e7 / 5.0
    run.panel.volume.loc[:, ["A", "B", "C", "D"]] = 1e8 / 100.0

    flags, _ = scan.run_flags(decompose(run), CFG)

    by = {f.name: f for f in flags}
    assert not by["implausible_day"].fired
    assert not by["untradeable_coin_days"].fired


def test_a_hedged_pair_is_one_coin_so_its_legs_do_not_inflate_the_concentration() -> None:
    frame = pd.DataFrame({"BTC": [1.0], "BTC-SPOT": [-0.9], "ETH": [0.1]})

    merged = scan.by_coin(frame)

    assert merged["BTC"].iloc[0] == pytest.approx(0.1) and list(merged.columns) == ["BTC", "ETH"]
    assert scan.coin_of("BTC-SPOT") == "BTC" and scan.coin_of("SPOTX") == "SPOTX"


# --- the flags that read the registry -----------------------------------------------------


def _store_series(trial_id: int, idea_id: str, daily_return: float, start: str = "2026-01-01",
                  days: int = 40) -> None:
    index = pd.date_range(start, periods=days, freq="D", tz="UTC")
    forward_series.save(forward_series.build_payload(
        trial_id=trial_id, idea_id=idea_id, venue="binance", interval="1d",
        net_return=pd.Series(daily_return, index=index), hold_return=None))


def test_the_two_stages_are_compared_on_their_stored_series(session) -> None:
    _idea(session, "gen")
    _idea(session, "gen-hl", parent="gen")
    t1 = _trial(session, "gen")
    t2 = _trial(session, "gen-hl", source="hyperliquid")
    _store_series(t1, "gen", 0.004)  # about +330% a year
    _store_series(t2, "gen-hl", 0.0002)

    pair = scan.counterpart_trial(session, t1)
    flag = scan.venue_flag(session, t1, CFG)

    assert pair == (t1, t2, "1")
    assert scan.counterpart_trial(session, t2) == (t1, t2, "2")
    assert flag.fired and flag.detail["stage1_trial"] == t1 and flag.detail["stage2_trial"] == t2
    assert flag.value > 0.5


def test_an_idea_with_one_stage_has_an_undefined_venue_flag(session) -> None:
    _idea(session, "solo")
    t = _trial(session, "solo")

    assert scan.counterpart_trial(session, t) is None
    flag = scan.venue_flag(session, t, CFG)
    assert flag.state == fl.UNDEFINED and "второй ступени" in flag.reason


def test_the_paper_check_is_used_only_for_the_run_it_was_made_on(session) -> None:
    _idea(session, "booked")
    t = _trial(session, "booked")
    session.add(PaperCheck(idea_id="booked", book="b", at=date(2026, 10, 28),
                           window_start=date(2026, 9, 1), window_end=date(2026, 10, 1),
                           paper_return=0.01, stand_return=0.09, stand_trial_id=t, note=""))
    session.flush()

    assert scan.paper_flag(session, "booked", t, CFG).fired
    other = scan.paper_flag(session, "booked", t + 1, CFG)
    assert other.state == fl.UNDEFINED and "другому прогону" in other.reason
    assert scan.paper_flag(session, "nobody", t, CFG).state == fl.UNDEFINED

    session.query(PaperCheck).one().note = "Отрезки разные: ..."
    assert "разные" in scan.paper_flag(session, "booked", t, CFG).reason


def test_the_family_is_the_other_ideas_of_the_same_strategy(session) -> None:
    for i in range(6):
        _idea(session, f"fam-{i}")
        _trial(session, f"fam-{i}", code_ref="qlab.strategies.xsmom:X",
               metrics={"ann_return_net": 0.10 + 0.02 * i})
    _idea(session, "star")
    star = _trial(session, "star", code_ref="qlab.strategies.xsmom:X",
                  metrics={"ann_return_net": 0.90})
    _idea(session, "noise-x")  # calibration noise is no family member
    _trial(session, "noise-x", code_ref="qlab.strategies.xsmom:X", metrics={"ann_return_net": 5.0})
    _idea(session, "stranger")  # another strategy
    _trial(session, "stranger", code_ref="qlab.strategies.trend:T",
           metrics={"ann_return_net": 7.0})

    others = scan.family_returns(session, "star")
    flag = scan.family_flag(session, "star", {"ann_return_net": 0.90}, CFG)

    assert sorted(others) == pytest.approx([0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    assert flag.fired and flag.detail["members"] == 6
    assert star  # the run exists
    assert scan.family_returns(session, "stranger") == []  # alone in its family


def test_a_run_that_cannot_be_re_run_still_gets_the_registry_flags(session) -> None:
    _idea(session, "gone")
    t = _trial(session, "gone")

    result = scan.scan_trial(session, t, CFG)

    by = {f.name: f for f in result.flags}
    assert [f.name for f in result.flags] == list(CFG.flags)  # all seven, in order
    assert by["implausible_day"].state == fl.UNDEFINED
    assert "не удалось повторить" in by["implausible_day"].reason
    assert "rerun_error" in result.facts and result.fired == []
    assert result.idea_id == "gone" and result.route == "paper"


# --- the flags added after the XSMOM investigation ------------------------------------------


def test_the_pump_is_a_tail_move_with_its_coin_move_volume_and_age() -> None:
    flags, _ = scan.run_flags(decompose(toy.toy_run()), CFG)

    by = {f.name: f for f in flags}
    tail = by["tail_move_coin_days"]
    assert tail.fired and tail.detail["tail_coin_days"] == 1
    top = tail.detail["top"][0]
    assert (top["instrument"], top["day"]) == ("PUMP", "2026-01-21")
    assert top["move"] == pytest.approx(2.0) and top["volume_usd"] == pytest.approx(2e5)
    assert top["age_days"] == toy.PUMP_BAR
    # The implausible day names the legs that made it.
    leg = by["implausible_day"].detail["top"][0]["legs"][0]
    assert leg["instrument"] == "PUMP" and leg["move"] == pytest.approx(2.0)


def test_coins_whose_price_never_moves_are_found_as_frozen_stretches() -> None:
    flags, _ = scan.run_flags(decompose(toy.toy_run()), CFG)

    frozen = {f.name: f for f in flags}["frozen_price"]
    # DEAD sits at 5.0 for its 30 priced bars and PUMP at 10.0 until the jump: flat prices.
    assert frozen.fired
    assert frozen.detail["longest"][0]["instrument"] == "DEAD"
    assert frozen.detail["longest"][0]["days"] >= 28


def test_frozen_stretches_need_the_minimum_run_in_a_row() -> None:
    idx = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    zero = pd.DataFrame({"A": [True, True, False, True, True, True, False, False]}, index=idx)

    inside, longest = scan.frozen_stretches(zero, 3)

    assert inside["A"].tolist() == [False, False, False, True, True, True, False, False]
    assert longest == [{"instrument": "A", "days": 3, "from": "2026-01-04"}]


def test_the_listing_flag_reads_the_execution_venues_list(monkeypatch) -> None:
    from qlab.anomalies import toolkit

    run = toy.toy_run()
    dec = decompose(run)
    listed = pd.DataFrame(True, index=run.panel.prices.index, columns=run.panel.prices.columns)
    listed["PUMP"] = False
    listed.iloc[:5] = False  # the venue's list starts on bar 5
    monkeypatch.setattr(toolkit, "listed_mask", lambda session, stored: listed)

    flag = scan.listing_flag(None, run, dec, CFG)

    assert flag.fired and flag.detail["from"] == "2026-01-06"
    assert flag.detail["top"][0]["instrument"] == "PUMP"
    assert flag.detail["unlisted_coin_days"] > 0

    monkeypatch.setattr(toolkit, "listed_mask", lambda session, stored: None)
    assert scan.listing_flag(None, run, dec, CFG).state == fl.UNDEFINED
    on_venue = toy.toy_run(venue="hyperliquid")
    assert "самой биржи исполнения" in scan.listing_flag(None, on_venue, dec, CFG).reason
