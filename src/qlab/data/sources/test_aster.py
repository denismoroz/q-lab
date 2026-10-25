"""Aster USDT perpetuals fetcher tests -- respx-mocked, no network access."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest
import respx
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.data import snapshot as snap
from qlab.data.sources import aster
from qlab.data.sources.base import align_funding_to_index
from qlab.registry.models import Base
from qlab.venues.config import load_venue


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(aster, "polite_sleep", lambda: None)
    monkeypatch.setattr(aster.binance_source, "polite_sleep", lambda: None)


def _ms(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000) if isinstance(ts, str) else int(
        ts.timestamp() * 1000
    )


_HOUR = pd.Timedelta(hours=1)


def _kline(ts: pd.Timestamp, close: float, *, volume: str = "10", trades: int = 5,
           high: float | None = None, step: pd.Timedelta = _HOUR) -> list:
    open_ms = _ms(ts)
    high = close if high is None else high
    return [
        open_ms, str(close), str(high), str(close), str(close), volume,
        open_ms + int(step.total_seconds() * 1000) - 1, "0", trades, "0", "0", "0",
    ]


def _funding(ts: pd.Timestamp, rate: float, symbol: str = "BTCUSDT") -> dict:
    return {"symbol": symbol, "fundingTime": _ms(ts), "fundingRate": f"{rate:.8f}"}


def _symbol_info(symbol: str, status: str = "TRADING", contract: str = "PERPETUAL",
                 quote: str = "USDT", onboard: str | None = "2021-07-30") -> dict:
    return {
        "symbol": symbol, "status": status, "contractType": contract, "quoteAsset": quote,
        "onboardDate": _ms(onboard) if onboard else None,
    }


def _serve(klines: dict[str, list], fundings: dict[str, list]) -> None:
    """Mock /klines and /fundingRate the way the venue answers: ascending,
    startTime/endTime inclusive, at most `limit` rows, 400 for an unknown
    symbol."""

    def klines_handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        rows = klines.get(p["symbol"])
        if rows is None:
            return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
        lo = int(p.get("startTime", 0))
        hi = int(p.get("endTime", 10**15))
        picked = [r for r in rows if lo <= r[0] <= hi]
        return httpx.Response(200, json=picked[: int(p["limit"])])

    def funding_handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        rows = fundings.get(p["symbol"], [])
        lo = int(p.get("startTime", 0))
        hi = int(p.get("endTime", 10**15))
        picked = [r for r in rows if lo <= r["fundingTime"] <= hi]
        return httpx.Response(200, json=picked[: int(p["limit"])])

    respx.get(f"{aster.BASE_URL}/fapi/v1/klines").mock(side_effect=klines_handler)
    respx.get(f"{aster.BASE_URL}/fapi/v1/fundingRate").mock(side_effect=funding_handler)


def test_symbol_and_instrument_names():
    assert aster._symbol("BTC") == "BTCUSDT"
    assert aster._symbol("1000sats") == "1000SATSUSDT"
    assert aster._symbol("BTCUSDT") == "BTCUSDT"
    assert aster._instrument("BTCUSDT") == "BTC"
    assert aster._instrument("1000SATSUSDT") == "1000SATS"
    assert aster._instrument("BTCUSD1") is None
    assert aster._instrument("USDT") is None
    assert aster._instrument("BTCUSDT_250328") is None


@respx.mock
def test_exchange_info_and_listed_universe_keep_only_usdt_perpetuals():
    payload = {"symbols": [
        _symbol_info("BTCUSDT"),
        _symbol_info("TONUSDT", status="SETTLING"),
        _symbol_info("BTCUSD1", quote="USD1"),
        _symbol_info("ASTERUSDC", quote="USDC"),
        _symbol_info("SQQQUSDT", status="PENDING_TRADING", contract=""),
    ]}
    respx.get(f"{aster.BASE_URL}/fapi/v1/exchangeInfo").mock(
        return_value=httpx.Response(200, json=payload)
    )

    with httpx.Client() as client:
        info = aster.fetch_exchange_info(client)
        universe = aster.listed_universe(client)

    assert info["BTCUSDT"]["onboard_date"] == pd.Timestamp("2021-07-30", tz="UTC")
    assert universe == [("BTC", False), ("TON", True)]


def test_universe_is_not_claimed_complete():
    """The venue cannot list its delisted contracts: no honest universe."""
    assert aster.describe_universe() is None
    assert aster.discover_universe() is None


@respx.mock
def test_fetch_candles_parses_close_volume_trades_high_and_pages(monkeypatch):
    monkeypatch.setattr(aster, "_MAX_KLINES_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    hours = [start + pd.Timedelta(hours=h) for h in range(5)]
    rows = [_kline(t, 100.0 + i, volume="2.5", trades=7, high=110.0 + i)
            for i, t in enumerate(hours)]
    _serve({"BTCUSDT": rows}, {})

    with httpx.Client() as client:
        frame = aster.fetch_candles(client, "BTC", "1h", start, hours[-1])

    assert list(frame.index) == hours
    assert frame["price"].iloc[2] == 102.0
    assert frame["high"].iloc[2] == 112.0
    assert frame["volume"].iloc[0] == 2.5
    assert frame["trade_count"].iloc[0] == 7
    assert len(respx.calls) == 3  # 2 + 2 + 1 rows


@respx.mock
def test_fetch_candles_stops_at_end_and_returns_typed_empty_frame():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    _serve({"BTCUSDT": [_kline(start + pd.Timedelta(days=9), 1.0)]}, {})

    with httpx.Client() as client:
        frame = aster.fetch_candles(client, "BTC", "1d", start, start + pd.Timedelta(days=1))

    assert frame.empty
    assert list(frame.columns) == ["price", "volume", "trade_count", "high"]


@respx.mock
def test_fetch_funding_pages_inclusive_start_without_repeats(monkeypatch):
    monkeypatch.setattr(aster, "_MAX_FUNDING_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    marks = [start + pd.Timedelta(hours=8 * i) for i in range(5)]
    # Real stamps carry a few ms of jitter.
    rows = [_funding(t + pd.Timedelta(milliseconds=i), 0.0001 * (i + 1))
            for i, t in enumerate(marks)]
    _serve({}, {"BTCUSDT": rows})

    with httpx.Client() as client:
        series = aster.fetch_funding(client, "BTC", start, marks[-1] + pd.Timedelta(seconds=1))

    assert len(series) == 5
    assert series.is_monotonic_increasing
    assert series.iloc[-1] == pytest.approx(0.0005)


def _daily_history(symbol_status: str | None, *, dead_after: int | None = None):
    """Seven daily bars; bars after `dead_after` (index) carry no trades."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    days = [start + pd.Timedelta(days=d) for d in range(7)]
    rows = []
    for i, d in enumerate(days):
        dead = dead_after is not None and i > dead_after
        rows.append(_kline(d, 10.0 + i, volume="0" if dead else "3", trades=0 if dead else 4,
                           step=pd.Timedelta(days=1)))
    marks = [start + pd.Timedelta(hours=8 * i) for i in range(1, 22)]
    fundings = [_funding(m, 0.0001) for m in marks]
    info = {} if symbol_status is None else {"XUSDT": {"status": symbol_status,
                                                       "onboard_date": None}}
    return start, days, rows, fundings, info


@respx.mock
def test_delisted_contract_is_cut_at_its_last_traded_bar_and_funding_follows():
    start, days, rows, fundings, info = _daily_history("SETTLING", dead_after=3)
    _serve({"XUSDT": rows}, {"XUSDT": fundings})

    with httpx.Client() as client:
        hist = aster.fetch_instrument_history(client, "X", "1d", start, days[-1], info)

    assert hist.is_delisted is True
    assert hist.last_seen == days[3]
    assert hist.prices.index.max() == days[3]
    assert hist.volume.index.max() == days[3]
    assert hist.funding.index.max() <= days[3] + pd.Timedelta(days=1)
    assert len(hist.funding) < len(fundings)
    assert hist.high is not None and hist.high.index.max() == days[3]


@respx.mock
def test_live_contract_keeps_its_zero_trade_bars():
    start, days, rows, fundings, info = _daily_history("TRADING", dead_after=3)
    _serve({"XUSDT": rows}, {"XUSDT": fundings})

    with httpx.Client() as client:
        hist = aster.fetch_instrument_history(client, "X", "1d", start, days[-1], info)

    assert hist.is_delisted is False
    assert hist.last_seen == days[-1]
    assert (hist.trade_count.iloc[4:] == 0).all()  # a quiet market, not a dead one
    assert (hist.volume.iloc[4:] == 0.0).all()


@respx.mock
def test_contract_absent_from_exchange_info_is_delisted_and_served_by_name():
    start, days, rows, fundings, _ = _daily_history(None)
    _serve({"XUSDT": rows}, {"XUSDT": fundings})

    with httpx.Client() as client:
        hist = aster.fetch_instrument_history(client, "X", "1d", start, days[-1], {})

    assert hist.is_delisted is True
    assert hist.last_seen == days[-1]


@respx.mock
def test_delisted_contract_with_no_traded_bar_in_window_is_missing():
    start, days, rows, fundings, info = _daily_history("SETTLING", dead_after=-1)
    _serve({"XUSDT": rows}, {"XUSDT": fundings})

    with httpx.Client() as client, pytest.raises(ValueError, match="no traded bar"):
        aster.fetch_instrument_history(client, "X", "1d", start, days[-1], info)


@respx.mock
def test_unknown_symbol_is_no_data_but_other_client_errors_are_not():
    _serve({}, {})
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = start + pd.Timedelta(days=1)

    with httpx.Client() as client:
        assert aster.fetch_candles(client, "NOPE", "1d", start, end).empty
        with pytest.raises(ValueError, match="no kline data"):
            aster.fetch_instrument_history(client, "NOPE", "1d", start, end, {})

    respx.get(f"{aster.BASE_URL}/fapi/v1/klines").mock(
        return_value=httpx.Response(400, json={"code": -1130, "msg": "bad limit"}))
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        aster.fetch_candles(client, "BTC", "1d", start, end)


@respx.mock
def test_no_klines_in_window_raises_value_error():
    _serve({"XUSDT": []}, {})
    start = pd.Timestamp("2024-01-01", tz="UTC")

    with httpx.Client() as client, pytest.raises(ValueError, match="no kline data"):
        aster.fetch_instrument_history(client, "X", "1d", start, start, {})


@respx.mock
def test_probe_asks_only_about_unknown_names_and_remembers_the_answers(tmp_path):
    start = pd.Timestamp("2022-05-01", tz="UTC")
    _serve({"LUNAUSDT": [_kline(start, 1.0, step=pd.Timedelta(days=1))]}, {})

    with httpx.Client() as client:
        first = aster.probe_symbols(client, ["LUNAUSDT", "SRMUSDT"], tmp_path)
        calls_after_first = len(respx.calls)
        second = aster.probe_symbols(client, ["LUNAUSDT", "SRMUSDT", "NOPEUSDT"], tmp_path)

    assert first == {"LUNAUSDT": _ms(start), "SRMUSDT": None}
    assert second == {"LUNAUSDT": _ms(start), "SRMUSDT": None, "NOPEUSDT": None}
    assert len(respx.calls) == calls_after_first + 1  # only NOPEUSDT was new
    cache = json.loads((tmp_path / "aster" / "_probe.json").read_text())
    assert set(cache) == {"LUNAUSDT", "SRMUSDT", "NOPEUSDT"}


@respx.mock
def test_probe_survives_a_corrupt_cache_file(tmp_path):
    (tmp_path / "aster").mkdir()
    (tmp_path / "aster" / "_probe.json").write_text("{not json")
    _serve({"LUNAUSDT": [_kline(pd.Timestamp("2022-05-01", tz="UTC"), 1.0)]}, {})

    with httpx.Client() as client:
        probed = aster.probe_symbols(client, ["LUNAUSDT"], tmp_path)

    assert probed["LUNAUSDT"] is not None


_ARCHIVE_PAGE = (
    "<ListBucketResult>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/LUNAUSDT/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/SRMUSDT/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT_250328/</Prefix>"
    "</CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHBUSD/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)


@respx.mock
def test_best_effort_universe_adds_binance_names_aster_still_serves(tmp_path):
    respx.get(f"{aster.BASE_URL}/fapi/v1/exchangeInfo").mock(return_value=httpx.Response(
        200, json={"symbols": [_symbol_info("BTCUSDT")]}))
    respx.get(aster.binance_source.ARCHIVE_LIST_URL).mock(
        return_value=httpx.Response(200, text=_ARCHIVE_PAGE))
    _serve({"LUNAUSDT": [_kline(pd.Timestamp("2022-05-01", tz="UTC"), 1.0)]}, {})

    universe = aster.best_effort_universe(tmp_path)

    # BTC is live; LUNA is served by name so it existed on Aster; SRM answers
    # "Invalid symbol"; quarterlies and BUSD quotes are not USDT perpetuals.
    assert universe == [("BTC", False), ("LUNA", True)]


# --- funding alignment: mixed settlement intervals on one 8-hour panel grid ---


def _settlements(start: str, n: int, every: pd.Timedelta, rate: float) -> pd.Series:
    t0 = pd.Timestamp(start, tz="UTC")
    return pd.Series(rate, index=[t0 + every * i for i in range(n)])


@pytest.mark.parametrize(("every", "per_day"), [
    (pd.Timedelta(hours=8), 3), (pd.Timedelta(hours=4), 6), (pd.Timedelta(hours=1), 24),
])
def test_daily_funding_sums_every_settlement_whatever_the_symbols_interval(every, per_day):
    raw = _settlements("2024-01-01 00:00", 4 * per_day, every, 0.0001)
    raw.index = raw.index + pd.Timedelta(milliseconds=3)  # venue jitter
    index = pd.date_range("2024-01-02", periods=3, freq="1D", tz="UTC")

    out = align_funding_to_index(raw, index, aster.FUNDING_NATIVE_INTERVAL)

    assert out.tolist() == pytest.approx([0.0001 * per_day] * 3)


def test_hourly_panel_over_4h_funding_lands_on_the_8h_slot_end_and_loses_nothing():
    """The panel's native interval is 8h, so a 4h settlement is attributed to
    the 8h slot it falls in: the 04:00 settlement arrives with the 08:00 one.
    A known limitation for HOURLY panels over sub-8h symbols (funding arrives
    up to 4h late, never early, and the total is kept); daily panels, the
    ones this source is used for, are exact."""
    raw = _settlements("2024-01-01 00:00", 6, pd.Timedelta(hours=4), 0.0002)
    index = pd.date_range("2024-01-01 01:00", periods=24, freq="1h", tz="UTC")

    out = align_funding_to_index(raw, index, aster.FUNDING_NATIVE_INTERVAL)

    assert out.notna().all()
    assert out.sum() == pytest.approx(0.0002 * 5)  # 00:00 closes the previous day
    assert out[pd.Timestamp("2024-01-01 08:00", tz="UTC")] == pytest.approx(0.0004)
    assert out[pd.Timestamp("2024-01-01 05:00", tz="UTC")] == 0.0


# --- registration, cache, snapshot -------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _serve_two_coins() -> pd.DatetimeIndex:
    days = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    marks = [days[0] + pd.Timedelta(hours=8 * i) for i in range(1, 6 * 3 + 1)]
    klines = {
        "BTCUSDT": [_kline(d, 100.0 + i, volume="5", trades=9, high=105.0 + i,
                           step=pd.Timedelta(days=1)) for i, d in enumerate(days)],
        "ETHUSDT": [_kline(d, 50.0 + i, volume="8", trades=3, step=pd.Timedelta(days=1))
                    for i, d in enumerate(days)],
    }
    fundings = {
        "BTCUSDT": [_funding(m, 0.0001) for m in marks],
        "ETHUSDT": [_funding(m, 0.0002, "ETHUSDT") for m in marks],
    }
    _serve(klines, fundings)
    respx.get(f"{aster.BASE_URL}/fapi/v1/exchangeInfo").mock(return_value=httpx.Response(
        200, json={"symbols": [_symbol_info("BTCUSDT"), _symbol_info("ETHUSDT")]}))
    return days


def test_aster_is_a_registered_source_with_the_8h_native_interval():
    spec = snap._SOURCES["aster"]
    assert spec["funding_native_interval"] == pd.Timedelta(hours=8)
    assert spec["fetch"] is aster.fetch_universe
    assert spec["discover_universe"]() is None
    assert "fetch_spot" not in spec


@respx.mock
def test_tiny_snapshot_build_and_load_round_trip(tmp_path, monkeypatch, session):
    monkeypatch.chdir(tmp_path)  # raw cache lands in ./data/raw, not the repo's
    days = _serve_two_coins()

    panel = snap.build_snapshot(
        "aster", ["BTC", "ETH"], days[0], days[-1], "1d",
        snapshots_dir=tmp_path / "snapshots", session=session,
    )

    assert panel.meta["venue"] == "aster"
    assert panel.meta["universe_complete"] is False
    assert list(panel.prices.columns) == ["BTC", "ETH"]
    assert panel.prices["BTC"].iloc[0] == 100.0
    assert panel.volume["ETH"].iloc[1] == 8.0
    assert panel.high is not None and panel.high["BTC"].iloc[2] == 107.0
    # Three 8-hour settlements per day: a full day is 3 x the rate. The first
    # bar's day began before the first record the venue serves (the fetch
    # starts at `days[0]`), so it is unknown -- NaN, never a partial sum.
    assert pd.isna(panel.funding["BTC"].iloc[0])
    assert panel.funding["BTC"].iloc[1:].tolist() == pytest.approx([0.0003] * 5)
    assert panel.funding["ETH"].iloc[1:].tolist() == pytest.approx([0.0006] * 5)
    assert not panel.tradeable["BTC"].iloc[0]
    assert panel.tradeable["BTC"].iloc[1:].all()

    loaded = snap.load_snapshot(panel.snapshot_id, session=session)
    pd.testing.assert_frame_equal(loaded.prices, panel.prices, check_freq=False)
    pd.testing.assert_frame_equal(loaded.funding, panel.funding, check_freq=False)
    assert loaded.meta["venue"] == "aster"
    assert loaded.meta["universe_complete"] is False


@respx.mock
def test_second_fetch_is_served_from_the_raw_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    days = _serve_two_coins()

    first = aster.fetch_universe(["BTC"], days[0], days[-1], "1d")
    calls = len(respx.calls)
    second = aster.fetch_universe(["BTC"], days[0], days[-1], "1d")

    # Only exchangeInfo is asked again; candles and funding come from disk.
    assert len(respx.calls) == calls + 1
    pd.testing.assert_series_equal(
        first["BTC"].prices, second["BTC"].prices, check_freq=False, check_names=False
    )
    assert (tmp_path / "data" / "raw" / "aster" / "1d").is_dir()


@respx.mock
def test_discovered_universe_is_refused_instead_of_claimed_complete(tmp_path, session):
    with pytest.raises(ValueError, match="does not expose a full point-in-time universe"):
        snap.build_snapshot("aster", None, "2024-01-01", "2024-01-03", "1d",
                            snapshots_dir=tmp_path, session=session)


def test_skip_mode_drops_an_unknown_name_and_raise_mode_does_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with respx.mock:
        days = _serve_two_coins()
        got = aster.fetch_universe(["BTC", "NOPE"], days[0], days[-1], "1d", on_missing="skip")
        assert list(got) == ["BTC"]
        with pytest.raises(ValueError, match="no kline data"):
            aster.fetch_universe(["NOPE"], days[0], days[-1], "1d")


def test_venue_config_records_facts_and_leaves_unknowns_unset():
    venue = load_venue("aster")

    assert venue is not None
    assert venue.id == "aster"
    assert venue.execution_adapter is False
    assert venue.free_public_data is True
    assert venue.min_leg_notional_usd == 5.0
    assert venue.supports_atomic_multileg is None
    assert venue.uncovered_leg_recovery is None
