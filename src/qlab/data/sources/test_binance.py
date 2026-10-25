"""Binance USDT-M perpetuals fetcher tests — respx-mocked, no network access."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from qlab.data.sources import binance as bn


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(bn, "polite_sleep", lambda: None)


def _kline(ts: pd.Timestamp, close: float) -> list:
    open_ms = int(ts.timestamp() * 1000)
    close_str = str(close)
    return [
        open_ms, close_str, close_str, close_str, close_str,
        "10", open_ms + 3_599_999, "0", 1, "0", "0", "0",
    ]


@respx.mock
def test_symbol_appends_usdt():
    assert bn._symbol("BTC") == "BTCUSDT"
    assert bn._symbol("btc") == "BTCUSDT"
    assert bn._symbol("BTCUSDT") == "BTCUSDT"


@respx.mock
def test_fetch_candles_single_page():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T02:00:00", tz="UTC")
    klines = [
        _kline(start, 100.0),
        _kline(start + pd.Timedelta(hours=1), 101.0),
        _kline(end, 102.0),
    ]
    respx.get(f"{bn.BASE_URL}/fapi/v1/klines").mock(return_value=httpx.Response(200, json=klines))

    with httpx.Client() as client:
        frame = bn.fetch_candles(client, "BTC", "1h", start, end)

    assert len(frame) == 3
    assert frame["price"].loc[start] == 100.0
    assert frame["price"].loc[end] == 102.0
    assert frame["volume"].loc[start] == 10.0
    assert frame["trade_count"].loc[start] == 1


@respx.mock
def test_fetch_candles_pages_until_short_page(monkeypatch):
    monkeypatch.setattr(bn, "_MAX_KLINES_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T03:00:00", tz="UTC")
    page1 = [_kline(start, 1.0), _kline(start + pd.Timedelta(hours=1), 2.0)]
    page2 = [_kline(start + pd.Timedelta(hours=2), 3.0), _kline(end, 4.0)]
    route = respx.get(f"{bn.BASE_URL}/fapi/v1/klines")
    route.side_effect = [httpx.Response(200, json=page1), httpx.Response(200, json=page2)]

    with httpx.Client() as client:
        frame = bn.fetch_candles(client, "BTC", "1h", start, end)

    assert len(frame) == 4
    assert route.call_count == 2


@respx.mock
def test_fetch_funding_pages_until_short_page(monkeypatch):
    monkeypatch.setattr(bn, "_MAX_FUNDING_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T16:00:00", tz="UTC")

    def entry(ts: pd.Timestamp, rate: float) -> dict:
        return {
            "symbol": "BTCUSDT",
            "fundingRate": str(rate),
            "fundingTime": int(ts.timestamp() * 1000),
        }

    page1 = [entry(start, 0.0001), entry(start + pd.Timedelta(hours=8), 0.0002)]
    page2 = [entry(start + pd.Timedelta(hours=16), 0.0003)]
    route = respx.get(f"{bn.BASE_URL}/fapi/v1/fundingRate")
    route.side_effect = [httpx.Response(200, json=page1), httpx.Response(200, json=page2)]

    with httpx.Client() as client:
        series = bn.fetch_funding(client, "BTC", start, end)

    assert len(series) == 3
    assert route.call_count == 2


@respx.mock
def test_fetch_exchange_info_parses_onboard_date():
    onboard = pd.Timestamp("2021-01-01", tz="UTC")
    payload = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "onboardDate": int(onboard.timestamp() * 1000),
            },
            {"symbol": "OLDUSDT", "status": "TRADING", "onboardDate": None},
        ]
    }
    exchange_info_url = f"{bn.BASE_URL}/fapi/v1/exchangeInfo"
    respx.get(exchange_info_url).mock(return_value=httpx.Response(200, json=payload))

    with httpx.Client() as client:
        info = bn.fetch_exchange_info(client)

    assert info["BTCUSDT"]["status"] == "TRADING"
    assert info["BTCUSDT"]["onboard_date"] == onboard
    assert info["OLDUSDT"]["onboard_date"] is None


@respx.mock
def test_fetch_instrument_history_marks_delisted_when_absent_from_exchange_info():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T02:00:00", tz="UTC")
    klines = [_kline(start, 1.0), _kline(end, 2.0)]
    respx.get(f"{bn.BASE_URL}/fapi/v1/klines").mock(return_value=httpx.Response(200, json=klines))
    respx.get(f"{bn.BASE_URL}/fapi/v1/fundingRate").mock(return_value=httpx.Response(200, json=[]))

    with httpx.Client() as client:
        hist = bn.fetch_instrument_history(client, "GONE", "1h", start, end, exchange_info={})

    assert hist.is_delisted is True
    assert hist.last_seen == end


@respx.mock
def test_a_delisted_contracts_flat_zero_trade_candles_are_not_its_life():
    """A contract taken off the market keeps answering with settlement-price candles that
    never traded (DENT, FIO, 2026-04); its life ends with its last traded bar."""
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T04:00:00", tz="UTC")
    klines = [_kline(start + pd.Timedelta(hours=h), 1.0 + 0.1 * h) for h in range(5)]
    for k in klines[3:]:
        k[8] = 0  # numberOfTrades: nothing traded
    respx.get(f"{bn.BASE_URL}/fapi/v1/klines").mock(return_value=httpx.Response(200, json=klines))
    respx.get(f"{bn.BASE_URL}/fapi/v1/fundingRate").mock(return_value=httpx.Response(200, json=[]))

    with httpx.Client() as client:
        gone = bn.fetch_instrument_history(client, "GONE", "1h", start, end, exchange_info={})
        alive = bn.fetch_instrument_history(
            client, "LIVE", "1h", start, end,
            exchange_info={"LIVEUSDT": {"status": "TRADING", "onboard_date": None}})

    assert gone.last_seen == start + pd.Timedelta(hours=2)
    assert len(gone.prices) == 3
    assert alive.last_seen == end and len(alive.prices) == 5  # a quiet hour is not a delisting


_PAGE_1 = (
    "<ListBucketResult><NextMarker>data/futures/um/monthly/klines/ETHUSDT/</NextMarker>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT_250328/</Prefix>"
    "</CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/SRMUSDT/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)
_PAGE_2 = (
    "<ListBucketResult>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHBUSD/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)


@respx.mock
def test_universe_is_the_archive_plus_exchange_info_with_delisted_flagged(monkeypatch):
    """The archive lists every contract ever listed (SRM is long delisted and
    absent from exchangeInfo); dated quarterlies and non-USDT quotes are not
    perpetual USDT instruments."""
    monkeypatch.setattr(bn, "polite_sleep", lambda: None)
    respx.get(bn.ARCHIVE_LIST_URL).mock(side_effect=lambda req: httpx.Response(
        200, text=_PAGE_2 if req.url.params.get("marker") else _PAGE_1))
    respx.get(f"{bn.BASE_URL}/fapi/v1/exchangeInfo").mock(return_value=httpx.Response(200, json={
        "symbols": [{"symbol": "BTCUSDT", "status": "TRADING"},
                    {"symbol": "FTTUSDT", "status": "SETTLING"}]}))

    described = bn.describe_universe()

    assert described == [("BTC", False), ("ETH", True), ("FTT", True), ("SRM", True)]
    assert bn.discover_universe() == ["BTC", "ETH", "FTT", "SRM"]


@respx.mock
def test_fetch_instrument_history_raises_on_no_klines():
    respx.get(f"{bn.BASE_URL}/fapi/v1/klines").mock(return_value=httpx.Response(200, json=[]))
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")

    with httpx.Client() as client, pytest.raises(ValueError, match="no kline data"):
        bn.fetch_instrument_history(client, "NOPE", "1h", start, end, exchange_info={})


# --------------------------------------------------------------------------
# Binance spot as the spot leg of Hyperliquid perps (docs/FRAB_BINANCE_SPOT.md)
# --------------------------------------------------------------------------


def test_a_thousand_unit_contract_maps_to_the_one_unit_token_with_a_scale():
    assert bn.spot_pair("kPEPE") == ("PEPE", 1000.0)
    assert bn.spot_pair("kBONK") == ("BONK", 1000.0)
    assert bn.spot_pair("BTC") == ("BTC", 1.0)
    # A lowercase k is only the thousand prefix in front of an upper-case ticker.
    assert bn.spot_pair("kaito") == ("kaito", 1.0)
    assert bn.spot_pair("k") == ("k", 1.0)


def test_pairing_keeps_only_perps_that_have_a_usdt_spot_pair_delisted_ones_included(monkeypatch):
    monkeypatch.setattr(bn, "spot_archive_symbols",
                        lambda client: ["BTCUSDT", "SRMUSDT", "PEPEUSDT", "ETHBTC", "HYPEUSDC"])
    found = bn.pair_spot_universe(["BTC", "SRM", "kPEPE", "PURR", "ETH", "HYPE"])
    # PURR has no pair at all; ETH only an ETHBTC pair; HYPE only a USDC pair.
    assert found == ["BTC-SPOT", "SRM-SPOT", "kPEPE-SPOT"]


@respx.mock
def test_spot_prices_of_a_thousand_unit_contract_are_scaled_to_its_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(bn, "fetch_universe_resumable",
                        lambda columns, start, end, interval, **kw: {
                            c: kw["fetch_one"](c) for c in columns})
    start = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2026-01-01T01:00:00", tz="UTC")
    seen = []

    def answer(request):
        seen.append(request.url.params["symbol"])
        return httpx.Response(200, json=[_kline(start, 0.004), _kline(end, 0.005)])

    respx.get(f"{bn.SPOT_BASE_URL}/api/v3/klines").mock(side_effect=answer)
    histories = bn.fetch_spot_universe(["kPEPE", "BTC"], start, end, "1h")
    assert seen == ["PEPEUSDT", "BTCUSDT"]  # the pair is PEPE's, not a "KPEPE" that does not exist
    pepe, btc = histories["kPEPE-SPOT"], histories["BTC-SPOT"]
    assert list(pepe.prices) == pytest.approx([4.0, 5.0])
    assert list(pepe.high) == pytest.approx([4.0, 5.0])
    assert list(pepe.volume) == pytest.approx([0.01, 0.01])  # 10 PEPE = 0.01 thousand
    assert list(btc.prices) == [0.004, 0.005]  # an ordinary coin is untouched
    assert not pepe.has_funding


def test_binances_own_numeric_prefixes_map_to_the_one_unit_spot_token():
    assert bn.spot_pair("1000PEPE") == ("PEPE", 1000.0)
    assert bn.spot_pair("1000000MOG") == ("MOG", 1_000_000.0)
    assert bn.spot_pair("1MBABYDOGE") == ("BABYDOGE", 1_000_000.0)
    assert bn.spot_pair("10000SATS") == ("SATS", 10_000.0)
    # An ordinary ticker that merely starts with a digit is untouched.
    assert bn.spot_pair("1INCH") == ("1INCH", 1.0)
    assert bn.spot_pair("1000") == ("1000", 1.0)


def test_pairing_a_binance_perp_with_its_own_spot_uses_the_same_mapping(monkeypatch):
    monkeypatch.setattr(bn, "spot_archive_symbols", lambda client: ["PEPEUSDT", "BTCUSDT"])
    assert bn.pair_spot_universe(["1000PEPE", "BTC", "BTCDOM"]) == ["1000PEPE-SPOT", "BTC-SPOT"]
