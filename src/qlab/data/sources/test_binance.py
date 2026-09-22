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


def test_discover_universe_is_none_no_network():
    """Binance's free API cannot expose delisted symbols at all -- this must
    return None unconditionally, without making any request."""
    assert bn.discover_universe() is None
    assert bn.discover_universe((pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))) is None


def test_describe_universe_is_none_no_network():
    assert bn.describe_universe() is None


@respx.mock
def test_fetch_instrument_history_raises_on_no_klines():
    respx.get(f"{bn.BASE_URL}/fapi/v1/klines").mock(return_value=httpx.Response(200, json=[]))
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")

    with httpx.Client() as client, pytest.raises(ValueError, match="no kline data"):
        bn.fetch_instrument_history(client, "NOPE", "1h", start, end, exchange_info={})
