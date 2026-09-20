"""Hyperliquid fetcher tests — respx-mocked, no network access."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from qlab.data.sources import hyperliquid as hl


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hl, "polite_sleep", lambda: None)


def _candle(ts: pd.Timestamp, close: float) -> dict:
    ts_ms = int(ts.timestamp() * 1000)
    return {
        "t": ts_ms,
        "T": ts_ms + 3_600_000,
        "s": "BTC",
        "i": "1h",
        "o": str(close),
        "c": str(close),
        "h": str(close),
        "l": str(close),
        "v": "1",
        "n": 1,
    }


@respx.mock
def test_fetch_candles_single_page():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T02:00:00", tz="UTC")
    candles = [
        _candle(start, 100.0),
        _candle(start + pd.Timedelta(hours=1), 101.0),
        _candle(end, 102.0),
    ]
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=candles))

    with httpx.Client() as client:
        series = hl.fetch_candles(client, "BTC", "1h", start, end)

    assert len(series) == 3
    assert series.loc[start] == 100.0
    assert series.loc[end] == 102.0
    assert series.index.tz is not None


@respx.mock
def test_fetch_candles_pages_until_short_page(monkeypatch):
    monkeypatch.setattr(hl, "_MAX_CANDLES_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T03:00:00", tz="UTC")
    page1 = [_candle(start, 1.0), _candle(start + pd.Timedelta(hours=1), 2.0)]
    page2 = [_candle(start + pd.Timedelta(hours=2), 3.0), _candle(end, 4.0)]
    route = respx.post(hl.BASE_URL)
    route.side_effect = [httpx.Response(200, json=page1), httpx.Response(200, json=page2)]

    with httpx.Client() as client:
        series = hl.fetch_candles(client, "BTC", "1h", start, end)

    assert len(series) == 4
    assert route.call_count == 2
    assert list(series.values) == [1.0, 2.0, 3.0, 4.0]


@respx.mock
def test_fetch_funding_pages_until_short_page(monkeypatch):
    monkeypatch.setattr(hl, "_MAX_FUNDING_PER_PAGE", 2)
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T03:00:00", tz="UTC")

    def entry(ts: pd.Timestamp, rate: float) -> dict:
        return {
            "coin": "BTC",
            "fundingRate": str(rate),
            "premium": "0",
            "time": int(ts.timestamp() * 1000),
        }

    page1 = [entry(start, 0.0001), entry(start + pd.Timedelta(hours=1), 0.0002)]
    page2 = [entry(start + pd.Timedelta(hours=2), 0.0003)]
    route = respx.post(hl.BASE_URL)
    route.side_effect = [httpx.Response(200, json=page1), httpx.Response(200, json=page2)]

    with httpx.Client() as client:
        series = hl.fetch_funding(client, "BTC", start, end)

    assert len(series) == 3
    assert route.call_count == 2
    assert series.loc[start] == 0.0001


@respx.mock
def test_fetch_meta_reports_delisted_flag():
    payload = [
        {
            "universe": [
                {"name": "BTC", "isDelisted": False, "maxLeverage": 50},
                {"name": "LUNA", "isDelisted": True, "maxLeverage": 10},
            ]
        },
        [{}, {}],
    ]
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=payload))

    with httpx.Client() as client:
        meta = hl.fetch_meta(client)

    assert meta["BTC"]["is_delisted"] is False
    assert meta["LUNA"]["is_delisted"] is True


@respx.mock
def test_describe_universe_includes_delisted():
    payload = [
        {
            "universe": [
                {"name": "BTC", "isDelisted": False, "maxLeverage": 50},
                {"name": "LUNA", "isDelisted": True, "maxLeverage": 10},
            ]
        },
        [{}, {}],
    ]
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=payload))

    described = hl.describe_universe()

    assert described == [("BTC", False), ("LUNA", True)]


@respx.mock
def test_discover_universe_returns_full_instrument_list_including_delisted():
    payload = [
        {
            "universe": [
                {"name": "BTC", "isDelisted": False, "maxLeverage": 50},
                {"name": "LUNA", "isDelisted": True, "maxLeverage": 10},
            ]
        },
        [{}, {}],
    ]
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=payload))

    discovered = hl.discover_universe()

    assert discovered == ["BTC", "LUNA"]  # LUNA (delisted) is NOT dropped


@respx.mock
def test_fetch_instrument_history_raises_on_no_candles():
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=[]))
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")

    with httpx.Client() as client, pytest.raises(ValueError, match="no candle data"):
        hl.fetch_instrument_history(client, "NOPE", "1h", start, end, meta={})
