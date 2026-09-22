"""Hyperliquid fetcher tests — respx-mocked, no network access."""

from __future__ import annotations

import json

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
        frame = hl.fetch_candles(client, "BTC", "1h", start, end)

    assert len(frame) == 3
    assert frame["price"].loc[start] == 100.0
    assert frame["price"].loc[end] == 102.0
    assert frame.index.tz is not None
    assert frame["volume"].loc[start] == 1.0
    assert frame["trade_count"].loc[start] == 1


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
        frame = hl.fetch_candles(client, "BTC", "1h", start, end)

    assert len(frame) == 4
    assert route.call_count == 2
    assert list(frame["price"].values) == [1.0, 2.0, 3.0, 4.0]


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


# --------------------------------------------------------------------------
# Spot markets (docs/TASKS.md, T17)
# --------------------------------------------------------------------------

_SPOT_META_PAYLOAD = {
    "tokens": [
        {"name": "USDC", "index": 0},
        {"name": "UBTC", "index": 197},
        {"name": "UETH", "index": 198},
        {"name": "HYPE", "index": 150},
        # A community coin with no perp counterpart -- must never surface
        # in `fetch_spot_meta`'s result.
        {"name": "HFUN", "index": 2},
    ],
    "universe": [
        {"tokens": [197, 0], "name": "@142", "index": 142},
        {"tokens": [198, 0], "name": "@151", "index": 151},
        {"tokens": [150, 0], "name": "HYPE/USDC", "index": 107},
        {"tokens": [2, 0], "name": "@1", "index": 1},
    ],
}

_PERP_META_PAYLOAD = [
    {
        "universe": [
            {"name": "BTC", "isDelisted": False, "maxLeverage": 50},
            {"name": "ETH", "isDelisted": False, "maxLeverage": 50},
            {"name": "HYPE", "isDelisted": False, "maxLeverage": 10},
        ]
    },
    [{}, {}, {}],
]


def _dispatch_by_type(responses: dict):
    """Build a respx side_effect that routes on the request body's "type"
    field -- needed once a test exercises more than one `/info` call shape
    (perp meta, spot meta, candles, ...) in sequence."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        key = body.get("type")
        if key not in responses:
            raise AssertionError(f"unexpected request type {key!r}: {body}")
        value = responses[key]
        return value(body) if callable(value) else httpx.Response(200, json=value)

    return _handler


@respx.mock
def test_fetch_spot_meta_maps_bridge_and_native_tokens():
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=_SPOT_META_PAYLOAD))

    with httpx.Client() as client:
        result = hl.fetch_spot_meta(client, ["BTC", "ETH", "HYPE", "SOL"])

    assert result == {"BTC": "@142", "ETH": "@151", "HYPE": "HYPE/USDC"}
    assert "SOL" not in result  # no matching token in the payload


@respx.mock
def test_describe_spot_universe_excludes_coins_without_perp_counterpart():
    respx.post(hl.BASE_URL).mock(
        side_effect=_dispatch_by_type(
            {"metaAndAssetCtxs": _PERP_META_PAYLOAD, "spotMeta": _SPOT_META_PAYLOAD}
        )
    )

    described = hl.describe_spot_universe()

    assert described == [("BTC-SPOT", False), ("ETH-SPOT", False), ("HYPE-SPOT", False)]


@respx.mock
def test_discover_spot_universe_returns_column_names():
    respx.post(hl.BASE_URL).mock(
        side_effect=_dispatch_by_type(
            {"metaAndAssetCtxs": _PERP_META_PAYLOAD, "spotMeta": _SPOT_META_PAYLOAD}
        )
    )

    assert hl.discover_spot_universe() == ["BTC-SPOT", "ETH-SPOT", "HYPE-SPOT"]


@respx.mock
def test_fetch_spot_instrument_history_has_no_funding():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T01:00:00", tz="UTC")
    candles = [_candle(start, 100.0), _candle(end, 101.0)]
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=candles))

    with httpx.Client() as client:
        hist = hl.fetch_spot_instrument_history(client, "BTC", "@142", "1h", start, end)

    assert hist.instrument == "BTC-SPOT"
    assert hist.has_funding is False
    assert hist.funding.empty
    assert hist.is_delisted is False
    assert len(hist.prices) == 2


@respx.mock
def test_fetch_spot_instrument_history_raises_on_no_candles():
    respx.post(hl.BASE_URL).mock(return_value=httpx.Response(200, json=[]))
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")

    with httpx.Client() as client, pytest.raises(ValueError, match="no spot candle data"):
        hl.fetch_spot_instrument_history(client, "NOPE", "@999", "1d", start, end)


@respx.mock
def test_fetch_spot_universe_returns_columns_keyed_by_coin_dash_spot():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T01:00:00", tz="UTC")

    def _candles_for(body: dict) -> httpx.Response:
        coin = body["req"]["coin"]
        price = 1.0 if coin == "@142" else 2.0
        return httpx.Response(200, json=[_candle(start, price), _candle(end, price + 1)])

    respx.post(hl.BASE_URL).mock(
        side_effect=_dispatch_by_type(
            {
                "metaAndAssetCtxs": _PERP_META_PAYLOAD,
                "spotMeta": _SPOT_META_PAYLOAD,
                "candleSnapshot": _candles_for,
            }
        )
    )

    result = hl.fetch_spot_universe(["BTC", "ETH"], start, end, "1h")

    assert set(result) == {"BTC-SPOT", "ETH-SPOT"}
    assert result["BTC-SPOT"].prices.iloc[0] == 1.0
    assert result["ETH-SPOT"].prices.iloc[0] == 2.0
    assert all(not h.has_funding for h in result.values())


@respx.mock
def test_fetch_spot_universe_raises_on_coin_with_no_spot_market():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T01:00:00", tz="UTC")
    respx.post(hl.BASE_URL).mock(
        side_effect=_dispatch_by_type(
            {"metaAndAssetCtxs": _PERP_META_PAYLOAD, "spotMeta": _SPOT_META_PAYLOAD}
        )
    )

    with pytest.raises(ValueError, match="no USDC-quoted spot market"):
        hl.fetch_spot_universe(["DOGE"], start, end, "1h", on_missing="raise")


@respx.mock
def test_fetch_spot_universe_skips_missing_when_on_missing_skip():
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01T01:00:00", tz="UTC")

    def _candles_for(body: dict) -> httpx.Response:
        return httpx.Response(200, json=[_candle(start, 1.0), _candle(end, 1.1)])

    respx.post(hl.BASE_URL).mock(
        side_effect=_dispatch_by_type(
            {
                "metaAndAssetCtxs": _PERP_META_PAYLOAD,
                "spotMeta": _SPOT_META_PAYLOAD,
                "candleSnapshot": _candles_for,
            }
        )
    )

    result = hl.fetch_spot_universe(["BTC", "DOGE"], start, end, "1h", on_missing="skip")

    assert set(result) == {"BTC-SPOT"}
