"""Tests for `qlab.data.recorder` against a fake Hyperliquid /info -- no network.

The property that matters most: a candle the venue stops serving stays in
our store (docs/TASKS.md T33)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qlab.data import recorder
from qlab.data.sources import hyperliquid as hl

DAY_MS = 86_400_000
T0 = pd.Timestamp("2026-01-01", tz="UTC")


class FakeVenue:
    """Serves daily candles for `xyz:A` on days `served_days` (offsets from
    T0), hourly funding on the same span, and a listing where `xyz:DEAD` is
    delisted. Records every request it sees."""

    def __init__(self, served_days: range) -> None:
        self.served_days = served_days
        self.requests: list[dict] = []

    def __call__(self, client, payload):
        self.requests.append(payload)
        kind = payload["type"]
        if kind == "metaAndAssetCtxs":
            universe = [{"name": "xyz:A"}, {"name": "xyz:DEAD", "isDelisted": True}]
            return [{"universe": universe}, [{"dayNtlVlm": "1"}, {"dayNtlVlm": "0"}]]
        if kind == "candleSnapshot":
            req = payload["req"]
            if req["coin"] != "xyz:A":
                return []
            out = []
            for d in self.served_days:
                t = int(T0.timestamp() * 1000) + d * DAY_MS
                if req["startTime"] <= t <= req["endTime"]:
                    px = 100.0 + d
                    out.append(
                        {"t": t, "o": px, "h": px + 1, "l": px - 1, "c": px, "v": 10, "n": 5}
                    )
            return out
        if kind == "fundingHistory":
            out = []
            first = int(T0.timestamp() * 1000) + self.served_days.start * DAY_MS
            last = int(T0.timestamp() * 1000) + self.served_days.stop * DAY_MS
            for t in range(first, last, 3_600_000):
                if payload["startTime"] <= t <= payload["endTime"]:
                    out.append({"time": t, "fundingRate": "0.00001"})
            return out
        raise AssertionError(f"unexpected request {payload}")


@pytest.fixture()
def venue(monkeypatch):
    fake = FakeVenue(range(0, 5))
    monkeypatch.setattr(hl, "_post", fake)
    monkeypatch.setattr(hl, "polite_sleep", lambda: None)
    return fake


def _candles(store, interval="1d"):
    return pd.read_parquet(store / "hyperliquid-xyz" / "candles" / interval / "xyz:A.parquet")


def test_first_run_keeps_only_closed_candles_and_writes_the_listing(venue, tmp_path) -> None:
    now = T0 + pd.Timedelta(days=4, hours=6)  # day 4's candle is still open
    report = recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=now)

    stored = _candles(tmp_path)
    assert list(stored.index) == [T0 + pd.Timedelta(days=d) for d in range(4)]
    assert list(stored.columns) == ["open", "high", "low", "close", "volume", "trade_count"]
    assert stored["high"].iloc[0] == 101.0
    assert report.listed == 1 and report.delisted == 1
    assert report.new_candles == {"1d": 4}

    funding = pd.read_parquet(tmp_path / "hyperliquid-xyz" / "funding" / "xyz:A.parquet")
    assert funding.index.min() == T0  # funding starts where candles start

    listing = json.loads((tmp_path / "hyperliquid-xyz" / "meta" / "2026-01-05.json").read_text())
    assert [e["name"] for e in listing["universe"]] == ["xyz:A", "xyz:DEAD"]


def test_delisted_instruments_are_not_requested(venue, tmp_path) -> None:
    recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=5))
    coins = {p["req"]["coin"] for p in venue.requests if p["type"] == "candleSnapshot"}
    assert coins == {"xyz:A"}


def test_rows_the_venue_stops_serving_stay_in_the_store(venue, tmp_path) -> None:
    recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=5))
    # Later the venue serves only days 3..7: days 0..2 are gone at the source.
    venue.served_days = range(3, 8)
    report = recorder.record(
        "hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=8)
    )

    stored = _candles(tmp_path)
    assert list(stored.index) == [T0 + pd.Timedelta(days=d) for d in range(8)]
    assert report.new_candles == {"1d": 3}  # days 5, 6, 7


def test_second_run_starts_from_the_last_stored_candle(venue, tmp_path) -> None:
    recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=5))
    venue.requests.clear()
    recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=5))
    starts = [p["req"]["startTime"] for p in venue.requests if p["type"] == "candleSnapshot"]
    assert starts == [int((T0 + pd.Timedelta(days=4)).timestamp() * 1000)]


def test_load_recorded_history_returns_the_range_or_none(venue, tmp_path) -> None:
    recorder.record("hyperliquid-xyz", ["1d"], store_dir=tmp_path, now=T0 + pd.Timedelta(days=5))

    hist = recorder.load_recorded_history(
        "hyperliquid-xyz", "xyz:A", "1d", T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=3),
        store_dir=tmp_path, is_delisted=True,
    )
    assert hist is not None and hist.is_delisted
    assert list(hist.prices.index) == [T0 + pd.Timedelta(days=d) for d in (1, 2, 3)]
    assert recorder.load_recorded_history(
        "hyperliquid-xyz", "xyz:NEVER", "1d", T0, T0 + pd.Timedelta(days=3),
        store_dir=tmp_path, is_delisted=True,
    ) is None


def test_unknown_source_is_refused() -> None:
    with pytest.raises(ValueError, match="does not support"):
        recorder.source_dex("binance")
