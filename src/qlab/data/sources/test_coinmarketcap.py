"""Tests for `qlab.data.sources.coinmarketcap`: page parsing, the Hyperliquid
symbol mapping, and above all the point-in-time visibility of `as_of_frame`."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from qlab.data.sources.coinmarketcap import (
    as_of_frame,
    hyperliquid_symbol,
    parse_snapshot,
    snapshot_dates,
)


def _page(rows: list[dict]) -> str:
    state = {"cryptocurrency": {"listingHistorical": {"data": rows}}}
    data = {"props": {"initialState": json.dumps(state)}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script>'


def _row(rank: int, symbol: str, cap: float, tags=()) -> dict:
    return {"cmcRank": rank, "symbol": symbol, "name": symbol, "slug": symbol.lower(),
            "dateAdded": "2020-01-01T00:00:00.000Z", "tags": list(tags),
            "circulatingSupply": 1.0, "quote": {"USD": {"marketCap": cap, "volume24h": 1.0,
                                                         "price": 1.0}}}


def test_parse_keeps_rank_cap_and_tags() -> None:
    rows = parse_snapshot(_page([_row(1, "BTC", 2e12), _row(7, "DOGE", 5e10, ["memes"])]))
    assert [(r["cmcRank"], r["symbol"], r["market_cap_usd"]) for r in rows] == [
        (1, "BTC", 2e12), (7, "DOGE", 5e10)]
    assert rows[1]["tags"] == ["memes"]


def test_changed_layout_fails_loudly() -> None:
    with pytest.raises(ValueError, match="layout"):
        parse_snapshot("<html>nothing here</html>")
    with pytest.raises(ValueError, match="layout"):
        parse_snapshot(_page([]))


def test_snapshot_dates_are_sundays() -> None:
    days = snapshot_dates(date(2025, 1, 1), date(2025, 1, 31))
    assert days == [date(2025, 1, 5), date(2025, 1, 12), date(2025, 1, 19), date(2025, 1, 26)]


def test_hyperliquid_symbol_handles_the_thousand_unit_prefix() -> None:
    names = {"BTC", "kPEPE"}
    assert hyperliquid_symbol("BTC", names) == "BTC"
    assert hyperliquid_symbol("PEPE", names) == "kPEPE"
    assert hyperliquid_symbol("XYZ", names) is None
    assert hyperliquid_symbol("SHIB", {"1000SHIB"}) == "1000SHIB"  # Binance's spelling


def test_as_of_frame_is_point_in_time_and_does_not_carry_dropped_coins() -> None:
    history = pd.DataFrame([
        {"snapshot": pd.Timestamp("2025-01-05", tz="UTC"), "symbol": "BTC", "cmcRank": 1},
        {"snapshot": pd.Timestamp("2025-01-05", tz="UTC"), "symbol": "ATOM", "cmcRank": 40},
        # A lower-ranked coin with the same symbol must not win the mapping.
        {"snapshot": pd.Timestamp("2025-01-05", tz="UTC"), "symbol": "ATOM", "cmcRank": 190},
        {"snapshot": pd.Timestamp("2025-01-12", tz="UTC"), "symbol": "BTC", "cmcRank": 1},
        # ATOM is outside the top 200 on 2025-01-12.
    ])
    index = pd.date_range("2025-01-04", "2025-01-14", freq="1D", tz="UTC")
    ranks = as_of_frame(history, index, ["BTC", "ATOM"], "cmcRank")

    assert ranks.loc["2025-01-05", "BTC"] != ranks.loc["2025-01-05", "BTC"]  # NaN: not yet known
    assert ranks.loc["2025-01-06", "BTC"] == 1
    assert ranks.loc["2025-01-06", "ATOM"] == 40
    assert ranks.loc["2025-01-12", "ATOM"] == 40  # the 01-12 snapshot is not visible yet
    assert pd.isna(ranks.loc["2025-01-13", "ATOM"])  # dropped out: no stale rank
    assert ranks.loc["2025-01-13", "BTC"] == 1
