"""Polymarket resolved-market fetcher tests -- respx-mocked, no network access.

The replayed bodies in ``fixtures/`` were recorded from the live public APIs on 2026-10-25
(Gamma ``/markets?closed=true`` for markets ending in March 2026, and CLOB ``/prices-history``
with ``interval=max&fidelity=1440`` for one of them). Cases the live list did not happen to
contain (a 50/50 payout, a categorical market) are made by editing a recorded market."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from qlab.data.sources import polymarket as pm

FIXTURES = Path(__file__).parent / "fixtures"
GAMMA = f"{pm.GAMMA_URL}/markets"
HISTORY = f"{pm.CLOB_URL}/prices-history"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(pm, "polite_sleep", lambda: None)


@pytest.fixture
def page() -> list[dict]:
    return json.loads((FIXTURES / "polymarket_gamma_markets_2026-03.json").read_text())


@pytest.fixture
def daily() -> dict:
    return json.loads((FIXTURES / "polymarket_prices_history_daily.json").read_text())


def _window() -> dict:
    return dict(end_after=pd.Timestamp("2026-03-01", tz="UTC"),
                end_before=pd.Timestamp("2026-03-31 23:59:59", tz="UTC"), min_volume_usd=100000.0)


# ---------------------------------------------------------------- parse_market


def test_every_recorded_market_parses_with_its_resolution(page) -> None:
    markets = [pm.parse_market(raw) for raw in page]
    assert all(m is not None for m in markets) and len(markets) == 5
    by_id = {m.market_id: m for m in markets}
    fed_hike = by_id["654415"]  # "Will the Fed increase ... after the March 2026 meeting?"
    assert fed_hike.resolved_yes is False and fed_hike.neg_risk is True
    assert fed_hike.closed_time == pd.Timestamp("2026-03-18 21:19:03", tz="UTC")
    khamenei = by_id["916732"]
    assert khamenei.resolved_yes is True and khamenei.neg_risk is False
    assert khamenei.order_min_size == 5 and khamenei.tick_size == 0.001
    assert khamenei.volume_usd == pytest.approx(63238697.721704)
    assert khamenei.yes_token_id != khamenei.no_token_id


def test_the_market_can_close_before_its_scheduled_end(page) -> None:
    # Resolved 2026-03-04 against a scheduled end of 2026-03-31: the cut for "prices before
    # resolution" must be closedTime, not endDate.
    khamenei = next(pm.parse_market(r) for r in page if r["id"] == "916732")
    assert khamenei.closed_time < khamenei.end_date


@pytest.mark.parametrize("edit", [
    {"outcomePrices": '["0.5", "0.5"]'},  # a 50/50 payout is not a clean outcome
    {"outcomePrices": '["0.31", "0.69"]'},  # not settled
    {"outcomes": '["Up", "Down"]'},  # not Yes/No
    {"outcomes": '["Yes", "No", "Maybe"]', "outcomePrices": '["1", "0", "0"]'},
    {"clobTokenIds": None},  # an old AMM market has no CLOB tokens
    {"closedTime": None},
    {"conditionId": None},
    {"umaResolutionStatus": "proposed"},
    {"outcomePrices": "not json"},
])
def test_a_market_that_did_not_end_in_a_clean_yes_or_no_is_dropped(page, edit) -> None:
    raw = copy.deepcopy(page[0])
    raw.update(edit)
    assert pm.parse_market(raw) is None


# ---------------------------------------------------------------- histories


def test_the_daily_history_is_a_utc_price_series(daily) -> None:
    series = pm._history_frame(daily["history"])
    assert len(series) == 83 and series.name == "price"
    assert str(series.index.tz) == "UTC" and series.index.is_monotonic_increasing
    assert series.iloc[0] == 0.075 and series.iloc[-1] == 0.9995
    assert series.between(0.0, 1.0).all()  # a binary contract's price is bounded by 0 and 1


def test_an_empty_history_is_an_empty_series() -> None:
    assert pm._history_frame([]).empty and pm._history_frame(None).empty


def test_prices_after_the_market_closed_are_cut(daily) -> None:
    series = pm._history_frame(daily["history"])
    cut = pd.Timestamp("2026-02-01", tz="UTC")
    kept = pm.prices_before_resolution(series, cut)
    assert kept.index.max() < cut and 0 < len(kept) < len(series)
    # the closing instant itself is not "before"
    assert pm.prices_before_resolution(series, series.index[10]).index.max() == series.index[9]


@respx.mock
def test_daily_history_asks_for_the_whole_life_at_day_fidelity(daily) -> None:
    route = respx.get(HISTORY).mock(return_value=httpx.Response(200, json=daily))
    with httpx.Client() as client:
        series = pm.fetch_daily_history(client, "123")
    params = route.calls[0].request.url.params
    assert params["market"] == "123" and params["interval"] == "max"
    assert params["fidelity"] == "1440"
    assert len(series) == 83


@respx.mock
def test_hourly_history_is_fetched_in_windows_the_endpoint_accepts() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTs"])
        end = int(request.url.params["endTs"])
        # the venue answers a window longer than ~15 days with an error body and status 400
        if end - start > 15 * 86400:
            return httpx.Response(400, json={"error": "invalid filters: interval is too long"})
        # an hourly point at the start and one at the end of each window
        return httpx.Response(200, json={"history": [{"t": start, "p": 0.4}, {"t": end, "p": 0.5}]})

    route = respx.get(HISTORY).mock(side_effect=answer)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-02-05", tz="UTC")
    with httpx.Client() as client:
        series = pm.fetch_hourly_history(client, "123", start, end)
    spans = [(int(c.request.url.params["startTs"]), int(c.request.url.params["endTs"]))
             for c in route.calls]
    assert len(spans) == 3  # 35 days in windows of 12
    assert spans[0][0] == int(start.timestamp()) and spans[-1][1] == int(end.timestamp())
    assert all(b - a <= pm.HOURLY_CHUNK_DAYS * 86400 for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))  # no gap
    assert series.index.is_unique and series.index.is_monotonic_increasing  # chunk joins merged
    assert {c.request.url.params["fidelity"] for c in route.calls} == {"60"}


@respx.mock
def test_an_error_body_from_the_history_endpoint_is_not_silently_empty() -> None:
    respx.get(HISTORY).mock(return_value=httpx.Response(200, json={"error": "invalid filters"}))
    with httpx.Client() as client, pytest.raises(RuntimeError, match="refused"):
        pm.fetch_hourly_history(client, "1", pd.Timestamp("2026-01-01", tz="UTC"),
                                pd.Timestamp("2026-01-05", tz="UTC"))


# ---------------------------------------------------------------- listing


@respx.mock
def test_listing_pages_by_offset_until_a_short_page(page) -> None:
    full = [copy.deepcopy(page[0]) for _ in range(pm.GAMMA_PAGE_SIZE)]
    short = [copy.deepcopy(page[1]) for _ in range(3)]
    route = respx.get(GAMMA).mock(side_effect=[
        httpx.Response(200, json=full), httpx.Response(200, json=full),
        httpx.Response(200, json=short)])
    with httpx.Client() as client:
        got = list(pm.iter_closed_markets(client, **_window()))
    assert len(got) == 203
    offsets = [c.request.url.params["offset"] for c in route.calls]
    assert offsets == ["0", "100", "200"]
    first = route.calls[0].request.url.params
    assert first["closed"] == "true" and first["limit"] == "100"
    assert first["end_date_min"] == "2026-03-01T00:00:00Z"
    assert first["volume_num_min"] == "100000.0" and first["order"] == "volumeNum"


@respx.mock
def test_listing_stops_on_an_exact_multiple_with_an_empty_page(page) -> None:
    full = [copy.deepcopy(page[0]) for _ in range(pm.GAMMA_PAGE_SIZE)]
    respx.get(GAMMA).mock(side_effect=[httpx.Response(200, json=full),
                                       httpx.Response(200, json=[])])
    with httpx.Client() as client:
        assert len(list(pm.iter_closed_markets(client, **_window()))) == 100


@respx.mock
def test_a_non_list_answer_is_an_error_not_an_empty_universe() -> None:
    respx.get(GAMMA).mock(return_value=httpx.Response(200, json={"error": "nope"}))
    with httpx.Client() as client, pytest.raises(RuntimeError, match="Gamma"):
        list(pm.iter_closed_markets(client, **_window()))


# ---------------------------------------------------------------- collecting


@respx.mock
def test_collecting_stores_each_market_once_and_resumes(page, daily, tmp_path) -> None:
    respx.get(GAMMA).mock(return_value=httpx.Response(200, json=page))
    hist = respx.get(HISTORY).mock(return_value=httpx.Response(200, json=daily))
    with httpx.Client() as client:
        first = pm.collect_resolved(client, cache_dir=tmp_path, max_markets=2, now=NOW,
                                    **_window())
        assert first["fetched"] == 2 and first["already_stored"] == 0 and first["in_store"] == 2
        assert len(hist.calls) == 2
        # the largest by volume come first in the list, and those are the ones kept
        assert first["listed"] == 3  # the loop saw a third market and stopped at the cap
        again = pm.collect_resolved(client, cache_dir=tmp_path, max_markets=2, now=NOW,
                                    **_window())
    assert again["fetched"] == 0 and again["already_stored"] == 2
    assert len(hist.calls) == 2  # nothing refetched

    meta, prices = pm.load_market(tmp_path, "1d", pm.parse_market(page[0]).condition_id)
    assert meta["resolved_yes"] is False and meta["points"] == 83 == len(prices)
    assert meta["fetched_at"] == "2026-10-25T12:00:00+00:00"
    assert meta["closed_time"] == "2026-03-18T21:19:03+00:00"
    assert "not a random sample" in meta["selection"]["note"]
    assert (tmp_path / "polymarket" / "1d").is_dir()


@respx.mock
def test_collecting_reports_the_range_and_the_outcome_split(page, daily, tmp_path) -> None:
    respx.get(GAMMA).mock(return_value=httpx.Response(200, json=page))
    respx.get(HISTORY).mock(return_value=httpx.Response(200, json=daily))
    with httpx.Client() as client:
        summary = pm.collect_resolved(client, cache_dir=tmp_path, max_markets=10, now=NOW,
                                      **_window())
    assert summary["in_store"] == 5 and summary["fetched"] == 5
    assert summary["yes_won"] == 1  # only the Khamenei market resolved Yes
    assert summary["closed_first"] == pd.Timestamp("2026-03-04 00:58:01", tz="UTC")
    assert summary["closed_last"] == pd.Timestamp("2026-04-05 06:15:11", tz="UTC")


@respx.mock
def test_a_market_without_history_is_counted_and_not_stored(page, tmp_path) -> None:
    respx.get(GAMMA).mock(return_value=httpx.Response(200, json=page[:1]))
    respx.get(HISTORY).mock(return_value=httpx.Response(200, json={"history": []}))
    with httpx.Client() as client:
        summary = pm.collect_resolved(client, cache_dir=tmp_path, max_markets=5, now=NOW,
                                      **_window())
    assert summary["no_history"] == 1 and summary["in_store"] == 0 and summary["fetched"] == 0
    assert not list((tmp_path / "polymarket").rglob("*.parquet"))


@respx.mock
def test_unusable_markets_in_the_list_are_dropped_and_counted(page, daily, tmp_path) -> None:
    broken = copy.deepcopy(page[1])
    broken["outcomePrices"] = '["0.5", "0.5"]'
    respx.get(GAMMA).mock(return_value=httpx.Response(200, json=[page[0], broken]))
    respx.get(HISTORY).mock(return_value=httpx.Response(200, json=daily))
    with httpx.Client() as client:
        summary = pm.collect_resolved(client, cache_dir=tmp_path, max_markets=5, now=NOW,
                                      **_window())
    assert summary["listed"] == 2 and summary["unusable_dropped"] == 1
    assert summary["in_store"] == 1


def test_an_entry_without_its_metadata_is_absent(page, daily, tmp_path) -> None:
    market = pm.parse_market(page[0])
    prices = pm._history_frame(daily["history"])
    pm.store_market(tmp_path, "1d", market, prices, fetched_at=NOW, selection={})
    assert pm.load_market(tmp_path, "1d", market.condition_id) is not None
    (tmp_path / "polymarket" / "1d" / f"{market.condition_id}.json").unlink()
    assert pm.load_market(tmp_path, "1d", market.condition_id) is None


def test_an_unknown_interval_is_refused(tmp_path) -> None:
    with httpx.Client() as client, pytest.raises(ValueError, match="interval"):
        pm.collect_resolved(client, cache_dir=tmp_path, max_markets=1, interval="5m", **_window())
