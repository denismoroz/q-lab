"""`GET /api/standard` and the `standard` block of `GET /api/ideas/{id}`: the fourteen things every
strategy page has, checked by code (docs/IDEA_STANDARD.md). Reuses the fixture registry of
`test_api`."""

from __future__ import annotations

import pandas as pd

from qlab import forward_series
from qlab.api import test_api, test_forward_api
from qlab.registry.models import (
    AssetClass,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
)

client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory


def _whole_series(trial_id: int, idea_id: str, venue: str) -> None:
    index = pd.date_range("2026-03-01", periods=6, freq="1D", tz="UTC")
    returns = pd.Series([0.01, 0.0, 0.02, -0.01, 0.0, 0.01], index=index)
    whole = forward_series.build_whole(
        net_return=returns, hold_return=None, hold_coins=None,
        bounds={"selection": (index[0], index[3]), "forward": (index[4], index[5])})
    forward_series.save(forward_series.build_payload(
        trial_id=trial_id, idea_id=idea_id, venue=venue, interval="1d",
        net_return=returns.iloc[4:], hold_return=None, whole=whole))


def _plain_idea(session, idea_id: str, status: IdeaStatus) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status))


def test_the_standard_lists_every_open_idea_against_fourteen_items(client):
    body = client.get("/api/standard").json()

    ids = {r["id"] for r in body["rows"]}
    assert ids == {"alpha", "omega"}
    assert [d["number"] for d in body["items"]] == list(range(1, 15))
    for row in body["rows"]:
        assert len(row["items"]) == 14
        assert {c["state"] for c in row["items"]} <= {"есть", "нет", "не нужно"}
        assert row["status"] in ("validated", "decayed") and "family" in row
    assert [t["number"] for t in body["totals"]] == list(range(1, 15))
    plain = body["totals"][0]
    assert plain["missing"] + plain["have"] + plain["not_needed"] == 2


def test_ideas_put_away_noise_books_and_second_stages_are_not_rows(client, session_factory):
    session = session_factory()
    _plain_idea(session, "gone", IdeaStatus.RETIRED)
    _plain_idea(session, "noise-book-1", IdeaStatus.BENCH)
    first = test_forward_api._stage(session, "gen-a", "binance")
    second = test_forward_api._stage(session, "gen-a-hl", "hyperliquid", parent="gen-a")
    session.commit()
    session.close()
    _whole_series(first, "gen-a", "binance")
    _whole_series(second, "gen-a-hl", "hyperliquid")

    body = client.get("/api/standard?fresh=true").json()

    ids = {r["id"] for r in body["rows"]}
    assert {"gone", "noise-book-1", "gen-a-hl"}.isdisjoint(ids) and "gen-a" in ids
    chart = next(r for r in body["rows"] if r["id"] == "gen-a")["items"][4]
    assert chart["number"] == 5 and chart["state"] == "есть"


def test_a_missing_second_stage_series_is_told_in_the_cell(client, session_factory):
    session = session_factory()
    first = test_forward_api._stage(session, "gen-b", "binance")
    test_forward_api._stage(session, "gen-b-hl", "hyperliquid", parent="gen-b")
    session.commit()
    session.close()
    _whole_series(first, "gen-b", "binance")

    body = client.get("/api/standard?fresh=true").json()

    chart = next(r for r in body["rows"] if r["id"] == "gen-b")["items"][4]
    assert chart["state"] == "нет"
    assert "Hyperliquid: ряд денег ещё не построен" in chart["text"]
    assert "Binance" not in chart["text"]


def test_the_idea_response_carries_its_checklist(client):
    body = client.get("/api/ideas/alpha").json()

    standard = body["standard"]
    assert standard["idea_id"] == "alpha" and standard["total"] == 14
    assert len(standard["items"]) == 14
    assert standard["have"] == sum(1 for i in standard["items"] if i["state"] == "есть")
    first = standard["items"][0]
    assert set(first) == {"number", "key", "title", "short", "state", "reason", "detail"}
    for item in standard["items"]:
        if item["state"] != "есть":
            assert item["reason"], item


def test_an_idea_put_away_has_no_checklist(client, session_factory):
    session = session_factory()
    _plain_idea(session, "gone", IdeaStatus.RETIRED)
    session.commit()
    session.close()

    assert client.get("/api/ideas/gone").json()["standard"] is None
