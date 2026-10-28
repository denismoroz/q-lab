"""`GET /api/ideas/{id}/forward`: the forward test's chart data per stage, or the reason there
is none (docs/FORWARD_CHART.md). Reuses the fixture registry of `test_api`."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from qlab import forward_series
from qlab.api import test_api
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialStatus,
)

# The fixture registry and the client of `test_api`, as fixtures of this module.
client, engine = test_api.client, test_api.engine
populated, session_factory = test_api.populated, test_api.session_factory


def _stage(session, idea_id: str, source: str, *, parent: str | None = None,
           status: IdeaStatus = IdeaStatus.VALIDATED, route: TrialRoute = TrialRoute.PAPER) -> int:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if parent else None))
    session.flush()
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref="x:Y")
    repo.add_data_snapshot(session, id=f"snap-{idea_id}", source=source, instruments={},
                           range_start=datetime(2026, 1, 1).date(),
                           range_end=datetime(2026, 2, 1).date(), path="p", rows=1)
    trial = repo.add_trial(session, spec_id=spec.id, config_hash=idea_id, code_sha="x", params={},
                           status=TrialStatus.OK, snapshot_id=f"snap-{idea_id}", route=route,
                           route_reason="r", started_at=datetime(2026, 10, 28, tzinfo=UTC))
    return trial.id


def _store(trial_id: int, idea_id: str) -> None:
    index = pd.date_range("2026-03-01", periods=6, freq="12h", tz="UTC")
    forward_series.save(forward_series.build_payload(
        trial_id=trial_id, idea_id=idea_id, venue="binance", interval="12h",
        net_return=pd.Series([0.01, 0.0, 0.02, -0.01, 0.0, 0.01], index=index),
        hold_return=pd.Series([0.0] * 6, index=index), hold_coins=["BTC-SPOT"]))


def test_a_passed_idea_returns_one_series_per_stage(client, session_factory):
    session = session_factory()
    t1 = _stage(session, "gen-a", "binance")
    t2 = _stage(session, "gen-a-hl", "hyperliquid", parent="gen-a")
    session.commit()
    session.close()
    _store(t1, "gen-a")
    _store(t2, "gen-a-hl")

    body = client.get("/api/ideas/gen-a/forward").json()

    assert body["passed"] is True and body["two_stage"] is True and body["reason"] is None
    first, second = body["stages"]
    assert (first["stage"], first["venue"], first["trial_id"]) == (1, "Binance", t1)
    assert (second["stage"], second["venue"], second["trial_id"]) == (2, "Hyperliquid", t2)
    series = first["series"]
    assert series["days"] == 3 and series["points"][0][1] == 0.0
    assert series["total_return"] == pytest.approx(series["points"][-1][1])
    assert series["hold_total"] == 0.0 and series["hold_coins"] == ["BTC-SPOT"]
    assert client.get("/api/ideas/gen-a-hl/forward").json()["stages"] == body["stages"]


def test_a_missing_series_is_explained_in_the_stage_not_as_an_error(client, session_factory):
    session = session_factory()
    t1 = _stage(session, "gen-b", "binance")
    _stage(session, "gen-b-hl", "hyperliquid", parent="gen-b")
    session.commit()
    session.close()
    _store(t1, "gen-b")

    response = client.get("/api/ideas/gen-b/forward")

    assert response.status_code == 200
    second = response.json()["stages"][1]
    assert second["series"] is None and "не сохранён" in second["message"]


def test_an_idea_that_did_not_pass_returns_an_empty_explained_result(client):
    # `omega` is decayed in the fixture registry, `alpha` validated but with no routed run.
    body = client.get("/api/ideas/omega/forward").json()

    assert body["passed"] is False and body["stages"] == [] and body["reason"]


def test_a_validated_idea_with_no_run_yet_says_the_stage_has_none(client):
    body = client.get("/api/ideas/alpha/forward").json()

    assert body["passed"] is True
    assert body["stages"][0]["trial_id"] is None
    assert "ещё нет прогона" in body["stages"][0]["message"]


def test_an_unknown_idea_is_404(client):
    assert client.get("/api/ideas/nobody/forward").status_code == 404
