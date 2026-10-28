"""`GET /api/ideas/{id}/overlays`: an idea's overlay variants against it, or a variant's against its
base, with the stored comparison and the base's described forward series (docs/OVERLAYS.md).
Reuses the fixture registry of `test_api`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from qlab import overlay_compare, overlay_diagnostics
from qlab.api import test_api
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    DataSnapshot,
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


def _idea(session, idea_id: str, parent: str | None = None, route: TrialRoute | None = None):
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     status=IdeaStatus.BENCH, parent_id=parent))
    session.flush()
    if route is None:
        return
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                         rebalance="1d", costs_model={}, code_ref="x:Y")
    if session.get(DataSnapshot, "snap-ov") is None:
        repo.add_data_snapshot(session, id="snap-ov", source="binance", instruments={},
                               range_start=datetime(2026, 1, 1).date(),
                               range_end=datetime(2026, 2, 1).date(), path="p", rows=1)
    repo.add_trial(session, spec_id=spec.id, config_hash=idea_id, code_sha="x", params={},
                   status=TrialStatus.OK, snapshot_id="snap-ov", route=route, route_reason="r",
                   started_at=datetime(2026, 10, 28, tzinfo=UTC))


def _noisy(seed: int, drift: float = 0.0005) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(drift, 0.02, 800),
                     index=pd.date_range("2021-01-01", periods=800, freq="1D", tz="UTC"))


def test_a_base_lists_its_variants_with_the_comparison_and_the_diagnostics(client, session_factory):
    session = session_factory()
    _idea(session, "strat", route=TrialRoute.PAPER)
    _idea(session, "strat-ov-own", parent="strat", route=TrialRoute.SHELF)
    _idea(session, "strat-ov-enter", parent="strat", route=TrialRoute.NEEDS_FORWARD)
    session.commit()
    session.close()
    record = overlay_compare.compare_series(_noisy(1), _noisy(1) + 0.004, n_boot=200)
    overlay_compare.save_record({**record, "base_id": "strat", "variant_id": "strat-ov-own"})
    overlay_diagnostics.save(overlay_diagnostics.build("strat", _noisy(2), _noisy(3), None))

    body = client.get("/api/ideas/strat/overlays").json()

    assert body["role"] == "base" and body["base_id"] == "strat"
    table = body["table"]
    assert table["family_size"] == 1 and table["family_trials"] == 3
    rows = {r["idea_id"]: r for r in table["variants"]}
    assert rows["strat-ov-own"]["name"] == "own" and rows["strat-ov-own"]["route"] == "shelf"
    assert rows["strat-ov-own"]["comparison"]["verdict_ru"] == "помогла"
    assert rows["strat-ov-enter"]["comparison"] is None
    assert body["diagnostics"]["idea_id"] == "strat" and body["diagnostics_lines"]
    assert body["alpha"] == 0.05


def test_a_variant_sees_its_base_and_its_neighbours(client, session_factory):
    session = session_factory()
    _idea(session, "strat2")
    _idea(session, "strat2-ov-own", parent="strat2")
    session.commit()
    session.close()

    body = client.get("/api/ideas/strat2-ov-own/overlays").json()

    assert body["role"] == "variant" and body["base_id"] == "strat2"
    assert [r["idea_id"] for r in body["table"]["variants"]] == ["strat2-ov-own"]
    assert body["diagnostics"] is None and body["diagnostics_lines"] is None


def test_an_idea_without_variants_returns_an_empty_table(client, session_factory):
    body = client.get("/api/ideas/alpha/overlays").json()

    assert body["role"] == "base" and body["table"]["variants"] == []


def test_an_unknown_idea_is_404(client):
    assert client.get("/api/ideas/nobody/overlays").status_code == 404
