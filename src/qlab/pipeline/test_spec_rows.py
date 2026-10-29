"""Tests for the lookup of a spec's registry row (`_get_or_create_spec_row`, `ensure_spec_rows`).

A noise idea has tens of thousands of versions, and loading all of them for each of 200 noise
books cost more than the books. The lookup reads only the rows that can match; these tests keep
it finding exactly the row the full scan found."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.pipeline.evaluate import (
    _candidate_rows,
    _get_or_create_spec_row,
    _plain_params,
    _same_content,
    _spec_row_content,
    ensure_spec_rows,
)
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Profile, SourceType
from qlab.registry.models import Spec as SpecRow

IDEA = "noise-test"


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    repo.upsert_idea(s, id=IDEA, title="t", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    repo.upsert_idea(s, id="other", title="t", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    yield s
    s.close()
    engine.dispose()


def _spec(params: dict, *, idea: str = IDEA, code_ref: str = "x.y:Z", end: date = date(2026, 1, 1)
          ) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": idea, "title": "t", "code_ref": code_ref, "params": params,
        "data": {"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1), "end": end,
                 "instruments": None},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 12.0,
        "unexpressed_mechanisms": [], "sources": {"p": "test"}})


def _full_scan(session, spec: StrategySpec) -> SpecRow | None:
    """What the lookup did before: every row of the idea, newest first, first one that matches."""
    data_requirements, costs_model = _spec_row_content(spec)
    for row in (session.query(SpecRow).filter(SpecRow.idea_id == spec.idea_id)
                .order_by(SpecRow.version.desc()).all()):
        if _same_content(row, spec, data_requirements, costs_model):
            return row
    return None


def test_a_spec_gets_one_row_and_finds_it_again(session) -> None:
    spec = _spec({"seed": 1, "generator": "g", "neutral": True})
    row = _get_or_create_spec_row(session, spec)
    assert row.version == 1
    assert _get_or_create_spec_row(session, spec).id == row.id
    other = _get_or_create_spec_row(session, _spec({"seed": 2, "generator": "g", "neutral": True}))
    assert other.version == 2 and other.id != row.id


def test_the_narrowed_lookup_finds_the_row_the_full_scan_finds(session) -> None:
    """Scalars of every kind, a nested value, a float, a boolean, two data ends."""
    specs = [
        _spec({"seed": s, "generator": g, "neutral": n, "scale": scale,
               "reference_params": {"a": [1, 2], "b": {"c": s}}}, end=end)
        for s in (0, 1, 2) for g in ("a", "b") for n in (True, False)
        for scale in (0.5, 1.0) for end in (date(2026, 1, 1), date(2026, 1, 2))]
    rows = [_get_or_create_spec_row(session, spec) for spec in specs]
    assert len({r.id for r in rows}) == len(specs)
    for spec, row in zip(specs, rows, strict=True):
        assert _get_or_create_spec_row(session, spec).id == row.id
        assert _full_scan(session, spec).id == row.id
    # A spec nobody stored: not found by either, and it gets the next version.
    new = _spec({"seed": 99, "generator": "a", "neutral": True, "scale": 0.5,
                 "reference_params": {}})
    assert _full_scan(session, new) is None
    assert _get_or_create_spec_row(session, new).version == len(specs) + 1


def test_the_lookup_keeps_the_newest_of_two_identical_rows(session) -> None:
    """Two rows with one content (an old race): the full scan took the newest, so does this."""
    spec = _spec({"seed": 1})
    data_requirements, costs_model = _spec_row_content(spec)
    for version in (1, 2):
        repo.add_spec(session, idea_id=IDEA, version=version, params=spec.params,
                      data_requirements=data_requirements, rebalance="1d",
                      costs_model=costs_model, code_ref=spec.code_ref)
    assert _get_or_create_spec_row(session, spec).version == 2
    assert _full_scan(session, spec).version == 2


def test_another_idea_or_another_code_is_never_a_match(session) -> None:
    row = _get_or_create_spec_row(session, _spec({"seed": 1}))
    assert _get_or_create_spec_row(session, _spec({"seed": 1}, idea="other")).id != row.id
    assert _get_or_create_spec_row(session, _spec({"seed": 1}, code_ref="x.y:W")).id != row.id


def test_only_plain_scalars_narrow_the_query() -> None:
    plain = _plain_params({"a": 1, "b": 1.5, "c": "s", "d": True, "e": None, "f": [1],
                           "g": {"h": 1}, "bad-key": 3, "nan": float("nan"), "big": 2**60})
    assert plain == {"a": 1, "b": 1.5, "c": "s", "d": True}


def test_candidate_rows_are_newest_first_and_leave_out_what_cannot_match(session) -> None:
    for seed, day in ((1, 2), (2, 3), (1, 9)):
        _get_or_create_spec_row(session, _spec({"seed": seed, "extra": seed},
                                               end=date(2026, 1, day)))
    rows = _candidate_rows(session, [_spec({"seed": 1, "extra": 1})])
    assert [r.params["seed"] for r in rows] == [1, 1]
    assert [r.version for r in rows] == sorted((r.version for r in rows), reverse=True)


def test_ensure_spec_rows_returns_ids_and_numbers_versions_like_one_by_one(session) -> None:
    first = [_spec({"seed": s}) for s in range(3)]
    ids = ensure_spec_rows(session, first)
    assert [session.get(SpecRow, i).version for i in ids] == [1, 2, 3]
    # The same specs again, with two new ones between them and a repeat.
    again = [first[0], _spec({"seed": 10}), first[2], _spec({"seed": 10}), _spec({"seed": 11})]
    ids2 = ensure_spec_rows(session, again)
    assert ids2[0] == ids[0] and ids2[2] == ids[2]
    assert ids2[1] == ids2[3]  # one row for a repeated spec
    assert [session.get(SpecRow, i).version for i in (ids2[1], ids2[4])] == [4, 5]
    # Each id is the row the lookup finds one by one.
    for spec, row_id in zip(again, ids2, strict=True):
        assert _get_or_create_spec_row(session, spec).id == row_id


def test_ensure_spec_rows_handles_several_ideas(session) -> None:
    specs = [_spec({"seed": 1}), _spec({"seed": 1}, idea="other"), _spec({"seed": 2})]
    ids = ensure_spec_rows(session, specs)
    versions = [(session.get(SpecRow, i).idea_id, session.get(SpecRow, i).version) for i in ids]
    assert versions == [(IDEA, 1), ("other", 1), (IDEA, 2)]
