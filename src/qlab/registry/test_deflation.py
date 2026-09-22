"""Tests for qlab.registry.deflation.

The arithmetic tests pin the ESTIMATORS' direction and magnitude rather than
just their shape: a deflation function that runs but moves the wrong way is
worse than none, because it would license exactly the searched-over results it
exists to catch. Each one states the hand computation it checks.
"""

from __future__ import annotations

import math
from datetime import date
from statistics import NormalDist

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.registry import repo
from qlab.registry.deflation import (
    DeflationError,
    deflated_sharpe,
    expected_max_sharpe,
    family_trials,
)
from qlab.registry.models import AssetClass, Base, Profile, SourceType, TrialStatus

_N = NormalDist()


@pytest.fixture()
def session():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng, expire_on_commit=False)()
    yield s
    s.close()
    eng.dispose()


# --- expected_max_sharpe ---------------------------------------------------


def test_expected_max_sharpe_matches_hand_computation_for_two_trials() -> None:
    """N=2, sharpe_std=1. Z(1 - 1/2) = Z(0.5) = 0 exactly, so the first term
    vanishes and the whole value is g * Z(1 - 1/(2e)) with g the
    Euler-Mascheroni constant."""
    expected = 0.5772156649015329 * _N.inv_cdf(1.0 - 1.0 / (2.0 * math.e))
    assert expected_max_sharpe(2, 1.0) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_rises_with_trials() -> None:
    """The entire point: searching more raises the best result you will see
    even when nothing has any edge."""
    values = [expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 50, 200)]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_expected_max_sharpe_scales_linearly_with_spread() -> None:
    assert expected_max_sharpe(20, 2.0) == pytest.approx(2.0 * expected_max_sharpe(20, 1.0))


def test_expected_max_sharpe_rejects_a_single_trial() -> None:
    with pytest.raises(DeflationError, match="at least 2 trials"):
        expected_max_sharpe(1, 1.0)


def test_expected_max_sharpe_rejects_zero_spread() -> None:
    with pytest.raises(DeflationError, match="positive and finite"):
        expected_max_sharpe(10, 0.0)


# --- deflated_sharpe -------------------------------------------------------


_BASE = dict(
    sharpe_net=1.0,
    sharpe_std_net=0.5,
    n_periods=627,
    periods_per_year=365.0,
    skew=0.0,
    kurtosis=3.0,  # NON-excess: 3.0 is the normal case
)


def test_deflated_sharpe_is_a_probability() -> None:
    p = deflated_sharpe(n_trials=10, **_BASE)
    assert 0.0 <= p <= 1.0


def test_more_trials_lower_the_deflated_sharpe() -> None:
    """Same result, more attempts behind it -> less evidence. This is the
    property the whole module exists for."""
    few = deflated_sharpe(n_trials=3, **_BASE)
    many = deflated_sharpe(n_trials=400, **_BASE)
    assert many < few


def test_negative_skew_lowers_the_deflated_sharpe() -> None:
    """Non-normality correction, in the direction that matters for a strategy
    whose return comes from a handful of days: negative skew makes a given
    Sharpe less trustworthy, never more."""
    kw = {k: v for k, v in _BASE.items() if k != "skew"}
    assert deflated_sharpe(n_trials=10, skew=-1.0, **kw) < deflated_sharpe(
        n_trials=10, skew=+1.0, **kw
    )


def test_fat_tails_lower_the_deflated_sharpe() -> None:
    kw = {k: v for k, v in _BASE.items() if k != "kurtosis"}
    assert deflated_sharpe(n_trials=10, kurtosis=12.0, **kw) < deflated_sharpe(
        n_trials=10, kurtosis=3.0, **kw
    )


def test_annualised_inputs_are_converted_not_taken_raw() -> None:
    """`sharpe_net` arrives annualised. If the conversion were dropped, the
    same numbers read as per-observation would give a wildly different answer
    -- this pins that the conversion happens."""
    daily = deflated_sharpe(n_trials=10, **_BASE)
    hourly = deflated_sharpe(n_trials=10, **{**_BASE, "periods_per_year": 8760.0})
    assert daily != pytest.approx(hourly)


def test_deflated_sharpe_rejects_a_degenerate_sample() -> None:
    with pytest.raises(DeflationError, match="n_periods"):
        deflated_sharpe(n_trials=10, **{**_BASE, "n_periods": 1})


# --- family_trials ---------------------------------------------------------


_SEEN_DRIVERS: set[str] = set()


def _idea(session, idea_id: str, driver_id: str | None) -> None:
    if driver_id is not None:
        repo.upsert_driver(
            session,
            id=driver_id,
            title=driver_id,
            description="test driver",
            kill_condition="test",
            observable="test",
        )
    repo.upsert_idea(
        session,
        id=idea_id,
        title=idea_id,
        source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.MOMENTUM,
        driver_id=driver_id,
    )


def _trial(session, idea_id: str, *, status: TrialStatus, version: int = 1) -> None:
    spec = repo.add_spec(
        session,
        idea_id=idea_id,
        version=version,
        params={},
        data_requirements={},
        rebalance="1d",
        costs_model={},
        code_ref="x:Y",
    )
    snap = repo.add_data_snapshot(
        session,
        id=f"snap-{idea_id}-{version}",
        source="test",
        instruments={"A": {}},
        range_start=date(2026, 1, 1),
        range_end=date(2026, 1, 2),
        path="/dev/null",
        rows=1,
    )
    repo.add_trial(
        session,
        spec_id=spec.id,
        config_hash="h",
        snapshot_id=snap.id,
        code_sha="sha",
        params={},
        status=status,
    )


def test_family_trials_idea_scope_counts_only_that_idea(session) -> None:
    _idea(session, "parent", "shared-driver")
    _idea(session, "variant", "shared-driver")
    _trial(session, "parent", status=TrialStatus.OK)
    _trial(session, "variant", status=TrialStatus.OK)
    session.commit()

    counted = family_trials(session, "parent", scope="idea")
    assert counted.n_trials == 1
    assert counted.idea_ids == ("parent",)


def test_family_trials_driver_scope_counts_every_variant(session) -> None:
    """The XSMOM case: three ids, one edge. Under idea scope each looks like a
    single attempt; under driver scope they are what they are -- repeated looks
    at the same data (docs/TASKS.md T27 item 4)."""
    _idea(session, "parent", "shared-driver")
    _idea(session, "variant", "shared-driver")
    _idea(session, "unrelated", "other-driver")
    _trial(session, "parent", status=TrialStatus.OK)
    _trial(session, "variant", status=TrialStatus.OK)
    _trial(session, "unrelated", status=TrialStatus.OK)
    session.commit()

    counted = family_trials(session, "parent", scope="driver")
    assert counted.n_trials == 2
    assert set(counted.idea_ids) == {"parent", "variant"}


def test_family_trials_counts_errored_runs_but_separates_them(session) -> None:
    """An errored run still consumed a look at the data, so it belongs in the
    trial count; it contributes no Sharpe, so it is excluded from n_ok."""
    _idea(session, "parent", "shared-driver")
    _trial(session, "parent", status=TrialStatus.OK, version=1)
    _trial(session, "parent", status=TrialStatus.ERROR, version=2)
    session.commit()

    counted = family_trials(session, "parent", scope="idea")
    assert (counted.n_trials, counted.n_ok) == (2, 1)


def test_family_trials_refuses_driver_scope_without_a_driver(session) -> None:
    _idea(session, "orphan", None)
    session.commit()
    with pytest.raises(DeflationError, match="no driver"):
        family_trials(session, "orphan", scope="driver")
