"""Tests for `qlab.calibration.run` (docs/TASKS.md, T16).

Follows the same fixture shape as `qlab.pipeline.test_evaluate`: an
in-memory SQLite registry, a hand-registered snapshot on disk (no network),
and a toy `Strategy` used as the "real" reference -- these tests exercise
`run_noise_series`/`run_real_strategies` wiring against the real
`evaluate_spec`, not against the real trend strategy or a real snapshot.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.calibration.noise import GENERATORS
from qlab.calibration.run import (
    NOISE_CODE_REF,
    build_noise_spec,
    ensure_noise_idea,
    noise_idea_id,
    run_noise_series,
    run_real_strategies,
)
from qlab.harness.panel import MarketPanel
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Idea, Profile, SourceType
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage

SOURCE = "hyperliquid"
INTERVAL = "1d"
START = date(2025, 1, 1)
END = date(2025, 3, 1)
INSTRUMENTS = [f"COIN{i}" for i in range(8)]
SNAPSHOT_ID = "snap-calibration-test"
REFERENCE_IDEA_ID = "toy-reference"


class ToyMomentumStrategy:
    """A tiny 'real' reference strategy: rolling-mean noise, cross-sectionally
    normalised. Not dollar-neutral by construction (like trend), varies over
    time (non-zero turnover), so it is a fair stand-in for the shape of a
    real strategy in these wiring tests."""

    name = "toy-momentum"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        seed = int(params.get("seed", 3))
        rng = np.random.default_rng(seed)
        raw = rng.normal(0.0, 1.0, size=(len(panel.prices.index), len(panel.prices.columns)))
        smoothed = (
            pd.DataFrame(raw, index=panel.prices.index, columns=panel.prices.columns)
            .rolling(4, min_periods=1)
            .mean()
        )
        gross = smoothed.abs().sum(axis=1).to_numpy()
        weights = smoothed.to_numpy() / gross[:, None] * 0.5
        weights = pd.DataFrame(weights, index=panel.prices.index, columns=panel.prices.columns)
        return weights.where(panel.tradeable, 0.0)


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "build_snapshot was called -- a matching snapshot should have been reused"
        )

    monkeypatch.setattr("qlab.pipeline.evaluate.build_snapshot", _boom)


@pytest.fixture(autouse=True)
def _reference_idea_row(session):
    """`build_noise_spec`/`run_real_strategies` reference this idea_id via
    `spec.idea_id`'s foreign key -- it must exist first, exactly like the
    noise idea rows `ensure_noise_idea` creates."""
    repo.upsert_idea(
        session,
        id=REFERENCE_IDEA_ID,
        title="Toy reference idea",
        source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.OTHER,
    )


@pytest.fixture(autouse=True)
def _register_snapshot(session, tmp_path: Path):
    index = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="1D")
    rng = np.random.default_rng(11)
    log_returns = rng.normal(0.0005, 0.02, size=(len(index), len(INSTRUMENTS)))
    prices = (1.0 + pd.DataFrame(log_returns, index=index, columns=INSTRUMENTS)).cumprod() * 100.0
    funding = pd.DataFrame(0.0001, index=index, columns=INSTRUMENTS)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)

    snap_dir = tmp_path / SNAPSHOT_ID
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prices", prices), ("funding", funding), ("tradeable", tradeable)):
        frame.to_parquet(snap_dir / f"{name}.parquet", engine="pyarrow", index=True)

    manifest = {
        "source": SOURCE,
        "instruments": sorted(INSTRUMENTS),
        "start": pd.Timestamp(START, tz="UTC").isoformat(),
        "end": pd.Timestamp(END, tz="UTC").isoformat(),
        "interval": INTERVAL,
        "universe_complete": True,
        "no_funding_instruments": [],
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    repo.add_data_snapshot(
        session,
        id=SNAPSHOT_ID,
        source=SOURCE,
        instruments=dict.fromkeys(sorted(INSTRUMENTS), True),
        range_start=START,
        range_end=END,
        path=str(snap_dir),
        rows=len(prices.index),
        fetched_at=datetime.now(UTC),
    )


def _reference_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "idea_id": REFERENCE_IDEA_ID,
            "title": "Toy reference strategy",
            "code_ref": "qlab.calibration.test_run:ToyMomentumStrategy",
            "params": {"seed": 3},
            "data": {
                "source": SOURCE,
                "interval": INTERVAL,
                "start": START,
                "end": END,
                "instruments": None,
            },
            "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9},
            "min_leg_notional": 10.0,
            "simultaneous_legs": 1,
        }
    )


def _permissive_ruleset() -> RuleSet:
    """A ruleset that admits almost everything, so most trials land on
    `paper`/`shelf` rather than `reject` -- these tests check WIRING
    (journal writes, idea auto-creation, spec construction), not the real
    ruleset's actual behaviour, which `qlab.calibration.test_report` and the
    real `qlab calibrate` run cover separately."""
    return RuleSet(
        version="2020-01-01.1",
        based_on=None,
        rules=[
            Rule(
                id="net_edge_positive",
                stage=Stage.EDGE,
                metric="ann_return_net",
                comparator=Comparator.GE,
                threshold=-1.0,
                fatal=True,
            )
        ],
        retired=[],
    )


# --------------------------------------------------------------------------
# noise_idea_id / build_noise_spec
# --------------------------------------------------------------------------


def test_noise_idea_id_is_stable_per_series_and_generator():
    assert noise_idea_id("dollar_neutral", "random_signs") == "noise-dollar_neutral-random_signs"
    assert noise_idea_id("dollar_neutral", "random_signs") == noise_idea_id(
        "dollar_neutral", "random_signs"
    )


def test_build_noise_spec_shares_data_costs_and_capital_with_reference():
    reference = _reference_spec()
    spec = build_noise_spec(
        series="dollar_neutral", generator="random_signs", seed=42, reference=reference
    )

    assert spec.idea_id == "noise-dollar_neutral-random_signs"
    assert spec.code_ref == NOISE_CODE_REF
    assert spec.data == reference.data
    assert spec.costs == reference.costs
    assert spec.min_leg_notional == reference.min_leg_notional
    assert spec.simultaneous_legs == reference.simultaneous_legs
    assert spec.params == {
        "generator": "random_signs",
        "seed": 42,
        "neutral": True,
        "reference_code_ref": reference.code_ref,
        "reference_params": dict(reference.params),
    }


def test_build_noise_spec_unconstrained_sets_neutral_false():
    reference = _reference_spec()
    spec = build_noise_spec(
        series="unconstrained", generator="bootstrap_time", seed=1, reference=reference
    )
    assert spec.params["neutral"] is False


def test_build_noise_spec_rejects_unknown_series():
    reference = _reference_spec()
    with pytest.raises(ValueError, match="unknown series"):
        build_noise_spec(
            series="not-a-series", generator="random_signs", seed=1, reference=reference
        )


def test_build_noise_spec_rejects_unknown_generator():
    reference = _reference_spec()
    with pytest.raises(ValueError, match="unknown generator"):
        build_noise_spec(
            series="dollar_neutral", generator="not-a-generator", seed=1, reference=reference
        )


# --------------------------------------------------------------------------
# ensure_noise_idea
# --------------------------------------------------------------------------


def test_ensure_noise_idea_creates_and_is_idempotent(session):
    idea_id = noise_idea_id("dollar_neutral", "random_weights")
    ensure_noise_idea(session, series="dollar_neutral", generator="random_weights")
    session.commit()
    ensure_noise_idea(session, series="dollar_neutral", generator="random_weights")
    session.commit()

    row = session.get(Idea, idea_id)
    assert row is not None
    assert "calibration noise" in row.title


# --------------------------------------------------------------------------
# run_noise_series / run_real_strategies wiring
# --------------------------------------------------------------------------


def test_run_noise_series_writes_one_trial_per_call_and_round_robins_generators(session):
    reference = _reference_spec()
    ruleset = _permissive_ruleset()

    trials = run_noise_series(
        series="dollar_neutral",
        n_trials=len(GENERATORS) * 2,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1_000_000.0,
        reference=reference,
        seed_offset=1000,
    )

    assert len(trials) == len(GENERATORS) * 2
    assert {t.generator for t in trials} == set(GENERATORS)
    assert all(t.series == "dollar_neutral" for t in trials)
    # every trial got a distinct, offset seed
    assert sorted(t.seed for t in trials) == list(range(1000, 1000 + len(GENERATORS) * 2))
    # every evaluation actually ran (no import/attribute error from the
    # noise code_ref, no crash from the generator/neutralize pipeline)
    for t in trials:
        assert t.evaluation.trial_id is not None
        assert t.evaluation.error is None, t.evaluation.error

    # idea rows exist for every generator under this series
    for generator in GENERATORS:
        assert session.get(Idea, noise_idea_id("dollar_neutral", generator)) is not None


def test_run_noise_series_seed_offset_avoids_collisions(session):
    reference = _reference_spec()
    ruleset = _permissive_ruleset()

    first = run_noise_series(
        series="unconstrained",
        n_trials=4,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1_000_000.0,
        reference=reference,
        seed_offset=0,
    )
    second = run_noise_series(
        series="unconstrained",
        n_trials=4,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1_000_000.0,
        reference=reference,
        seed_offset=4,
    )

    all_seeds = [t.seed for t in first] + [t.seed for t in second]
    assert len(all_seeds) == len(set(all_seeds))


def test_run_noise_series_rejects_non_positive_n_trials(session):
    reference = _reference_spec()
    ruleset = _permissive_ruleset()
    with pytest.raises(ValueError, match="n_trials must be positive"):
        run_noise_series(
            series="dollar_neutral",
            n_trials=0,
            session=session,
            ruleset=ruleset,
            deployable_capital_usd=1000.0,
            reference=reference,
        )


def test_run_real_strategies_runs_every_spec_path(session, tmp_path):
    reference_spec = _reference_spec()
    spec_path = tmp_path / "toy.yaml"
    spec_path.write_text(
        yaml.safe_dump(reference_spec.model_dump(mode="json")), encoding="utf-8"
    )
    ruleset = _permissive_ruleset()

    results = run_real_strategies(
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1_000_000.0,
        spec_paths={REFERENCE_IDEA_ID: spec_path},
    )

    assert set(results) == {REFERENCE_IDEA_ID}
    evaluation = results[REFERENCE_IDEA_ID]
    assert evaluation.error is None
    assert evaluation.metrics is not None
    assert evaluation.routing.route in {"paper", "shelf"}
