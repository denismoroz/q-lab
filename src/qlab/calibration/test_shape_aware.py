"""Tests for `qlab.calibration.shape_aware` (docs/TASKS.md T27/T28).

Same fixture shape as `qlab.calibration.test_run`: in-memory SQLite, a
hand-registered snapshot on disk (no network), and a toy reference
strategy -- these tests exercise the WIRING between `run_noise_series`,
`evaluate_spec`'s `extra_metrics` hook, and `qlab.rules.engine.evaluate`,
not the real strategies or a real snapshot.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
from qlab.harness.panel import MarketPanel
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Profile, SourceType
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage

SOURCE = "hyperliquid"
INTERVAL = "1d"
START = date(2025, 1, 1)
END = date(2025, 4, 1)
INSTRUMENTS = [f"COIN{i}" for i in range(8)]
SNAPSHOT_ID = "snap-shape-aware-test"
IDEA_ID = "toy-shape-aware"


class ToyMomentumStrategy:
    """Same shape as `qlab.calibration.test_run.ToyMomentumStrategy`: a
    tiny 'real' reference with non-zero turnover, so noise generated
    against it structurally matches without special-casing."""

    name = "toy-momentum"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        seed = int(params.get("seed", 7))
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
def _idea_row(session):
    repo.upsert_idea(
        session,
        id=IDEA_ID,
        title="Toy shape-aware candidate",
        source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.OTHER,
    )


@pytest.fixture(autouse=True)
def _register_snapshot(session, tmp_path: Path):
    index = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="1D")
    rng = np.random.default_rng(23)
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


def _spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "idea_id": IDEA_ID,
            "title": "Toy shape-aware candidate",
            "code_ref": "qlab.calibration.test_shape_aware:ToyMomentumStrategy",
            "params": {"seed": 7},
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


def _shape_aware_ruleset(*, threshold: float = 0.99) -> RuleSet:
    return RuleSet(
        version="2020-01-01.1",
        based_on=None,
        rules=[
            Rule(
                id="net_edge_positive",
                stage=Stage.EDGE,
                metric="ann_return_net",
                comparator=Comparator.GE,
                threshold=-1.0,  # permissive: these tests are about the percentile rule
                fatal=True,
            ),
            Rule(
                id="shape_aware_edge",
                stage=Stage.EDGE,
                metric="noise_return_percentile",
                comparator=Comparator.GE,
                threshold=threshold,
                fatal=True,
            ),
        ],
        retired=[],
    )


# --------------------------------------------------------------------------


def test_writes_n_trials_plus_one_and_folds_percentile_into_candidate_metrics(session):
    spec = _spec()
    ruleset = _shape_aware_ruleset(threshold=0.0)  # trivially satisfied either way

    result = evaluate_spec_with_shape_aware_bar(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1_000_000.0,
        n_trials=8,
        seed_offset=0,
    )

    assert len(result.noise_trials) == 8
    assert result.candidate.error is None
    assert "noise_return_percentile" in result.candidate.metrics
    assert result.percentiles is not None
    assert result.percentiles.n_requested == 8
    assert 0.0 <= result.percentiles.return_percentile <= 1.0

    # exactly n_trials + 1 real trial rows: the 8 noise trials plus the
    # candidate itself (CLAUDE.md: "trial пишется всегда").
    from qlab.registry.models import Trial

    assert session.query(Trial).count() == 9


def test_candidate_below_threshold_is_rejected_by_the_percentile_rule(session):
    spec = _spec()
    # A threshold of 1.01 cannot be met by any candidate (percentile is
    # capped at 1.0) -- guarantees rejection regardless of the actual noise
    # draw, so this test is about ROUTING, not about hitting a precise
    # percentile value.
    ruleset = _shape_aware_ruleset(threshold=1.01)

    result = evaluate_spec_with_shape_aware_bar(
        spec, session=session, ruleset=ruleset, deployable_capital_usd=1_000_000.0, n_trials=8
    )

    assert result.candidate.error is None
    assert result.candidate.rules_result.overall_passed is False
    assert "shape_aware_edge" in result.candidate.rules_result.failed_rule_ids
    assert result.candidate.routing.route == "reject"


def test_candidate_at_or_above_threshold_is_admitted(session):
    spec = _spec()
    ruleset = _shape_aware_ruleset(threshold=-1.0)  # trivially satisfied

    result = evaluate_spec_with_shape_aware_bar(
        spec, session=session, ruleset=ruleset, deployable_capital_usd=1_000_000.0, n_trials=8
    )

    assert result.candidate.error is None
    assert result.candidate.rules_result.overall_passed is True
    assert result.candidate.routing.route in ("paper", "shelf")


def test_percentile_metric_absent_without_shape_aware_evaluation(session):
    """A plain `evaluate_spec` call (no shape-aware wrapper) must never see
    a `noise_return_percentile` key -- it is not fabricated, and its
    absence must read as unknown, not as a pass, under a ruleset that
    references it."""
    from qlab.pipeline.evaluate import evaluate_spec

    spec = _spec()
    ruleset = _shape_aware_ruleset(threshold=0.0)

    evaluation = evaluate_spec(
        spec, session=session, ruleset=ruleset, deployable_capital_usd=1_000_000.0
    )

    assert evaluation.error is None
    assert "noise_return_percentile" not in evaluation.metrics
    assert evaluation.rules_result.decisive is False
    assert "noise_return_percentile" in evaluation.rules_result.unknown_metrics
    assert evaluation.routing.route == "needs-more-data"
