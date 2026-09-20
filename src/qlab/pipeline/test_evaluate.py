"""End-to-end tests for `evaluate_spec`: a toy strategy and a fixture panel
run through the full spec -> data -> backtest -> metrics -> rules -> verdict
pipeline, against an in-memory SQLite registry. No network: every snapshot
used here is pre-written to `tmp_path` and pre-registered in `data_snapshot`
directly, and `qlab.pipeline.evaluate.build_snapshot` is monkeypatched to
fail loudly if the pipeline ever tries to fetch instead of reusing it.
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

from qlab.data.panel import MarketPanel
from qlab.pipeline.evaluate import (
    StrategyResolutionError,
    decide_route,
    evaluate_spec,
    resolve_strategy,
)
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Profile,
    SourceType,
    Trial,
    TrialSource,
    TrialStatus,
    Verdict,
)
from qlab.registry.models import (
    Spec as SpecRow,
)
from qlab.rules.engine import EvaluationResult
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage

IDEA_ID = "toy-idea"
SOURCE = "hyperliquid"
INTERVAL = "1h"
START = date(2026, 1, 1)
END = date(2026, 1, 2)
INSTRUMENTS = ["BTC", "ETH"]


# --------------------------------------------------------------------------
# Toy strategies, resolved by dotted `code_ref` in tests below.
# --------------------------------------------------------------------------


class ToyStrategy:
    """Constant weight on every tradeable instrument. `params["weight"]"
    controls the magnitude, which in turn controls `min_capital_usd`
    (`min_leg_notional / weight`) — small weight -> big required capital."""

    name = "toy"
    default_weight = 0.4

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        weight = float(params.get("weight", self.default_weight))
        weights = pd.DataFrame(weight, index=panel.prices.index, columns=panel.prices.columns)
        return weights.where(panel.tradeable, 0.0)


class RaisingStrategy:
    """Always raises -- exercises the 'trial written even on failure' path."""

    name = "raising"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        raise RuntimeError("strategy blew up")


class NotAStrategy:
    """Resolvable by `code_ref` but does not implement the Strategy protocol."""


# --------------------------------------------------------------------------
# Fixtures: DB, idea row, fixture panel written straight to disk.
# --------------------------------------------------------------------------


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
def _idea_row(session):
    """Every spec below references this idea_id; `spec`/`trial`/`verdict`
    all FK to it, so it must exist before `evaluate_spec` is called."""
    repo.upsert_idea(
        session,
        id=IDEA_ID,
        title="Toy idea",
        source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.OTHER,
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any attempt to actually fetch from a venue fails the test loudly,
    rather than hanging on a real HTTP call."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "build_snapshot was called -- a matching snapshot should have been reused"
        )

    monkeypatch.setattr("qlab.pipeline.evaluate.build_snapshot", _boom)


def _fixture_frames(
    instruments: list[str],
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(
        pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="1h"
    )
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0015, 0.01, size=(len(index), len(instruments)))
    prices = (1.0 + pd.DataFrame(log_returns, index=index, columns=instruments)).cumprod() * 100.0
    funding = pd.DataFrame(0.0001, index=index, columns=instruments)
    tradeable = pd.DataFrame(True, index=index, columns=instruments)
    return index, prices, funding, tradeable


def _register_snapshot(
    session,
    tmp_path: Path,
    *,
    snapshot_id: str,
    instruments: list[str],
    universe_complete: bool,
) -> None:
    """Write a snapshot's parquet + manifest to disk and register its
    `data_snapshot` row directly -- the same shape `build_snapshot` would
    have produced, without touching the network."""
    _, prices, funding, tradeable = _fixture_frames(instruments)

    snap_dir = tmp_path / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prices", prices), ("funding", funding), ("tradeable", tradeable)):
        frame.to_parquet(snap_dir / f"{name}.parquet", engine="pyarrow", index=True)

    manifest = {
        "source": SOURCE,
        "instruments": sorted(instruments),
        "start": pd.Timestamp(START, tz="UTC").isoformat(),
        "end": pd.Timestamp(END, tz="UTC").isoformat(),
        "interval": INTERVAL,
        "universe_complete": universe_complete,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    repo.add_data_snapshot(
        session,
        id=snapshot_id,
        source=SOURCE,
        instruments=dict.fromkeys(sorted(instruments), True),
        range_start=START,
        range_end=END,
        path=str(snap_dir),
        rows=len(prices.index),
        fetched_at=datetime.now(UTC),
    )


DISCOVERED_ID = "snap-discovered"
EXPLICIT_ID = "snap-explicit"


def _register_discovered(session, tmp_path: Path) -> None:
    """Shorthand for the common case: a discovered-universe snapshot
    already on disk, registered under `DISCOVERED_ID`."""
    _register_snapshot(
        session,
        tmp_path,
        snapshot_id=DISCOVERED_ID,
        instruments=INSTRUMENTS,
        universe_complete=True,
    )


def _make_spec(**overrides: object) -> StrategySpec:
    base: dict[str, object] = {
        "idea_id": IDEA_ID,
        "title": "Toy strategy",
        "code_ref": "qlab.pipeline.test_evaluate:ToyStrategy",
        "params": {},
        "data": {
            "source": SOURCE,
            "interval": INTERVAL,
            "start": START,
            "end": END,
            "instruments": None,
        },
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0},
        "min_leg_notional": 12.0,
    }
    base.update(overrides)
    return StrategySpec.model_validate(base)


CAPITAL_FIT_GENEROUS = Rule(
    id="capital_fit",
    stage=Stage.PREFLIGHT,
    metric="min_capital_usd",
    comparator=Comparator.LE,
    threshold=1_000_000,
    fatal=True,
)
HONEST_UNIVERSE = Rule(
    id="honest_universe",
    stage=Stage.EDGE,
    metric="point_in_time_universe",
    comparator=Comparator.EQ,
    threshold=1,
    fatal=True,
)


def _ruleset(rules: list[Rule]) -> RuleSet:
    return RuleSet(version="2026-01-01.1", rules=rules)


# --------------------------------------------------------------------------
# resolve_strategy
# --------------------------------------------------------------------------


def test_resolve_strategy_colon_form() -> None:
    strategy = resolve_strategy("qlab.pipeline.test_evaluate:ToyStrategy")
    assert strategy.name == "toy"


def test_resolve_strategy_dotted_form() -> None:
    strategy = resolve_strategy("qlab.pipeline.test_evaluate.ToyStrategy")
    assert strategy.name == "toy"


def test_resolve_strategy_missing_module_fails_clearly() -> None:
    with pytest.raises(StrategyResolutionError, match="cannot import"):
        resolve_strategy("qlab.pipeline.no_such_module:Whatever")


def test_resolve_strategy_missing_attribute_fails_clearly() -> None:
    with pytest.raises(StrategyResolutionError, match="no attribute"):
        resolve_strategy("qlab.pipeline.test_evaluate:NoSuchStrategy")


def test_resolve_strategy_wrong_shape_rejected() -> None:
    with pytest.raises(StrategyResolutionError, match="Strategy protocol"):
        resolve_strategy("qlab.pipeline.test_evaluate:NotAStrategy")


# --------------------------------------------------------------------------
# decide_route (unit-level, synthetic EvaluationResult)
# --------------------------------------------------------------------------


def _result(
    *, decisive: bool, overall_passed: bool, failed_fatal=None, failed=(), unknown=()
) -> EvaluationResult:
    return EvaluationResult(
        rows=(),
        overall_passed=overall_passed,
        failed_fatal_rule_id=failed_fatal,
        failed_rule_ids=tuple(failed),
        unknown_metrics=tuple(unknown),
        decisive=decisive,
    )


def test_decide_route_needs_more_data_takes_priority() -> None:
    result = _result(decisive=False, overall_passed=False, unknown=("mystery",))
    routing = decide_route(result, {}, deployable_capital_usd=1000)
    assert routing.route == "needs-more-data"


def test_decide_route_reject_on_failed_rule() -> None:
    result = _result(
        decisive=True, overall_passed=False, failed_fatal="capital_fit", failed=("capital_fit",)
    )
    routing = decide_route(result, {"min_capital_usd": 1.0}, deployable_capital_usd=1000)
    assert routing.route == "reject"
    assert "capital_fit" in routing.reason


def test_decide_route_shelf_when_over_deployable() -> None:
    result = _result(decisive=True, overall_passed=True)
    routing = decide_route(result, {"min_capital_usd": 5000.0}, deployable_capital_usd=1000)
    assert routing.route == "shelf"
    assert routing.required_capital_usd == 5000.0


def test_decide_route_paper_when_affordable() -> None:
    result = _result(decisive=True, overall_passed=True)
    routing = decide_route(result, {"min_capital_usd": 500.0}, deployable_capital_usd=1000)
    assert routing.route == "paper"


def test_decide_route_never_returns_live() -> None:
    for decisive, overall_passed in [(False, False), (True, False), (True, True)]:
        result = _result(decisive=decisive, overall_passed=overall_passed)
        routing = decide_route(result, {"min_capital_usd": 1.0}, deployable_capital_usd=1_000_000)
        assert routing.route != "live"


# --------------------------------------------------------------------------
# evaluate_spec: end-to-end routing branches
# --------------------------------------------------------------------------


def test_route_paper_when_passed_and_affordable(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.routing.route == "paper"
    assert result.metrics["min_capital_usd"] == pytest.approx(12.0 / 0.4)
    assert result.metrics["point_in_time_universe"] == 1.0
    assert result.metrics["accrual_applied"] == 1.0

    trial = session.get(Trial, result.trial_id)
    assert trial.status == TrialStatus.OK
    assert trial.snapshot_id == DISCOVERED_ID
    assert trial.metrics is not None

    verdicts = session.query(Verdict).filter(Verdict.trial_id == result.trial_id).all()
    assert len(verdicts) == 2
    for v in verdicts:
        assert v.source == TrialSource.QLAB
        assert v.rules_version == "2026-01-01.1"
        assert v.data_range_start == START
        assert v.data_range_end == END
        assert v.passed is True


def test_route_shelf_when_min_capital_exceeds_deployable(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.001})  # min_capital_usd = 12000
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.routing.route == "shelf"
    assert result.routing.required_capital_usd == pytest.approx(12000.0)
    assert result.rules_result.overall_passed is True


def test_route_reject_on_fatal_capital_fit(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 1e-6})  # min_capital_usd = 12,000,000
    strict_capital_fit = Rule(
        id="capital_fit",
        stage=Stage.PREFLIGHT,
        metric="min_capital_usd",
        comparator=Comparator.LE,
        threshold=120_000,
        fatal=True,
    )
    ruleset = _ruleset([strict_capital_fit, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.routing.route == "reject"
    assert result.rules_result.failed_fatal_rule_id == "capital_fit"


def test_route_needs_more_data_when_metric_unknown(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    mystery_rule = Rule(
        id="mystery",
        stage=Stage.PREFLIGHT,
        metric="metric_this_pipeline_never_computes",
        comparator=Comparator.GE,
        threshold=0.0,
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, mystery_rule])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.rules_result.decisive is False
    assert result.routing.route == "needs-more-data"


def test_honest_universe_rejects_hand_picked_universe(session, tmp_path) -> None:
    _register_snapshot(
        session, tmp_path, snapshot_id=EXPLICIT_ID, instruments=INSTRUMENTS, universe_complete=False
    )
    spec = _make_spec(
        params={"weight": 0.4},
        data={
            "source": SOURCE,
            "interval": INTERVAL,
            "start": START,
            "end": END,
            "instruments": INSTRUMENTS,
        },
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["point_in_time_universe"] == 0.0
    assert result.routing.route == "reject"
    assert result.rules_result.failed_fatal_rule_id == "honest_universe"


def test_trial_written_even_when_strategy_raises(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:RaisingStrategy")
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is not None
    assert "blew up" in result.error
    assert result.metrics is None
    assert result.rules_result is None
    assert result.routing.route == "error"

    trial = session.get(Trial, result.trial_id)
    assert trial is not None
    assert trial.status == TrialStatus.ERROR
    assert trial.metrics is None
    assert trial.snapshot_id == DISCOVERED_ID

    verdicts = session.query(Verdict).filter(Verdict.trial_id == result.trial_id).all()
    assert verdicts == []


def test_missing_code_ref_is_recorded_as_error_trial(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.no_such_module:Nope")
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is not None
    trial = session.get(Trial, result.trial_id)
    assert trial.status == TrialStatus.ERROR


def test_snapshot_not_refetched_when_already_registered(session, tmp_path) -> None:
    # `_no_network` (autouse) already fails the test if build_snapshot is
    # called; this test just exercises the ordinary success path to prove
    # that never happens when a matching snapshot is already registered.
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.routing.route == "paper"


def test_repeated_evaluation_reuses_registry_spec_row(session, tmp_path) -> None:
    """Two evaluations of byte-identical spec content append two `trial`
    rows (every run counts) but only one `spec` row (content-versioned)."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    first = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)
    second = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert first.trial_id != second.trial_id
    first_trial = session.get(Trial, first.trial_id)
    second_trial = session.get(Trial, second.trial_id)
    assert first_trial.spec_id == second_trial.spec_id

    spec_rows = session.query(SpecRow).filter(SpecRow.idea_id == IDEA_ID).all()
    assert len(spec_rows) == 1
