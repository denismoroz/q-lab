"""End-to-end tests for `evaluate_spec`: a toy strategy and a fixture panel
run through the full spec -> data -> backtest -> metrics -> rules -> verdict
pipeline, against an in-memory SQLite registry. No network: every snapshot
used here is pre-written to `tmp_path` and pre-registered in `data_snapshot`
directly, and `qlab.pipeline.evaluate.build_snapshot` is monkeypatched to
fail loudly if the pipeline ever tries to fetch instead of reusing it.
"""

from __future__ import annotations

import dataclasses
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
from qlab.data.snapshot import PANEL_RULES_VERSION
from qlab.pipeline.evaluate import (
    FORWARD_NOTE,
    SELECTION_NOTE,
    PeriodSplit,
    StrategyResolutionError,
    _find_matching_snapshot,
    complete_book_window,
    decide_fit_forward_route,
    decide_route,
    evaluate_spec,
    forward_years_needed,
    resolve_strategy,
    split_at_fixed_date,
)
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    StageTransition,
    Trial,
    TrialRoute,
    TrialSource,
    TrialStatus,
    Verdict,
)
from qlab.registry.models import (
    Spec as SpecRow,
)
from qlab.rules.engine import EvaluationResult
from qlab.rules.schema import (
    Comparator,
    FitPeriodUse,
    ForwardResolution,
    Rule,
    RuleKind,
    RuleSet,
    ShortForwardUse,
    Stage,
)

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
    instruments: list[str], end: date | None = None
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(
        pd.Timestamp(START, tz="UTC"), pd.Timestamp(end or END, tz="UTC"), freq="1h"
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
    tradeable: pd.DataFrame | None = None,
    end: date | None = None,
) -> None:
    """Write a snapshot's parquet + manifest to disk and register its
    `data_snapshot` row directly -- the same shape `build_snapshot` would
    have produced, without touching the network. `tradeable` overrides the
    default all-True listing mask."""
    end = end or END
    _, prices, funding, default_tradeable = _fixture_frames(instruments, end)
    tradeable = default_tradeable if tradeable is None else tradeable

    snap_dir = tmp_path / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prices", prices), ("funding", funding), ("tradeable", tradeable)):
        frame.to_parquet(snap_dir / f"{name}.parquet", engine="pyarrow", index=True)

    manifest = {
        "source": SOURCE,
        "instruments": sorted(instruments),
        "start": pd.Timestamp(START, tz="UTC").isoformat(),
        "end": pd.Timestamp(end, tz="UTC").isoformat(),
        "interval": INTERVAL,
        "universe_complete": universe_complete,
        "panel_rules": PANEL_RULES_VERSION,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    repo.add_data_snapshot(
        session,
        id=snapshot_id,
        source=SOURCE,
        instruments=dict.fromkeys(sorted(instruments), True),
        range_start=START,
        range_end=end,
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
        "unexpressed_mechanisms": [],
        # Test fixtures cite the test author (docs/SOURCES.md).
        "sources": {"weight": "owner 2026-01-01: test fixture",
                    "costs": "owner 2026-01-01: test fixture",
                    "min_leg_notional": "owner 2026-01-01: test fixture"},
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
CAPITAL_FIT_TIGHT = Rule(
    # min_capital_usd is always >= the 12.0 min leg notional, so this always fails.
    id="capital_fit",
    stage=Stage.PREFLIGHT,
    metric="min_capital_usd",
    comparator=Comparator.LE,
    threshold=1.0,
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


def test_extra_metrics_mapping_is_merged_and_persisted(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1000.0,
        extra_metrics={"noise_return_percentile": 0.995},
    )

    assert result.error is None
    assert result.metrics["noise_return_percentile"] == pytest.approx(0.995)
    trial = session.get(Trial, result.trial_id)
    assert trial.metrics["noise_return_percentile"] == pytest.approx(0.995)


def test_extra_metrics_callable_sees_the_computed_base_metrics(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])
    seen: dict[str, float] = {}

    def _extra(base_metrics: dict[str, float]) -> dict[str, float]:
        seen.update(base_metrics)
        return {"double_ann_return_net": base_metrics["ann_return_net"] * 2}

    result = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1000.0,
        extra_metrics=_extra,
    )

    assert result.error is None
    assert "ann_return_net" in seen  # the callable saw the real backtest metrics
    assert result.metrics["double_ann_return_net"] == pytest.approx(
        result.metrics["ann_return_net"] * 2
    )


def test_extra_metrics_omitted_key_reads_as_unknown_not_a_pass(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    percentile_rule = Rule(
        id="shape_aware_edge",
        stage=Stage.EDGE,
        metric="noise_return_percentile",
        comparator=Comparator.GE,
        threshold=0.99,
        fatal=True,
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, percentile_rule])

    # extra_metrics deliberately returns no key for a metric it could not
    # compute (e.g. no usable matched-noise sample) -- must not read as a
    # pass just because nothing failed.
    result = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1000.0,
        extra_metrics=lambda _base: {},
    )

    assert result.error is None
    assert "noise_return_percentile" not in result.metrics
    assert result.rules_result.decisive is False
    assert "noise_return_percentile" in result.rules_result.unknown_metrics
    assert result.routing.route == "needs-more-data"


def test_extra_metrics_raising_is_recorded_as_an_error_trial(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    def _boom(_base: dict[str, float]) -> dict[str, float]:
        raise ValueError("no usable noise sample")

    result = evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=1000.0,
        extra_metrics=_boom,
    )

    assert result.error is not None
    assert "no usable noise sample" in result.error
    assert result.routing.route == "error"
    trial = session.get(Trial, result.trial_id)
    assert trial.status == TrialStatus.ERROR
    assert trial.metrics is None


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


# --------------------------------------------------------------------------
# venue metrics (docs/TASKS.md T13): venue_supported / data_forward_available
# / atomic_execution, sourced from `venues/*.yaml` via `qlab.venues`.
# --------------------------------------------------------------------------


def test_venue_metrics_present_for_configured_hyperliquid_source(session, tmp_path) -> None:
    """The real `venues/hyperliquid.yaml` this task ships is read straight
    off disk (no network) and turns into all three preflight metrics for a
    single-leg spec."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["venue_supported"] == 1.0
    assert result.metrics["data_forward_available"] == 1.0
    assert result.metrics["atomic_execution"] == 1.0


def test_frab_like_multileg_spec_passes_atomic_execution_via_recovery(session, tmp_path) -> None:
    """The task's central question: a two-leg (spot + perp short) spec on
    Hyperliquid must NOT be rejected just because the two legs fill as
    separate, sequential orders (`venues/hyperliquid.yaml`:
    `supports_atomic_multileg: false`). It passes because the live engine
    names a real recovery mechanism (`uncovered_leg_recovery`) that unwinds
    an uncovered leg after a failure -- exactly what SCREENING.md's
    atomicity rule actually asks for. This is the FRAB scenario from
    docs/TASKS.md T13."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4}, simultaneous_legs=2)
    atomic_rule = Rule(
        id="atomic_execution",
        stage=Stage.PREFLIGHT,
        metric="atomic_execution",
        comparator=Comparator.EQ,
        threshold=1,
        fatal=True,
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, atomic_rule])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["atomic_execution"] == 1.0
    assert result.routing.route == "paper"


def test_venue_metrics_absent_when_venue_unconfigured(session, tmp_path) -> None:
    """A spec naming a venue with no `venues/<id>.yaml` file must not crash
    the pipeline -- venue_supported/data_forward_available/atomic_execution
    are simply absent from `metrics`, so a preflight rule on any of them
    honestly reports `needs-more-data` (docs/TASKS.md T13) instead of a
    guessed pass or fail."""
    unconfigured_source = "no-such-venue"
    _, prices, funding, tradeable = _fixture_frames(INSTRUMENTS)
    snap_dir = tmp_path / "snap-unconfigured"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prices", prices), ("funding", funding), ("tradeable", tradeable)):
        frame.to_parquet(snap_dir / f"{name}.parquet", engine="pyarrow", index=True)
    manifest = {
        "source": unconfigured_source,
        "instruments": sorted(INSTRUMENTS),
        "start": pd.Timestamp(START, tz="UTC").isoformat(),
        "end": pd.Timestamp(END, tz="UTC").isoformat(),
        "interval": INTERVAL,
        "universe_complete": True,
        "panel_rules": PANEL_RULES_VERSION,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    repo.add_data_snapshot(
        session,
        id="snap-unconfigured",
        source=unconfigured_source,
        instruments=dict.fromkeys(sorted(INSTRUMENTS), True),
        range_start=START,
        range_end=END,
        path=str(snap_dir),
        rows=len(prices.index),
        fetched_at=datetime.now(UTC),
    )
    spec = _make_spec(
        params={"weight": 0.4},
        data={
            "source": unconfigured_source,
            "interval": INTERVAL,
            "start": START,
            "end": END,
            "instruments": None,
        },
    )
    venue_rule = Rule(
        id="venue_supported",
        stage=Stage.PREFLIGHT,
        metric="venue_supported",
        comparator=Comparator.EQ,
        threshold=1,
        fatal=True,
    )
    ruleset = _ruleset([venue_rule])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert "venue_supported" not in result.metrics
    assert "data_forward_available" not in result.metrics
    # atomic_execution IS present here, honestly: this spec's default
    # `simultaneous_legs=1` satisfies it trivially regardless of venue
    # (qlab.venues.derive.atomic_execution) -- it is venue_supported that
    # stays absent, since no `venues/no-such-venue.yaml` file exists.
    assert result.metrics["atomic_execution"] == 1.0
    assert result.rules_result.decisive is False
    assert result.routing.route == "needs-more-data"


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


# --------------------------------------------------------------------------
# T24: not-evaluable is an outcome, decided before anything is computed.
# --------------------------------------------------------------------------


def test_unexpressed_mechanism_routes_not_evaluable_without_running_the_strategy(
    session, tmp_path
) -> None:
    """The strategy here raises if called. A not-evaluable route rather than
    an error proves the pipeline stopped before running it -- no number about
    a different strategy is computed, let alone judged."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(
        code_ref="qlab.pipeline.test_evaluate:RaisingStrategy",
        unexpressed_mechanisms=["per-position breakeven state machine"],
    )

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "per-position breakeven state machine" in evaluation.routing.reason
    assert evaluation.error is None
    assert evaluation.metrics is None
    trial = session.get(Trial, evaluation.trial_id)
    assert trial.status == TrialStatus.NOT_EVALUABLE
    assert trial.route == TrialRoute.NOT_EVALUABLE
    assert session.query(Verdict).filter(Verdict.trial_id == trial.id).count() == 0


def test_not_evaluable_never_moves_an_idea_to_rejected(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(unexpressed_mechanisms=["something the interface cannot carry"])

    evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert session.get(Idea, IDEA_ID).status == IdeaStatus.CANDIDATE
    assert session.query(StageTransition).count() == 0


def test_required_instrument_absent_from_data_is_not_evaluable(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(required_instruments=["BTC", "AVAX-SPOT"])

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "AVAX-SPOT" in evaluation.routing.reason


def test_book_never_complete_is_not_evaluable(session, tmp_path) -> None:
    index, *_ = _fixture_frames(INSTRUMENTS)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)
    tradeable.iloc[: len(index) // 2, 0] = False  # BTC lists halfway through
    tradeable.iloc[len(index) // 2 :, 1] = False  # ETH delists halfway through
    _register_snapshot(
        session,
        tmp_path,
        snapshot_id=DISCOVERED_ID,
        instruments=INSTRUMENTS,
        universe_complete=True,
        tradeable=tradeable,
    )
    spec = _make_spec(required_instruments=["BTC", "ETH"])

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "never complete" in evaluation.routing.reason
    assert session.get(Trial, evaluation.trial_id).metrics == {"book_coverage": 0.0}


def test_partial_book_is_judged_only_where_it_was_complete(session, tmp_path) -> None:
    """ETH lists late. The verdict's data range must start where the whole
    book first existed, and coverage must be reported, not thresholded."""
    index, *_ = _fixture_frames(INSTRUMENTS)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)
    late = 10
    tradeable.iloc[:late, 1] = False
    _register_snapshot(
        session,
        tmp_path,
        snapshot_id=DISCOVERED_ID,
        instruments=INSTRUMENTS,
        universe_complete=True,
        tradeable=tradeable,
    )
    spec = _make_spec(required_instruments=["BTC", "ETH"])

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "paper"
    assert evaluation.metrics["book_coverage"] == pytest.approx((len(index) - late) / len(index))
    verdict = session.query(Verdict).filter(Verdict.trial_id == evaluation.trial_id).one()
    assert verdict.data_range_start == index[late].date()
    assert verdict.data_range_end == index[-1].date()


def test_complete_book_window_picks_the_longest_unbroken_stretch() -> None:
    index, prices, funding, _ = _fixture_frames(INSTRUMENTS)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)
    tradeable.iloc[3, 0] = False  # stretches: 0..2 (3 bars) and 4..end
    tradeable.iloc[-5, 1] = False  # ... which splits into 4..n-6 and n-4..n-1
    panel = MarketPanel(
        snapshot_id="x",
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta={"universe_complete": True},
    )

    coverage = complete_book_window(panel, INSTRUMENTS)

    assert coverage.window == (4, len(index) - 6)
    assert coverage.coverage == pytest.approx((len(index) - 2) / len(index))
    assert coverage.missing == ()


def test_complete_book_window_full_coverage_is_the_whole_panel() -> None:
    index, prices, funding, tradeable = _fixture_frames(INSTRUMENTS)
    panel = MarketPanel(
        snapshot_id="x",
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta={"universe_complete": True},
    )

    coverage = complete_book_window(panel, INSTRUMENTS)

    assert coverage.window == (0, len(index) - 1)
    assert coverage.coverage == 1.0


# --------------------------------------------------------------------------
# T31: the route is stored on the trial and moves the idea's status.
# --------------------------------------------------------------------------


def test_reject_moves_candidate_to_rejected_and_names_the_trial(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec()

    evaluation = evaluate_spec(
        spec,
        session=session,
        ruleset=_ruleset([CAPITAL_FIT_TIGHT]),
        deployable_capital_usd=1e9,
    )

    assert evaluation.routing.route == "reject"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.REJECTED
    transition = session.query(StageTransition).one()
    assert transition.from_status == IdeaStatus.CANDIDATE
    assert transition.to_status == IdeaStatus.REJECTED
    assert transition.trial_id == evaluation.trial_id
    assert transition.rules_version == "2026-01-01.1"
    assert session.get(Trial, evaluation.trial_id).route == TrialRoute.REJECT


def test_paper_route_stops_at_validated_because_paper_is_the_owners_gate(
    session, tmp_path
) -> None:
    _register_discovered(session, tmp_path)

    evaluation = evaluate_spec(
        _make_spec(),
        session=session,
        ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
        deployable_capital_usd=1e9,
    )

    assert evaluation.routing.route == "paper"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.VALIDATED


def test_shelf_route_moves_to_bench(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)

    evaluation = evaluate_spec(
        _make_spec(),
        session=session,
        ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
        deployable_capital_usd=0.01,
    )

    assert evaluation.routing.route == "shelf"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.BENCH


def test_backtest_never_overrides_a_production_status(session, tmp_path) -> None:
    """An idea running in paper got there by the owner's decision; a
    rejecting backtest is recorded but does not move it."""
    _register_discovered(session, tmp_path)
    repo.set_status(session, idea_id=IDEA_ID, new_status=IdeaStatus.PAPER, reason="owner")

    evaluation = evaluate_spec(
        _make_spec(),
        session=session,
        ruleset=_ruleset([CAPITAL_FIT_TIGHT]),
        deployable_capital_usd=1e9,
    )

    assert evaluation.routing.route == "reject"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.PAPER
    assert session.get(Trial, evaluation.trial_id).route == TrialRoute.REJECT


def test_update_idea_status_false_leaves_status_alone(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)

    evaluation = evaluate_spec(
        _make_spec(),
        session=session,
        ruleset=_ruleset([CAPITAL_FIT_TIGHT]),
        deployable_capital_usd=1e9,
        update_idea_status=False,
    )

    assert evaluation.routing.route == "reject"
    assert evaluation.status_decision is None
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.CANDIDATE
    # The route is still recorded: skipping the status move is not skipping the record.
    assert session.get(Trial, evaluation.trial_id).route == TrialRoute.REJECT


def test_error_trial_records_its_route_and_moves_nothing(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:RaisingStrategy")

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    trial = session.get(Trial, evaluation.trial_id)
    assert trial.route == TrialRoute.ERROR
    assert "strategy blew up" in trial.route_reason
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.CANDIDATE


def test_universe_with_erased_delisted_history_is_not_evaluable(session, tmp_path) -> None:
    """Survivors-only data is not a test of the strategy: not-evaluable, with
    the instruments named, before any backtest."""
    _register_discovered(session, tmp_path)
    manifest_path = tmp_path / DISCOVERED_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["universe_complete"] = False
    manifest["delisted_without_history"] = ["xyz:LRCX", "xyz:GLW"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:RaisingStrategy")

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "xyz:LRCX" in evaluation.routing.reason and "only survivors" in evaluation.routing.reason


def test_exploratory_run_computes_everything_but_decides_nothing(session, tmp_path) -> None:
    """Owner-authorised run past a named reason: metrics and verdict rows
    exist, the rules' own answer is visible, the route stays not-evaluable
    and the idea's status does not move."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(
        unexpressed_mechanisms=["a mechanism the toy leaves out"],
        exploratory="owner wants to see the number on the data that exists",
    )

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "rules alone would route paper" in evaluation.routing.reason
    assert "a mechanism the toy leaves out" in evaluation.routing.reason
    assert evaluation.metrics is not None and "ann_return_net" in evaluation.metrics
    trial = session.get(Trial, evaluation.trial_id)
    assert trial.status == TrialStatus.OK
    assert trial.route == TrialRoute.NOT_EVALUABLE
    assert session.query(Verdict).filter(Verdict.trial_id == trial.id).count() == 1
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.CANDIDATE


def test_exploratory_flag_changes_nothing_when_there_is_no_reason(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(exploratory="nothing to override")

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "paper"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.VALIDATED


# --------------------------------------------------------------------------
# T35: missing infrastructure is not a rejection.
# --------------------------------------------------------------------------

NO_SUCH_VENUE_METRIC = Rule(
    # Always fails: the toy pipeline never emits this metric as 1.
    id="venue_supported",
    stage=Stage.PREFLIGHT,
    metric="min_capital_usd",
    comparator=Comparator.LE,
    threshold=1.0,
    fatal=True,
    kind=RuleKind.INFRASTRUCTURE,
)


def test_decide_route_needs_infrastructure_when_only_infrastructure_fails() -> None:
    result = _result(decisive=True, overall_passed=False, failed=("venue_supported",))
    result = dataclasses.replace(result, failed_infrastructure_rule_ids=("venue_supported",))
    routing = decide_route(result, {"min_capital_usd": 50.0}, deployable_capital_usd=1000)
    assert routing.route == "needs-infrastructure"
    assert "venue_supported" in routing.reason


def test_decide_route_rejects_when_a_strategy_rule_also_fails() -> None:
    result = _result(
        decisive=True, overall_passed=False, failed_fatal="net_edge_positive",
        failed=("venue_supported", "net_edge_positive"),
    )
    result = dataclasses.replace(result, failed_infrastructure_rule_ids=("venue_supported",))
    routing = decide_route(result, {"min_capital_usd": 50.0}, deployable_capital_usd=1000)
    assert routing.route == "reject"
    assert "net_edge_positive" in routing.reason and "missing infrastructure" in routing.reason


def test_missing_infrastructure_benches_the_idea_and_keeps_the_strategy_verdict(
    session, tmp_path
) -> None:
    _register_discovered(session, tmp_path)

    evaluation = evaluate_spec(
        _make_spec(),
        session=session,
        ruleset=_ruleset([NO_SUCH_VENUE_METRIC, CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE]),
        deployable_capital_usd=1e9,
    )

    assert evaluation.routing.route == "needs-infrastructure"
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.BENCH
    rule_ids = {
        v.rule_id for v in session.query(Verdict).filter(Verdict.trial_id == evaluation.trial_id)
    }
    assert "honest_universe" in rule_ids  # the edge stage was still evaluated


class DailyOnlyToy(ToyStrategy):
    valid_intervals = ("1d",)


def test_strategy_not_valid_on_the_spec_interval_is_not_evaluable(session, tmp_path) -> None:
    """T25: this module's fixture panel is hourly; a strategy that declares
    itself valid on daily bars only must not be run on it."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:DailyOnlyToy")

    evaluation = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]), deployable_capital_usd=1e9
    )

    assert evaluation.routing.route == "not-evaluable"
    assert "not valid on '1h'" in evaluation.routing.reason
    assert session.get(Trial, evaluation.trial_id).status == TrialStatus.NOT_EVALUABLE


def test_truncated_history_moves_the_judged_window_to_where_it_is_honest(
    session, tmp_path
) -> None:
    """T25: verdicts are rendered only from the last truncated instrument's
    first served bar onward."""
    _register_discovered(session, tmp_path)
    index, *_ = _fixture_frames(INSTRUMENTS)
    honest_from = index[10]
    manifest_path = tmp_path / DISCOVERED_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["history_truncated"] = {"ETH": honest_from.isoformat(), "BTC": index[3].isoformat()}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evaluation = evaluate_spec(
        _make_spec(), session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
        deployable_capital_usd=1e9,
    )

    assert evaluation.metrics["history_truncated_instruments"] == 2.0
    # Judged from bar 10 (the later of the two truncation points) to the end.
    assert evaluation.metrics["n_periods"] <= len(index) - 10
    verdict = session.query(Verdict).filter(Verdict.trial_id == evaluation.trial_id).one()
    assert verdict.data_range_start == honest_from.date()


# --------------------------------------------------------------------------
# Selection period vs forward test (docs/FIT_VS_FORWARD.md)
# --------------------------------------------------------------------------

LONG_END = date(2026, 1, 11)  # ten days of hourly bars
RESOLUTION = ForwardResolution(confidence=0.95, power=0.80, rationale="test")
HONEST_UNIVERSE_INFO = HONEST_UNIVERSE.model_copy(
    update={"fit_period": FitPeriodUse.INFORMATIONAL}
)


def _split_ruleset(rules: list[Rule]) -> RuleSet:
    return RuleSet(version="2026-01-01.1", rules=rules, forward_resolution=RESOLUTION)


def _explicit_long(session, tmp_path, **spec_overrides) -> StrategySpec:
    _register_snapshot(
        session,
        tmp_path,
        snapshot_id="snap-long",
        instruments=INSTRUMENTS,
        universe_complete=False,
        end=LONG_END,
    )
    data = {"source": SOURCE, "interval": INTERVAL, "start": START, "end": LONG_END,
            "instruments": INSTRUMENTS}
    return _make_spec(params={"weight": 0.4}, data=data, **spec_overrides)


def test_split_at_fixed_date_bounds() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    assert split_at_fixed_date(index, 0, 47, None) == PeriodSplit((0, 47), None)
    assert split_at_fixed_date(index, 0, 47, date(2026, 1, 2)) == PeriodSplit((0, 23), (24, 47))
    # Fixed before the data began: all forward. Fixed after it ended: all selection.
    assert split_at_fixed_date(index, 0, 47, date(2025, 12, 1)) == PeriodSplit(None, (0, 47))
    assert split_at_fixed_date(index, 0, 47, date(2026, 2, 1)) == PeriodSplit((0, 47), None)


def test_forward_years_needed_matches_the_preregistered_horizon() -> None:
    # seed/preregistration/2026-09-21-paper-bv2-trend.yaml: trend at Sharpe
    # 0.305 needs 66.5 years.
    assert forward_years_needed(0.305, RESOLUTION) == pytest.approx(66.5, abs=0.1)
    assert forward_years_needed(0.0, RESOLUTION) is None
    assert forward_years_needed(float("nan"), RESOLUTION) is None


def test_unknown_fixed_date_makes_a_pass_wait_for_forward_not_paper(session, tmp_path) -> None:
    spec = _explicit_long(session, tmp_path)
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    # The hand-picked universe fails honest_universe, but on the selection
    # period that is informational: it neither rejects nor admits.
    assert result.routing.route == "needs-forward"
    assert "honest_universe" in result.routing.reason
    assert result.metrics["judged_on_forward"] == 0.0
    notes = {v.note for v in session.query(Verdict).filter(Verdict.trial_id == result.trial_id)}
    assert notes == {SELECTION_NOTE}


def test_conclusive_failure_on_the_selection_period_rejects(session, tmp_path) -> None:
    spec = _explicit_long(session, tmp_path)
    ruleset = _split_ruleset([CAPITAL_FIT_TIGHT, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.routing.route == "reject"
    assert "selection period" in result.routing.reason
    assert "capital_fit" in result.routing.reason


def test_forward_test_is_judged_and_a_fixed_list_is_honest_there(session, tmp_path) -> None:
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test"
    )
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.metrics["judged_on_forward"] == 1.0
    assert result.metrics["point_in_time_universe"] == 1.0  # list fixed before the forward test
    assert result.metrics["fit_point_in_time_universe"] == 0.0  # chosen with hindsight before
    assert result.metrics["selection_days"] == pytest.approx(5.0)
    assert result.metrics["forward_days"] == pytest.approx(5.0 + 1 / 24)
    # Five days pass every rule, and still decide nothing.
    assert result.routing.route == "needs-forward"
    assert "would route paper" in result.routing.reason

    rows = session.query(Verdict).filter(Verdict.trial_id == result.trial_id).all()
    forward = [v for v in rows if v.note == FORWARD_NOTE]
    selection = [v for v in rows if v.note == SELECTION_NOTE]
    assert {v.data_range_start for v in forward} == {date(2026, 1, 6)}
    assert {v.data_range_end for v in selection} == {date(2026, 1, 5)}


def _rows(*rows: tuple[str, bool | None]) -> EvaluationResult:
    from qlab.rules.engine import VerdictRow

    built = tuple(
        VerdictRow(stage=Stage.EDGE, rule_id=rid, rules_version="t", metric=rid, value=None,
                   comparator=">=", threshold=0.0, passed=passed)
        for rid, passed in rows
    )
    failed = tuple(r.rule_id for r in built if r.passed is False)
    return EvaluationResult(
        rows=built,
        overall_passed=not failed and all(r.passed for r in built),
        failed_fatal_rule_id=failed[0] if failed else None,
        failed_rule_ids=failed,
        unknown_metrics=tuple(r.metric for r in built if r.passed is None),
        decisive=all(r.passed is not None for r in built),
    )


NET_EDGE = Rule(id="net_edge", stage=Stage.EDGE, metric="net_edge",
                comparator=Comparator.GE, threshold=0.04, fatal=True)


def test_short_failing_forward_test_waits_long_failing_one_rejects() -> None:
    ruleset = _split_ruleset([NET_EDGE])
    selection = _rows(("net_edge", True))
    forward = _rows(("net_edge", False))
    base = {"fit_sharpe_net": 0.5, "forward_days_needed": 9000.0, "min_capital_usd": 10.0}

    short = decide_fit_forward_route(
        ruleset=ruleset, selection=selection, forward=forward,
        metrics={**base, "forward_days": 30.0}, deployable_capital_usd=1000.0,
        params_fixed_at=date(2026, 1, 1),
    )
    assert short.route == "needs-forward"
    assert "too short" in short.reason

    long = decide_fit_forward_route(
        ruleset=ruleset, selection=selection, forward=forward,
        metrics={**base, "forward_days": 9500.0}, deployable_capital_usd=1000.0,
        params_fixed_at=date(2026, 1, 1),
    )
    assert long.route == "reject"
    assert "forward test failed" in long.reason


def test_short_passing_forward_test_also_waits() -> None:
    ruleset = _split_ruleset([NET_EDGE])
    passing = _rows(("net_edge", True))
    base = {"fit_sharpe_net": 0.5, "forward_days_needed": 9000.0, "min_capital_usd": 10.0}
    kwargs = dict(ruleset=ruleset, selection=passing, forward=passing,
                  deployable_capital_usd=1000.0, params_fixed_at=date(2026, 1, 1))

    assert decide_fit_forward_route(**kwargs, metrics={**base, "forward_days": 18.0}).route \
        == "needs-forward"
    assert decide_fit_forward_route(**kwargs, metrics={**base, "forward_days": 9500.0}).route \
        == "paper"


def test_without_forward_resolution_the_window_is_judged_as_before(session, tmp_path) -> None:
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test"
    )
    result = evaluate_spec(spec, session=session, ruleset=_ruleset([HONEST_UNIVERSE]),
                           deployable_capital_usd=1000.0)
    assert result.routing.route == "reject"
    assert "judged_on_forward" not in result.metrics


def test_fixed_date_without_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="params_fixed_evidence"):
        _make_spec(params_fixed_at=date(2026, 1, 1))


def test_causal_selection_is_judged_from_its_first_choice_without_waiting(session, tmp_path):
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 3), params_fixed_evidence="test",
        selects_causally=True,
    )
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    # No selection period: the forward test's own route stands, even when short.
    assert result.routing.route == "paper"
    assert result.metrics["warmup_days"] == pytest.approx(2.0)
    assert result.metrics["selection_days"] == 0.0
    assert not any(k.startswith("fit_") for k in result.metrics)


def test_causal_selection_needs_its_first_choice_date() -> None:
    with pytest.raises(ValueError, match="first causal choice"):
        _make_spec(selects_causally=True)


NOISE_BAR = Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="shape_aware_edge",
                 comparator=Comparator.GE, threshold=0.99, fatal=True,
                 on_short_forward=ShortForwardUse.WAIT)


def test_forward_only_noise_miss_on_a_short_window_waits() -> None:
    ruleset = _split_ruleset([NET_EDGE, NOISE_BAR])
    noise_miss = _rows(("net_edge", True), ("shape_aware_edge", False))
    kwargs = dict(ruleset=ruleset, selection=None, forward=noise_miss,
                  deployable_capital_usd=1000.0, params_fixed_at=date(2025, 7, 1))
    base = {"sharpe_net": 0.75, "forward_days_needed": 4000.0, "min_capital_usd": 10.0}

    short = decide_fit_forward_route(**kwargs, metrics={**base, "forward_days": 447.0})
    assert short.route == "needs-forward"
    assert "shape_aware_edge" in short.reason
    long = decide_fit_forward_route(**kwargs, metrics={**base, "forward_days": 4500.0})
    assert long.route == "reject"


def test_forward_only_economic_floor_miss_still_rejects() -> None:
    ruleset = _split_ruleset([NET_EDGE, NOISE_BAR])
    floor_miss = _rows(("net_edge", False), ("shape_aware_edge", False))
    route = decide_fit_forward_route(
        ruleset=ruleset, selection=None, forward=floor_miss, deployable_capital_usd=1000.0,
        params_fixed_at=date(2025, 7, 1),
        metrics={"sharpe_net": 0.2, "forward_days_needed": 9e4, "forward_days": 447.0},
    )
    assert route.route == "reject"


class PeekingStrategy:
    """Holds a coin tomorrow's price says will rise -- reads the future."""

    name = "peeking"

    def target_weights(self, panel, params):
        nxt = panel.prices.shift(-1) > panel.prices
        return (nxt.astype(float) * 0.4).where(panel.tradeable, 0.0)


def test_a_strategy_that_reads_the_future_is_an_error_not_a_verdict(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:PeekingStrategy")

    result = evaluate_spec(spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
                           deployable_capital_usd=1000.0)

    assert result.routing.route == "error"
    assert "look-ahead" in result.routing.reason
    assert session.query(Verdict).filter(Verdict.trial_id == result.trial_id).count() == 0


def test_a_spec_with_an_unsourced_number_does_not_start(session, tmp_path) -> None:
    from qlab.pipeline.sources import SpecSourceError

    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4}, sources={"costs": "owner 2026-01-01: test"})

    with pytest.raises(SpecSourceError, match="no source for weight"):
        evaluate_spec(spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
                      deployable_capital_usd=1000.0)
    assert session.query(Trial).count() == 0


def test_a_forward_test_that_missed_a_regime_waits() -> None:
    ruleset = RuleSet(version="2026-01-01.1", rules=[NET_EDGE],
                      forward_resolution=RESOLUTION, regime_coverage_days=30)
    passing = _rows(("net_edge", True))
    base = {"sharpe_net": 2.0, "forward_days_needed": 100.0, "forward_days": 400.0,
            "min_capital_usd": 10.0, "regime_bull_days": 120.0, "regime_flat_days": 200.0}
    kwargs = dict(ruleset=ruleset, selection=None, forward=passing,
                  deployable_capital_usd=1000.0, params_fixed_at=date(2025, 1, 1))

    missed = decide_fit_forward_route(**kwargs, metrics={**base, "regime_bear_days": 12.0})
    assert missed.route == "needs-forward" and "bear 12 days" in missed.reason
    seen = decide_fit_forward_route(**kwargs, metrics={**base, "regime_bear_days": 80.0})
    assert seen.route == "paper"
    unlabeled = {k: v for k, v in base.items() if not k.startswith("regime_")}
    assert decide_fit_forward_route(**kwargs, metrics=unlabeled).route == "paper"


def test_a_snapshot_built_under_other_panel_rules_is_not_reused(session, tmp_path) -> None:
    """A fix to the data layer must reach the next run of an unchanged
    request: a snapshot built under other panel rules is not an answer."""
    _register_snapshot(session, tmp_path, snapshot_id="snap-rules", instruments=INSTRUMENTS,
                       universe_complete=False)
    manifest_path = tmp_path / "snap-rules" / "manifest.json"
    found = _find_matching_snapshot(
        session, source=SOURCE, start=pd.Timestamp(START, tz="UTC"),
        end=pd.Timestamp(END, tz="UTC"), interval=INTERVAL, instruments=INSTRUMENTS,
        include_spot=False, min_daily_volume_usd=None)
    assert found == "snap-rules"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["panel_rules"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _find_matching_snapshot(
        session, source=SOURCE, start=pd.Timestamp(START, tz="UTC"),
        end=pd.Timestamp(END, tz="UTC"), interval=INTERVAL, instruments=INSTRUMENTS,
        include_spot=False, min_daily_volume_usd=None) is None


def test_a_fit_window_puts_history_after_it_in_the_forward_test() -> None:
    """Owner, 2026-10-04: «разделять backtest and forward test». Bv2's
    parameters were fitted on 2023-06 … 2025-05 and committed in 2026-09: the
    sixteen months after the fit are a forward test on history."""
    index = pd.date_range("2020-09-01", "2026-09-20", freq="1D", tz="UTC")
    split = split_at_fixed_date(index, 0, len(index) - 1, date(2025, 6, 1), date(2023, 6, 1))
    assert index[split.before[0]].date() == date(2020, 9, 1)
    assert index[split.before[1]].date() == date(2023, 5, 31)
    assert index[split.selection[0]].date() == date(2023, 6, 1)
    assert index[split.selection[1]].date() == date(2025, 5, 31)
    assert index[split.forward[0]].date() == date(2025, 6, 1)
    # Without a fit start, everything before the fixed date is selection.
    plain = split_at_fixed_date(index, 0, len(index) - 1, date(2025, 6, 1))
    assert plain.before is None and plain.selection[0] == 0


def test_a_fit_start_needs_an_end_after_it() -> None:
    base = dict(idea_id="x", title="x", code_ref="m:C", params={},
                data={"source": "binance", "interval": "1d", "start": "2020-01-01",
                      "end": "2026-01-01"},
                costs={"taker_fee_bps": 1.0, "slippage_bps": 0.0}, min_leg_notional=10.0,
                unexpressed_mechanisms=[])
    with pytest.raises(ValueError, match="needs params_fixed_at"):
        StrategySpec(**base, params_fit_from=date(2023, 6, 1))
    with pytest.raises(ValueError, match="before params_fixed_at"):
        StrategySpec(**base, params_fit_from=date(2025, 6, 1), params_fixed_at=date(2023, 6, 1),
                     params_fixed_evidence="doc")
    StrategySpec(**base, params_fit_from=date(2023, 6, 1), params_fixed_at=date(2025, 6, 1),
                 params_fixed_evidence="doc")


class CashAfter(ToyStrategy):
    """Holds the toy book until `params["cash_from"]`, then nothing."""

    name = "cash-after"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        weights = super().target_weights(panel, params)
        held = weights.index < pd.Timestamp(params["cash_from"], tz="UTC")
        return weights.mul(pd.Series(held, index=weights.index).astype(float), axis=0)


def test_a_forward_test_spent_in_cash_is_measured_not_an_error(session, tmp_path) -> None:
    """2026-10-06: trend's two levels sat in cash through their first forward
    days and every night read `error` -- an all-flat part has no smallest
    leg. The strategy's capital need comes from its whole run."""
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test",
        code_ref="qlab.pipeline.test_evaluate:CashAfter",
    )
    spec = spec.model_copy(update={"params": {"weight": 0.4, "cash_from": "2026-01-06"}})
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0,
                           check_lookahead=False, check_sources=False)

    assert result.error is None
    assert result.metrics["judged_on_forward"] == 1.0
    assert result.metrics["min_capital_usd"] == result.metrics["fit_min_capital_usd"]
