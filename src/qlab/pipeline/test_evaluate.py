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
import math
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
    NOISE_UNMATCHED_REASON,
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
    HoldoutShare,
    NoisePower,
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


class BasedToy(ToyStrategy):
    """A book that sets money aside (docs/CAPITAL_BASE.md): its weights are shares of a balance
    that is `params["base"]` times the capital, and it does not reinvest its profit."""

    name = "based-toy"
    reinvests_profit = False

    def own_base(self, panel: MarketPanel, params) -> pd.Series:
        from qlab.harness.strategy import CapitalBaseRefused

        if params.get("refuse"):
            raise CapitalBaseRefused("deposits")
        return pd.Series(float(params["base"]), index=panel.prices.index)


class OrderedToy(ToyStrategy):
    """A book whose weights hold a speck left behind by a cut (`params["weight"]`, tiny), and
    that says it opened its leg with an order of `params["order"]` of its money
    (docs/MIN_CAPITAL.md)."""

    name = "ordered-toy"

    def placed_orders(self, panel: MarketPanel, params):
        from qlab.harness.strategy import PlacedOrders

        sizes = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        sizes.iloc[0, 0] = float(params["order"])
        sizes.iloc[5, 1] = 2.0 * float(params["order"])
        skipped = pd.Series(0.0, index=panel.prices.index)
        skipped.iloc[3:6] = 1.0
        return PlacedOrders(sizes=sizes, skipped=skipped)


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
    source: str = SOURCE,
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
        "source": source,
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
        source=source,
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


def test_the_capital_a_book_needs_is_measured_from_its_orders_not_its_weights(
        session, tmp_path) -> None:
    """The weights hold a speck a cut left behind: read as a leg to buy it asks for $12M and
    shelves the run. The book never sent such an order (docs/MIN_CAPITAL.md)."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:OrderedToy",
                      params={"weight": 1e-6, "order": 0.3},
                      sources={"weight": "owner 2026-01-01: test fixture",
                               "order": "owner 2026-01-01: test fixture",
                               "costs": "owner 2026-01-01: test fixture",
                               "min_leg_notional": "owner 2026-01-01: test fixture"})
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["min_capital_usd"] == pytest.approx(12.0 / 0.3)
    assert result.metrics["min_capital_weights_usd"] == pytest.approx(12.0 / 1e-6)
    assert result.metrics["min_capital_from_orders"] == 1.0
    assert result.metrics["orders_opening"] == 2.0
    assert result.metrics["orders_skipped"] == 3.0
    assert result.routing.route == "paper"


def test_a_book_that_declares_no_orders_keeps_the_weights_number(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    result = evaluate_spec(_make_spec(params={"weight": 0.4}), session=session,
                           ruleset=_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE]),
                           deployable_capital_usd=1000.0)
    assert result.metrics["min_capital_usd"] == pytest.approx(30.0)
    assert not any(k in result.metrics for k in ("min_capital_weights_usd",
                                                 "min_capital_from_orders", "orders_opening",
                                                 "orders_skipped"))


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


def test_failed_venue_supported_names_the_venue_and_the_instruments(session, tmp_path) -> None:
    """A Binance-data spec fails `venue_supported` because `venues/binance.yaml` records no
    order adapter of ours -- not because of the strategy and not because of access. The
    verdict row and the route must say so: which venue, what is missing, which
    instruments the book held (the 2026-10-25 generator ideas read as a bare
    `venue_supported = 0`)."""
    _register_snapshot(
        session, tmp_path, snapshot_id="snap-binance", instruments=INSTRUMENTS,
        universe_complete=True, source="binance",
    )
    spec = _make_spec(
        params={"weight": 0.4},
        data={"source": "binance", "interval": INTERVAL, "start": START, "end": END,
              "instruments": None},
    )
    venue_rule = Rule(
        id="venue_supported", stage=Stage.PREFLIGHT, kind=RuleKind.INFRASTRUCTURE,
        metric="venue_supported", comparator=Comparator.EQ, threshold=1, fatal=True,
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, venue_rule])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["venue_supported"] == 0.0
    # Every strategy rule passed: an infrastructure fact never rejects.
    assert result.routing.route == "needs-infrastructure"
    assert "venue binance has no order-placing adapter" in result.routing.reason
    assert INSTRUMENTS[0] in result.routing.reason
    row = (session.query(Verdict)
           .filter(Verdict.trial_id == result.trial_id, Verdict.rule_id == "venue_supported")
           .one())
    assert row.passed is False
    assert "venues/binance.yaml execution_adapter: false" in row.note
    assert "not about access" in row.note


_ATOMIC_RULE = Rule(
    id="atomic_execution", stage=Stage.PREFLIGHT, kind=RuleKind.INFRASTRUCTURE,
    metric="atomic_execution", comparator=Comparator.EQ, threshold=1, fatal=True,
)
_VENUE_RULE = Rule(
    id="venue_supported", stage=Stage.PREFLIGHT, kind=RuleKind.INFRASTRUCTURE,
    metric="venue_supported", comparator=Comparator.EQ, threshold=1, fatal=True,
)


def _binance_spec(session, tmp_path, **overrides: object) -> StrategySpec:
    _register_snapshot(
        session, tmp_path, snapshot_id="snap-binance", instruments=INSTRUMENTS,
        universe_complete=True, source="binance",
    )
    return _make_spec(
        params={"weight": 0.4},
        data={"source": "binance", "interval": INTERVAL, "start": START, "end": END,
              "instruments": None},
        **overrides,
    )


def test_two_leg_binance_spec_without_an_execution_venue_stays_undecided(session, tmp_path) -> None:
    """Unchanged behaviour: Binance names no recovery mechanism, so `atomic_execution` of a
    two-leg spec is unknown and the run routes `needs-more-data`."""
    spec = _binance_spec(session, tmp_path, simultaneous_legs=2)
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, _VENUE_RULE, _ATOMIC_RULE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert "atomic_execution" not in result.metrics
    assert result.metrics["venue_supported"] == 0.0
    assert result.routing.route == "needs-more-data"
    assert "read from" not in result.routing.reason


def test_two_leg_binance_proxy_reads_execution_facts_from_hyperliquid(session, tmp_path) -> None:
    """Data from Binance, orders to Hyperliquid: Hyperliquid names `uncovered_leg_recovery`
    and has an adapter, so both facts are known and true, the forward data is still
    Binance's, and the route says where the execution facts came from."""
    spec = _binance_spec(session, tmp_path, simultaneous_legs=2, execution_venue="hyperliquid")
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, _VENUE_RULE, _ATOMIC_RULE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.error is None
    assert result.metrics["atomic_execution"] == 1.0
    assert result.metrics["venue_supported"] == 1.0
    assert result.metrics["data_forward_available"] == 1.0
    assert result.routing.route == "paper"
    assert "read from hyperliquid, the spec's execution_venue; data from binance" in (
        result.routing.reason)


def test_single_leg_binance_spec_is_unchanged_without_an_execution_venue(session, tmp_path) -> None:
    spec = _binance_spec(session, tmp_path)
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, _VENUE_RULE, _ATOMIC_RULE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.metrics["atomic_execution"] == 1.0
    assert result.metrics["venue_supported"] == 0.0
    assert result.routing.route == "needs-infrastructure"
    assert "venue binance has no order-placing adapter" in result.routing.reason


def test_an_execution_venue_equal_to_the_data_source_changes_nothing(session, tmp_path) -> None:
    """Stage 2 copies a stage 1 spec onto Hyperliquid and keeps the field."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4}, simultaneous_legs=2, execution_venue=SOURCE)
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, _VENUE_RULE, _ATOMIC_RULE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.routing.route == "paper"
    assert "execution_venue" not in result.routing.reason


def test_an_unconfigured_execution_venue_is_named_in_the_route(session, tmp_path) -> None:
    """A proxy whose execution venue has no config leaves `venue_supported` unknown, and the
    route says which venue the execution facts were looked up on, not the data source's."""
    spec = _binance_spec(session, tmp_path, execution_venue="no-such-venue")
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, _VENUE_RULE])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert "venue_supported" not in result.metrics
    assert result.routing.route == "needs-more-data"
    assert "read from no-such-venue, the spec's execution_venue; data from binance" in (
        result.routing.reason)


def test_passed_venue_supported_has_no_venue_explanation(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    venue_rule = Rule(
        id="venue_supported", stage=Stage.PREFLIGHT, kind=RuleKind.INFRASTRUCTURE,
        metric="venue_supported", comparator=Comparator.EQ, threshold=1, fatal=True,
    )
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE, venue_rule])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.routing.route == "paper"
    assert "adapter" not in result.routing.reason
    row = (session.query(Verdict)
           .filter(Verdict.trial_id == result.trial_id, Verdict.rule_id == "venue_supported")
           .one())
    assert row.note is None


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


LIST_SOURCES = {"weight": "owner 2026-01-01: test fixture",
                "costs": "owner 2026-01-01: test fixture",
                "min_leg_notional": "owner 2026-01-01: test fixture",
                "instruments_chosen_at": "owner 2026-01-01: test fixture"}


def test_a_list_chosen_after_the_judged_part_began_is_not_point_in_time(session, tmp_path) -> None:
    """A causal twin's list is chosen today and applied to older history
    (`instruments_chosen_at`): it knows how those days ended."""
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test",
        instruments_chosen_at=date(2026, 3, 1), sources=LIST_SOURCES)
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.metrics["judged_on_forward"] == 1.0
    assert result.metrics["point_in_time_universe"] == 0.0  # the list is younger than the days


def test_a_list_chosen_when_the_judged_part_began_is_point_in_time(session, tmp_path) -> None:
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test",
        instruments_chosen_at=date(2026, 1, 6), sources=LIST_SOURCES)
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.metrics["point_in_time_universe"] == 1.0  # fixed when the forward test began


def test_a_causal_twin_with_a_late_list_is_judged_only_from_the_day_the_list_was_known(
        session, tmp_path) -> None:
    """The twin's first causal choice is early (`params_fixed_at`), its hand-picked list is
    younger (`instruments_chosen_at`); `judged_from` = the list's day cuts the judged part, so
    the universe is point-in-time there, and the days before are warm-up, not selection."""
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])
    twin = dict(params_fixed_at=date(2026, 1, 2), params_fixed_evidence="first causal choice",
                selects_causally=True, instruments_chosen_at=date(2026, 1, 6),
                sources=LIST_SOURCES)

    spec = _explicit_long(session, tmp_path, **twin)
    unjudged = evaluate_spec(spec, session=session, ruleset=ruleset,
                             deployable_capital_usd=1000.0)
    assert unjudged.metrics["point_in_time_universe"] == 0.0  # judged from the first choice

    cut = evaluate_spec(spec.model_copy(update={"judged_from": date(2026, 1, 6)}),
                        session=session, ruleset=ruleset, deployable_capital_usd=1000.0)
    assert cut.metrics["point_in_time_universe"] == 1.0
    assert cut.metrics["judged_on_forward"] == 1.0
    assert cut.metrics["selection_days"] == 0.0  # a causal chooser has no selection period
    assert cut.metrics["forward_days"] == pytest.approx(5.0 + 1 / 24)
    assert cut.metrics["skipped_warmup_days"] == pytest.approx(5.0)
    rows = session.query(Verdict).filter(Verdict.trial_id == cut.trial_id).all()
    assert {v.data_range_start for v in rows if v.note == FORWARD_NOTE} == {date(2026, 1, 6)}


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


def test_the_year_view_describes_the_judged_part_alone(session, tmp_path) -> None:
    spec = _explicit_long(
        session, tmp_path, params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test"
    )
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    m = result.metrics
    assert m["judged_on_forward"] == 1.0
    # The forward test lies inside 2026: one partial year, listed, and too short to be counted.
    assert m["year_2026_days"] == 5 and "year_2026_return" in m and "year_2026_max_dd" in m
    assert "years_counted" not in m
    # The selection period's copy of the metrics carries no year view.
    assert not any(k.startswith("fit_year_") or k.startswith("fit_years_") for k in m)
    # And the view changed no verdict: it is not a rule.
    assert result.routing.route == "needs-forward"


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


def test_spot_priced_at_the_perp_and_at_its_own_market_are_different_requests(
        session, tmp_path) -> None:
    """A snapshot is reused only for the spot-price convention it was built
    under; one with no such key (built before the option) was priced at the perp."""
    _register_snapshot(session, tmp_path, snapshot_id="snap-spot", instruments=INSTRUMENTS,
                       universe_complete=False)
    manifest_path = tmp_path / "snap-spot" / "manifest.json"

    def find(spot_prices: str) -> str | None:
        return _find_matching_snapshot(
            session, source=SOURCE, start=pd.Timestamp(START, tz="UTC"),
            end=pd.Timestamp(END, tz="UTC"), interval=INTERVAL, instruments=INSTRUMENTS,
            include_spot=False, min_daily_volume_usd=None, spot_prices=spot_prices)

    assert find("perp") == "snap-spot" and find("own") is None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spot_prices"] = "own"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert find("own") == "snap-spot" and find("perp") is None


def test_the_default_spot_price_is_not_written_into_the_spec_row(session) -> None:
    """Spec rows recorded before the option existed must still match, or every
    idea would grow a new version on its next run."""
    from qlab.pipeline.evaluate import _get_or_create_spec_row

    perp = _make_spec()
    assert perp.data.spot_prices == "perp"
    row = _get_or_create_spec_row(session, perp)
    assert "spot_prices" not in row.data_requirements
    assert _get_or_create_spec_row(session, perp).id == row.id

    own = _make_spec(data={**perp.data.model_dump(), "include_spot": True, "spot_prices": "own"})
    own_row = _get_or_create_spec_row(session, own)
    assert own_row.id != row.id and own_row.data_requirements["spot_prices"] == "own"


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


class UnmatchedNoise:
    """A noise book discarded for its shape."""

    name = "unmatched-noise"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        from qlab.calibration.noise import StructuralMismatchError

        raise StructuralMismatchError("gross ratio 0.053 is outside the structural-match band")


def test_a_noise_book_of_the_wrong_shape_is_not_evaluable_not_an_error(session, tmp_path) -> None:
    """Owner, 2026-10-09: «в прогонах вижу тучу ошибок». A random book that
    does not have its reference's shape is discarded by design; that is not
    a crash."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:UnmatchedNoise")

    result = evaluate_spec(spec, session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS]),
                           deployable_capital_usd=1000.0)

    assert result.error is None and result.metrics is None
    assert result.routing.route == "not-evaluable"
    assert result.routing.reason.startswith(NOISE_UNMATCHED_REASON)
    assert session.get(Trial, result.trial_id).status.value == "not-evaluable"


def test_failed_informational_rules_are_named_not_called_nothing(session, tmp_path) -> None:
    """Owner, 2026-10-09, on xsmom-frozen-universe: a page said the selection
    period "shows nothing against it" beside two failed rules, and promised
    a forward test that could never start."""
    spec = _explicit_long(session, tmp_path)  # no params_fixed_at
    ruleset = _split_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE_INFO])

    result = evaluate_spec(spec, session=session, ruleset=ruleset, deployable_capital_usd=1000.0)

    assert result.routing.route == "needs-forward"
    assert "nothing against" not in result.routing.reason
    assert "FAILED honest_universe" in result.routing.reason
    assert "until params_fixed_at is declared" in result.routing.reason
    assert result.metrics["fix_date_known"] == 0.0
    assert result.metrics["selection_rules_failed"] == 1.0


# --------------------------------------------------------------------------
# A third of the data as the forward test (rules 2026-10-24.1)
# --------------------------------------------------------------------------

HOLDOUT = HoldoutShare(share=1 / 3, rationale="test")


def _holdout_ruleset(rules: list[Rule]) -> RuleSet:
    return RuleSet(version="2026-01-01.1", rules=rules, holdout_share=HOLDOUT)


def test_a_forward_test_a_third_of_the_data_decides() -> None:
    """Owner, 2026-10-24: «поделить 2/3 и 1/3 ... оставшуюся 1/3 для прогона
    вперед» -- 731 days of fitting need 366 forward, not the years a Sharpe
    takes to resolve."""
    ruleset = _holdout_ruleset([NET_EDGE])
    passing = _rows(("net_edge", True))
    failing = _rows(("net_edge", False))
    base = {"fit_sharpe_net": 0.5, "fit_window_days": 731.0,
            "forward_days_needed": HOLDOUT.days_needed(731.0), "min_capital_usd": 10.0}
    kwargs = dict(ruleset=ruleset, selection=passing, deployable_capital_usd=1000.0,
                  params_fixed_at=date(2025, 6, 1))

    early = decide_fit_forward_route(**kwargs, forward=passing,
                                     metrics={**base, "forward_days": 169.0})
    assert early.route == "needs-forward"
    assert "366 days, 33% of fitting plus forward (731 days of fitting)" in early.reason
    assert decide_fit_forward_route(**kwargs, forward=passing,
                                    metrics={**base, "forward_days": 366.0}).route == "paper"
    assert decide_fit_forward_route(**kwargs, forward=failing,
                                    metrics={**base, "forward_days": 366.0}).route == "reject"


def test_the_holdout_is_measured_against_the_declared_fitting_window(session, tmp_path) -> None:
    """A declared fitting window counts in full even where the run's data does
    not cover it; without one, the run's own data before the fix date."""
    ruleset = _holdout_ruleset([CAPITAL_FIT_GENEROUS])
    kwargs = dict(session=session, ruleset=ruleset, deployable_capital_usd=1000.0,
                  check_lookahead=False, check_sources=False)

    declared = _explicit_long(session, tmp_path, params_fit_from=date(2025, 12, 1),
                              params_fixed_at=date(2026, 1, 6), params_fixed_evidence="test")
    result = evaluate_spec(declared, **kwargs)
    assert result.metrics["fit_window_days"] == 36.0
    assert result.metrics["forward_days_needed"] == pytest.approx(18.0)

    own = declared.model_copy(update={"params_fit_from": None})
    result = evaluate_spec(own, **kwargs)
    assert result.metrics["fit_window_days"] == pytest.approx(result.metrics["selection_days"])
    assert result.metrics["forward_days_needed"] == math.ceil(
        result.metrics["selection_days"] / 2)


def test_a_ruleset_has_one_forward_length_rule() -> None:
    with pytest.raises(ValueError, match="not both"):
        RuleSet(version="2026-01-01.1", forward_resolution=RESOLUTION, holdout_share=HOLDOUT)


# --------------------------------------------------------------------------
# The noise bar decides only where it can tell (rules 2026-10-24.2)
# --------------------------------------------------------------------------

POWER = NoisePower(good_return=0.07, min_power=0.8, rationale="test")
NOISE_BAR = Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="noise_return_percentile",
                 comparator=Comparator.GE, threshold=0.99, fatal=True,
                 fit_period=FitPeriodUse.INFORMATIONAL, on_short_forward=ShortForwardUse.WAIT)


def test_a_noise_bar_that_cannot_tell_does_not_reject() -> None:
    """Owner, 2026-10-24: «шарп отличный... а стенд берет и режет» -- on
    strategy-b-v2's 170 days a book truly earning 7% a year passed the bar 1%
    of the time; its failure there is too early, not a rejection."""
    ruleset = RuleSet(version="2026-01-01.1", rules=[NET_EDGE, NOISE_BAR],
                      holdout_share=HOLDOUT, noise_power=POWER)
    selection = _rows(("net_edge", True))
    failing_bar = _rows(("net_edge", True), ("shape_aware_edge", False))
    base = {"fit_sharpe_net": 1.0, "fit_window_days": 100.0, "forward_days_needed": 50.0,
            "forward_days": 400.0, "min_capital_usd": 10.0}
    kwargs = dict(ruleset=ruleset, selection=selection, forward=failing_bar,
                  deployable_capital_usd=1000.0, params_fixed_at=date(2025, 6, 1))

    weak = decide_fit_forward_route(**kwargs, metrics={**base, "noise_power": 0.01})
    assert weak.route == "needs-forward"
    assert "only 1% of the time" in weak.reason
    assert decide_fit_forward_route(**kwargs, metrics={**base, "noise_power": 0.9}).route \
        == "reject"
    # Unmeasured power cannot show the bar can tell.
    assert decide_fit_forward_route(**kwargs, metrics=base).route == "needs-forward"
    # A failure of a rule that does not wait still rejects.
    lost = _rows(("net_edge", False), ("shape_aware_edge", False))
    assert decide_fit_forward_route(**{**kwargs, "forward": lost},
                                    metrics={**base, "noise_power": 0.01}).route == "reject"


def test_with_no_selection_period_a_powerless_bar_waits_too() -> None:
    ruleset = RuleSet(version="2026-01-01.1", rules=[NET_EDGE, NOISE_BAR],
                      holdout_share=HOLDOUT, noise_power=POWER)
    failing_bar = _rows(("net_edge", True), ("shape_aware_edge", False))
    routing = decide_fit_forward_route(
        ruleset=ruleset, selection=None, forward=failing_bar, deployable_capital_usd=1000.0,
        params_fixed_at=date(2025, 6, 1),
        metrics={"forward_days": 400.0, "noise_power": 0.03, "min_capital_usd": 10.0})
    assert routing.route == "needs-forward"
    assert routing.reason.startswith("too early to tell: 400 days fail only shape_aware_edge")


def test_spot_legs_from_another_venue_are_a_different_request(session, tmp_path) -> None:
    """A snapshot is reused only for the spot venue it was built with; one with
    no such key (every snapshot built before the option) used the source's own."""
    _register_snapshot(session, tmp_path, snapshot_id="snap-venue", instruments=INSTRUMENTS,
                       universe_complete=False)
    manifest_path = tmp_path / "snap-venue" / "manifest.json"

    def find(spot_source: str | None) -> str | None:
        return _find_matching_snapshot(
            session, source=SOURCE, start=pd.Timestamp(START, tz="UTC"),
            end=pd.Timestamp(END, tz="UTC"), interval=INTERVAL, instruments=INSTRUMENTS,
            include_spot=False, min_daily_volume_usd=None, spot_source=spot_source)

    assert find(None) == "snap-venue" and find("binance") is None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spot_source"] = "binance"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert find("binance") == "snap-venue" and find(None) is None


def test_no_spot_source_is_not_written_into_the_spec_row(session) -> None:
    """Spec rows recorded before the option existed must still match."""
    from qlab.pipeline.evaluate import _get_or_create_spec_row

    plain = _make_spec()
    row = _get_or_create_spec_row(session, plain)
    assert "spot_source" not in row.data_requirements
    assert _get_or_create_spec_row(session, plain).id == row.id

    mixed = _make_spec(data={**plain.data.model_dump(), "include_spot": True,
                             "spot_prices": "own", "spot_source": "binance"})
    mixed_row = _get_or_create_spec_row(session, mixed)
    assert mixed_row.id != row.id and mixed_row.data_requirements["spot_source"] == "binance"


class ToyWithFlatFees(ToyStrategy):
    """The toy book that also pays a flat fee, as a share of equity, on bar 3."""

    name = "toy-flat-fees"

    def extra_costs(self, panel: MarketPanel, params) -> pd.Series:
        fees = pd.Series(0.0, index=panel.prices.index)
        fees.iloc[3] = float(params["flat_fee"])
        return fees


def test_a_strategys_own_costs_reach_the_net_return(session, tmp_path) -> None:
    """Flat dollar fees (moving money between venues) are not a rate on turnover;
    a strategy that has them says so with `extra_costs` and the run pays them."""
    _register_discovered(session, tmp_path)
    rules = _ruleset([CAPITAL_FIT_GENEROUS])

    def run(**params):
        spec = _make_spec(code_ref="qlab.pipeline.test_evaluate:ToyWithFlatFees",
                          params={"flat_fee": 0.0, **params},
                          sources={"flat_fee": "owner 2026-01-01: test fixture",
                                   "costs": "owner 2026-01-01: test fixture",
                                   "min_leg_notional": "owner 2026-01-01: test fixture"})
        return evaluate_spec(spec, session=session, ruleset=rules, deployable_capital_usd=1000.0,
                             check_lookahead=False)

    free = run()
    paying = run(flat_fee=0.01)
    assert paying.error is None
    n_periods = free.metrics["n_periods"]
    assert paying.metrics["ann_return_net"] < free.metrics["ann_return_net"]
    # One bar paid 1% of equity: the total return fell by about that much.
    total_free = (1 + free.metrics["ann_return_net"]) ** (n_periods / 8760.0) - 1
    total_paying = (1 + paying.metrics["ann_return_net"]) ** (n_periods / 8760.0) - 1
    assert total_free - total_paying == pytest.approx(0.01, rel=0.1)


def test_judged_from_starts_the_judged_window_after_the_warm_up(session, tmp_path) -> None:
    """Bars before `judged_from` are warm-up: the verdict's data range and the metrics
    start at it (docs/TWO_STAGE.md), and the run says how much it skipped."""
    end = date(2026, 1, 5)
    _register_snapshot(session, tmp_path, snapshot_id=DISCOVERED_ID, instruments=INSTRUMENTS,
                       universe_complete=True, end=end)
    rules = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])
    data = {"source": SOURCE, "interval": INTERVAL, "start": START, "end": end}

    whole = evaluate_spec(_make_spec(params={"weight": 0.4}, data=data), session=session,
                          ruleset=rules, deployable_capital_usd=1000.0, check_lookahead=False)
    judged = evaluate_spec(_make_spec(params={"weight": 0.4}, data=data,
                                      judged_from=date(2026, 1, 3)),
                           session=session, ruleset=rules, deployable_capital_usd=1000.0,
                           check_lookahead=False)

    assert "skipped_warmup_days" not in whole.metrics
    assert judged.metrics["skipped_warmup_days"] == pytest.approx(2.0)
    assert judged.metrics["n_periods"] < whole.metrics["n_periods"]
    rows = session.query(Verdict).filter(Verdict.trial_id == judged.trial_id).all()
    assert {v.data_range_start for v in rows} == {date(2026, 1, 3)}
    assert {v.data_range_end for v in rows} == {end}


def test_judged_from_never_moves_the_forward_boundary(session, tmp_path) -> None:
    """`judged_from` only moves the start of the window; the split at `params_fixed_at`
    stays where the spec put it (the parameters were fixed before the market existed)."""
    end = date(2026, 1, 5)
    _register_snapshot(session, tmp_path, snapshot_id=DISCOVERED_ID, instruments=INSTRUMENTS,
                       universe_complete=True, end=end)
    ruleset = RuleSet(version="2026-01-01.1", rules=[CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE],
                      forward_resolution=RESOLUTION)
    spec = _make_spec(
        params={"weight": 0.4}, data={"source": SOURCE, "interval": INTERVAL, "start": START,
                                      "end": end},
        judged_from=date(2026, 1, 3), params_fixed_at=date(2025, 6, 1),
        params_fixed_evidence="owner 2026-01-01: test fixture")
    result = evaluate_spec(spec, session=session, ruleset=ruleset,
                           deployable_capital_usd=1000.0, check_lookahead=False)
    assert result.metrics["judged_on_forward"] == 1.0
    assert result.metrics["selection_days"] == 0.0
    assert result.metrics["forward_days"] == pytest.approx(2.0, abs=0.1)  # 3 Jan .. 5 Jan 00:00


def test_extra_not_evaluable_reasons_write_a_not_evaluable_trial_and_run_nothing(
        session, tmp_path) -> None:
    """A reason the caller established from the data (too few instruments for the
    portfolio rule) is a not-evaluable trial: written, never a fail, no verdict rows."""
    _register_discovered(session, tmp_path)
    spec = _make_spec(params={"weight": 0.4})
    result = evaluate_spec(
        spec, session=session, ruleset=_ruleset([CAPITAL_FIT_TIGHT]),
        deployable_capital_usd=1000.0, not_evaluable_extra=["the market had 3 coins, 5 needed"])

    assert result.routing.route == "not-evaluable"
    assert "the market had 3 coins, 5 needed" in result.routing.reason
    assert result.metrics is None
    trial = session.get(Trial, result.trial_id)
    assert trial.status == TrialStatus.NOT_EVALUABLE
    assert session.query(Verdict).filter(Verdict.trial_id == result.trial_id).count() == 0
    assert session.get(Idea, IDEA_ID).status == IdeaStatus.CANDIDATE  # never "rejected"


def test_a_list_that_names_spot_columns_finds_the_snapshot_built_for_it(
        session, tmp_path) -> None:
    """specs/bv2-binance.yaml lists BTC-SPOT itself and leaves include_spot off.
    The snapshot built for that request carries spot columns; it must be found,
    or every run (each noise book too) refetches and rebuilds it (2026-10-26)."""
    listed = ["BTC", "BTC-SPOT", "ETH", "ETH-SPOT"]
    _register_snapshot(session, tmp_path, snapshot_id="snap-listed", instruments=listed,
                       universe_complete=False)

    def find(instruments: list[str], include_spot: bool) -> str | None:
        return _find_matching_snapshot(
            session, source=SOURCE, start=pd.Timestamp(START, tz="UTC"),
            end=pd.Timestamp(END, tz="UTC"), interval=INTERVAL, instruments=instruments,
            include_spot=include_spot, min_daily_volume_usd=None)

    assert find(listed, include_spot=False) == "snap-listed"
    assert find(listed, include_spot=True) == "snap-listed"
    assert find(["BTC", "ETH"], include_spot=False) is None  # a perp-only request


_BASED_SOURCES = {key: "owner 2026-01-01: test fixture"
                  for key in ("weight", "costs", "min_leg_notional", "base", "refuse")}


def test_the_run_is_also_measured_on_the_strategys_own_money(session, tmp_path) -> None:
    _register_discovered(session, tmp_path)
    ruleset = _ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE])

    plain = evaluate_spec(_make_spec(params={"weight": 0.4}), session=session, ruleset=ruleset,
                          deployable_capital_usd=1000.0)
    m = plain.metrics
    # no base and a reinvesting book: the capital figures ARE the stand's
    assert m["cap_reinvested"] == 1.0
    assert m["cap_ann_return"] == pytest.approx(m["ann_return_net"])
    assert m["cap_total_return"] == pytest.approx(m["total_return_net"])
    assert m["cap_max_dd"] == pytest.approx(m["max_dd"])
    assert m["cap_sharpe"] == pytest.approx(m["sharpe_net"])

    based = evaluate_spec(
        _make_spec(code_ref="qlab.pipeline.test_evaluate:BasedToy",
                   params={"weight": 0.4, "base": 3.0}, sources=_BASED_SOURCES),
        session=session, ruleset=ruleset, deployable_capital_usd=1000.0)
    b = based.metrics
    assert b["ann_return_net"] == pytest.approx(m["ann_return_net"]), "the stand is unchanged"
    assert b["cap_reinvested"] == 0.0
    n, ppy = b["n_periods"], 365.25 * 24
    # summed, three times the shares: the total is about three times the stand's
    assert b["cap_total_return"] == pytest.approx(3.0 * b["total_return_net"], rel=0.05)
    assert b["cap_ann_return"] == pytest.approx(b["cap_total_return"] * ppy / n)
    trial = session.get(Trial, based.trial_id)
    assert trial.metrics["cap_ann_return"] == pytest.approx(b["cap_ann_return"])


def test_a_refused_base_leaves_the_capital_figures_out(session, tmp_path) -> None:
    """The stand's figures must not be passed off as the strategy's in their place."""
    _register_discovered(session, tmp_path)
    result = evaluate_spec(
        _make_spec(code_ref="qlab.pipeline.test_evaluate:BasedToy",
                   params={"weight": 0.4, "base": 3.0, "refuse": True},
                   sources=_BASED_SOURCES),
        session=session, ruleset=_ruleset([CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE]),
        deployable_capital_usd=1000.0)
    assert result.error is None
    assert result.metrics["cap_base_refused"] == 1.0
    assert not any(k.startswith("cap_") and k != "cap_base_refused" for k in result.metrics)


def _money_edge_rules():
    return [CAPITAL_FIT_GENEROUS, HONEST_UNIVERSE,
            Rule(id="net_edge_positive", stage=Stage.EDGE, metric="cap_ann_return",
                 comparator=Comparator.GT, threshold=0.0, fatal=True)]


def test_a_rule_on_the_strategys_money_reads_the_money_figure(session, tmp_path) -> None:
    """Rules 2026-10-29.2: `net_edge_positive` reads `cap_ann_return`, not `ann_return_net`."""
    _register_discovered(session, tmp_path)
    result = evaluate_spec(
        _make_spec(code_ref="qlab.pipeline.test_evaluate:BasedToy",
                   params={"weight": 0.4, "base": 3.0}, sources=_BASED_SOURCES),
        session=session, ruleset=_ruleset(_money_edge_rules()), deployable_capital_usd=1000.0)
    row = next(r for r in result.rules_result.rows if r.rule_id == "net_edge_positive")
    assert row.metric == "cap_ann_return"
    assert row.value == pytest.approx(result.metrics["cap_ann_return"])
    assert row.value != pytest.approx(result.metrics["ann_return_net"])  # not the stand's figure


def test_a_book_with_no_base_of_percent_is_not_judged_on_the_stand_instead(
        session, tmp_path) -> None:
    """The honest answer for a book with deposits (rules 2026-10-29.2): the money figure is
    missing, the rule is unknown and the run routes needs-more-data with the reason named --
    the stand's figure, which exists, is not put in its place."""
    _register_discovered(session, tmp_path)
    result = evaluate_spec(
        _make_spec(code_ref="qlab.pipeline.test_evaluate:BasedToy",
                   params={"weight": 0.4, "base": 3.0, "refuse": True}, sources=_BASED_SOURCES),
        session=session, ruleset=_ruleset(_money_edge_rules()), deployable_capital_usd=1000.0)
    assert result.error is None and "ann_return_net" in result.metrics
    assert result.routing.route == "needs-more-data"
    assert "cap_ann_return" in result.routing.reason
    assert "base of percent is undefined" in result.routing.reason
    assert result.rules_result.unknown_metrics == ("cap_ann_return",)
