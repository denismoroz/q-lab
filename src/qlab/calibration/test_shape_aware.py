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
from qlab.data.snapshot import PANEL_RULES_VERSION
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
        "panel_rules": PANEL_RULES_VERSION,
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
            "unexpressed_mechanisms": [],
            "sources": {"seed": "owner 2026-01-01: test fixture",
                        "costs": "owner 2026-01-01: test fixture",
                        "min_leg_notional": "owner 2026-01-01: test fixture"},
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
    # The same rank on the strategy's own money (docs/CAPITAL_BASE.md), from noise books that
    # carry the capital figures too: a strategy without a base reads the same either way.
    assert all("cap_ann_return" in t.evaluation.metrics for t in result.noise_trials)
    assert result.candidate.metrics["noise_cap_return_percentile"] == pytest.approx(
        result.candidate.metrics["noise_return_percentile"])

    # exactly n_trials + 1 real trial rows: the 8 noise trials plus the
    # candidate itself (CLAUDE.md: "trial пишется всегда").
    from qlab.registry.models import Trial

    assert session.query(Trial).count() == 9


def test_rules_that_read_the_strategys_money_get_the_rank_and_the_power_on_it(session):
    """Rules 2026-10-29.2: the noise rank and the power of the bar are those of the yearly return
    on the strategy's own money, stored as `noise_cap_return_percentile` and `noise_cap_power`
    (the stand's `noise_power` is kept beside them for the older rules)."""
    from qlab.calibration.shape_aware import power_targets
    from qlab.rules.schema import NoisePower

    ruleset = _shape_aware_ruleset(threshold=0.0)
    cap_rules = [r.model_copy(update={"metric": {"ann_return_net": "cap_ann_return",
                                                 "noise_return_percentile":
                                                 "noise_cap_return_percentile"}[r.metric]})
                 for r in ruleset.rules]
    setting = NoisePower(good_return=0.07, min_power=0.8, return_metric="cap_ann_return",
                         percentile_metric="noise_cap_return_percentile",
                         power_metric="noise_cap_power", rationale="t")
    ruleset = ruleset.model_copy(update={"rules": cap_rules, "noise_power": setting})
    assert power_targets(ruleset) == [("cap_ann_return", "noise_cap_power"),
                                      ("ann_return_net", "noise_power")]
    old = ruleset.model_copy(update={"noise_power": NoisePower(
        good_return=0.07, min_power=0.8, rationale="t")})
    assert power_targets(old) == [("ann_return_net", "noise_power")]

    result = evaluate_spec_with_shape_aware_bar(
        _spec(), session=session, ruleset=ruleset, deployable_capital_usd=1_000_000.0,
        n_trials=8, seed_offset=0)

    m = result.candidate.metrics
    assert result.candidate.error is None
    assert "noise_cap_power" in m and "noise_power" in m
    assert 0.0 <= m["noise_cap_power"] <= 1.0
    rows = {r.rule_id: r for r in result.candidate.rules_result.rows}
    assert rows["shape_aware_edge"].metric == "noise_cap_return_percentile"
    assert rows["shape_aware_edge"].value == pytest.approx(m["noise_cap_return_percentile"])


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


def test_not_evaluable_spec_writes_one_trial_and_runs_no_noise(session):
    """200 noise backtests against a strategy that cannot be tested are pure
    waste: the wrapper must stop before them (docs/TASKS.md T24)."""
    from qlab.registry.models import Trial, TrialStatus

    spec = _spec().model_copy(update={"unexpressed_mechanisms": ["a missing state machine"]})

    result = evaluate_spec_with_shape_aware_bar(
        spec,
        session=session,
        ruleset=_shape_aware_ruleset(),
        deployable_capital_usd=1_000_000.0,
        n_trials=8,
    )

    assert result.candidate.routing.route == "not-evaluable"
    assert result.noise_trials == ()
    assert result.percentiles is None
    trials = session.query(Trial).all()
    assert len(trials) == 1
    assert trials[0].status == TrialStatus.NOT_EVALUABLE


def test_a_book_without_a_number_is_replaced_and_the_counts_are_recorded(session, monkeypatch):
    """The bar is measured on the 8 books asked for, not on 7: the book that gave no number is
    replaced and the trial says how many were asked for, usable, unusable and replaced."""
    import dataclasses

    import qlab.calibration.run as run_module

    real = run_module._evaluate_noise

    def wrapped(spec, session, ruleset, deployable_capital_usd):
        evaluation = real(spec, session, ruleset, deployable_capital_usd)
        if int(spec.params["seed"]) == 2:
            return dataclasses.replace(evaluation, metrics=None, error="StructuralMismatchError")
        return evaluation

    monkeypatch.setattr(run_module, "_evaluate_noise", wrapped)

    result = evaluate_spec_with_shape_aware_bar(
        _spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
        deployable_capital_usd=1_000_000.0, n_trials=8)

    m = result.candidate.metrics
    assert (m["noise_n_requested"], m["noise_n_usable"], m["noise_n_unusable"],
            m["noise_n_replaced"]) == (8, 8, 1, 1)
    assert result.percentiles.n_requested == 8
    assert result.percentiles.n_usable_return == 8
    assert result.percentiles.n_replaced == 1
    # the percentile is a multiple of 1/8: the denominator is the books asked for
    scaled = m["noise_return_percentile"] * 8
    assert scaled == pytest.approx(round(scaled))


def test_a_clean_run_records_the_counts_with_nothing_replaced(session):
    result = evaluate_spec_with_shape_aware_bar(
        _spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
        deployable_capital_usd=1_000_000.0, n_trials=8)
    m = result.candidate.metrics
    assert (m["noise_n_requested"], m["noise_n_usable"], m["noise_n_unusable"],
            m["noise_n_replaced"]) == (8, 8, 0, 0)


def test_a_run_of_the_bv2_family_records_the_protection_measurement(session, monkeypatch):
    """docs/PROTECTION.md: the judged part of a Bv2-family run carries the `prot_*` metrics,
    read by no rule; any other strategy's run carries none."""
    from qlab.pipeline.evaluate import evaluate_spec

    plain = evaluate_spec(_spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
                          deployable_capital_usd=1_000_000.0)
    assert not any(k.startswith("prot_") for k in plain.metrics)

    monkeypatch.setattr("qlab.protection.BV2_CODE_REFS", frozenset(
        {"qlab.calibration.test_shape_aware:ToyMomentumStrategy"}))
    family = evaluate_spec(_spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
                           deployable_capital_usd=1_000_000.0)
    m = family.metrics
    assert m["prot_weeks"] >= 8 and m["prot_days"] > 56
    # the strategy's own return over the judged part is the run's own total (no base, compounded)
    assert m["prot_total_return"] == pytest.approx(m["total_return_net"], abs=1e-9)
    assert m["prot_reinvested"] == 1.0
    assert "prot_long_worst_fall" in m and "prot_worst_fall" in m


def test_the_candidates_row_is_committed_before_the_noise_books_start(session, monkeypatch):
    """The books run in worker processes that write the registry; a write lock this session
    held from the candidate's own spec row would stall them until it timed out."""
    import qlab.calibration.shape_aware as module
    from qlab.registry.models import Spec as SpecRow

    seen: dict = {}
    real = module.start_noise_series

    def watching(**kw):
        seen["pending"] = bool(kw["session"].new or kw["session"].dirty)
        seen["rows"] = kw["session"].query(SpecRow).filter(
            SpecRow.idea_id == IDEA_ID).count()
        return real(**kw)

    monkeypatch.setattr(module, "start_noise_series", watching)
    evaluate_spec_with_shape_aware_bar(
        _spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
        deployable_capital_usd=1_000_000.0, n_trials=4)
    assert seen == {"pending": False, "rows": 1}


def test_a_failure_of_the_noise_books_is_raised_after_the_candidate_is_recorded(
        session, monkeypatch):
    """A series that cannot finish (a dead pool) fails the item as it did when the books ran
    first -- the item is tried again the next night -- and the candidate's own trial is written
    as the error it became."""
    import qlab.calibration.shape_aware as module
    from qlab.calibration.run import NoiseRun
    from qlab.registry.models import Trial

    class Boom(Exception):
        pass

    class Failing(NoiseRun):
        def _finish(self):
            raise Boom("pool died")

    def start(**kw):
        return Failing(series=kw["series"], n_trials=kw["n_trials"], seed_offset=0,
                       replace_unusable=True, session=kw["session"], ruleset=kw["ruleset"],
                       deployable_capital_usd=kw["deployable_capital_usd"],
                       reference=kw["reference"], first=[])

    monkeypatch.setattr(module, "start_noise_series", start)
    with pytest.raises(Boom):
        evaluate_spec_with_shape_aware_bar(
            _spec(), session=session, ruleset=_shape_aware_ruleset(threshold=0.0),
            deployable_capital_usd=1_000_000.0, n_trials=4)
    trials = session.query(Trial).all()
    assert len(trials) == 1 and trials[0].route.value == "error"
