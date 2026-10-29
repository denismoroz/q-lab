"""Tests for `qlab.calibration.run` (docs/TASKS.md, T16).

Follows the same fixture shape as `qlab.pipeline.test_evaluate`: an
in-memory SQLite registry, a hand-registered snapshot on disk (no network),
and a toy `Strategy` used as the "real" reference -- these tests exercise
`run_noise_series`/`run_real_strategies` wiring against the real
`evaluate_spec`, not against the real trend strategy or a real snapshot.
"""

from __future__ import annotations

import json
from concurrent.futures.process import BrokenProcessPool
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
    GENERATOR_NAMES,
    NOISE_CODE_REF,
    REPLACEMENT_SEED_BASE,
    NoiseTrial,
    build_noise_spec,
    ensure_noise_idea,
    noise_idea_id,
    run_noise_series,
    run_real_strategies,
    usable_noise_book,
)
from qlab.data.snapshot import PANEL_RULES_VERSION
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
            "unexpressed_mechanisms": [],
            "sources": {"seed": "owner 2026-01-01: test fixture",
                        "costs": "owner 2026-01-01: test fixture",
                        "min_leg_notional": "owner 2026-01-01: test fixture"},
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
        reference=reference,
        seed_offset=0,
    )
    second = run_noise_series(
        series="unconstrained",
        n_trials=4,
        session=session,
        ruleset=ruleset,
        reference=reference,
        seed_offset=4,
    )

    all_seeds = [t.seed for t in first] + [t.seed for t in second]
    assert len(all_seeds) == len(set(all_seeds))


def _failing_books(monkeypatch, *, fail) -> None:
    """Noise books for which `fail(seed)` is true give no number, as a book refused by the
    structural band does: the run is real, then its metrics are dropped and an error set."""
    import dataclasses

    import qlab.calibration.run as run_module

    real = run_module._evaluate_noise

    def wrapped(spec, session, ruleset):
        evaluation = real(spec, session, ruleset)
        if fail(int(spec.params["seed"])):
            return dataclasses.replace(evaluation, metrics=None, error="StructuralMismatchError")
        return evaluation

    monkeypatch.setattr(run_module, "_evaluate_noise", wrapped)


def _run(session, *, n_trials=8, seed_offset=0, replace_unusable=True):
    return run_noise_series(
        series="dollar_neutral", n_trials=n_trials, session=session,
        ruleset=_permissive_ruleset(),
        reference=_reference_spec(), seed_offset=seed_offset,
        replace_unusable=replace_unusable)


def test_a_book_that_gives_no_number_is_replaced_by_a_fresh_one(session, monkeypatch):
    _failing_books(monkeypatch, fail=lambda seed: seed == 3)
    trials = _run(session)

    usable = [t for t in trials if usable_noise_book(t)]
    assert len(usable) == 8
    assert len(trials) == 9
    spare = [t for t in trials if t.replacement]
    assert len(spare) == 1
    # a seed far from the requested range, so the replacement was never in the requested set
    assert spare[0].seed == REPLACEMENT_SEED_BASE
    assert spare[0].seed not in {t.seed for t in trials if not t.replacement}
    # the book that gave nothing stays a trial of its own
    broken = next(t for t in trials if not usable_noise_book(t))
    assert broken.seed == 3 and broken.evaluation.trial_id is not None
    assert broken.evaluation.trial_id != spare[0].evaluation.trial_id


def test_replacements_are_deterministic_for_the_same_call(session, monkeypatch):
    _failing_books(monkeypatch, fail=lambda seed: seed in (11, 14, 16))
    first = _run(session, seed_offset=10)
    second = _run(session, seed_offset=10)

    def key(trials):
        return [(t.generator, t.seed, t.replacement, usable_noise_book(t)) for t in trials]

    assert key(first) == key(second)
    assert [t.seed for t in first if t.replacement] == [
        REPLACEMENT_SEED_BASE + 10 + j for j in range(3)]
    # a different offset draws different replacements
    other = _run(session, seed_offset=50)
    assert {t.seed for t in other if t.replacement}.isdisjoint(
        {t.seed for t in first if t.replacement})


def test_replacements_continue_the_round_robin_of_generators(session, monkeypatch):
    _failing_books(monkeypatch, fail=lambda seed: seed < 3)
    trials = _run(session, n_trials=8)
    spare = [t for t in trials if t.replacement]
    assert [t.generator for t in spare] == [
        GENERATOR_NAMES[(8 + j) % len(GENERATOR_NAMES)] for j in range(3)]


def test_replacement_is_bounded_by_the_number_of_books_asked_for(session, monkeypatch):
    # every requested book and every replacement gives nothing: draws stop at 2 * n
    _failing_books(monkeypatch, fail=lambda seed: True)
    trials = _run(session, n_trials=5)
    assert len(trials) == 10
    assert sum(t.replacement for t in trials) == 5
    assert not any(usable_noise_book(t) for t in trials)


def test_a_partial_shortage_keeps_the_honest_smaller_count(session, monkeypatch):
    # seeds 0..3 fail and so do the first two replacements: 4 missing, 4 extra draws allowed
    fails = {0, 1, 2, 3, REPLACEMENT_SEED_BASE, REPLACEMENT_SEED_BASE + 1}
    _failing_books(monkeypatch, fail=lambda seed: seed in fails)
    trials = _run(session, n_trials=4)
    assert sum(usable_noise_book(t) for t in trials) == 2
    assert sum(t.replacement for t in trials) == 4


def test_nothing_is_replaced_when_not_asked_or_when_every_book_is_usable(session, monkeypatch):
    _failing_books(monkeypatch, fail=lambda seed: seed == 2)
    off = _run(session, replace_unusable=False)
    assert len(off) == 8 and not any(t.replacement for t in off)

    monkeypatch.undo()
    clean = _run(session, seed_offset=100)
    assert len(clean) == 8 and not any(t.replacement for t in clean)


def test_run_noise_series_rejects_non_positive_n_trials(session):
    reference = _reference_spec()
    ruleset = _permissive_ruleset()
    with pytest.raises(ValueError, match="n_trials must be positive"):
        run_noise_series(
            series="dollar_neutral",
            n_trials=0,
            session=session,
            ruleset=ruleset,
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
        spec_paths={REFERENCE_IDEA_ID: spec_path},
    )

    assert set(results) == {REFERENCE_IDEA_ID}
    evaluation = results[REFERENCE_IDEA_ID]
    assert evaluation.error is None
    assert evaluation.metrics is not None
    assert evaluation.routing.route in {"paper", "shelf"}


def test_noise_runs_never_move_a_noise_idea_status(session):
    """A noise idea is a measuring instrument (docs/TASKS.md T31). Even under
    a ruleset that rejects every run, its status stays where it was created
    and no transition is written."""
    from qlab.registry.models import IdeaStatus, StageTransition, TrialRoute

    rejecting = RuleSet(
        version="2026-01-01.1",
        rules=[
            Rule(
                id="capital_fit",
                stage=Stage.PREFLIGHT,
                metric="min_capital_usd",
                comparator=Comparator.LE,
                threshold=1.0,
                fatal=True,
            )
        ],
    )

    trials = run_noise_series(
        series="dollar_neutral",
        n_trials=len(GENERATORS),
        session=session,
        ruleset=rejecting,
        reference=_reference_spec(),
    )

    assert {t.evaluation.routing.route for t in trials} == {"reject"}
    for generator in GENERATORS:
        idea = session.get(Idea, noise_idea_id("dollar_neutral", generator))
        assert idea.status == IdeaStatus.CANDIDATE
    assert session.query(StageTransition).count() == 0
    # The route itself is still on every trial.
    from qlab.registry.models import Trial

    routes = {trial.route for trial in session.query(Trial).all()}
    assert routes == {TrialRoute.REJECT}


def test_noise_spec_does_not_inherit_the_reference_unexpressed_mechanisms():
    reference = _reference_spec().model_copy(
        update={"unexpressed_mechanisms": ["a state machine the reference leaves out"]}
    )
    spec = build_noise_spec(
        series="dollar_neutral", generator=next(iter(GENERATORS)), seed=0, reference=reference
    )
    assert spec.unexpressed_mechanisms == []


def test_noise_does_not_repeat_the_references_hedge_timing() -> None:
    from qlab.calibration.run import build_noise_spec, load_reference_spec

    reference = load_reference_spec().model_copy(update={"measure_hedge_timing": True})
    noise = build_noise_spec(series="dollar_neutral", generator="random_signs", seed=1,
                             reference=reference)
    assert noise.measure_hedge_timing is False


def test_noise_books_in_worker_processes_give_the_same_numbers(tmp_path, monkeypatch):
    """Owner, 2026-10-26: «делай параллельно на 10 ядрах». A pool of workers, each
    with its own registry session on the same file, writes one trial per book
    and gives exactly the numbers of a run in one process."""
    import qlab.registry.db as db_module
    from qlab.calibration.run import NOISE_WORKERS_ENV

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())

    def run(workers: int, offset: int) -> list:
        monkeypatch.setenv(NOISE_WORKERS_ENV, str(workers))
        with db_module.session_scope() as s:
            if s.get(Idea, REFERENCE_IDEA_ID) is None:
                repo.upsert_idea(s, id=REFERENCE_IDEA_ID, title="Toy reference idea",
                                 source_type=SourceType.INTERNAL,
                                 asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
                repo.add_data_snapshot(
                    s, id=SNAPSHOT_ID, source=SOURCE,
                    instruments=dict.fromkeys(sorted(INSTRUMENTS), True),
                    range_start=START, range_end=END, path=str(tmp_path / SNAPSHOT_ID),
                    rows=60, fetched_at=datetime.now(UTC))
            return run_noise_series(series="dollar_neutral", n_trials=6, session=s,
                                    ruleset=_permissive_ruleset(),
                                    reference=_reference_spec(), seed_offset=offset)

    import qlab.calibration.run as run_module

    pools: list[int] = []
    real_submit = run_module._submit_jobs
    monkeypatch.setattr(run_module, "_submit_jobs", lambda jobs, **kw: (
        pools.append(kw["workers"]) or real_submit(jobs, **kw)))
    serial = run(1, 500)
    pooled = run(3, 500)
    assert pools == [3]
    assert [(t.generator, t.seed) for t in pooled] == [(t.generator, t.seed) for t in serial]
    assert [t.evaluation.metrics for t in pooled] == [t.evaluation.metrics for t in serial]
    assert all(t.evaluation.error is None for t in pooled)
    assert len({t.evaluation.trial_id for t in serial + pooled}) == 12
    run_module.shutdown_pool()  # its workers hold the temporary registry file
    db_module.get_engine().dispose()
    db_module._engine = None
    db_module._SessionLocal = None


def _shared_registry(tmp_path, monkeypatch):
    """A registry FILE the worker processes can open, with the toy reference registered."""
    import qlab.registry.db as db_module

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    with db_module.session_scope() as s:
        repo.upsert_idea(s, id=REFERENCE_IDEA_ID, title="Toy reference idea",
                         source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                         profile=Profile.OTHER)
        repo.add_data_snapshot(
            s, id=SNAPSHOT_ID, source=SOURCE, instruments=dict.fromkeys(sorted(INSTRUMENTS), True),
            range_start=START, range_end=END, path=str(tmp_path / SNAPSHOT_ID), rows=60,
            fetched_at=datetime.now(UTC))
    return db_module


def test_started_books_run_in_one_pool_that_is_kept_and_give_the_numbers_of_one_process(
        tmp_path, monkeypatch):
    """The books are handed to the pool at once (the caller runs the candidate meanwhile), the
    pool is the same for the next call, and the numbers are those of a run in one process."""
    import qlab.calibration.run as run_module
    from qlab.calibration.run import NOISE_WORKERS_ENV, shared_pool, start_noise_series

    db_module = _shared_registry(tmp_path, monkeypatch)

    def start(workers: int, offset: int):
        monkeypatch.setenv(NOISE_WORKERS_ENV, str(workers))
        s = db_module.get_sessionmaker()()
        return s, start_noise_series(
            series="dollar_neutral", n_trials=6, session=s, ruleset=_permissive_ruleset(),
                reference=_reference_spec(), seed_offset=offset)

    try:
        s1, pooled = start(2, 700)
        first_pool = run_module._POOL
        assert first_pool is not None and shared_pool(2) is first_pool
        assert isinstance(pooled._first[0], run_module.Future)  # not waited for yet
        got = pooled.result()
        assert pooled.result() is got  # the same answer every time it is asked
        s2, again = start(2, 800)
        assert run_module._POOL is first_pool  # not spawned again
        again.result()
        serial_session, serial = start(1, 700)
        serial_trials = serial.result()
        assert [(t.generator, t.seed) for t in got] == [(t.generator, t.seed)
                                                        for t in serial_trials]
        assert [t.evaluation.metrics for t in got] == [t.evaluation.metrics
                                                       for t in serial_trials]
        assert all(t.evaluation.error is None for t in got)
        for s in (s1, s2, serial_session):
            s.close()
        # Another registry file is another pool: its workers would write to the old one.
        monkeypatch.setenv("QLAB_DB", str(tmp_path / "other.db"))
        assert shared_pool(2) is not first_pool
    finally:
        run_module.shutdown_pool()
        db_module.get_engine().dispose()
        db_module._engine = None
        db_module._SessionLocal = None
    assert run_module._POOL is None


def test_a_broken_pool_is_dropped_so_the_next_call_starts_a_new_one(session, monkeypatch):
    import qlab.calibration.run as run_module

    class Dead:
        shut = False

        def shutdown(self, **kw):
            Dead.shut = True

    failed = run_module.Future()
    failed.set_exception(BrokenProcessPool("a worker died"))
    monkeypatch.setattr(run_module, "_POOL", Dead())
    monkeypatch.setattr(run_module, "_POOL_KEY", (2, "x"))
    handle = run_module.NoiseRun(
        series="dollar_neutral", n_trials=1, seed_offset=0, replace_unusable=False,
        session=session, ruleset=_permissive_ruleset(),
        reference=_reference_spec(), first=[failed])
    with pytest.raises(BrokenProcessPool):
        handle.result()
    with pytest.raises(BrokenProcessPool):
        handle.result()  # asked again: the same failure, the books are not run twice
    assert Dead.shut and run_module._POOL is None


def test_start_noise_series_in_one_process_is_ready_on_return(session):
    from qlab.calibration.run import start_noise_series

    handle = start_noise_series(
        series="dollar_neutral", n_trials=4, session=session, ruleset=_permissive_ruleset(),
            reference=_reference_spec())
    assert all(isinstance(t, NoiseTrial) for t in handle._first)
    assert len(handle.result()) == 4
