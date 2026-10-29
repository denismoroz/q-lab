"""Run noise strategies -- and the three real strategies -- through the
real `evaluate_spec` (docs/TASKS.md, T16).

Every noise trial is an ordinary `StrategySpec` pointed at
`qlab.calibration.noise.NoiseStrategy`, sharing its data/costs/capital block
with the real strategy it is structurally matched against so the SAME data
snapshot, costs, and accrual apply (docs/TASKS.md: "чтобы применялись те же
правила, косты и accrual"). It goes through `evaluate_spec` exactly like any
other candidate -- same rules engine, same trial-journal write, same routing
logic -- there is no separate "noise path" through the pipeline anywhere in
this module.
"""

from __future__ import annotations

import atexit
import math
import os
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.calibration.noise import GENERATORS
from qlab.pipeline.evaluate import Evaluation, evaluate_spec
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Profile, SourceType
from qlab.registry.queries import CALIBRATION_IDEA_PREFIX
from qlab.rules.schema import RuleSet

# src/qlab/calibration/run.py -> parents[3] is the project root (q-lab/).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

NOISE_CODE_REF = "qlab.calibration.noise:NoiseStrategy"

# The real strategy every noise book in this module is structurally matched
# against (docs/TASKS.md T16: "каждая шумовая стратегия обязана быть
# структурно похожа на настоящую"). Trend is the only one of the three specs
# that runs on data with no outstanding data-layer blocker end to end
# (Bv2/FRAB both need a spot leg whose coverage is still partial -- see
# docs/ACCEPTANCE_M2.md and docs/TASKS.md T19) and it is also the one
# candidate with a passing verdict today (4.75% net) -- exactly the number
# this calibration exists to put in context.
REFERENCE_SPEC_PATH = PROJECT_ROOT / "specs" / "trend.yaml"

# The three real strategies the report puts side by side with noise
# (docs/TASKS.md T16: "те же фигуры для трёх реальных стратегий для
# сравнения"). All three run end to end against snapshots already on disk
# (docs/ACCEPTANCE_M2.md) -- no network access happens here.
REAL_SPEC_PATHS: dict[str, Path] = {
    "trend-tsmom-crypto": PROJECT_ROOT / "specs" / "trend.yaml",
    "strategy-b-v2": PROJECT_ROOT / "specs" / "bv2.yaml",
    "frab": PROJECT_ROOT / "specs" / "frab.yaml",
}

# docs/TASKS.md T16's two required series: dollar-neutral (pure
# instrument-selection noise) and unconstrained (the same noise without the
# neutrality constraint, so it carries market beta and funding).
SERIES: tuple[str, ...] = ("dollar_neutral", "unconstrained")

GENERATOR_NAMES: tuple[str, ...] = tuple(GENERATORS)

# Noise books are independent of each other, so they run in worker processes
# (owner, 2026-10-26: «да, делай параллельно на 10 ядрах»). The environment
# variable overrides the count; 1 runs them in this process, as before.
NOISE_WORKERS_ENV = "QLAB_NOISE_WORKERS"
DEFAULT_NOISE_WORKERS = 10

# A replacement book (a book drawn because an earlier one gave no number) takes its seed from
# here, far above the seeds of the requested books (`seed_offset + i`), so the two ranges never
# meet and a replacement is a book that was never in the requested set. Replacement `r` of a call
# uses `REPLACEMENT_SEED_BASE + seed_offset + r`: fixed by the call's arguments alone, so a rerun
# draws the same replacements.
REPLACEMENT_SEED_BASE = 1_000_000


@dataclass(frozen=True, slots=True)
class NoiseTrial:
    """One noise trial's identity, alongside the `Evaluation` `evaluate_spec`
    produced for it -- kept together so `qlab.calibration.report` can group
    by series/generator without re-deriving them from `idea_id` strings."""

    series: str
    generator: str
    seed: int
    evaluation: Evaluation
    # True for a book drawn to stand in for one that gave no number (see `run_noise_series`).
    replacement: bool = False


def usable_noise_book(trial: NoiseTrial) -> bool:
    """Whether the book produced the number the noise bar ranks on: an annual return.

    A book gives none when its run errored or was refused (a `StructuralMismatchError`: its
    shape fell outside the band around the strategy it imitates), or when the return came back
    NaN. `qlab.calibration.percentile` drops exactly these books.
    """
    metrics = trial.evaluation.metrics
    if metrics is None:
        return False
    value = metrics.get("ann_return_net")
    return isinstance(value, int | float) and not math.isnan(value)


def load_reference_spec(path: Path | str = REFERENCE_SPEC_PATH) -> StrategySpec:
    """Load the real spec every noise trial is structurally matched against."""
    return load_spec(path)


def noise_idea_id(series: str, generator: str) -> str:
    """The `idea` id shared by every seed of one (series, generator)
    combination. One `idea` per combination (8 total: 2 series x 4
    generators), each seed becoming a new `spec` VERSION under it -- the
    same "one idea, many spec versions over time" shape the registry already
    uses for real strategies (`qlab.pipeline.evaluate._get_or_create_spec_row`),
    rather than one throwaway idea per seed.
    """
    return f"{CALIBRATION_IDEA_PREFIX}{series}-{generator}"


def ensure_noise_idea(session: Session, *, series: str, generator: str) -> None:
    """Upsert the `idea` row `noise_idea_id(series, generator)` needs to
    exist before any `spec`/`trial` referencing it can be written (`spec.
    idea_id` is a foreign key -- docs/REGISTRY.md's schema, unchanged by
    this task). Idempotent: safe to call before every trial.

    `source_type=INTERNAL` and the `notes` field mark this row as
    calibration tooling, not a real candidate -- so `qlab funnel` and other
    registry queries can tell a noise idea apart from an actual strategy
    under consideration at a glance, rather than silently inflating the
    candidate count.
    """
    idea_id = noise_idea_id(series, generator)
    repo.upsert_idea(
        session,
        id=idea_id,
        title=f"[calibration noise] {series}/{generator}",
        source_type=SourceType.INTERNAL,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.OTHER,
        notes=(
            "Synthetic noise strategy generated for ruleset calibration "
            "(docs/TASKS.md T16, qlab.calibration.noise). Not a real "
            "candidate -- structurally matched to a real strategy's gross "
            "exposure/turnover/position count but carries no signal. Each "
            "seed is a separate `spec` version under this idea."
        ),
    )


def build_noise_spec(
    *, series: str, generator: str, seed: int, reference: StrategySpec
) -> StrategySpec:
    """One noise `StrategySpec`.

    Shares `reference`'s `data`/`costs`/`min_leg_notional`/`simultaneous_legs`
    verbatim -- docs/TASKS.md T16's requirement that noise runs through "те
    же правила, косты и accrual" as the real strategy -- but points
    `code_ref` at `NoiseStrategy`, carrying the generator name, seed, and
    neutrality flag needed to reproduce this exact book deterministically
    (`qlab.calibration.noise.NoiseStrategy`'s own `params` contract).
    """
    if series not in SERIES:
        raise ValueError(f"unknown series {series!r}; expected one of {SERIES}")
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATOR_NAMES}")

    return reference.model_copy(
        update={
            "idea_id": noise_idea_id(series, generator),
            "title": f"noise calibration ({series}/{generator}, seed={seed})",
            "code_ref": NOISE_CODE_REF,
            # The noise book is fully expressed by NoiseStrategy -- what the
            # reference's implementation leaves out of ITS strategy says
            # nothing about the noise. `required_instruments` IS inherited:
            # noise must be judged on the same complete-book stretch as the
            # reference, or the percentile compares two different windows.
            "unexpressed_mechanisms": [],
            # A measurement of the reference's own decisions, not of noise.
            "measure_hedge_timing": False,
            "params": {
                "generator": generator,
                "seed": seed,
                "neutral": series == "dollar_neutral",
                "reference_code_ref": reference.code_ref,
                "reference_params": dict(reference.params),
                # Only when set, so the noise specs of every other strategy
                # keep their identity.
                **({"universe": "held"} if reference.noise_universe == "held" else {}),
            },
        }
    )


def run_noise_series(
    *,
    series: str,
    n_trials: int,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    reference: StrategySpec,
    seed_offset: int = 0,
    replace_unusable: bool = False,
) -> list[NoiseTrial]:
    """Run `n_trials` noise strategies for one series through the real
    `evaluate_spec` (docs/TASKS.md T16: "не меньше 200 шумовых стратегий на
    серию").

    Generators are round-robined (`GENERATOR_NAMES[i % 4]`) so each of the
    four contributes as evenly as possible across `n_trials`; every trial's
    seed is `seed_offset + i`, so calling this twice with a different
    `seed_offset` never repeats the same noise book, and every run remains
    reproducible from `(series, generator, seed)` alone.

    `replace_unusable` (docs/T28_SHAPE_AWARE_BAR.md, "Учёт шумовых книг"): a book can give no
    number -- its shape fell outside the structural band, its run failed -- and then the bar
    would be measured on fewer books than were asked for, where a percentile of "loses to 2 of
    N" reads differently for N = 199 and N = 200. With this on, every book that gave none is
    followed by a replacement, drawn the same way with a fresh seed (`REPLACEMENT_SEED_BASE`),
    until `n_trials` books are usable or as many extra books as were requested have been drawn;
    then the honest smaller count stays. Replacements continue the round-robin of generators.
    The unusable books stay in the registry as the trials they are, and in the returned list
    (replacements marked `replacement=True`).

    Blocks until every book is done; `start_noise_series` lets the caller do other work meanwhile.
    """
    return start_noise_series(
        series=series, n_trials=n_trials, session=session, ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd, reference=reference,
        seed_offset=seed_offset, replace_unusable=replace_unusable).result()


class NoiseRun:
    """The noise books of one `run_noise_series` call, possibly still running in the worker
    pool. `result()` waits for them and returns what `run_noise_series` returns."""

    def __init__(self, *, series: str, n_trials: int, seed_offset: int, replace_unusable: bool,
                 session: Session, ruleset: RuleSet, deployable_capital_usd: float,
                 reference: StrategySpec, first: list[NoiseTrial] | list[Future]) -> None:
        self._series = series
        self._n_trials = n_trials
        self._seed_offset = seed_offset
        self._replace_unusable = replace_unusable
        self._session = session
        self._ruleset = ruleset
        self._capital = deployable_capital_usd
        self._reference = reference
        self._first = first
        self._trials: list[NoiseTrial] | None = None
        self._failure: BaseException | None = None

    def result(self) -> list[NoiseTrial]:
        if self._failure is not None:
            raise self._failure
        if self._trials is None:
            try:
                self._trials = self._finish()
            except BaseException as exc:
                self._failure = exc
                if isinstance(exc, BrokenProcessPool):
                    shutdown_pool()  # the next call starts a new one
                raise
        return self._trials

    def _finish(self) -> list[NoiseTrial]:
        trials = [f.result() if isinstance(f, Future) else f for f in self._first]
        if not self._replace_unusable:
            return trials
        n_trials = self._n_trials
        drawn_extra = 0
        while drawn_extra < n_trials:
            missing = n_trials - sum(usable_noise_book(t) for t in trials)
            if missing <= 0:
                break
            count = min(missing, n_trials - drawn_extra)
            extra_jobs = [
                (self._series, GENERATOR_NAMES[(n_trials + drawn_extra + j) % len(GENERATOR_NAMES)],
                 REPLACEMENT_SEED_BASE + self._seed_offset + drawn_extra + j)
                for j in range(count)
            ]
            drawn_extra += count
            trials += [replace(t, replacement=True) for t in _run_jobs(
                extra_jobs, session=self._session, ruleset=self._ruleset,
                deployable_capital_usd=self._capital, reference=self._reference)]
        return trials


def start_noise_series(
    *,
    series: str,
    n_trials: int,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    reference: StrategySpec,
    seed_offset: int = 0,
    replace_unusable: bool = False,
) -> NoiseRun:
    """`run_noise_series` split in two: this starts the books and returns at once when they run
    in the worker pool (`noise_workers() > 1` and a registry file the workers share), so the
    caller can run the candidate meanwhile; `NoiseRun.result()` waits for them. In one process
    (tests, one worker) the books run here and the result is ready on return. Starting commits
    `session`: the workers write the registry from their own connections, and a lock this
    session held would stall them."""
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    for generator in GENERATOR_NAMES:
        ensure_noise_idea(session, series=series, generator=generator)

    jobs = [
        (series, GENERATOR_NAMES[i % len(GENERATOR_NAMES)], seed_offset + i)
        for i in range(n_trials)
    ]
    workers = min(noise_workers(), len(jobs))
    if workers > 1 and _shared_database(session):
        session.commit()
        first: list = _submit_jobs(jobs, workers=workers, ruleset=ruleset,
                                   deployable_capital_usd=deployable_capital_usd,
                                   reference=reference)
    else:
        first = _run_jobs(jobs, session=session, ruleset=ruleset,
                          deployable_capital_usd=deployable_capital_usd, reference=reference)
    return NoiseRun(series=series, n_trials=n_trials, seed_offset=seed_offset,
                    replace_unusable=replace_unusable, session=session, ruleset=ruleset,
                    deployable_capital_usd=deployable_capital_usd, reference=reference,
                    first=first)


def _run_jobs(jobs: list[tuple[str, str, int]], *, session: Session, ruleset: RuleSet,
              deployable_capital_usd: float, reference: StrategySpec) -> list[NoiseTrial]:
    """The books `jobs` name -- (series, generator, seed) -- in worker processes when the
    registry is shared, else in this process."""
    workers = min(noise_workers(), len(jobs))
    if workers > 1 and _shared_database(session):
        futures = _submit_jobs(jobs, workers=workers, ruleset=ruleset,
                               deployable_capital_usd=deployable_capital_usd,
                               reference=reference)
        return [f.result() for f in futures]
    trials: list[NoiseTrial] = []
    for series_, generator, seed in jobs:
        spec = build_noise_spec(
            series=series_, generator=generator, seed=seed, reference=reference
        )
        trials.append(NoiseTrial(
            series=series_, generator=generator, seed=seed,
            evaluation=_evaluate_noise(spec, session, ruleset, deployable_capital_usd)))
    return trials


def noise_workers() -> int:
    """How many processes run the noise books (`QLAB_NOISE_WORKERS`, default 10)."""
    return max(1, int(os.environ.get(NOISE_WORKERS_ENV, DEFAULT_NOISE_WORKERS)))


def _shared_database(session: Session) -> bool:
    """Whether worker processes, each opening the registry from `QLAB_DB`, would
    write to the same file this session writes to. An in-memory database or a
    session bound elsewhere (tests) runs the books in this process."""
    from qlab.registry.db import get_database_url

    url = session.get_bind().url
    return url.database not in (None, "", ":memory:") and \
        url.render_as_string(hide_password=False) == get_database_url()


def _evaluate_noise(spec: StrategySpec, session: Session, ruleset: RuleSet,
                    deployable_capital_usd: float, spec_row_id: int | None = None) -> Evaluation:
    return evaluate_spec(
        spec,
        session=session,
        ruleset=ruleset,
        deployable_capital_usd=deployable_capital_usd,
        # A noise idea is a measuring instrument, not a candidate: its
        # status must never move (docs/TASKS.md T31).
        update_idea_status=False,
        # Its generators are q-lab's own and tested; the candidate it is
        # matched to is checked on its own run (docs/LOOKAHEAD.md).
        check_lookahead=False,
        # Noise params (generator, seed) are the instrument's own.
        check_sources=False,
        spec_row_id=spec_row_id,
    )


def _noise_job(job: tuple[str, str, int, StrategySpec, RuleSet, float, int]) -> NoiseTrial:
    """One noise book in a worker process, in its own registry session."""
    from qlab.registry.db import session_scope

    series, generator, seed, reference, ruleset, deployable_capital_usd, spec_row_id = job
    spec = build_noise_spec(series=series, generator=generator, seed=seed, reference=reference)
    with session_scope() as session:
        evaluation = _evaluate_noise(spec, session, ruleset, deployable_capital_usd, spec_row_id)
    return NoiseTrial(series=series, generator=generator, seed=seed, evaluation=evaluation)


def _warm_worker(parent_pid: int) -> None:
    """Runs once in each worker as it starts: the imports a book needs, so the first book does
    not pay for them, and a watch on the parent -- a pool that lives all night must not outlive
    a night that was killed (ten workers of an earlier killed run were found idle a day and a
    half later)."""
    import threading
    import time

    import qlab.calibration.noise  # noqa: F401
    import qlab.pipeline.evaluate  # noqa: F401
    import qlab.registry.db  # noqa: F401

    def watch() -> None:
        while True:
            time.sleep(5)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch, daemon=True, name="qlab-parent-watch").start()


# One pool for the whole process, kept between calls: a pool made per call spawned its workers
# and had each import the stand again for every spec of the night (about 40 times a night).
# Workers read the registry file named by `QLAB_DB` when they start, so the pool is tied to it.
_POOL: ProcessPoolExecutor | None = None
_POOL_KEY: tuple[int, str] | None = None


def shared_pool(workers: int) -> ProcessPoolExecutor:
    """The process-wide pool of `workers` noise workers, made on first use and again when the
    count or the registry file changes."""
    global _POOL, _POOL_KEY
    from qlab.registry.db import get_database_url

    key = (workers, get_database_url())
    if _POOL is not None and _POOL_KEY != key:
        shutdown_pool()
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                    initializer=_warm_worker, initargs=(os.getpid(),))
        _POOL_KEY = key
    return _POOL


def shutdown_pool() -> None:
    """Stop the shared pool (a new one is made on the next use). Runs at exit too."""
    global _POOL, _POOL_KEY
    pool, _POOL, _POOL_KEY = _POOL, None, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown_pool)


def _submit_jobs(jobs: list[tuple[str, str, int]], *, workers: int, ruleset: RuleSet,
                 deployable_capital_usd: float, reference: StrategySpec) -> list[Future]:
    """Hand the books to the shared pool; returns their futures in the order of `jobs`.

    The numbers are the same as in one process: each book is fixed by (series, generator,
    seed). What is shared is written here first, in a registry connection of its own: every
    book's `spec` row (versions are numbered per idea, so two workers adding one would collide),
    committed so the workers -- each writing only its own trial and verdict rows -- never wait
    on it. A worker is given the id of its row and does not look for it."""
    from qlab.pipeline.evaluate import ensure_spec_rows
    from qlab.registry.db import session_scope

    with session_scope() as own:
        row_ids = ensure_spec_rows(own, [
            build_noise_spec(series=series, generator=generator, seed=seed, reference=reference)
            for series, generator, seed in jobs])
    work = [(series, generator, seed, reference, ruleset, deployable_capital_usd, row_id)
            for (series, generator, seed), row_id in zip(jobs, row_ids, strict=True)]
    try:
        pool = shared_pool(workers)
        return [pool.submit(_noise_job, w) for w in work]
    except BrokenProcessPool:
        shutdown_pool()
        pool = shared_pool(workers)
        return [pool.submit(_noise_job, w) for w in work]


def run_real_strategies(
    *,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    spec_paths: dict[str, Path] | None = None,
) -> dict[str, Evaluation]:
    """Run the real strategies (default: trend, Bv2, FRAB) through the same
    `evaluate_spec` noise uses, for the report's side-by-side comparison
    (docs/TASKS.md T16). Every run is a fresh recorded trial -- this module
    never reads a past number out of a document instead of computing it.
    """
    paths = spec_paths if spec_paths is not None else REAL_SPEC_PATHS
    results: dict[str, Evaluation] = {}
    for idea_id, path in paths.items():
        spec = load_spec(path)
        results[idea_id] = evaluate_spec(
            spec, session=session, ruleset=ruleset, deployable_capital_usd=deployable_capital_usd
        )
    return results


__all__ = [
    "GENERATOR_NAMES",
    "NOISE_CODE_REF",
    "REAL_SPEC_PATHS",
    "REFERENCE_SPEC_PATH",
    "SERIES",
    "REPLACEMENT_SEED_BASE",
    "NoiseTrial",
    "build_noise_spec",
    "ensure_noise_idea",
    "load_reference_spec",
    "noise_idea_id",
    "noise_workers",
    "run_noise_series",
    "run_real_strategies",
    "shared_pool",
    "shutdown_pool",
    "start_noise_series",
    "NoiseRun",
    "usable_noise_book",
]
