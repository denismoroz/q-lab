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

import math
import os
from concurrent.futures import ProcessPoolExecutor
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
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    for generator in GENERATOR_NAMES:
        ensure_noise_idea(session, series=series, generator=generator)

    jobs = [
        (series, GENERATOR_NAMES[i % len(GENERATOR_NAMES)], seed_offset + i)
        for i in range(n_trials)
    ]
    trials = _run_jobs(jobs, session=session, ruleset=ruleset,
                       deployable_capital_usd=deployable_capital_usd, reference=reference)
    if not replace_unusable:
        return trials
    drawn_extra = 0
    while drawn_extra < n_trials:
        missing = n_trials - sum(usable_noise_book(t) for t in trials)
        if missing <= 0:
            break
        count = min(missing, n_trials - drawn_extra)
        extra_jobs = [
            (series, GENERATOR_NAMES[(n_trials + drawn_extra + j) % len(GENERATOR_NAMES)],
             REPLACEMENT_SEED_BASE + seed_offset + drawn_extra + j)
            for j in range(count)
        ]
        drawn_extra += count
        trials += [replace(t, replacement=True) for t in _run_jobs(
            extra_jobs, session=session, ruleset=ruleset,
            deployable_capital_usd=deployable_capital_usd, reference=reference)]
    return trials


def _run_jobs(jobs: list[tuple[str, str, int]], *, session: Session, ruleset: RuleSet,
              deployable_capital_usd: float, reference: StrategySpec) -> list[NoiseTrial]:
    """The books `jobs` name -- (series, generator, seed) -- in worker processes when the
    registry is shared, else in this process."""
    workers = min(noise_workers(), len(jobs))
    if workers > 1 and _shared_database(session):
        return _run_parallel(jobs, workers=workers, session=session, ruleset=ruleset,
                             deployable_capital_usd=deployable_capital_usd, reference=reference)
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
                    deployable_capital_usd: float) -> Evaluation:
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
    )


def _noise_job(job: tuple[str, str, int, StrategySpec, RuleSet, float]) -> NoiseTrial:
    """One noise book in a worker process, in its own registry session."""
    from qlab.registry.db import session_scope

    series, generator, seed, reference, ruleset, deployable_capital_usd = job
    spec = build_noise_spec(series=series, generator=generator, seed=seed, reference=reference)
    with session_scope() as session:
        evaluation = _evaluate_noise(spec, session, ruleset, deployable_capital_usd)
    return NoiseTrial(series=series, generator=generator, seed=seed, evaluation=evaluation)


def _run_parallel(jobs: list[tuple[str, str, int]], *, workers: int, session: Session,
                  ruleset: RuleSet, deployable_capital_usd: float,
                  reference: StrategySpec) -> list[NoiseTrial]:
    """The noise books in `workers` processes. The numbers are the same as in one
    process: each book is fixed by (series, generator, seed). What is shared is
    written here first: every book's `spec` row (versions are numbered per idea,
    so two workers adding one would collide), then a commit, so the workers --
    each writing only its own trial and verdict rows -- never wait on this
    session's lock. That commit also commits what the caller wrote before."""
    from qlab.pipeline.evaluate import ensure_spec_rows

    ensure_spec_rows(session, [
        build_noise_spec(series=series, generator=generator, seed=seed, reference=reference)
        for series, generator, seed in jobs])
    session.commit()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(
            _noise_job,
            [(series, generator, seed, reference, ruleset, deployable_capital_usd)
             for series, generator, seed in jobs]))


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
    "usable_noise_book",
]
