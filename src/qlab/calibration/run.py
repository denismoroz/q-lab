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

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.calibration.noise import GENERATORS
from qlab.pipeline.evaluate import Evaluation, evaluate_spec
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Profile, SourceType
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


@dataclass(frozen=True, slots=True)
class NoiseTrial:
    """One noise trial's identity, alongside the `Evaluation` `evaluate_spec`
    produced for it -- kept together so `qlab.calibration.report` can group
    by series/generator without re-deriving them from `idea_id` strings."""

    series: str
    generator: str
    seed: int
    evaluation: Evaluation


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
    return f"noise-{series}-{generator}"


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
            "params": {
                "generator": generator,
                "seed": seed,
                "neutral": series == "dollar_neutral",
                "reference_code_ref": reference.code_ref,
                "reference_params": dict(reference.params),
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
) -> list[NoiseTrial]:
    """Run `n_trials` noise strategies for one series through the real
    `evaluate_spec` (docs/TASKS.md T16: "не меньше 200 шумовых стратегий на
    серию").

    Generators are round-robined (`GENERATOR_NAMES[i % 4]`) so each of the
    four contributes as evenly as possible across `n_trials`; every trial's
    seed is `seed_offset + i`, so calling this twice with a different
    `seed_offset` never repeats the same noise book, and every run remains
    reproducible from `(series, generator, seed)` alone.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    for generator in GENERATOR_NAMES:
        ensure_noise_idea(session, series=series, generator=generator)

    trials: list[NoiseTrial] = []
    for i in range(n_trials):
        generator = GENERATOR_NAMES[i % len(GENERATOR_NAMES)]
        seed = seed_offset + i
        spec = build_noise_spec(
            series=series, generator=generator, seed=seed, reference=reference
        )
        evaluation = evaluate_spec(
            spec, session=session, ruleset=ruleset, deployable_capital_usd=deployable_capital_usd
        )
        trials.append(
            NoiseTrial(series=series, generator=generator, seed=seed, evaluation=evaluation)
        )
    return trials


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
    "NoiseTrial",
    "build_noise_spec",
    "ensure_noise_idea",
    "load_reference_spec",
    "noise_idea_id",
    "run_noise_series",
    "run_real_strategies",
]
