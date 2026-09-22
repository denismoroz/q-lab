"""Strategy spec file format: `<spec>.yaml` -> `StrategySpec`.

This is the ONE input `qlab evaluate` accepts (docs/PLAN.md, "Что такое
фреймворк"). It is deliberately narrow: enough to name a strategy
implementation, its parameters, what data it runs on, and the costs/capital
facts the harness refuses to assume (`qlab.harness.costs.CostModel`,
`qlab.harness.metrics.min_capital_usd`).

No field here has a default for costs or `min_leg_notional` — a spec that
omits them fails validation at load time, matching the harness's own refusal
to run without them. Guessing a "reasonable" fee or leg size would silently
turn a real cost into a fabricated one, exactly the failure mode
`qlab.harness.costs.CostModel` and `min_leg_notional` are designed to make
impossible by construction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpecData(BaseModel):
    """Where the panel comes from: `qlab.data.snapshot.build_snapshot`'s
    own arguments, one to one.

    `instruments=None` (the default) means "the source's discovered
    universe" — `build_snapshot`'s recommended, `universe_complete=True`
    path. Naming an explicit list is a hand-picked, potentially
    survivorship-biased universe (`universe_complete=False`), which is
    exactly what the `honest_universe` rule exists to catch — this spec
    format does not paper over that choice, it only records it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    interval: str
    include_spot: bool = False
    """Fetch the venue's spot markets alongside its perpetuals.

    A strategy that holds spot against a perp short — FRAB, Bv2 — needs two
    columns per coin (`BTC` and `BTC-SPOT`) and cannot run without this. It
    defaults to off because spot doubles the fetch and most strategies never
    touch it, and because a panel should carry only what the spec asked for.
    """
    start: date
    end: date
    instruments: list[str] | None = None
    min_daily_volume_usd: float | None = None
    """Point-in-time liquidity floor (docs/TASKS.md, T27, gap 2), one to one
    with `qlab.data.snapshot.build_snapshot`'s own argument of the same
    name — see that function's docstring for the exact trailing-window
    arithmetic and why it cannot look ahead. `None` (the default) means no
    liquidity filter at all, unchanged behaviour from before this field
    existed.

    This is a MEASUREMENT input, not a verdict: it narrows
    `panel.tradeable` (an instrument reads untradeable wherever its own
    trailing volume was too thin), the same mechanism already used for
    delisting/funding-gap/bad-price exclusions — it never removes an
    instrument from the panel and never touches `universe_complete`. It
    exists because `qlab.harness.metrics.min_capital_usd` only checks a
    venue's minimum ORDER size, which a thin memecoin clears as easily as
    BTC — exactly the gap that let a 20-leg XSMOM book of illiquid names
    read as "affordable" up to $120k (docs/XSMOM_T21.md, "Ревизия").
    """

    @field_validator("source", "interval")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("min_daily_volume_usd")
    @classmethod
    def _positive_if_given(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError(f"min_daily_volume_usd must be > 0 when given, got {value}")
        return value


class SpecCosts(BaseModel):
    """Trading costs. Both fields required, no defaults — mirrors
    `qlab.harness.costs.CostModel`, which has no defaults either, so that a
    spec cannot accidentally simulate free trading by omission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    taker_fee_bps: float
    slippage_bps: float

    @field_validator("taker_fee_bps", "slippage_bps")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value


class StrategySpec(BaseModel):
    """One strategy spec, as formalized in a `<spec>.yaml` file.

    `code_ref` is a dotted path to a `qlab.harness.strategy.Strategy`
    implementation living under `qlab.strategies`, e.g.
    `qlab.strategies.xsmom:XSMomStrategy` (module:attribute, the
    unambiguous form when either name contains dots) or
    `qlab.strategies.xsmom.XSMomStrategy` (plain dotted path, split on the
    last dot). See `qlab.pipeline.evaluate.resolve_strategy`.

    `min_leg_notional` has no default, for the same reason as `costs`: it
    feeds `qlab.harness.metrics.min_capital_usd`, which raises rather than
    silently assume a venue's minimum leg size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    idea_id: str
    title: str
    code_ref: str
    params: dict[str, object] = Field(default_factory=dict)
    data: SpecData
    costs: SpecCosts
    min_leg_notional: float
    simultaneous_legs: int = Field(
        default=1,
        description=(
            "How many legs this strategy must have open TOGETHER for one "
            "entry to make economic sense — the input `qlab.venues.derive."
            "atomic_execution` needs to know whether an uncovered-leg-after-"
            "failure question even applies. Defaults to 1: a strategy that "
            "never needs more than one leg open at a time satisfies "
            "atomic_execution trivially, since there is no partner leg a "
            "failure could ever strand. A MULTI-leg strategy (e.g. FRAB's "
            "spot-plus-perp-short, where a lone perp short or a lone spot "
            "long is not the strategy) MUST say so explicitly here — "
            "silently defaulting to 1 for a multi-leg strategy would make "
            "atomic_execution pass by construction, which is exactly the "
            "kind of guessed metric this pipeline exists to refuse (see "
            "qlab.venues.derive.atomic_execution)."
        ),
    )

    @field_validator("idea_id", "title", "code_ref")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("min_leg_notional")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"min_leg_notional must be > 0, got {value}")
        return value

    @field_validator("simultaneous_legs")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"simultaneous_legs must be >= 1, got {value}")
        return value


def load_spec(path: Path | str) -> StrategySpec:
    """Load and validate a `StrategySpec` from a YAML file.

    Raises:
        FileNotFoundError: no file at `path`.
        pydantic.ValidationError: the file is missing a required field
            (including `costs` or `min_leg_notional`) or has an unknown one.
    """
    spec_path = Path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"no spec file at {spec_path}")
    with spec_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"spec file {spec_path} must contain a YAML mapping at top level")
    return StrategySpec.model_validate(raw)


__all__ = ["SpecCosts", "SpecData", "StrategySpec", "load_spec"]
