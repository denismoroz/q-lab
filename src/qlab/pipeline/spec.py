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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    unexpressed_mechanisms: list[str]
    """Mechanisms of the strategy `idea_id` names that `code_ref` does NOT
    express -- the "Left OUT" list a transcription already writes in its
    docstring, made machine-readable (docs/TASKS.md T24).

    Required, with no default, for the same reason `costs` has none: an
    implementation that silently drops part of the strategy it is named
    after produces a number about a DIFFERENT strategy, attributed to this
    idea. A spec that expresses the whole strategy says so with an explicit
    empty list; a spec that does not must name what it leaves out.

    Non-empty means the run cannot test this idea's strategy, so the
    pipeline routes it to `not-evaluable` before computing anything. That
    is not a judgement on the simplified version: if the simplified version
    is worth testing on its own, it is a different candidate and gets its
    own `idea_id` with an empty list here (the same move T19 asks for with
    "соседний кандидат").
    """

    required_instruments: list[str] | None = None
    """Instruments (panel column names) the strategy needs ALL of at once to
    be the strategy it describes (docs/TASKS.md T19/T24). `None` (the
    default) means no single instrument is required: a cross-sectional book
    over whatever universe exists, e.g. trend or XSMOM with a discovered
    universe.

    When set, the verdict is computed only on the longest unbroken stretch
    of bars where every one of them was tradeable -- see
    `qlab.pipeline.evaluate.complete_book_window`. There is deliberately no
    coverage threshold: the owner declined to set one (2026-09-21, "я не хочу
    поиметь систему которая будет резать валидные стратегии"), and a
    window on which the specified book existed in full needs none -- it is
    the strategy as specified, judged by the same rules as any other window.
    """

    exploratory: str | None = None
    """The owner's explicit permission to run despite named not-evaluable
    reasons, and why (owner, 2026-10-01: "не тупо следовать описанию а
    пробывать прогнать стратегию на доступных данных").

    When set and `qlab.pipeline.evaluate.not_evaluable_reasons` is non-empty,
    the run is computed in full -- backtest, metrics, matched noise, every
    rule -- and the verdict rows are written, but the route stays
    `not-evaluable`: a number on survivors-only data, or on a strategy the
    code does not fully express, can inform a decision but cannot be one. The
    route reason carries what the rules alone would have said. When there
    are no such reasons, this field changes nothing.
    """

    params_fixed_at: date | None = None
    """The date on which this spec's parameters AND its instrument list (or
    instrument-selection rule) were last chosen -- the end of the period the
    choice could have been fitted to (docs/FIT_VS_FORWARD.md).

    Owner, 2026-10-02: "нужно разделять период выбора параметров стратегии и
    тестирования". Data before this date is the SELECTION period: a result
    there shows how well the choice fits the data it was made on, and a pass
    there proves nothing. Data from this date on is the FORWARD test: only
    there can the strategy show something it was not chosen to show.

    `None` means the date is unknown, and then the whole window counts as the
    selection period -- otherwise a strategy whose fitting history nobody
    wrote down would look better than one described honestly.
    """

    params_fit_from: date | None = None
    """The first date of the data the parameters were fitted on, when the
    source says so (owner, 2026-10-04: «588 дней — 2 года я не буду ждать ...
    разделять backtest and forward test»).

    `params_fixed_at` is then the END of that data -- the day after the last
    one the choice saw, NOT the day the code was committed: frab chose Bv2's
    parameters on 2023-06 … 2025-05 (research/strategy_b_v2/REPORT.md) and
    committed them in September 2026, and dating the boundary by the commit
    threw sixteen months of history the choice never saw into the selection
    period. With this field the window has three parts: data BEFORE the fit
    (reported as `pre_*`, not judged: an instrument list chosen later may
    know who survived), the SELECTION period `[params_fit_from,
    params_fixed_at)`, and the FORWARD test from `params_fixed_at` on --
    history as well as fresh data. `params_fixed_evidence` must cite both
    dates."""

    params_fixed_evidence: str | None = None
    """Where `params_fixed_at` comes from (a commit, a document, a database
    row). Required whenever the date is given: an undocumented date would
    let anyone move the boundary to where the results look best."""

    sources: dict[str, str] = Field(default_factory=dict)
    """Where every number and choice comes from: dotted path -> citation
    (`qlab.pipeline.sources`, docs/SOURCES.md). A citation at a path covers
    everything under it. Checked before any run starts."""

    regime_claim: dict[str, str] = Field(default_factory=dict)
    """How the source says the strategy behaves in each market regime
    (`qlab.regimes`: bull / flat / bear), declared before the run and cited in
    `sources` like any parameter (docs/REGIMES.md). Words the code can check:
    "earns" (compounded return in the regime > 0), "loses" (< 0),
    "beats_market" (better than BTC on the same days). Each run records
    whether the claim held; no rule reads it."""

    @field_validator("regime_claim")
    @classmethod
    def _known_regimes_and_words(cls, value: dict[str, str]) -> dict[str, str]:
        for regime, word in value.items():
            if regime not in ("bull", "flat", "bear"):
                raise ValueError(f"unknown regime {regime!r} (bull, flat, bear)")
            if word not in ("earns", "loses", "beats_market"):
                raise ValueError(f"unknown claim {word!r} (earns, loses, beats_market)")
        return value

    review_answers: dict[str, str] = Field(default_factory=dict)
    """Answers to the reviewer agent's findings (docs/REVIEWER.md): finding
    summary -> why it is not a gap in this implementation. An accepted
    finding without an answer makes the run not evaluable."""

    selects_causally: bool = False
    """The strategy chooses its own parameters from past data only, by code
    (`qlab.strategies.retune.Retune`), and `params_fixed_at` is the bar of its
    FIRST choice. Before that it is warming up, not fitted, so the data
    before it is neither judged nor called a selection period; from it on
    every day trades a choice made without that day -- the forward test,
    judged now on history (owner, 2026-10-02: "а нельзя ли делать backtest на
    3 месяца назад, а forward делать на оставшихся?").

    What stays in-sample is the grid the spec author offered; the evidence
    must say where its values come from."""

    @model_validator(mode="after")
    def _causal_selection_needs_a_start(self) -> StrategySpec:
        if self.selects_causally and self.params_fixed_at is None:
            raise ValueError("selects_causally needs params_fixed_at: the first causal choice")
        return self

    @model_validator(mode="after")
    def _fit_window_is_ordered(self) -> StrategySpec:
        if self.params_fit_from is None:
            return self
        if self.params_fixed_at is None:
            raise ValueError("params_fit_from needs params_fixed_at: the end of the fitted data")
        if self.params_fit_from >= self.params_fixed_at:
            raise ValueError("params_fit_from must be before params_fixed_at")
        if self.selects_causally:
            raise ValueError("a causal selection is fitted on nothing: drop params_fit_from")
        return self

    @model_validator(mode="after")
    def _fixed_date_has_evidence(self) -> StrategySpec:
        if self.params_fixed_at is not None and not (self.params_fixed_evidence or "").strip():
            raise ValueError(
                "params_fixed_at needs params_fixed_evidence: say where the date comes from"
            )
        return self

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

    @field_validator("unexpressed_mechanisms")
    @classmethod
    def _named_mechanisms(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("each unexpressed mechanism must be named, not blank")
        return value

    @field_validator("exploratory")
    @classmethod
    def _exploratory_says_why(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("exploratory must say why the owner accepts the run")
        return value

    @field_validator("required_instruments")
    @classmethod
    def _non_empty_if_given(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError(
                "required_instruments must list at least one instrument when given; "
                "omit it (None) when no single instrument is required"
            )
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
