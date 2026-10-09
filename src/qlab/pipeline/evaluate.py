"""`evaluate_spec` — the glue that turns a `StrategySpec` into a `verdict`.

This is `qlab evaluate`'s engine (docs/PLAN.md, "Что такое фреймворк"):

    spec -> panel -> weights -> backtest -> metrics -> rules -> verdict

Every step below reuses an existing, already-merged building block
(`qlab.data.snapshot`, `qlab.harness`, `qlab.rules`, `qlab.registry`,
`qlab.venues`) — this module's only job is sequencing them and persisting
the result. See each function's docstring for the judgment calls this
task's spec left open (pipeline ordering around the `trial.snapshot_id`
constraint, how an existing snapshot is found without a network round
trip, and which extra preflight metrics this module does — and does not —
synthesize; `venue_supported`/`data_forward_available`/`atomic_execution`
are read from `qlab.venues`, never synthesized here).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import json
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from qlab.data.panel import MarketPanel
from qlab.data.snapshot import PANEL_RULES_VERSION, build_snapshot, load_snapshot
from qlab.data.sources.base import INTERVAL_TO_TIMEDELTA, SPOT_COLUMN_SUFFIX
from qlab.harness.costs import CostModel
from qlab.harness.gaps import holdable_gaps
from qlab.harness.lookahead import lookahead_violation
from qlab.harness.metrics import compute_metrics, min_capital_usd, periods_per_year
from qlab.harness.run import run_backtest
from qlab.harness.strategy import Strategy, validate_weights
from qlab.pipeline.sources import require_sources
from qlab.pipeline.spec import StrategySpec
from qlab.regimes import breakdown as regime_breakdown
from qlab.regimes import claim_checks as regime_claim_checks
from qlab.registry import repo
from qlab.registry.lifecycle import StatusDecision, apply_route
from qlab.registry.models import DataSnapshot, TrialRoute, TrialSource, TrialStatus
from qlab.registry.models import Spec as SpecRow
from qlab.rules.engine import EvaluationResult
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.schema import (
    FitPeriodUse,
    ForwardResolution,
    RuleKind,
    RuleSet,
    ShortForwardUse,
)
from qlab.venues.config import load_venue
from qlab.venues.derive import derive_venue_metrics

Route = Literal[
    "reject",
    "needs-more-data",
    "needs-forward",
    "not-evaluable",
    "needs-infrastructure",
    "shelf",
    "paper",
    "error",
]


class StrategyResolutionError(ImportError):
    """`spec.code_ref` does not resolve to a usable `Strategy`."""


class _NotValidOnInterval(Exception):
    """The strategy declares `valid_intervals` and the spec's interval is not
    one of them (T25) -- a not-evaluable outcome, not a crash."""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """What `evaluate_spec` decided to do with a run, and why.

    `route` is never `"live"` — see `decide_route`'s docstring for why that
    is a structural guarantee, not an oversight. `"error"` and
    `"not-evaluable"` are not routes the rules engine can produce: `"error"`
    means the run crashed before reaching it (see `Evaluation.error`),
    `"not-evaluable"` means the pipeline established before computing
    anything that this run could not test its idea's strategy
    (docs/TASKS.md T24, see `not_evaluable_reasons`).
    """

    route: Route
    reason: str
    required_capital_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The full result of one `evaluate_spec` call.

    `trial_id` is always set: a `trial` row is written whether or not the
    run succeeded (docs/REGISTRY.md: "trial — append-only, ALL runs are
    written"). `metrics`/`rules_result` are `None` exactly when `error` is
    set — the run never produced anything to feed the rules engine.
    """

    trial_id: int
    metrics: dict[str, float] | None
    rules_result: EvaluationResult | None
    routing: RoutingDecision
    error: str | None = None
    status_decision: StatusDecision | None = None
    """What the run did to its idea's status (docs/TASKS.md T31), or None
    when the caller asked `evaluate_spec` not to touch statuses at all
    (calibration noise)."""


def resolve_strategy(code_ref: str) -> Strategy:
    """Resolve `code_ref` to a `Strategy` instance.

    Accepts `"module.path:AttrName"` (unambiguous, preferred when either
    half could itself contain dots) or a plain dotted path
    `"module.path.AttrName"` (split on the last dot). If the resolved
    attribute is a class, it is instantiated with no arguments — a
    `Strategy` is defined as a pure function with no other state
    (`qlab.harness.strategy.Strategy`), so a no-arg constructor is the
    contract; if it is already an instance (a module-level singleton), it is
    used as-is.

    Raises:
        StrategyResolutionError: the module cannot be imported, the
            attribute does not exist, or the resolved object does not
            implement the `Strategy` protocol (`name` + `target_weights`).
    """
    module_path, sep, attr_path = code_ref.partition(":")
    if not sep:
        if "." not in code_ref:
            raise StrategyResolutionError(
                f"code_ref {code_ref!r} is not a dotted path or 'module:attr' reference"
            )
        module_path, _, attr_path = code_ref.rpartition(".")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise StrategyResolutionError(
            f"cannot import module {module_path!r} for code_ref {code_ref!r}: {exc}"
        ) from exc

    try:
        obj = getattr(module, attr_path)
    except AttributeError as exc:
        raise StrategyResolutionError(
            f"module {module_path!r} has no attribute {attr_path!r} (code_ref {code_ref!r})"
        ) from exc

    strategy = obj() if isinstance(obj, type) else obj
    if not isinstance(strategy, Strategy):
        raise StrategyResolutionError(
            f"{code_ref!r} resolved to {strategy!r}, which does not implement the "
            "Strategy protocol (needs a `name` attribute and a `target_weights` method)"
        )
    return strategy


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _find_matching_snapshot(
    session: Session,
    *,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    instruments: list[str] | None,
    include_spot: bool,
    min_daily_volume_usd: float | None,
) -> str | None:
    """Find a `data_snapshot` already on disk that answers this exact data
    request, without ever hitting the network.

    `build_snapshot` can only compute a snapshot's id from the fetched bytes
    themselves (see its docstring), so it always re-fetches even when the
    result would be identical. A snapshot already on disk with the same id
    must not be refetched (this task's requirement), so this function does
    the matching a different way: it narrows candidates by the registry
    columns that ARE queryable (`source`, `range_start`, `range_end`), then
    reads each candidate's `manifest.json` off disk — a local read, not a
    network call — to compare `interval`, `universe_complete`, and (for an
    explicit instrument list) the instrument set itself.

    `instruments=None` (discovered universe) matches any candidate with
    `universe_complete=True` for the same source/range/interval, without
    needing to know the discovered list in advance: `build_snapshot`
    documents that re-running the identical request is idempotent, so an
    existing "discovered universe" snapshot for this exact request is by
    construction the same request already answered.
    """
    wanted_instruments = (
        sorted({i.upper() for i in instruments}) if instruments is not None else None
    )

    candidates = (
        session.query(DataSnapshot)
        .filter(
            DataSnapshot.source == source,
            DataSnapshot.range_start == start.date(),
            DataSnapshot.range_end == end.date(),
        )
        # Newest build first: when the same request has been rebuilt, the
        # later build carries what was learned since (e.g. the
        # `delisted_without_history` check, added after a first xyz build
        # had already claimed a complete universe). The earlier one is left
        # on disk -- trials may reference it -- but is no longer chosen.
        .order_by(DataSnapshot.fetched_at.desc())
        .all()
    )
    matches: list[str] = []
    for row in candidates:
        manifest_path = Path(row.path) / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("interval") != interval:
            continue
        # Built under other panel rules (bad bars, holdability, funding,
        # spot marks): the same request would be answered differently today.
        if manifest.get("panel_rules") != PANEL_RULES_VERSION:
            continue
        # A perp-only snapshot is not a substitute for a spot-inclusive one,
        # and vice versa: a spec that asks for spot must not be silently
        # handed a panel without it (the run would die on a missing column),
        # and one that does not must not inherit columns it never requested.
        # The manifest records the instrument list, so the presence of any
        # "-SPOT" name answers this without a separate field.
        names = manifest.get("instruments", [])
        has_spot = any(str(name).endswith(SPOT_COLUMN_SUFFIX) for name in names)
        if has_spot != include_spot:
            continue
        # A liquidity-filtered snapshot and an unfiltered one answer
        # different requests even when source/range/interval/instruments all
        # agree: the filter lives in `tradeable`, so reusing an unfiltered
        # snapshot for a filtered spec would silently apply NO filter and
        # produce a result labelled as filtered. `manifest.get` returns None
        # for snapshots built before the field existed, which compares equal
        # to an unfiltered request and unequal to any filtered one -- the
        # safe direction.
        if manifest.get("min_daily_volume_usd") != min_daily_volume_usd:
            continue
        if wanted_instruments is None:
            # A discovered-universe request is answered by a discovered build:
            # complete, or incomplete ONLY because the venue erased delisted
            # instruments' history (`delisted_without_history`). The latter is
            # still the answer to this request -- and the one that must be
            # found, so the run is routed not-evaluable instead of refetching.
            if not (manifest.get("universe_complete") or manifest.get("delisted_without_history")):
                continue
        else:
            # Neither a complete discovered build nor one marked for erased
            # history answers a hand-written list: the latter's flag is a
            # statement about the discovered universe, not about this list.
            if manifest.get("universe_complete") or manifest.get("delisted_without_history"):
                continue
            if sorted(manifest.get("instruments", [])) != wanted_instruments:
                continue
        matches.append(str(row.id))

    if not matches:
        return None
    # Prefer a snapshot that carries volume. One that does is a strict
    # superset of one that does not -- same panel, same tradeable mask, plus a
    # frame -- so preferring it is never wrong, and NOT preferring it silently
    # shadows the newer snapshot with an older volume-less one for every spec
    # whose strategy needs volume (`top_k_by_volume` raised rather than
    # guessing, which is how this was found; "unknown volume is not zero
    # volume" is the same rule `qlab.harness.accrual` applies to funding).
    for snapshot_id in matches:
        row = session.get(DataSnapshot, snapshot_id)
        if row is not None and (Path(row.path) / "volume.parquet").is_file():
            return snapshot_id
    return matches[0]


def resolve_panel(session: Session, data: StrategySpec) -> MarketPanel:
    """Load an existing matching snapshot, or build (and fetch) a new one."""
    data_block = data.data
    start_ts = _to_utc_timestamp(data_block.start)
    end_ts = _to_utc_timestamp(data_block.end)

    existing_id = _find_matching_snapshot(
        session,
        source=data_block.source,
        start=start_ts,
        end=end_ts,
        interval=data_block.interval,
        instruments=data_block.instruments,
        include_spot=data_block.include_spot,
        min_daily_volume_usd=data_block.min_daily_volume_usd,
    )
    if existing_id is not None:
        return load_snapshot(existing_id, session=session)

    return build_snapshot(
        data_block.source,
        data_block.instruments,
        data_block.start,
        data_block.end,
        data_block.interval,
        include_spot=data_block.include_spot,
        min_daily_volume_usd=data_block.min_daily_volume_usd,
        session=session,
    )


def _get_or_create_spec_row(session: Session, spec: StrategySpec) -> SpecRow:
    """Find an existing registry `spec` row with identical content for this
    `idea_id`, or append a new version.

    The registry's `spec` table is versioned by content, not by call: two
    `evaluate_spec` calls with byte-identical params/data/costs/code_ref
    reuse the same row rather than growing a new version on every trial.
    `rebalance` has no equivalent field in `StrategySpec` — this pipeline
    records the data interval there (`data.interval`), which is the closest
    available proxy; a strategy with a genuinely different rebalance cadence
    from its data interval should say so in `params` instead.
    """
    data_requirements = spec.data.model_dump(mode="json")
    costs_model = spec.costs.model_dump(mode="json")

    existing = (
        session.query(SpecRow)
        .filter(SpecRow.idea_id == spec.idea_id)
        .order_by(SpecRow.version.desc())
        .all()
    )
    for row in existing:
        if (
            row.code_ref == spec.code_ref
            and row.params == spec.params
            and row.data_requirements == data_requirements
            and row.costs_model == costs_model
        ):
            return row

    next_version = (existing[0].version + 1) if existing else 1
    return repo.add_spec(
        session,
        idea_id=spec.idea_id,
        version=next_version,
        params=spec.params,
        data_requirements=data_requirements,
        rebalance=spec.data.interval,
        costs_model=costs_model,
        code_ref=spec.code_ref,
    )


def _code_sha() -> str:
    """Best-effort git commit of the running code, for `trial.code_sha`.

    Falls back to `"unknown"` rather than raising: a trial's reproducibility
    record should not be the reason an evaluation fails to complete."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config_hash(params: dict[str, object]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BookCoverage:
    """How much of the panel the specified book existed on in full.

    `coverage` is the share of panel bars on which every required instrument
    was tradeable -- a measurement, reported, never compared to a threshold
    (see `StrategySpec.required_instruments`). `window` is the longest
    unbroken run of such bars as inclusive `(first, last)` row positions,
    or None if the book was never complete. `missing` lists required
    instruments the panel has no column for at all.
    """

    coverage: float
    window: tuple[int, int] | None
    missing: tuple[str, ...]


def complete_book_window(panel: MarketPanel, required: Sequence[str]) -> BookCoverage:
    """Measure where the book `required` describes existed in full.

    The verdict is rendered on the longest unbroken stretch, not on every
    complete bar: returns are compounded bar to bar, and stitching separate
    stretches together would invent price moves across the gaps between them.
    """
    missing = tuple(name for name in required if name not in panel.prices.columns)
    if missing:
        return BookCoverage(coverage=0.0, window=None, missing=missing)

    # A bar the book is held through (no price, docs/TASKS.md T36) does not
    # break the stretch: the harness holds and marks the position across it.
    gaps = holdable_gaps(panel.prices, panel.funding,
                         panel.meta.get("no_funding_instruments", ()))
    present = panel.tradeable | gaps
    complete = present[list(required)].all(axis=1).to_numpy(dtype=bool)
    n_bars = len(complete)
    coverage = float(complete.mean()) if n_bars else 0.0

    best: tuple[int, int] | None = None
    run_start: int | None = None
    for i, is_complete in enumerate(complete):
        if is_complete and run_start is None:
            run_start = i
        if run_start is not None and (not is_complete or i == n_bars - 1):
            run_end = i if is_complete else i - 1
            if best is None or (run_end - run_start) > (best[1] - best[0]):
                best = (run_start, run_end)
            run_start = None
    return BookCoverage(coverage=coverage, window=best, missing=())


def not_evaluable_reasons(
    spec: StrategySpec,
    coverage: BookCoverage | None,
    panel_meta: Mapping[str, object] | None = None,
) -> list[str]:
    """Why this run cannot test the strategy `spec.idea_id` names (T24).

    Empty means the run can go ahead. Each reason is checked BEFORE any
    backtest: a number computed on a different strategy, or on a book that
    never existed, would be recorded as if it were an answer -- which is how
    FRAB came to be "rejected" for a state machine the interface never ran,
    and Bv2 for a book complete on a fifth of its window (T26).
    """
    reasons: list[str] = []
    erased = list((panel_meta or {}).get("delisted_without_history") or [])
    if erased:
        # The universe the spec asked for existed, but part of it can no
        # longer be priced: a test on the survivors would carry exactly the
        # bias `honest_universe` exists to catch, so it is not run at all
        # (qlab.data.sources.hyperliquid.has_history_before).
        reasons.append(
            f"the source serves no price history for {len(erased)} delisted instrument(s) "
            f"of this universe, so only survivors could be tested: " + ", ".join(erased)
        )
    if spec.unexpressed_mechanisms:
        reasons.append(
            "implementation does not express: " + "; ".join(spec.unexpressed_mechanisms)
        )
    if coverage is not None:
        if coverage.missing:
            reasons.append(
                "required instrument(s) absent from the data: " + ", ".join(coverage.missing)
            )
        elif coverage.window is None:
            reasons.append("the specified book was never complete in the data")
    return reasons


@functools.lru_cache(maxsize=1)
def _regimes():
    """The stored regime labels (`qlab regimes build`), or None: without them
    a run simply carries no regime breakdown."""
    from qlab.regimes import load

    return load()


def review_reasons(session: Session, spec: StrategySpec) -> list[str]:
    """Not evaluable while the latest review of exactly this implementation
    holds a verified blocking finding the spec does not answer
    (docs/REVIEWER.md). The agent proposed it; the evidence was checked by
    code; this rule, not the agent, decides."""
    from qlab.agents.reviewer import code_files, review_key, unanswered_blocking
    from qlab.registry.models import Review

    latest = (
        session.query(Review)
        .filter(Review.idea_id == spec.idea_id,
                Review.review_key == review_key(spec, code_files(spec)))
        .order_by(Review.id.desc())
        .first()
    )
    if latest is None:
        return []
    return [
        f"reviewer (review {latest.id}) found, with verified evidence, {f['kind']}: "
        f"{f['summary']} -- declare it in unexpressed_mechanisms or answer it in review_answers"
        for f in unanswered_blocking(spec, latest.accepted)
    ]


def _slice_rows(panel: MarketPanel, first: int, last: int) -> MarketPanel:
    """`panel` restricted to rows `first..last` inclusive. Works for both the
    real `qlab.data.panel.MarketPanel` and the harness stand-in: both are
    frozen dataclasses over the same four aligned frames."""
    rows = slice(first, last + 1)
    return dataclasses.replace(
        panel,
        prices=panel.prices.iloc[rows],
        funding=panel.funding.iloc[rows],
        tradeable=panel.tradeable.iloc[rows],
        volume=panel.volume.iloc[rows],
        high=panel.high.iloc[rows],
    )


def decide_route(
    result: EvaluationResult, metrics: dict[str, float], deployable_capital_usd: float
) -> RoutingDecision:
    """Turn a rules `EvaluationResult` into a routing decision.

    Order matters and is fail-closed throughout:

      1. `not result.decisive` -> `"needs-more-data"`. Some metric the
         ruleset needs could not be computed at all. This is checked FIRST
         because an indecisive result is also, mechanically, an
         `overall_passed=False` result (`qlab.rules.engine.evaluate`) — but
         "we don't know" and "it failed" are different findings, and only
         the former should ever send a candidate back for more data instead
         of rejecting it outright.
      2. A STRATEGY rule failed -> `"reject"`. Every metric was
         computed and at least one strategy rule failed — fatal or not.
         2b. Only INFRASTRUCTURE rules failed -> `"needs-infrastructure"`
         (docs/TASKS.md T35): the strategy passed everything it was asked;
         what is missing is our ability to run it. The task
         describes `"reject"` via the fatal case (`capital_fit` needing more
         than the whole capital horizon), but a ruleset can define a
         non-fatal rule too (`Rule.fatal=False`); a candidate that fails a
         real, computed rule was screened out either way, so both are
         reported the same route here, distinguished by
         `result.failed_rule_ids` / `result.failed_fatal_rule_id` in the
         reason string for anyone inspecting the verdict.
      3. Otherwise every rule passed. If `min_capital_usd` (present in
         `metrics` on every successful run — see `evaluate_spec`) exceeds
         `deployable_capital_usd`, `"shelf"`: affordable in principle
         (it already cleared `capital_fit`'s whole-horizon ceiling), just
         not with money on hand right now. `required_capital_usd` carries
         the number so the caller can queue it.
      4. Otherwise `"paper"`: passed, and affordable now.

    `"live"` is never returned — promotion to real money is a human
    decision informed by forward (paper) evidence, never a direct output of
    a single backtest verdict (docs/PLAN.md: "живёт... paper — единственная
    стадия, дающая forward-evidence").
    """
    if not result.decisive:
        return RoutingDecision(
            route="needs-more-data",
            reason=f"metric(s) could not be computed: {', '.join(result.unknown_metrics)}",
        )

    if result.failed_strategy_rule_ids:
        failed = ", ".join(result.failed_strategy_rule_ids)
        fatal_note = (
            f"; fatal rule: {result.failed_fatal_rule_id}"
            if result.failed_fatal_rule_id
            else ""
        )
        infra = (
            f"; also missing infrastructure: {', '.join(result.failed_infrastructure_rule_ids)}"
            if result.failed_infrastructure_rule_ids
            else ""
        )
        return RoutingDecision(
            route="reject", reason=f"rule(s) failed: {failed}{fatal_note}{infra}"
        )

    if result.failed_infrastructure_rule_ids:
        # Every strategy rule was computed and passed; only our setup falls
        # short. Not a rejection (T35): the claim "tested, does not work"
        # would be false.
        return RoutingDecision(
            route="needs-infrastructure",
            reason=(
                "every strategy rule passed; missing infrastructure: "
                + ", ".join(result.failed_infrastructure_rule_ids)
            ),
            required_capital_usd=metrics.get("min_capital_usd"),
        )

    required = metrics.get("min_capital_usd")
    if required is not None and required > deployable_capital_usd:
        return RoutingDecision(
            route="shelf",
            reason=(
                f"passed but needs ${required:,.0f} minimum capital; "
                f"only ${deployable_capital_usd:,.0f} is deployable now"
            ),
            required_capital_usd=required,
        )

    return RoutingDecision(route="paper", reason="passed and affordable")


# --------------------------------------------------------------------------
# Selection period vs forward test (docs/FIT_VS_FORWARD.md)
# --------------------------------------------------------------------------

SELECTION_NOTE = "selection period"
PRE_FIT_KEYS = ("ann_return_net", "sharpe_net", "max_dd", "ann_return_net_ex_best_1pct",
                "fragility_days")
FORWARD_NOTE = "forward test"


@dataclass(frozen=True, slots=True)
class PeriodSplit:
    """Row bounds (inclusive) of the judged window's parts: data BEFORE the
    fit (only with `params_fit_from`), the SELECTION period up to the spec's
    `params_fixed_at`, and the FORWARD test from it on. A part shorter than
    two bars has no return to measure and is None."""

    selection: tuple[int, int] | None
    forward: tuple[int, int] | None
    before: tuple[int, int] | None = None


def split_at_fixed_date(
    index: pd.DatetimeIndex, first: int, last: int, params_fixed_at: date | None,
    params_fit_from: date | None = None,
) -> PeriodSplit:
    """Split rows `first..last` at the first bar on or after `params_fixed_at`
    (and, when given, at the first bar on or after `params_fit_from`). An
    unknown date (None) makes the whole window the selection period."""
    if params_fixed_at is None:
        return PeriodSplit(selection=(first, last), forward=None)

    def _row(day: date) -> int:
        return min(max(int(index.searchsorted(pd.Timestamp(day, tz="UTC"))), first), last + 1)

    k = _row(params_fixed_at)
    k0 = first if params_fit_from is None else min(_row(params_fit_from), k)
    before = (first, k0 - 1) if k0 - 1 > first else None
    selection = (k0, k - 1) if k - 1 > k0 else None
    forward = (k, last) if last > k else None
    return PeriodSplit(selection=selection, forward=forward, before=before)


def forward_years_needed(
    selection_sharpe: float | None, resolution: ForwardResolution
) -> float | None:
    """Years of forward data needed to tell a strategy whose selection-period
    Sharpe is `selection_sharpe` from zero: ((z_conf + z_power) / S)^2.
    None when the Sharpe is unknown or not positive -- there is then no
    positive claim for a forward test to resolve."""
    if selection_sharpe is None or selection_sharpe != selection_sharpe or selection_sharpe <= 0:
        return None
    z = statistics.NormalDist()
    z_sum = z.inv_cdf(resolution.confidence) + z.inv_cdf(resolution.power)
    return (z_sum / selection_sharpe) ** 2


def _for_selection_period(ruleset: RuleSet) -> RuleSet:
    """The ruleset as applied to the selection period: an informational rule
    is never fatal there, so its failure cannot stop the engine before a
    conclusive rule in a later stage is evaluated."""
    rules = [
        rule.model_copy(update={"fatal": False})
        if rule.fit_period == FitPeriodUse.INFORMATIONAL
        else rule
        for rule in ruleset.rules
    ]
    return ruleset.model_copy(update={"rules": rules})


def decide_fit_forward_route(
    *,
    ruleset: RuleSet,
    selection: EvaluationResult | None,
    forward: EvaluationResult | None,
    metrics: dict[str, float],
    deployable_capital_usd: float,
    params_fixed_at: date | None,
) -> RoutingDecision:
    """`_decide_fit_forward`, then -- when the ruleset asks for it
    (`regime_coverage_days`) -- a forward test that would decide anything but
    has seen some market regime for fewer days waits instead
    (docs/REGIMES.md). A rejection on the selection period stands: it does
    not rest on the forward test."""
    routing = _decide_fit_forward(
        ruleset=ruleset, selection=selection, forward=forward, metrics=metrics,
        deployable_capital_usd=deployable_capital_usd, params_fixed_at=params_fixed_at,
    )
    need = ruleset.regime_coverage_days
    if (need is None or forward is None or routing.route == "needs-forward"
            or routing.reason.startswith("fails on the selection period")):
        return routing
    from qlab.regimes import REGIMES

    days = {r: metrics.get(f"regime_{r}_days") for r in REGIMES}
    if any(v is None for v in days.values()):
        return routing  # no regime labels for this run: nothing to check
    short = {r: int(v) for r, v in days.items() if v < need}
    if not short:
        return routing
    return RoutingDecision(
        route="needs-forward",
        reason=(
            "has not seen every market regime for " + f"{need} days: "
            + ", ".join(f"{r} {n} days" for r, n in short.items())
            + f"; so far it would route {routing.route} ({routing.reason})"
        ),
        required_capital_usd=metrics.get("min_capital_usd"),
    )


def _decide_fit_forward(
    *,
    ruleset: RuleSet,
    selection: EvaluationResult | None,
    forward: EvaluationResult | None,
    metrics: dict[str, float],
    deployable_capital_usd: float,
    params_fixed_at: date | None,
) -> RoutingDecision:
    """The route of a run judged under a ruleset with `forward_resolution`.

    1. On the selection period, a failed CONCLUSIVE strategy rule rejects:
       the period flatters the strategy, so falling short even there is a
       finding no forward data can undo. An unknown conclusive metric with no
       forward test routes `needs-more-data`.
    2. No forward test -> `needs-forward`: a selection-period pass proves
       nothing, whatever the informational rules said.
    3. A forward test is judged by every rule (`decide_route`). While it is
       shorter than `forward_years_needed` of the selection-period Sharpe it
       decides nothing in EITHER direction -> `needs-forward`, with what it
       would route so far in the reason: 18 days of trend once read +446% a
       year, passed every rule and would have gone to `paper`. Once long
       enough, its route stands. With no selection period (parameters fixed
       before the data began) the forward route stands as is.
    """
    assert ruleset.forward_resolution is not None
    by_id = {rule.id: rule for rule in ruleset.rules}

    def _conclusive(row) -> bool:
        rule = by_id.get(row.rule_id)
        return (
            rule is not None
            and rule.kind == RuleKind.STRATEGY
            and rule.fit_period == FitPeriodUse.CONCLUSIVE
        )

    when = (
        f"parameters fixed on {params_fixed_at}"
        if params_fixed_at is not None
        else "the date the parameters were fixed is unknown, so the whole window is the "
        "selection period"
    )

    if selection is not None:
        failed = [r.rule_id for r in selection.rows if r.passed is False and _conclusive(r)]
        if failed:
            return RoutingDecision(
                route="reject",
                reason=(
                    "fails on the selection period, the data its parameters were chosen on "
                    f"and which flatters it: {', '.join(failed)}"
                ),
            )
        unknown = [r.metric for r in selection.rows if r.passed is None and _conclusive(r)]
        if unknown and forward is None:
            return RoutingDecision(
                route="needs-more-data",
                reason=f"metric(s) could not be computed: {', '.join(unknown)}",
            )

    if forward is None:
        informational = (
            [r.rule_id for r in selection.rows if r.passed is False and not _conclusive(r)]
            if selection is not None
            else []
        )
        note = (
            f"; informational on that period, not judged there: {', '.join(informational)} failed"
            if informational
            else ""
        )
        return RoutingDecision(
            route="needs-forward",
            reason=(
                f"no forward test yet ({when}); the selection period shows nothing against "
                f"it{note}"
            ),
            required_capital_usd=metrics.get("min_capital_usd"),
        )

    routing = decide_route(forward, metrics, deployable_capital_usd)
    if selection is None:
        # No selection period: the forward test is judged on its own. A
        # failure only of rules that ask "separable from luck?" on a window
        # too short to resolve its own Sharpe is too early to tell.
        if routing.route != "reject":
            return routing
        failed = [r for r in forward.rows if r.passed is False]
        waits = all(
            (rule := by_id.get(r.rule_id)) is not None
            and rule.on_short_forward == ShortForwardUse.WAIT
            for r in failed
        )
        days = metrics.get("forward_days", 0.0)
        needed = metrics.get("forward_days_needed")
        if failed and waits and needed is not None and days < needed:
            return RoutingDecision(
                route="needs-forward",
                reason=(
                    f"forward test too short to tell: {days:,.0f} days fail only "
                    f"{', '.join(r.rule_id for r in failed)}; telling a Sharpe of "
                    f"{metrics.get('sharpe_net', float('nan')):.2f} from zero takes about "
                    f"{needed:,.0f} days"
                ),
                required_capital_usd=metrics.get("min_capital_usd"),
            )
        return routing

    days = metrics.get("forward_days", 0.0)
    needed = metrics.get("forward_days_needed")
    if needed is None or days < needed:
        sharpe = metrics.get("fit_sharpe_net")
        how_long = (
            f"telling a selection-period Sharpe of {sharpe:.2f} from zero takes about "
            f"{needed:,.0f} days"
            if needed is not None and sharpe is not None
            else "the selection period shows no positive Sharpe for it to resolve"
        )
        return RoutingDecision(
            route="needs-forward",
            reason=(
                f"forward test too short to tell either way: {days:,.0f} days; {how_long}; "
                f"so far it would route {routing.route} ({routing.reason})"
            ),
            required_capital_usd=metrics.get("min_capital_usd"),
        )
    if routing.route == "reject":
        return RoutingDecision(
            route="reject", reason=f"forward test failed over {days:,.0f} days: {routing.reason}"
        )
    return routing

def evaluate_spec(
    spec: StrategySpec,
    *,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    extra_metrics: (
        Mapping[str, float] | Callable[[dict[str, float]], Mapping[str, float]] | None
    ) = None,
    update_idea_status: bool = True,
    check_lookahead: bool = True,
    check_sources: bool = True,
) -> Evaluation:
    """Run `spec` end to end: data -> weights -> backtest -> metrics -> verdict.

    Unless `check_sources` is False, every number and choice in the spec must
    cite a checkable source (`qlab.pipeline.sources`) or the run does not
    start: `SpecSourceError` is raised and no trial is written.

    Unless `check_lookahead` is False, the strategy is run again on panels
    whose data after a cut was changed, and its weights up to the cut must
    not move (`qlab.harness.lookahead`); a strategy that reads the future
    becomes an `error` trial with the bar named. Only calibration noise, whose
    generators are q-lab's own and tested, passes False.

    The run's route is stored on its trial and, unless `update_idea_status`
    is False, applied to the idea's status in the same transaction
    (`qlab.registry.lifecycle.apply_route`, docs/TASKS.md T31). Only
    calibration noise passes False: a noise idea is a measuring instrument,
    and a "rejected" noise idea would read in the funnel as a finding.

    Before any computation, `not_evaluable_reasons` checks whether this run
    can test the strategy its idea names at all (T24). If not, a trial with
    status and route `not-evaluable` is written, no backtest is run and no
    verdict rows are written -- there is no number to judge. When the spec
    declares `required_instruments`, the book's coverage is measured and
    stored as `book_coverage`, and the backtest is evaluated only on the
    longest unbroken stretch where the whole book existed
    (`complete_book_window`); the verdict's data range is that stretch.
    Weights are still computed on the full panel, so lookback warm-up is
    not lost at the stretch's start.

    `extra_metrics` (docs/TASKS.md T28, the shape-aware admission bar) lets a
    caller fold metrics this pipeline cannot compute on its own into the
    SAME metrics mapping the rules engine sees and the SAME trial row that
    gets persisted -- rather than evaluating rules a second time out of
    band, which would produce a second, competing verdict for one trial.
    The motivating case: a percentile against matched noise
    (`qlab.calibration.percentile.compute_shape_aware_percentiles`) needs
    this run's OWN `ann_return_net`/`sharpe_net` to be computed first, so it
    cannot be supplied as a plain dict up front -- pass a callable, invoked
    with the metrics dict exactly as computed below (already including
    `min_capital_usd`, `point_in_time_universe`, `accrual_applied`, and the
    venue facts) and expected to return the extra keys to merge in. A plain
    `Mapping` is accepted too, for metrics that do not depend on this run's
    own numbers.

    Invoked, and merged via `dict.update`, INSIDE the same try/except this
    function already uses to isolate strategy/backtest failures: if
    computing the extra metrics raises, this run is recorded as an `error`
    trial exactly like a strategy that raises or a backtest that fails --
    it is one more way "the metrics needed for a verdict were unobtainable
    this run", not a different failure class needing its own handling.

    A key `extra_metrics` chooses not to return for some run (e.g. no
    usable matched-noise sample, `ShapeAwarePercentiles.return_percentile is
    None`) must simply be ABSENT from the returned mapping, never present
    with a `None`/`NaN` value -- `qlab.rules.engine.evaluate` treats a
    missing key exactly like an explicit `None` (metric "unknown", never a
    pass), but a `NaN` value would instead compare `False` against every
    threshold and read as an ordinary rule failure, which is not the same
    finding as "this could not be computed" (docs/REGISTRY.md's `unknown`
    verdict kind exists precisely to keep those two apart).

    Pipeline order (a deliberate deviation from a strictly literal reading
    of this task's step list, forced by `docs/REGISTRY.md`'s own schema):
    the registry `spec` row is resolved first (pure DB work, needed for
    `trial.spec_id`), then the market data snapshot is resolved, and ONLY
    THEN is the strategy code resolved and run. `trial.snapshot_id` is
    required whenever `trial.source == 'qlab'`
    (`ck_trial_snapshot_required_unless_imported` in
    `qlab.registry.models.Trial`), so a trial row can only be legally
    written once a snapshot exists. Resolving data before code means a
    missing/broken `code_ref` — or the strategy raising, or an invalid
    weights frame, or a backtest/metrics failure — all land inside the
    error-handling block below, where a snapshot (and therefore a valid
    `trial` row) is already guaranteed to exist. A failure to obtain a
    snapshot at all (bad source name, network failure) is the one failure
    mode this function does NOT convert into an `error` trial: there is no
    honest `qlab`-sourced trial row without a data anchor, so that exception
    propagates instead of being written as a schema-violating row.

    Always writes exactly one `trial` row. Writes one `verdict` row per rule
    the ruleset actually evaluated, but only on a successful run — a failed
    run has no metrics to hand the rules engine, so there is nothing
    honest to record as a rule verdict (see docs/REGISTRY.md's three
    legitimate verdict kinds; there is no fourth kind for "run crashed").
    """
    if check_sources:
        # Before anything else: a spec with a number from nowhere does not
        # start (docs/SOURCES.md), so no trial is written for it.
        require_sources(spec)
    spec_row = _get_or_create_spec_row(session, spec)
    panel = resolve_panel(session, spec)

    range_start = pd.Timestamp(panel.meta["range_start"]).date()
    range_end = pd.Timestamp(panel.meta["range_end"]).date()

    truncated = dict(panel.meta.get("history_truncated") or {})
    if truncated:
        # The venue served only its last N candles (T25): before the latest
        # truncated instrument's first served bar, instruments that were
        # alive are missing. Cutting the judged window is not enough -- a
        # strategy's own warm-up would still start early on whatever history
        # happened to be served further back, which is the history of
        # instruments that died sooner. So EVERY instrument starts at the same
        # bar: the panel itself begins there, before any weight is computed.
        honest_from = pd.Timestamp(max(truncated.values()))
        cut = int(panel.prices.index.searchsorted(honest_from))
        panel = _slice_rows(panel, cut, len(panel.prices.index) - 1)
        range_start = panel.prices.index[0].date()
    started_at = datetime.now(UTC)
    config_hash = _config_hash(spec.params)
    code_sha = _code_sha()

    def _record(
        *,
        status: TrialStatus,
        metrics: dict[str, float] | None,
        routing: RoutingDecision,
        data_end: date | None,
    ) -> tuple[int, StatusDecision | None]:
        trial = repo.add_trial(
            session,
            spec_id=spec_row.id,
            config_hash=config_hash,
            code_sha=code_sha,
            params=spec.params,
            status=status,
            snapshot_id=panel.snapshot_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            metrics=metrics,
            source=TrialSource.QLAB,
            route=TrialRoute(routing.route),
            route_reason=routing.reason,
        )
        decision = None
        if update_idea_status:
            decision = apply_route(
                session,
                idea_id=spec.idea_id,
                route=TrialRoute(routing.route),
                route_reason=routing.reason,
                trial_id=trial.id,
                rules_version=ruleset.version,
                run_data_end=data_end,
            )
        return trial.id, decision

    coverage = (
        complete_book_window(panel, spec.required_instruments)
        if spec.required_instruments is not None
        else None
    )
    reasons = not_evaluable_reasons(spec, coverage, panel.meta)
    reasons += review_reasons(session, spec)
    exploratory = bool(reasons) and spec.exploratory is not None
    if reasons and not exploratory:
        reason = "; ".join(reasons)
        if coverage is not None and not coverage.missing:
            reason += f" (complete book on {coverage.coverage:.1%} of bars)"
        routing = RoutingDecision(route="not-evaluable", reason=reason)
        measured = {"book_coverage": coverage.coverage} if coverage is not None else None
        trial_id, decision = _record(
            status=TrialStatus.NOT_EVALUABLE, metrics=measured, routing=routing, data_end=None
        )
        return Evaluation(
            trial_id=trial_id,
            metrics=None,
            rules_result=None,
            routing=routing,
            error=None,
            status_decision=decision,
        )

    try:
        strategy = resolve_strategy(spec.code_ref)
        valid = getattr(strategy, "valid_intervals", None)
        if valid is not None and spec.data.interval not in valid:
            raise _NotValidOnInterval(
                f"strategy {spec.code_ref} is not valid on {spec.data.interval!r} bars "
                f"(declares valid_intervals={tuple(valid)}): its params would not mean "
                "what they say"
            )
        weights = strategy.target_weights(panel, spec.params)
        validate_weights(panel, weights)
        if check_lookahead:
            # Every candidate, every run (docs/LOOKAHEAD.md). A fresh instance,
            # so a strategy that remembers its last run (`diagnostics`) still
            # reports the untampered one.
            violation = lookahead_violation(
                resolve_strategy(spec.code_ref), panel, spec.params, weights
            )
            if violation is not None:
                raise ValueError(violation)

        first, last = 0, len(panel.prices.index) - 1
        if coverage is not None and coverage.coverage < 1.0 and coverage.window is not None:
            first, last = coverage.window

        costs = CostModel(
            taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps
        )
        index = panel.prices.index
        bar = INTERVAL_TO_TIMEDELTA.get(spec.data.interval)

        def _measure(lo: int, hi: int, *, forward_part: bool = False) -> dict[str, float]:
            """Backtest metrics plus the facts the rules read, on rows lo..hi."""
            if (lo, hi) == (0, len(index) - 1):
                part_panel, part_weights = panel, weights
            else:
                part_panel = _slice_rows(panel, lo, hi)
                part_weights = weights.iloc[lo : hi + 1]
            result = run_backtest(part_panel, part_weights, costs, part_panel.funding)
            measured = compute_metrics(result)
            # How the result splits across BTC's bull / flat / bear days
            # (docs/REGIMES.md): a description, read by no rule.
            regime_series = _regimes()
            if regime_series is not None and len(result.net_return) >= 2:
                measured.update(regime_breakdown(
                    result.net_return, regime_series, periods_per_year(part_panel.prices.index)
                ))
                measured.update(regime_claim_checks(spec.regime_claim, measured))
            # The capital a strategy needs is the strategy's, not one part's:
            # a part in which it held nothing (trend's two levels sat in cash
            # through their first forward days, 2026-10-06) is a legitimate
            # stretch, measured with the whole run's book. A book flat
            # everywhere still raises.
            flat = not (part_weights.abs() > 0).to_numpy().any()
            measured["min_capital_usd"] = min_capital_usd(
                weights if flat else part_weights, spec.min_leg_notional)
            if coverage is not None:
                measured["book_coverage"] = coverage.coverage
            if truncated:
                measured["history_truncated_instruments"] = float(len(truncated))
            # A fixed instrument list chosen on `params_fixed_at` cannot know
            # what happened after that date: on the forward test it is a
            # point-in-time universe by construction (docs/FIT_VS_FORWARD.md).
            # A discovered universe keeps the panel's own answer, which also
            # covers instruments the venue erased.
            honest = bool(panel.meta.get("universe_complete")) or (
                forward_part and spec.data.instruments is not None
            )
            measured["point_in_time_universe"] = 1.0 if honest else 0.0
            # This pipeline always runs the backtest with the panel's real
            # funding as accrual (never `NO_ACCRUAL`, see the call above) —
            # `accrual_applied` is therefore honestly always 1 for any run that
            # reaches this point, not a fabricated pass-through value.
            measured["accrual_applied"] = 1.0
            # venue_supported / data_forward_available / atomic_execution
            # (docs/TASKS.md T13): infrastructure facts a backtest cannot
            # derive, read from `venues/<source>.yaml` (qlab.venues.config) and
            # combined with this spec's `simultaneous_legs`
            # (qlab.venues.derive.derive_venue_metrics). A venue with no config
            # file, or a fact left unset in it, is silently absent from
            # `metrics` here -- never guessed -- so `qlab.rules.engine.evaluate`
            # honestly reports the corresponding rule as unknown rather than
            # passing or failing it.
            measured.update(
                derive_venue_metrics(
                    load_venue(spec.data.source),
                    snapshot_source=str(panel.meta.get("venue", spec.data.source)),
                    venue_id=spec.data.source,
                    simultaneous_legs=spec.simultaneous_legs,
                )
            )
            return measured

        def _days(part: tuple[int, int]) -> float:
            span = index[part[1]] - index[part[0]] + (bar or pd.Timedelta(0))
            return span / pd.Timedelta(days=1)

        split_mode = ruleset.forward_resolution is not None
        split = (
            split_at_fixed_date(index, first, last, spec.params_fixed_at, spec.params_fit_from)
            if split_mode
            else PeriodSplit(selection=None, forward=(first, last))
        )
        warmup = None
        if split_mode and spec.selects_causally:
            # The strategy picks its own parameters causally (T38): before its
            # first choice it is warming up, not fitted -- nothing to judge
            # there, and no selection period to protect.
            warmup, split = split.selection, PeriodSplit(selection=None, forward=split.forward)
        judged = split.forward if split.forward is not None else split.selection
        if judged is None:  # the whole window is shorter than two bars
            judged = (first, last)
        metrics = _measure(*judged, forward_part=split_mode and split.forward is not None)
        if judged != (0, len(index) - 1):
            range_start = index[judged[0]].date()
            range_end = index[judged[1]].date()

        selection_metrics: dict[str, float] | None = None
        if split_mode:
            if split.forward is not None and split.selection is not None:
                selection_metrics = _measure(*split.selection)
                metrics.update({f"fit_{k}": v for k, v in selection_metrics.items()})
            elif split.forward is None:
                selection_metrics = metrics
            if split.before is not None:
                # Data the choice did not see, but from before it: reported,
                # not judged -- an instrument list chosen later may know who
                # survived (`StrategySpec.params_fit_from`).
                before_metrics = _measure(*split.before)
                metrics.update({f"pre_{k}": v for k, v in before_metrics.items()
                                if k in PRE_FIT_KEYS or k.startswith("regime_")})
                metrics["pre_days"] = _days(split.before)
            if warmup is not None:
                metrics["warmup_days"] = _days(warmup)
            metrics["judged_on_forward"] = 1.0 if split.forward is not None else 0.0
            metrics["selection_days"] = _days(split.selection) if split.selection else 0.0
            metrics["forward_days"] = _days(split.forward) if split.forward else 0.0
            if split.forward is not None:
                # The claim a forward test must resolve: the selection period's
                # Sharpe, or with no selection period the forward test's own.
                claim = (selection_metrics or metrics).get("sharpe_net")
                years = forward_years_needed(claim, ruleset.forward_resolution)
                if years is not None:
                    metrics["forward_days_needed"] = years * 365.0
        # A strategy may report facts about its own run (a re-tuning wrapper:
        # how many re-tunings, how often the choice changed -- T38).
        diagnostics = getattr(strategy, "diagnostics", None)
        if callable(diagnostics):
            metrics.update(diagnostics())
        if extra_metrics is not None:
            resolved_extra = extra_metrics(metrics) if callable(extra_metrics) else extra_metrics
            metrics.update(resolved_extra)
    except _NotValidOnInterval as exc:
        routing = RoutingDecision(route="not-evaluable", reason=str(exc))
        trial_id, decision = _record(
            status=TrialStatus.NOT_EVALUABLE, metrics=None, routing=routing, data_end=None
        )
        return Evaluation(
            trial_id=trial_id,
            metrics=None,
            rules_result=None,
            routing=routing,
            error=None,
            status_decision=decision,
        )
    except Exception as exc:  # noqa: BLE001 - strategy code is arbitrary; this
        # is the pipeline's error-isolation boundary. Anything from here on
        # (a missing code_ref, a strategy that raises, invalid weights, a
        # backtest/metrics failure) must still produce a trial row -- see
        # this function's docstring -- so it is caught as broadly as
        # `docs/REGISTRY.md`'s "trial written for every run" promise
        # requires, not narrowed to the exception types this module happens
        # to know about today.
        routing = RoutingDecision(route="error", reason=str(exc))
        trial_id, decision = _record(
            status=TrialStatus.ERROR, metrics=None, routing=routing, data_end=None
        )
        return Evaluation(
            trial_id=trial_id,
            metrics=None,
            rules_result=None,
            routing=routing,
            error=str(exc),
            status_decision=decision,
        )

    # Rules and route are pure functions of `metrics`, so both are decided
    # before the trial is written: the row carries its route from the moment
    # it exists, and the status change below sees a complete trial.
    rules_result = evaluate_rules(metrics, ruleset)
    selection_result: EvaluationResult | None = None
    if split_mode:
        if split.forward is None:
            # The judged window IS the selection period.
            rules_result = evaluate_rules(metrics, _for_selection_period(ruleset))
            selection_result = rules_result
        elif selection_metrics is not None:
            selection_result = evaluate_rules(selection_metrics, _for_selection_period(ruleset))
        routing = decide_fit_forward_route(
            ruleset=ruleset,
            selection=selection_result,
            forward=rules_result if split.forward is not None else None,
            metrics=metrics,
            deployable_capital_usd=deployable_capital_usd,
            params_fixed_at=spec.params_fixed_at,
        )
    else:
        routing = decide_route(rules_result, metrics, deployable_capital_usd)
    if exploratory:
        # Computed in full on the owner's say-so, but not a decision: keep the
        # rules' answer visible and the route where the reasons put it.
        routing = RoutingDecision(
            route="not-evaluable",
            reason=(
                f"exploratory run ({spec.exploratory}); rules alone would route "
                f"{routing.route} ({routing.reason}); not evaluable because: "
                + "; ".join(reasons)
            ),
        )

    trial_id, decision = _record(
        status=TrialStatus.OK, metrics=metrics, routing=routing, data_end=range_end
    )

    decided_at = datetime.now(UTC)

    def _verdict_rows(result: EvaluationResult, start: date, end: date, period: str | None):
        return [
            {
                "idea_id": spec.idea_id,
                "spec_id": spec_row.id,
                "trial_id": trial_id,
                "stage": row.stage.value,
                "rule_id": row.rule_id,
                "rules_version": row.rules_version,
                "metric": row.metric,
                "value": row.value,
                "comparator": row.comparator,
                "threshold": row.threshold,
                "passed": row.passed,
                "data_range_start": start,
                "data_range_end": end,
                "decided_at": decided_at,
                "note": "; ".join(n for n in (period, row.note) if n) or None,
                "source": TrialSource.QLAB,
            }
            for row in result.rows
        ]

    if not split_mode:
        verdict_rows = _verdict_rows(rules_result, range_start, range_end, None)
    elif split.forward is None:
        verdict_rows = _verdict_rows(rules_result, range_start, range_end, SELECTION_NOTE)
    else:
        verdict_rows = _verdict_rows(rules_result, range_start, range_end, FORWARD_NOTE)
        if selection_result is not None and split.selection is not None:
            verdict_rows += _verdict_rows(
                selection_result,
                index[split.selection[0]].date(),
                index[split.selection[1]].date(),
                SELECTION_NOTE,
            )
    if verdict_rows:
        repo.add_verdicts(session, verdict_rows)

    return Evaluation(
        trial_id=trial_id,
        metrics=metrics,
        rules_result=rules_result,
        routing=routing,
        error=None,
        status_decision=decision,
    )


__all__ = [
    "FORWARD_NOTE",
    "SELECTION_NOTE",
    "BookCoverage",
    "PeriodSplit",
    "decide_fit_forward_route",
    "forward_years_needed",
    "split_at_fixed_date",
    "Evaluation",
    "RoutingDecision",
    "StrategyResolutionError",
    "complete_book_window",
    "decide_route",
    "evaluate_spec",
    "not_evaluable_reasons",
    "resolve_panel",
    "resolve_strategy",
]
