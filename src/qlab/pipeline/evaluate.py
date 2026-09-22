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

import hashlib
import importlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from qlab.data.panel import MarketPanel
from qlab.data.snapshot import build_snapshot, load_snapshot
from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics, min_capital_usd
from qlab.harness.run import run_backtest
from qlab.harness.strategy import Strategy, validate_weights
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import DataSnapshot, TrialSource, TrialStatus
from qlab.registry.models import Spec as SpecRow
from qlab.rules.engine import EvaluationResult
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.schema import RuleSet
from qlab.venues.config import load_venue
from qlab.venues.derive import derive_venue_metrics

Route = Literal["reject", "needs-more-data", "shelf", "paper", "error"]


class StrategyResolutionError(ImportError):
    """`spec.code_ref` does not resolve to a usable `Strategy`."""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """What `evaluate_spec` decided to do with a run, and why.

    `route` is never `"live"` — see `decide_route`'s docstring for why that
    is a structural guarantee, not an oversight. `"error"` is not one of the
    four routes the rules engine can produce; it means the run itself never
    reached the rules engine (see `Evaluation.error`).
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
        .all()
    )
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
        if wanted_instruments is None:
            if not manifest.get("universe_complete"):
                continue
        else:
            if manifest.get("universe_complete"):
                continue
            if sorted(manifest.get("instruments", [])) != wanted_instruments:
                continue
        return str(row.id)
    return None


def _resolve_panel(session: Session, data: StrategySpec) -> MarketPanel:
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
      2. `not result.overall_passed` -> `"reject"`. Every metric was
         computed and at least one rule failed — fatal or not. The task
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

    if not result.overall_passed:
        failed = ", ".join(result.failed_rule_ids) or "(none listed)"
        fatal_note = (
            f"; fatal rule: {result.failed_fatal_rule_id}"
            if result.failed_fatal_rule_id
            else ""
        )
        return RoutingDecision(route="reject", reason=f"rule(s) failed: {failed}{fatal_note}")

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


def evaluate_spec(
    spec: StrategySpec,
    *,
    session: Session,
    ruleset: RuleSet,
    deployable_capital_usd: float,
    extra_metrics: (
        Mapping[str, float] | Callable[[dict[str, float]], Mapping[str, float]] | None
    ) = None,
) -> Evaluation:
    """Run `spec` end to end: data -> weights -> backtest -> metrics -> verdict.

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
    spec_row = _get_or_create_spec_row(session, spec)
    panel = _resolve_panel(session, spec)

    range_start = pd.Timestamp(panel.meta["range_start"]).date()
    range_end = pd.Timestamp(panel.meta["range_end"]).date()
    started_at = datetime.now(UTC)
    config_hash = _config_hash(spec.params)
    code_sha = _code_sha()

    try:
        strategy = resolve_strategy(spec.code_ref)
        weights = strategy.target_weights(panel, spec.params)
        validate_weights(panel, weights)

        costs = CostModel(
            taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps
        )
        result = run_backtest(panel, weights, costs, panel.funding)

        metrics = compute_metrics(result)
        metrics["min_capital_usd"] = min_capital_usd(weights, spec.min_leg_notional)
        metrics["point_in_time_universe"] = (
            1.0 if bool(panel.meta.get("universe_complete")) else 0.0
        )
        # This pipeline always runs the backtest with the panel's real
        # funding as accrual (never `NO_ACCRUAL`, see the call above) —
        # `accrual_applied` is therefore honestly always 1 for any run that
        # reaches this point, not a fabricated pass-through value.
        metrics["accrual_applied"] = 1.0
        # venue_supported / data_forward_available / atomic_execution
        # (docs/TASKS.md T13): infrastructure facts a backtest cannot
        # derive, read from `venues/<source>.yaml` (qlab.venues.config) and
        # combined with this spec's `simultaneous_legs`
        # (qlab.venues.derive.derive_venue_metrics). A venue with no config
        # file, or a fact left unset in it, is silently absent from
        # `metrics` here -- never guessed -- so `qlab.rules.engine.evaluate`
        # honestly reports the corresponding rule as unknown rather than
        # passing or failing it.
        metrics.update(
            derive_venue_metrics(
                load_venue(spec.data.source),
                snapshot_source=str(panel.meta.get("venue", spec.data.source)),
                venue_id=spec.data.source,
                simultaneous_legs=spec.simultaneous_legs,
            )
        )
        if extra_metrics is not None:
            resolved_extra = extra_metrics(metrics) if callable(extra_metrics) else extra_metrics
            metrics.update(resolved_extra)
    except Exception as exc:  # noqa: BLE001 - strategy code is arbitrary; this
        # is the pipeline's error-isolation boundary. Anything from here on
        # (a missing code_ref, a strategy that raises, invalid weights, a
        # backtest/metrics failure) must still produce a trial row -- see
        # this function's docstring -- so it is caught as broadly as
        # `docs/REGISTRY.md`'s "trial written for every run" promise
        # requires, not narrowed to the exception types this module happens
        # to know about today.
        trial = repo.add_trial(
            session,
            spec_id=spec_row.id,
            config_hash=config_hash,
            code_sha=code_sha,
            params=spec.params,
            status=TrialStatus.ERROR,
            snapshot_id=panel.snapshot_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            metrics=None,
            source=TrialSource.QLAB,
        )
        return Evaluation(
            trial_id=trial.id,
            metrics=None,
            rules_result=None,
            routing=RoutingDecision(route="error", reason=str(exc)),
            error=str(exc),
        )

    trial = repo.add_trial(
        session,
        spec_id=spec_row.id,
        config_hash=config_hash,
        code_sha=code_sha,
        params=spec.params,
        status=TrialStatus.OK,
        snapshot_id=panel.snapshot_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        metrics=metrics,
        source=TrialSource.QLAB,
    )

    rules_result = evaluate_rules(metrics, ruleset)

    decided_at = datetime.now(UTC)
    verdict_rows = [
        {
            "idea_id": spec.idea_id,
            "spec_id": spec_row.id,
            "trial_id": trial.id,
            "stage": row.stage.value,
            "rule_id": row.rule_id,
            "rules_version": row.rules_version,
            "metric": row.metric,
            "value": row.value,
            "comparator": row.comparator,
            "threshold": row.threshold,
            "passed": row.passed,
            "data_range_start": range_start,
            "data_range_end": range_end,
            "decided_at": decided_at,
            "note": row.note,
            "source": TrialSource.QLAB,
        }
        for row in rules_result.rows
    ]
    if verdict_rows:
        repo.add_verdicts(session, verdict_rows)

    routing = decide_route(rules_result, metrics, deployable_capital_usd)

    return Evaluation(
        trial_id=trial.id,
        metrics=metrics,
        rules_result=rules_result,
        routing=routing,
        error=None,
    )


__all__ = [
    "Evaluation",
    "RoutingDecision",
    "StrategyResolutionError",
    "decide_route",
    "evaluate_spec",
    "resolve_strategy",
]
