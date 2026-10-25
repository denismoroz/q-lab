"""The second stage of the idea generator's evaluation (docs/TWO_STAGE.md, T45).

Stage 1 judges a generated spec as written, on Binance's long history (perpetuals
since 2019, delisted ones included: an honest universe). An idea that survives it
gets its forward check where q-lab's paper and live execution is: the SAME
mechanism -- the same blocks and the same parameter values, nothing refitted --
is built on Hyperliquid and judged there, with Hyperliquid's instruments, fees
and funding, on data the stage 1 fit never saw (everything after the
`params_fixed_at` stage 1 inherited). The idea needs BOTH stages to survive.

Everything here is code. No agent is called and no token is spent, so the token
budget guard has nothing to guard (a test pins that).

* `translate` -- the stage 1 spec -> the stage 2 spec (pure);
* `book_window` -- from which day the market holds enough coins for the portfolio
  rule, and whether it ever does (pure, on a panel);
* `combine` -- the idea's route from the two stages' routes (pure);
* `run_stage2` -- the whole step: child idea, translation, window, run, verdict,
  the parent's status, all of it written to the registry;
* `report_lines` -- the stages side by side, in plain Russian.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from qlab.pipeline.spec import SpecCosts, SpecData, StrategySpec
from qlab.registry.stages import CHILD_SUFFIX, NOTES_PREFIX

CONFIG = Path("night/stage2.yaml")
SPEC_DIR = Path("specs/generated/stage2")
COMPOSED_CODE_REF = "qlab.strategies.blocks:Composed"
RESULT_METRICS = ("ann_return_net", "sharpe_net", "max_dd", "noise_return_percentile",
                  "min_capital_usd", "forward_days", "forward_days_needed", "selection_days")
UNDECIDED = ("not-evaluable", "needs-more-data", "error")
"""Stage 2 routes that decided nothing: the idea waits."""


# --------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class Stage2Config:
    enabled: bool
    venue: str
    data_start: date
    taker_fee_bps: float
    slippage_bps: float
    cost_sources: Mapping[str, str]
    stage1_survives: tuple[str, ...]


def load_config(path: Path = CONFIG) -> Stage2Config:
    """`night/stage2.yaml`. A missing key is an error: nothing here has a default
    value, for the same reason a spec's costs have none."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        costs = raw["costs"]
        config = Stage2Config(
            enabled=bool(raw["enabled"]), venue=str(raw["venue"]),
            data_start=date.fromisoformat(str(raw["data_start"])),
            taker_fee_bps=float(costs["taker_fee_bps"]),
            slippage_bps=float(costs["slippage_bps"]),
            cost_sources={str(k): " ".join(str(v).split()) for k, v in costs["sources"].items()},
            stage1_survives=tuple(str(r) for r in raw["stage1_survives"]))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    absent = {"costs.taker_fee_bps", "costs.slippage_bps"} - set(config.cost_sources)
    if absent:
        raise ValueError(f"{path}: no source for {sorted(absent)}")
    return config


def survives(route: str | None, config: Stage2Config) -> bool:
    """Whether a stage 1 route goes on to stage 2."""
    return route in config.stage1_survives


# --------------------------------------------------------------------------
# Instruments: stage 1's names on the other venue

_UNIT_PREFIXES = ("1000000", "1000", "1M", "k")
SPOT_SUFFIX = "-SPOT"


def venue_name(name: str, venue_names: Collection[str]) -> str | None:
    """The column the other venue calls `name`, or None when it lists no such coin.

    Venues list a low-priced coin per many units under different names (Binance
    1000PEPE, 1000000MOG, 1MBABYDOGE; Hyperliquid kPEPE); the prefix is dropped
    and the coin looked up again (`coinmarketcap.venue_symbol`, the mapping the
    lab already uses for market-cap weights). Prices scale, returns do not."""
    from qlab.data.sources.coinmarketcap import venue_symbol

    suffix = SPOT_SUFFIX if name.endswith(SPOT_SUFFIX) else ""
    base = name[: -len(SPOT_SUFFIX)] if suffix else name
    # Spot columns are matched among spot columns, perpetuals among perpetuals.
    names = ({n[: -len(SPOT_SUFFIX)] for n in venue_names if n.endswith(SPOT_SUFFIX)} if suffix
             else {n for n in venue_names if not n.endswith(SPOT_SUFFIX)})
    candidates = [base]
    for prefix in _UNIT_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix):
            candidates.append(base[len(prefix):])
    for candidate in candidates:
        found = venue_symbol(candidate, names)
        if found is not None:
            return found + suffix
    return None


def map_instruments(instruments: list[str], venue_names: Collection[str]
                    ) -> tuple[list[str], list[str]]:
    """(mapped names without repeats, in order; the stage 1 names the venue lacks)."""
    mapped: list[str] = []
    dropped: list[str] = []
    for name in instruments:
        found = venue_name(name, venue_names)
        if found is None:
            dropped.append(name)
        elif found not in mapped:
            mapped.append(found)
    return mapped, dropped


# --------------------------------------------------------------------------
# The translation


@dataclass(frozen=True)
class Translation:
    spec: StrategySpec | None
    problems: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def translate(spec: StrategySpec, config: Stage2Config, *,
              venue_names: Collection[str] | None = None) -> Translation:
    """The stage 2 spec of a stage 1 spec.

    Kept exactly: the code, every parameter value (a deep copy, never an edit),
    the interval, the end of the data, `params_fixed_at` and its evidence, the
    fit window, the noise universe, the sources of the parameters. Switched: the
    venue (`data.source`), the start of the data (the later of stage 1's and the
    config's), the costs (Hyperliquid's, cited), the idea id and title.

    With `instruments` or `required_instruments` named, `venue_names` (what the
    venue lists) maps them; a coin it does not list is dropped from the list --
    not traded -- and named in `dropped`. `problems` non-empty means no spec could
    be made (the whole list is missing)."""
    problems: list[str] = []
    dropped: list[str] = []
    data = spec.data
    instruments = data.instruments
    if instruments is not None:
        if venue_names is None:
            raise ValueError("the spec names instruments: venue_names is needed to map them")
        instruments, dropped = map_instruments(instruments, venue_names)
        if not instruments:
            problems.append(f"none of the {len(data.instruments)} instruments of stage 1 is "
                            f"listed on {config.venue}")
    required = spec.required_instruments
    if required is not None and venue_names is not None:
        mapped = [venue_name(n, venue_names) or n for n in required]
        required = list(dict.fromkeys(mapped))  # an unmapped name stays: the run says it is absent
    if problems:
        return Translation(None, problems, dropped)

    child_data = SpecData(
        source=config.venue, interval=data.interval, include_spot=data.include_spot,
        spot_prices=data.spot_prices, spot_source=None,
        start=max(data.start, config.data_start), end=data.end, instruments=instruments,
        min_daily_volume_usd=data.min_daily_volume_usd)
    notes = [
        f"Stage 2 of the evaluation (docs/TWO_STAGE.md): the same mechanism and the same "
        f"parameter values as stage 1 ({spec.idea_id}) on {config.venue}'s own perpetual "
        f"contracts listed at each bar; nothing was refitted.",
        f"The costs are {config.venue}'s (published base-tier taker fee, frab's slippage "
        f"estimate), not stage 1's.",
    ]
    if dropped:
        notes.append(f"{len(dropped)} instrument(s) of stage 1 are not listed on {config.venue} "
                     f"and are not traded: {', '.join(dropped)}.")
    sources = {k: v for k, v in spec.sources.items() if k != "costs" and not k.startswith("costs.")}
    sources.update(config.cost_sources)
    evidence = spec.params_fixed_evidence
    child = spec.model_copy(update={
        "idea_id": f"{spec.idea_id}{CHILD_SUFFIX}",
        "title": f"{spec.title} (второй этап: {config.venue})",
        "params": copy.deepcopy(spec.params),
        "data": child_data,
        "costs": SpecCosts(taker_fee_bps=config.taker_fee_bps,
                           slippage_bps=config.slippage_bps),
        "required_instruments": required,
        "exploratory": None,
        "judged_from": None,
        "differs_from_source": [*spec.differs_from_source, *notes],
        "params_fixed_evidence": (None if evidence is None
                                  else f"Inherited from stage 1 ({spec.idea_id}): {evidence}"),
        "sources": sources,
    })
    return Translation(child, [], dropped)


# --------------------------------------------------------------------------
# From which day the market holds the book


@dataclass(frozen=True)
class BookWindow:
    needed: int
    block: str
    most: int
    judged_from: date | None
    judged_bars: int
    full_share: float | None
    reason: str | None
    """Why the mechanism cannot be tested on this market; None when it can."""


def book_window(panel, params: Mapping[str, object]) -> BookWindow:
    """From which bar the market has, at once, as many coins as the portfolio rule
    needs (`PORTFOLIO_MIN_COINS`), and whether it ever does.

    The panel may start later than the mechanism can trade (a coin needs
    `min_history_days` of prices before it is admitted): the first bars carry no
    book. Judging them as flat days would dilute the return and the Sharpe, so the
    judged window starts at the first bar with a full book (`StrategySpec.
    judged_from`); a market that never has one cannot test the mechanism, and
    that is "not evaluable", never a fail."""
    from qlab.strategies.blocks.composed import eligible_coins
    from qlab.strategies.blocks.portfolio import min_coins

    portfolio = params["portfolio"]
    block = str(portfolio["block"])  # type: ignore[index]
    needed = min_coins(portfolio)  # type: ignore[arg-type]
    count = eligible_coins(panel, params).sum(axis=1)
    most = int(count.max()) if len(count) else 0
    full = (count >= needed).to_numpy()
    if not full.any():
        return BookWindow(needed, block, most, None, 0, None, (
            f"the market never had {needed} admissible coins at once, which the portfolio "
            f"rule `{block}` needs to hold its book (most on any day: {most}), so the "
            "mechanism cannot be held there"))
    position = int(full.argmax())
    bars = len(full) - position
    share = float(full[position:].mean())
    if bars < 2:
        return BookWindow(needed, block, most, None, bars, share, (
            f"the market had {needed} admissible coins at once only on the last day of the "
            "data: too short to measure a return"))
    return BookWindow(needed, block, most, count.index[position].date(), bars, share, None)


# --------------------------------------------------------------------------
# Both stages together


@dataclass(frozen=True)
class Combined:
    route: str
    """The route that moves the idea's status (a `TrialRoute` value)."""
    reason: str


def combine(stage1_route: str, stage2_route: str | None, *, config: Stage2Config,
            stage2_reason: str = "") -> Combined:
    """The idea's route from the two stages. It needs both to survive:

    * a reject at either stage rejects (stage 2's with the data range of stage 2:
      the child's verdicts carry it, and `apply_route(also_ideas=...)` reads it);
    * stage 1 still waiting for forward data keeps the idea waiting, whatever
      stage 2 says (it passed nothing yet);
    * otherwise stage 2's route stands: it holds everything stage 1 checked, on the
      venue where the orders go (a stage 1 `needs-infrastructure` is Binance's
      missing adapter, which stage 2 settles);
    * a stage 2 that decided nothing (not evaluable, no data, a crash) never
      passes the idea: it waits, with the reason."""
    if stage1_route not in config.stage1_survives:
        return Combined(stage1_route, f"stage 1 {stage1_route}: stage 2 not run")
    if stage2_route == "reject":
        return Combined("reject", f"stage 2 ({config.venue}) rejected: {stage2_reason}")
    if stage2_route is None or stage2_route in UNDECIDED:
        why = stage2_reason or "stage 2 did not finish"
        return Combined("needs-forward", f"stage 1 {stage1_route}; stage 2 decided nothing "
                        f"({stage2_route or 'no run'}): {why}")
    if stage1_route == "needs-forward":
        return Combined("needs-forward", f"stage 1 still waits for forward data; stage 2 "
                        f"{stage2_route}: {stage2_reason}")
    return Combined(stage2_route, f"stage 1 {stage1_route}; stage 2 {stage2_route}: "
                    f"{stage2_reason}")


# --------------------------------------------------------------------------
# The step


@dataclass
class Stage2Result:
    parent_id: str
    idea_id: str
    stage1_route: str | None = None
    stage1_trial_id: int | None = None
    status: str = "skipped"
    """skipped | evaluated | not-evaluable | error"""
    measurement: bool = False
    """Run by hand on an idea that did not survive stage 1: a measurement of the
    mechanism on the second venue. The trial and the child's own verdicts are written
    (every run is), the parent's status is not touched."""
    skipped: str | None = None
    trial_id: int | None = None
    route: str | None = None
    reason: str | None = None
    metrics: dict = field(default_factory=dict)
    venue: str | None = None
    data_start: str | None = None
    data_end: str | None = None
    judged_from: str | None = None
    needed_coins: int | None = None
    most_coins: int | None = None
    full_share: float | None = None
    dropped: list = field(default_factory=list)
    spec_path: str | None = None
    combined_route: str | None = None
    combined_reason: str | None = None
    parent_status: str | None = None
    error: str | None = None


def write_spec(spec: StrategySpec, parent_id: str, spec_dir: Path) -> Path:
    """The stage 2 spec as a file, for whoever wants to read or re-run it."""
    path = spec_dir / f"{spec.idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False,
                          allow_unicode=True, width=100)
    path.write_text(
        f"# Stage 2 of {parent_id} (qlab.pipeline.stage2, docs/TWO_STAGE.md): made by code from\n"
        f"# the stage 1 spec. The parameter values are stage 1's, untouched; do not edit by hand.\n"
        + body, encoding="utf-8")
    return path


def _ensure_child(session, spec: StrategySpec, parent_id: str, *, measurement: bool) -> None:
    from qlab.registry import repo
    from qlab.registry.models import AssetClass, Idea

    parent = session.get(Idea, parent_id)
    if parent is None:
        raise ValueError(f"stage 2 of {parent_id!r}: the idea is not in the registry")
    child = session.get(Idea, spec.idea_id)
    if child is None:
        child = repo.upsert_idea(
            session, id=spec.idea_id, title=spec.title, source_type=parent.source_type,
            asset_class=AssetClass.CRYPTO_PERP, profile=parent.profile,
            notes=f"{NOTES_PREFIX}{parent_id} (docs/TWO_STAGE.md): the same mechanism and "
                  "parameters on Hyperliquid"
                  + ("; a MEASUREMENT run by hand on an idea that did not survive stage 1, "
                     "its parent's status is not touched" if measurement else ""))
    child.parent_id = parent_id
    session.flush()


def _settle(session, result: Stage2Result, combined: Combined, *, rules_version: str | None,
            end: date | None, parent_id: str, child_id: str) -> None:
    """Move the parent's status by the two stages together."""
    from qlab.registry.lifecycle import apply_route
    from qlab.registry.models import Idea, TrialRoute

    apply_route(
        session, idea_id=parent_id, route=TrialRoute(combined.route),
        route_reason=combined.reason, trial_id=result.trial_id, rules_version=rules_version,
        run_data_end=end, also_ideas=(child_id,))
    result.combined_route = combined.route
    result.combined_reason = combined.reason
    result.parent_status = session.get(Idea, parent_id).status.value


def run_stage2(spec: StrategySpec, *, stage1_route: str | None, stage1_trial_id: int | None,
               noise_trials: int, deployable_capital_usd: float,
               config: Stage2Config | None = None, spec_dir: Path = SPEC_DIR,
               venue_names: Collection[str] | None = None,
               measure: bool = False) -> Stage2Result:
    """Stage 2 of one idea whose stage 1 (`spec`, run as trial `stage1_trial_id`)
    routed `stage1_route`. Never raises: a failure is the result's `error`, and the
    idea waits (stage 2 is tried again on the next run).

    `measure=True` runs the stage on an idea that did not survive stage 1 (by hand,
    `qlab stage2 --measure`): the mechanism is measured on the second venue and the
    parent's status is left alone."""
    result = Stage2Result(parent_id=spec.idea_id, idea_id=f"{spec.idea_id}{CHILD_SUFFIX}",
                          stage1_route=stage1_route, stage1_trial_id=stage1_trial_id,
                          measurement=measure)
    try:
        config = config or load_config()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # Stage 1 is on record and must stay so: a broken config is this result's error.
        result.status = "error"
        result.error = f"night/stage2.yaml: {type(exc).__name__}: {exc}"
        return result
    result.venue = config.venue
    if not config.enabled:
        result.skipped = "stage 2 is switched off (night/stage2.yaml)"
        return result
    if not measure and not survives(stage1_route, config):
        result.skipped = f"stage 1 routed {stage1_route}: only survivors go on"
        return result
    try:
        _run(spec, result, config, noise_trials=noise_trials,
             deployable_capital_usd=deployable_capital_usd, spec_dir=spec_dir,
             venue_names=venue_names, measure=measure)
    except Exception as exc:  # noqa: BLE001 - recorded in the report, the idea waits
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        if not measure:
            _settle_after_error(result, config)
    return result


def _settle_after_error(result: Stage2Result, config: Stage2Config) -> None:
    """A stage 2 that crashed does not pass the idea: it waits (status bench)."""
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    combined = combine(result.stage1_route or "", None, config=config,
                       stage2_reason=result.error or "")
    try:
        with session_scope() as session:
            _settle(session, result, combined, rules_version=load_latest().version,
                    end=None, parent_id=result.parent_id, child_id=result.idea_id)
    except Exception as exc:  # noqa: BLE001 - the error above is the one worth reporting
        result.error = f"{result.error}; and the idea's status was not updated: {exc}"


def _run(spec: StrategySpec, result: Stage2Result, config: Stage2Config, *,
         noise_trials: int, deployable_capital_usd: float, spec_dir: Path,
         venue_names: Collection[str] | None, measure: bool) -> None:
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.pipeline.evaluate import evaluate_spec, resolve_panel
    from qlab.pipeline.sources import source_problems
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    if spec.data.instruments is not None or spec.required_instruments is not None:
        if venue_names is None:
            from qlab.data.sources.hyperliquid import discover_universe

            venue_names = discover_universe()
    translation = translate(spec, config, venue_names=venue_names)
    result.dropped = translation.dropped
    if translation.spec is None:
        result.status = "not-evaluable"
        result.reason = "; ".join(translation.problems)
        return  # nothing to anchor a trial on: no data was ever requested
    child = translation.spec
    ruleset = load_latest()

    with session_scope() as session:
        _ensure_child(session, child, spec.idea_id, measurement=measure)
        panel = resolve_panel(session, child)
        window = None
        if spec.code_ref == COMPOSED_CODE_REF:
            window = book_window(panel, child.params)
            result.needed_coins, result.most_coins = window.needed, window.most
            result.full_share = window.full_share
            if window.judged_from is not None:
                child = child.model_copy(update={
                    "judged_from": window.judged_from,
                    "differs_from_source": [*child.differs_from_source, (
                        f"Judged from {window.judged_from}: the first day {config.venue} has "
                        f"the {window.needed} coins the portfolio rule needs at once "
                        f"(snapshot {panel.snapshot_id}); the days before are warm-up with no "
                        "book.")]})
                result.judged_from = window.judged_from.isoformat()
        result.data_start = child.data.start.isoformat()
        result.data_end = child.data.end.isoformat()
        result.spec_path = str(write_spec(child, spec.idea_id, spec_dir))
        problems = source_problems(child)
        if problems:
            raise ValueError("the stage 2 spec cites no checkable source for: "
                             + "; ".join(problems))

        if window is not None and window.reason is not None:
            evaluation = evaluate_spec(
                child, session=session, ruleset=ruleset,
                deployable_capital_usd=deployable_capital_usd,
                not_evaluable_extra=[window.reason])
        else:
            evaluation = evaluate_spec_with_shape_aware_bar(
                child, session=session, ruleset=ruleset,
                deployable_capital_usd=deployable_capital_usd, n_trials=noise_trials).candidate
        result.trial_id = evaluation.trial_id
        result.route = str(evaluation.routing.route)
        result.reason = evaluation.routing.reason
        result.status = "not-evaluable" if result.route == "not-evaluable" else "evaluated"
        metrics = evaluation.metrics or {}
        result.metrics = {k: metrics[k] for k in RESULT_METRICS if k in metrics}
        if not measure:
            combined = combine(result.stage1_route or "", result.route, config=config,
                               stage2_reason=result.reason or "")
            _settle(session, result, combined, rules_version=ruleset.version,
                    end=child.data.end, parent_id=spec.idea_id, child_id=child.idea_id)


# --------------------------------------------------------------------------
# The morning report: both stages side by side, in plain Russian


def _pct(value: object) -> str:
    return f"{value:+.1%}" if isinstance(value, int | float) and value == value else "—"


def _route_ru(route: str | None) -> str:
    from qlab.funnel import ROUTE_TITLE

    return ROUTE_TITLE.get(route or "", route or "—")


def _numbers_ru(m: Mapping[str, object]) -> str:
    sharpe = m.get("sharpe_net")
    beat = m.get("noise_return_percentile")
    return (f"в год {_pct(m.get('ann_return_net'))}, "
            f"Шарп {f'{sharpe:.2f}' if isinstance(sharpe, int | float) else '—'}, "
            f"просадка {_pct(m.get('max_dd'))}, "
            f"обгоняет шум {_pct(beat).lstrip('+') if beat is not None else '—'}")


def report_lines(stage1: Mapping[str, object], stage2: Mapping[str, object] | None) -> list[str]:
    """The two stages of one idea as report lines. `stage1` holds `route` and `metrics`
    (a `GeneratedResult` as a dict); `stage2` is a `Stage2Result` as a dict, or None for
    a run made before stage 2 existed (no lines)."""
    if stage2 is None:
        return []
    lines: list[str] = []
    stage1_route = stage2.get("stage1_route") or stage1.get("route")
    if stage2.get("status") == "skipped":
        if stage2.get("skipped", "").startswith("stage 2 is switched off"):
            return ["второй этап (проверка на Hyperliquid) выключен в `night/stage2.yaml`"]
        return [f"второй этап (тот же механизм на Hyperliquid) не проводился: на первом этапе "
                f"исход «{_route_ru(str(stage1_route))}», а на Hyperliquid проверяют только "
                "тех, кто прошёл первый"]
    head = ("второй этап, Hyperliquid (тот же механизм и те же числа; комиссии Hyperliquid; "
            "код без агента, токены не тратятся)")
    if stage2.get("measurement"):
        head = ("ЗАМЕР на Hyperliquid (тот же механизм и те же числа; запущен вручную, статус "
                "идеи не меняет)")
    if stage2.get("status") == "error":
        lines.append(f"{head}: не завершился — {stage2.get('error')}; повторится при следующем "
                     "запуске, пока идея ждёт")
    elif stage2.get("status") == "not-evaluable":
        needed, most = stage2.get("needed_coins"), stage2.get("most_coins")
        why = (f"на Hyperliquid ни в один день не набиралось {needed} монет сразу, которые "
               f"нужны правилу портфеля (больше всего было {most})"
               if needed is not None and stage2.get("judged_from") is None
               and stage2.get("full_share") is None else stage2.get("reason"))
        lines.append(f"{head}: проверить нечем — {why}. Это не отказ: идея просто ждёт")
    else:
        from_day = stage2.get("judged_from")
        window = (f"считается с {from_day} — первого дня, когда на Hyperliquid хватает монет "
                  f"({stage2.get('needed_coins')} сразу), до {stage2.get('data_end')}"
                  if from_day else f"{stage2.get('data_start')} — {stage2.get('data_end')}")
        metrics = stage2.get("metrics") or {}
        days = metrics.get("forward_days")
        forward = (f"; проверка вперёд — {days:,.0f} дн. после даты фиксации параметров"
                   if isinstance(days, int | float) and days else "")
        numbers = f" — {_numbers_ru(metrics)}" if metrics else ""
        lines.append(f"{head}, прогон {stage2.get('trial_id')}: {_route_ru(stage2.get('route'))}"
                     f"{numbers}; {window}{forward}")
        if stage2.get("route") not in ("paper", "shelf"):
            lines.append(f"причина: {stage2.get('reason')}")
    if stage2.get("dropped"):
        lines.append("монет первого этапа, которых нет на Hyperliquid, не торгуем: "
                     f"{len(stage2['dropped'])}")  # type: ignore[arg-type]
    if stage2.get("combined_route") is not None:
        lines.append("итог по двум этапам: " + _combined_ru(stage1_route, stage2))
    return lines


def _combined_ru(stage1_route: object, stage2: Mapping[str, object]) -> str:
    route = stage2.get("combined_route")
    status = stage2.get("parent_status")
    if route == "reject":
        return (f"отвергнута на втором этапе — на данных Hyperliquid по {stage2.get('data_end')}; "
                "воскресить можно только на данных, которых тогда не было")
    if route in ("paper", "shelf", "needs-infrastructure") and stage2.get("route") == route:
        return (f"прошла оба этапа ({_route_ru(str(route))}), статус «{status}»")
    return (f"оба этапа вместе пока не решают: {_route_ru(str(route))}; статус «{status}». "
            "Пройти надо обоим")


__all__ = ["BookWindow", "CONFIG", "Combined", "SPEC_DIR", "Stage2Config", "Stage2Result",
           "Translation", "book_window", "combine", "load_config", "map_instruments",
           "report_lines", "run_stage2", "survives", "translate", "venue_name", "write_spec"]
