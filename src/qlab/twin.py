"""The causal twin of a strategy spec (docs/RETUNE.md, «Двойник: числа выбирает код»).

An idea whose numbers a person or an agent chose recently (`params_fixed_at` is a recent date)
has to wait for data that comes after that date: the history before it was seen when the
numbers were chosen. If the numbers are chosen BY CODE instead, every month, from the past
only (`qlab.strategies.retune.Retune`, `selects_causally`), then every day after the first
choice is a forward test, and Binance's history from 2019 gives years of it at once. The
researcher built such a twin by hand for XSMOM (specs/research/xsmom/
xsmom-research-nobull-retune.yaml); this module builds it for any spec.

What the code chooses between is DECLARED, never invented:

1. a research experiment and the strategy spec it modifies (`base`): the two are the two
   candidates, the base's numbers unchanged and the experiment's numbers. An experiment that
   wraps the base in `RegimeGate` is the same pair: the base is the wrapper with every regime
   traded;
2. the parameters for which `night/own_numbers.yaml` declares a list: the WHOLE list, as
   `qlab.own_numbers` does, within `max_candidates`;
3. nothing of the two: no twin, and the reason says so.

No agent is called, no token is spent, no database is touched: the module only reads specs and
the declared lists, and writes a spec file when asked.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

from qlab.own_numbers import CONFIG, RETUNE, _label, load_config, open_params, problems
from qlab.pipeline.sources import QLAB_ROOT, covering_key
from qlab.pipeline.spec import SpecData, StrategySpec
from qlab.regimes import WINDOW_DAYS
from qlab.strategies.retune import expand_grid

TWIN_SUFFIX = "-causal"
BINANCE_ARCHIVE_START = date(2019, 9, 8)
"""The first day of Binance's perpetual candles in the lab's archive (docs/VENUES.md)."""
TWIN_VENUE = "binance"
REGIME_GATE = "qlab.strategies.regime_gate:RegimeGate"
ALL_REGIMES = ["bull", "flat", "bear"]
RETUNE_DOC = "docs/RETUNE.md"
VENUES_DOC = "docs/VENUES.md"
MONEY_SIZES = ("budget_cap_usdc", "capital_usd", "strategy_capital_usd")
"""Parameters that say how much money the book is: the owner chooses them (CLAUDE.md, the
strategy's money), code does not pick them by the Sharpe they happened to give."""
PERIODS = {"M": ("month", "раз в месяц"), "W": ("week", "раз в неделю"),
           "D": ("day", "каждый день")}
NO_ALTERNATIVE = ("the numbers were picked by hand and there is no declared alternative; the "
                  "history before params_fixed_at was seen when choosing them")


@dataclass(frozen=True)
class TwinResult:
    """The twin spec, or the plain reason why none can be made (exactly one of the two)."""

    spec: StrategySpec | None
    reason: str | None = None
    kind: str | None = None
    """`pair` (a base and its experiment), `wrapper` (a regime gate over the base) or
    `lists` (the whole declared lists of night/own_numbers.yaml)."""
    candidates: tuple[str, ...] = ()
    origin: str | None = None

    def __post_init__(self) -> None:
        if (self.spec is None) == (self.reason is None):
            raise ValueError("a twin result carries a spec or a reason, not both and not none")


@dataclass(frozen=True)
class BaseChoice:
    """`find_base`: the base spec of an experiment, or why none could be picked."""

    base: StrategySpec | None
    reason: str | None = None


@dataclass(frozen=True)
class _Grid:
    kind: str
    inner_code_ref: str
    base_params: dict
    grid: dict
    candidates: dict[str, dict]
    """{candidate name: params} exactly as `Retune` expands them."""
    sources: dict[str, str] = field(default_factory=dict)
    trees: list[dict] = field(default_factory=list)
    """Parameter trees whose day, hour and window values bound the warm-up."""
    description: str = ""
    caveat: str = ""
    differs: str = ""


# --------------------------------------------------------------------------
# Small readers


def normalize_ref(code_ref: str) -> str:
    """`module:attr` for both spellings a spec may use."""
    if ":" in code_ref:
        return code_ref
    module, _, attr = code_ref.rpartition(".")
    return f"{module}:{attr}"


def _code_file(code_ref: str) -> str | None:
    module = normalize_ref(code_ref).split(":")[0].replace(".", "/")
    for tail in (".py", "/__init__.py"):
        if (QLAB_ROOT / "src" / f"{module}{tail}").is_file():
            return f"src/{module}{tail}"
    return None


def _leaf_items(value: object, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaf_items(item, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def _get(params: Mapping, path: str) -> tuple[bool, object]:
    node: object = params
    for name in path.split("."):
        if not isinstance(node, Mapping) or name not in node:
            return False, None
        node = node[name]
    return True, node


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _plain(value: float) -> int | float:
    return int(value) if value == int(value) else value


def differing_leaves(a: Mapping, b: Mapping) -> int:
    """How many parameter leaves differ between two params (a list is one leaf; a key present
    on one side only counts once)."""
    left, right = dict(_leaf_items(a)), dict(_leaf_items(b))
    return sum(1 for path in left.keys() | right.keys() if left.get(path, _MISSING)
               != right.get(path, _MISSING))


_MISSING = object()


# --------------------------------------------------------------------------
# What code chooses between: a base and its experiment


def _overrides(base: Mapping, exp: Mapping) -> tuple[dict[str, object], str | None]:
    """The dotted overrides that turn `base` into `exp`, or why that cannot be said as
    overrides. A key the experiment drops is expressed by replacing the dict that held it (an
    override can set a value, not delete one); at the top level that is impossible."""
    out: dict[str, object] = {}
    dropped_in: set[str] = set()

    def walk(a: Mapping, b: Mapping, prefix: str) -> None:
        for key in b:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a:
                out[path] = copy.deepcopy(b[key])
            elif isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
                walk(a[key], b[key], path)
            elif a[key] != b[key]:
                out[path] = copy.deepcopy(b[key])
        if any(key not in b for key in a):
            dropped_in.add(prefix)

    walk(base, exp, "")
    if "" in dropped_in:
        return {}, ("the experiment drops a top-level parameter of the base, which an "
                    "override cannot say")
    for parent in sorted(dropped_in, key=len):
        if any(parent == p or parent.startswith(p + ".") for p in dropped_in if p != parent):
            continue
        found, whole = _get(exp, parent)
        out = {p: v for p, v in out.items() if not p.startswith(parent + ".") and p != parent}
        out[parent] = copy.deepcopy(whole) if found else None
    return out, None


def _carry_sources(sources: Mapping[str, str], override: str, into: str) -> dict[str, str]:
    """The citations of the spec that cover the override's path, moved under `into`."""
    out: dict[str, str] = {}
    key = covering_key(override, sources)
    if key is not None and key != override and not key.startswith(override + "."):
        out[f"{into}.{override}"] = sources[key]
    for k, citation in sources.items():
        if k == override or k.startswith(override + "."):
            out[f"{into}.{k}"] = citation
    return out


def _pair_grid(spec: StrategySpec, base: StrategySpec, spec_path: str | None,
               base_path: str | None) -> tuple[_Grid | None, str | None]:
    """The two-candidate grid {base's numbers, experiment's numbers}, or why not."""
    if base.idea_id == spec.idea_id:
        return None, "the base is the spec itself: there is nothing to choose between"
    if normalize_ref(base.code_ref) == RETUNE or base.selects_causally:
        return None, f"the base {base.idea_id} already chooses its numbers by code"
    mismatch = _setup_mismatch(spec, base)
    if mismatch:
        return None, mismatch
    base_cite = base_path or RETUNE_DOC
    spec_cite = spec_path or RETUNE_DOC
    if normalize_ref(spec.code_ref) == normalize_ref(base.code_ref):
        return _same_code_pair(spec, base, base_cite, spec_cite)
    return _wrapper_pair(spec, base, base_cite, spec_cite)


def _setup_mismatch(spec: StrategySpec, base: StrategySpec) -> str | None:
    """Both candidates run on the twin's one data and costs, so an experiment built on other
    data, costs or leg size than its base is not a choice between two numbers."""
    mine, theirs = spec.data.model_dump(), base.data.model_dump()
    differing = sorted(k for k in mine if k not in ("start", "end") and mine[k] != theirs[k])
    if spec.costs != base.costs:
        differing.append("costs")
    for name in ("min_leg_notional", "simultaneous_legs", "required_instruments"):
        if getattr(spec, name) != getattr(base, name):
            differing.append(name)
    if differing:
        return (f"{spec.idea_id} and {base.idea_id} differ in {', '.join(differing)}, not in "
                "numbers alone: the twin has one data set and one set of costs")
    return None


def _same_code_pair(spec: StrategySpec, base: StrategySpec, base_cite: str, spec_cite: str
                    ) -> tuple[_Grid | None, str | None]:
    overrides, problem = _overrides(base.params, spec.params)
    if problem:
        return None, problem
    if not overrides:
        return None, (f"{spec.idea_id} and {base.idea_id} have the same numbers: there is "
                      "nothing to choose between")
    sized = sorted(p for p in overrides if p.split(".")[-1] in MONEY_SIZES)
    if sized:
        return None, (f"the experiment differs from {base.idea_id} in the size of the money "
                      f"({', '.join(sized)}): the owner chooses the size, code does not pick "
                      "it by the Sharpe it gave")
    grid = {"variant": {"base": {}, "experiment": overrides}}
    base_params = copy.deepcopy(base.params)
    candidates = expand_grid(base_params, grid)
    if candidates["experiment"] != spec.params:
        return None, "the overrides do not rebuild the experiment's numbers exactly"
    sources = {f"base_params.{k}": v for k, v in base.sources.items()
               if _param_source(k, base.params)}
    sources["inner_code_ref"] = (f"{_code_file(spec.code_ref) or RETUNE_DOC}: the strategy both "
                                 f"{base.idea_id} and {spec.idea_id} run")
    sources["grid.variant.base"] = f"{base_cite}: the base's numbers, unchanged"
    for path in overrides:
        sources.update(_carry_sources(spec.sources, path, "grid.variant.experiment"))
    names = ", ".join(sorted(overrides))
    return _Grid(
        kind="pair", inner_code_ref=spec.code_ref, base_params=base_params, grid=grid,
        candidates=candidates, sources=sources, trees=[base.params, spec.params],
        description=(f"two candidates declared whole: the numbers of {base.idea_id} unchanged, "
                     f"and the numbers of {spec.idea_id} ({names})"),
        caveat=_seen_caveat(spec),
        differs=f"the base {base.idea_id} against its experiment ({names})"), None


def _unwrap(spec: StrategySpec) -> tuple[str, dict] | None:
    """(inner code_ref, inner params) of a known wrapper whose 'unchanged' base is one of its
    own candidates; None when the spec is no such wrapper."""
    if normalize_ref(spec.code_ref) != REGIME_GATE:
        return None
    params = spec.params
    if not all(key in params for key in ("inner_code_ref", "inner_params", "trade_in")):
        return None
    if str(params.get("detector", "direction_terciles")).startswith("meta:"):
        return None
    return normalize_ref(str(params["inner_code_ref"])), dict(params["inner_params"])


def _wrapper_pair(spec: StrategySpec, base: StrategySpec, base_cite: str, spec_cite: str
                  ) -> tuple[_Grid | None, str | None]:
    unwrapped = _unwrap(spec)
    if unwrapped is None:
        return None, (f"{spec.idea_id} and {base.idea_id} run different code "
                      f"({spec.code_ref}, {base.code_ref}) and {spec.code_ref} is not a wrapper "
                      "whose unchanged base is one of its own candidates")
    inner_ref, inner_params = unwrapped
    if inner_ref != normalize_ref(base.code_ref):
        return None, (f"{spec.idea_id} wraps {inner_ref}, not the base's {base.code_ref}")
    if inner_params != base.params:
        return None, (f"the strategy {spec.idea_id} wraps has other numbers than {base.idea_id} "
                      f"({differing_leaves(inner_params, base.params)} differ): the wrapper is "
                      "not the base plus one choice")
    trade_in = list(spec.params["trade_in"])  # type: ignore[call-overload]
    if sorted(trade_in) == sorted(ALL_REGIMES):
        return None, "the gate already trades in every regime: it is the base"
    name = "only_" + "_".join(trade_in)
    base_params = {k: copy.deepcopy(v) for k, v in spec.params.items() if k != "trade_in"}
    grid = {"gate": {"always": {"trade_in": list(ALL_REGIMES)}, name: {"trade_in": trade_in}}}
    candidates = expand_grid(base_params, grid)
    if candidates[name] != spec.params:
        return None, "the overrides do not rebuild the experiment's numbers exactly"
    sources = {f"base_params.{k}": v for k, v in spec.sources.items()
               if k != "trade_in" and _param_source(k, spec.params)}
    sources["inner_code_ref"] = (f"{_code_file(spec.code_ref) or RETUNE_DOC}: the wrapper of "
                                 f"{spec.idea_id}")
    sources["grid.gate.always"] = (f"{base_cite}: the unchanged book of {base.idea_id}, every "
                                   "regime traded")
    sources.update(_carry_sources(spec.sources, "trade_in", f"grid.gate.{name}"))
    if f"grid.gate.{name}.trade_in" not in sources:
        sources[f"grid.gate.{name}"] = f"{spec_cite}: the experiment's regimes"
    return _Grid(
        kind="wrapper", inner_code_ref=spec.code_ref, base_params=base_params, grid=grid,
        candidates=candidates, sources=sources, trees=[spec.params],
        description=(f"two candidates declared whole: the book of {base.idea_id} traded in every "
                     f"regime (always), and held only in {', '.join(trade_in)} ({spec.idea_id})"),
        caveat=_seen_caveat(spec),
        differs=f"the base {base.idea_id} against its regime-gated experiment"), None


def _param_source(key: str, params: Mapping) -> bool:
    """Whether a source key names a parameter (a leaf of `params` or a branch above one)."""
    return any(path == key or path.startswith(key + ".") or key.startswith(path + ".")
               for path, _ in _leaf_items(params))


def _seen_caveat(spec: StrategySpec) -> str:
    fixed, start = spec.params_fixed_at, spec.data.start
    if fixed is None:
        return ("the date the experiment's numbers were chosen is not recorded: the whole "
                "history may have been seen when they were chosen")
    if spec.data.source == TWIN_VENUE:
        return (f"the experiment's own data was already Binance's: {start} .. {fixed} may have "
                f"been seen when its numbers were chosen, so that stretch is in-sample for the "
                f"grid and the twin adds only the days before {start}")
    return (f"the experiment's numbers were chosen by hand, looking at {spec.data.source} data "
            f"up to {fixed} (its window starts {start}): the same market on Binance from {start} "
            f"to {fixed} is in-sample for the grid (it may have been seen), while the days "
            f"before {start} and after {fixed} were not")


# --------------------------------------------------------------------------
# What code chooses between: the declared lists


def _declared_grid(spec: StrategySpec, config: dict, config_path: Path
                   ) -> tuple[_Grid | None, str | None]:
    lists = config.get("lists") or {}
    open_found = open_params(spec, config_path)
    if open_found:
        found_problems = problems(spec, config, config_path)
        if found_problems:
            return None, "; ".join(found_problems)
    axes: dict[str, list[int | float]] = {}
    wrapped = normalize_ref(spec.code_ref) == REGIME_GATE
    for path in open_found:
        axes[path] = [_plain(v) for v in lists[path]]
    for name, values in lists.items():
        for path in (name, f"inner_params.{name}") if wrapped else (name,):
            present, current = _get(spec.params, path)
            if present and _number(current) and path not in axes:
                axes[path] = [_plain(float(v)) for v in values]
    if not axes:
        return None, NO_ALTERNATIVE
    count = math.prod(len(v) for v in axes.values())
    cap = int(config["max_candidates"])
    if count > cap:
        return None, (f"{count} candidates from the declared lists of {', '.join(axes)}; at "
                      f"most {cap} ({config_path})")
    base_params = copy.deepcopy(spec.params)
    for path, values in axes.items():
        *parents, leaf = path.split(".")
        node = base_params
        for name in parents:
            node = node[name]
        node[leaf] = values[0]
    grid = {path: {_label(path, v): {path: v} for v in values} for path, values in axes.items()}
    own = (f"{config_path} (version {config['version']}): the declared list for each of "
           f"{', '.join(axes)}; owner 2026-10-26: own numbers chosen on data before the cut")
    sources = {f"base_params.{k}": v for k, v in spec.sources.items()
               if _param_source(k, spec.params) and k not in axes}
    for path in axes:
        sources[f"base_params.{path}"] = own
    sources["inner_code_ref"] = f"{_code_file(spec.code_ref) or RETUNE_DOC}: the code of the spec"
    sources["grid"] = own
    candidates = expand_grid(base_params, grid)
    return _Grid(
        kind="lists", inner_code_ref=spec.code_ref, base_params=base_params, grid=grid,
        candidates=candidates, sources=sources,
        trees=[_with_largest(base_params, axes)],
        description=(f"the whole declared list of each open parameter ({', '.join(axes)}) from "
                     f"{config_path}, {count} candidates"),
        caveat=(f"the lists were declared in {config_path} (version {config['version']}) "
                "before this twin ran; what was seen of the history when they were declared is "
                "not recorded, so the choice is only as honest as the lists are"),
        differs=f"the declared lists of {config_path}"), None


def _with_largest(params: dict, axes: Mapping[str, list]) -> dict:
    out = copy.deepcopy(params)
    for path, values in axes.items():
        *parents, leaf = path.split(".")
        node = out
        for name in parents:
            node = node[name]
        node[leaf] = max(values)
    return out


# --------------------------------------------------------------------------
# The first causal choice


def _interval_days(interval: str) -> float:
    match = re.fullmatch(r"(\d+)\s*([mhdw])", interval.strip().lower())
    if match is None:
        raise ValueError(f"cannot read the interval {interval!r}")
    unit = {"m": 1 / 1440, "h": 1 / 24, "d": 1.0, "w": 7.0}[match.group(2)]
    return int(match.group(1)) * unit


def warmup_days(trees: list[dict], interval: str) -> tuple[float, str]:
    """The longest look-back the parameters name, in days, and which parameter it is.

    Read generically, so conservatively: every number under a key ending `_days` or `_hours`,
    and every number under a key that mentions a lookback, a window or a history (counted in
    bars of the data's interval), whatever the strategy does with it. The regime gate adds its
    detector's own window (`qlab.regimes.WINDOW_DAYS`)."""
    bar = _interval_days(interval)
    best, label = 0.0, "no look-back in the parameters"

    def consider(path: str, key: str, value: object) -> None:
        nonlocal best, label
        key = key.lower()
        if key.endswith("_days"):
            unit = 1.0
        elif key.endswith("_hours"):
            unit = 1 / 24
        elif key.endswith("_minutes"):
            unit = 1 / 1440
        elif any(word in key for word in ("lookback", "window", "history", "warmup")):
            unit = bar
        else:
            return
        numbers = [value] if _number(value) else (
            [v for v in value if _number(v)] if isinstance(value, list | tuple) else [])
        if numbers and max(numbers) * unit > best:
            best, label = max(numbers) * unit, f"{path} = {_plain(float(max(numbers)))}"

    def scan(node: object, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(item, Mapping):
                    scan(item, path)
                else:
                    consider(path, str(key), item)

    for tree in trees:
        scan(tree, "")
        if normalize_ref(str(tree.get("inner_code_ref", ""))) == REGIME_GATE or "trade_in" in tree:
            if WINDOW_DAYS > best:
                best, label = float(WINDOW_DAYS), "the regime detector's window"
    return best, label


def _next_boundary(day: date, refit_every: str) -> date:
    """The first re-tuning bar strictly after `day` (Retune: a calendar day, a week from
    Monday, a month from its first day, UTC)."""
    if refit_every == "D":
        return day + timedelta(days=1)
    if refit_every == "W":
        return day + timedelta(days=7 - day.weekday())
    return date(day.year + day.month // 12, day.month % 12 + 1, 1)


def first_choice(start: date, warm_days: float, refit_every: str) -> tuple[date, date, date]:
    """(the first day the book can trade, the first re-tuning after it, the first choice).

    A candidate is eligible only after it has traded through one whole re-tuning period
    (`qlab.strategies.retune`): it starts trading at `start + warm-up`, the next re-tuning
    ends its first whole period only after another one, and that second re-tuning is where
    the first choice is made."""
    active = start + timedelta(days=math.ceil(warm_days))
    first = _next_boundary(active, refit_every)
    return active, first, _next_boundary(first, refit_every)


# --------------------------------------------------------------------------
# The twin


def _refusal(spec: StrategySpec) -> str | None:
    if normalize_ref(spec.code_ref) == RETUNE or spec.selects_causally:
        return (f"{spec.idea_id} already chooses its numbers by code from the past only "
                "(a twin of a twin would add nothing)")
    if spec.idea_id.endswith(TWIN_SUFFIX):
        return f"{spec.idea_id} is a causal twin already"
    if spec.data.source not in ("hyperliquid", TWIN_VENUE):
        return (f"the data comes from {spec.data.source}, which has no Binance counterpart "
                "in the lab to give the long history")
    if spec.data.spot_source not in (None, TWIN_VENUE):
        return f"the spot legs come from {spec.data.spot_source}, not from Binance"
    return None


def _twin_data(spec: StrategySpec, venue_names: Collection[str] | None
               ) -> tuple[SpecData | None, list[str] | None, str | None, list[str]]:
    """(the twin's data, the mapped required instruments, a refusal, the dropped names)."""
    from qlab.pipeline.stage2 import map_instruments, venue_name

    data = spec.data
    instruments, required, dropped = data.instruments, spec.required_instruments, []
    if data.source != TWIN_VENUE and instruments is not None:
        if venue_names is None:
            return None, None, (f"the spec names instruments by {data.source}'s columns and the "
                                "Binance listing is needed to map them (venue_names)"), []
        instruments, dropped = map_instruments(instruments, venue_names)
        if not instruments:
            return None, None, "none of the named instruments is listed on Binance", dropped
    if data.source != TWIN_VENUE and required is not None and venue_names is not None:
        required = list(dict.fromkeys(venue_name(n, venue_names) or n for n in required))
    start = min(data.start, BINANCE_ARCHIVE_START) if data.source == TWIN_VENUE \
        else BINANCE_ARCHIVE_START
    return SpecData(
        source=TWIN_VENUE, interval=data.interval, include_spot=data.include_spot,
        spot_prices=data.spot_prices, spot_source=None, start=start, end=data.end,
        instruments=instruments, min_daily_volume_usd=data.min_daily_volume_usd,
    ), required, None, dropped


def _evidence(spec: StrategySpec, grid: _Grid, start: date, warm: float, label: str,
              dates: tuple[date, date, date], refit_every: str) -> str:
    active, first, choice = dates
    period = PERIODS[refit_every][0]
    return (
        f"The wrapper's first causal choice (qlab.strategies.retune, {RETUNE_DOC}): the data "
        f"starts {start}; the longest warm-up read from the parameters is {_plain(warm)} days "
        f"({label}), so the book can first trade on {active}; a candidate is eligible only "
        f"after one whole {period} re-tuning period, so the first re-tuning after {active} is "
        f"{first} and the first choice is made at the one after it, {choice}. This is warm-up "
        "arithmetic read generically from the parameters (every day, hour, window and lookback "
        "number, whatever the strategy does with it), not a result. If the strategy needs more "
        "than the parameters show (enough instruments listed on Binance, a signal that waits "
        "for history), the true first choice is later and the days before it are judged as "
        "flat days. The grid is " + grid.description + ". What this does not prove: "
        + grid.caveat + (f" (the spec's own date for its numbers: {spec.params_fixed_at})"
                         if spec.params_fixed_at else "") + ".")


def _sources(spec: StrategySpec, grid: _Grid, refit_every: str, spec_cite: str
             ) -> dict[str, str]:
    period = PERIODS[refit_every][0]
    sources = {
        "code_ref": "src/qlab/strategies/retune.py (existing wrapper)",
        "refit_every": f"{RETUNE_DOC}: re-tuned every {period}; the owner 2026-10-02 named "
                       "«раз в неделю/день или месяц»",
        "window": f"{RETUNE_DOC} (expanding window: no number introduced)",
        "objective": f"{RETUNE_DOC} (Sharpe after costs)",
        "data.start": f"{VENUES_DOC}: Binance USDT perpetuals, «свечи с 2019-09-08 (первый день "
                      "в архиве лаборатории)»",
        "data.end": f"{spec_cite}: the same window end",
    }
    sources.update(grid.sources)
    for name in ("taker_fee_bps", "slippage_bps"):
        cost = spec.sources.get(f"costs.{name}") or spec.sources.get("costs")
        sources[f"score_costs.{name}"] = cost or f"{RETUNE_DOC}: scores use the spec's costs"
        if cost:
            sources[f"costs.{name}"] = cost
    for key, citation in spec.sources.items():
        if key.startswith(("min_leg_notional", "regime_claim.")):
            sources[key] = citation
    return sources


def causal_twin(spec: StrategySpec, *, base: StrategySpec | None = None,
                spec_path: Path | str | None = None, base_path: Path | str | None = None,
                config_path: Path = CONFIG, venue_names: Collection[str] | None = None,
                refit_every: str = "M") -> TwinResult:
    """The spec whose numbers code chooses every month from the past only, on Binance's history.

    `base` is the strategy spec an experiment `spec` modifies (see the module docstring).
    `spec_path` and `base_path` are cited as the places the candidates' numbers come from.
    `venue_names` (Binance's listing) is needed only to map instruments a Hyperliquid spec
    names. `refit_every` is `M`, `W` or `D`, as `Retune`'s."""
    if refit_every not in PERIODS:
        raise ValueError(f"refit_every must be one of {sorted(PERIODS)}, got {refit_every!r}")
    refusal = _refusal(spec)
    if refusal:
        return TwinResult(None, refusal)
    reasons: list[str] = []
    grid = None
    if base is not None and (open_params(spec, config_path) or open_params(base, config_path)):
        # A spec that leaves numbers open (qlab.own_numbers) has declared its candidates already:
        # the whole lists are the choice, and a list is not a number a pair could override.
        reasons.append("no pair with the base: a spec that leaves numbers open (night/"
                       "own_numbers.yaml) offers its declared lists, not a base and one override")
    elif base is not None:
        grid, why = _pair_grid(spec, base, _text(spec_path), _text(base_path))
        if grid is None:
            reasons.append(f"no pair with the base: {why}")
    if grid is None:
        grid, why = _declared_grid(spec, load_config(config_path), config_path)
        if grid is None:
            reasons.append(why or NO_ALTERNATIVE)
    if grid is None:
        return TwinResult(None, "; ".join(dict.fromkeys(reasons)))

    data, required, why, dropped = _twin_data(spec, venue_names)
    if data is None:
        return TwinResult(None, why)
    warm, label = warmup_days(grid.trees, spec.data.interval)
    dates = first_choice(data.start, warm, refit_every)
    if dates[2] >= data.end:
        return TwinResult(None, f"the first causal choice ({dates[2]}) falls on or after the end "
                                f"of the data ({data.end}): no day would be a forward test")

    params = {
        "inner_code_ref": grid.inner_code_ref,
        "base_params": grid.base_params,
        "grid": grid.grid,
        "refit_every": refit_every,
        "window": "expanding",
        "objective": "sharpe_net",
        "score_costs": {"taker_fee_bps": spec.costs.taker_fee_bps,
                        "slippage_bps": spec.costs.slippage_bps},
    }
    notes = [
        f"Causal twin of {spec.idea_id}: the numbers it trades are "
        f"chosen by code, every {PERIODS[refit_every][0]}, from returns realised before the "
        f"choice ({grid.differs}); the data is Binance's history, a proxy for {spec.data.source}'s "
        "market, with the original's costs and minimum leg.",
    ]
    if spec.required_instruments and spec.data.source != TWIN_VENUE and venue_names is None:
        notes.append("The required instruments keep the original venue's names (the Binance "
                     "listing was not consulted): a name Binance does not use shows in the run "
                     "as absent, it is not traded silently.")
    if dropped:
        notes.append(f"{len(dropped)} named instrument(s) are not listed on Binance and are "
                     f"not traded: {', '.join(dropped)}.")
    update = {
        "idea_id": f"{spec.idea_id}{TWIN_SUFFIX}",
        "title": f"{spec.title}: числа выбирает код {PERIODS[refit_every][1]} только по "
                 f"прошлому, история Binance с {data.start.year}",
        "code_ref": RETUNE,
        "params": params,
        "data": data,
        "required_instruments": required,
        "exploratory": None,
        "judged_from": None,
        "params_fit_from": None,
        "params_fixed_at": dates[2],
        "params_fixed_evidence": _evidence(spec, grid, data.start, warm, label, dates,
                                           refit_every),
        "selects_causally": True,
        "sources": _sources(spec, grid, refit_every, _text(spec_path) or RETUNE_DOC),
        "differs_from_source": [*spec.differs_from_source, *notes],
    }
    if "execution_venue" in StrategySpec.model_fields:
        update["execution_venue"] = getattr(spec, "execution_venue", None) or (
            spec.data.source if spec.data.source != TWIN_VENUE else None)
    update["data"] = data.model_dump(mode="python")
    twin = StrategySpec.model_validate({**spec.model_dump(mode="python"), **update})
    return TwinResult(twin, None, grid.kind, tuple(grid.candidates), spec.idea_id)


def _text(path: Path | str | None) -> str | None:
    """A spec's path as the sources cite it: relative to the project when it lies inside."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute() and path.is_relative_to(QLAB_ROOT):
        return path.relative_to(QLAB_ROOT).as_posix()
    return path.as_posix()


# --------------------------------------------------------------------------
# Finding the base of an experiment


def _distance(spec: StrategySpec, candidate: StrategySpec) -> int | None:
    """How many parameter leaves separate the spec from a candidate base, or None when the
    candidate cannot be its base (other code that is not the spec's wrapped strategy)."""
    if normalize_ref(spec.code_ref) == normalize_ref(candidate.code_ref):
        return differing_leaves(candidate.params, spec.params)
    unwrapped = _unwrap(spec)
    if unwrapped is not None and unwrapped[0] == normalize_ref(candidate.code_ref):
        return differing_leaves(candidate.params, unwrapped[1])
    return None


def find_base(spec: StrategySpec, candidates: list[StrategySpec]) -> BaseChoice:
    """The strategy spec an experiment modifies: among `candidates` (night/research.yaml lists
    each strategy's specs) the one with the same code (or the code the spec wraps) and the
    fewest differing parameter leaves. A tie, or no candidate, is no base, with the reason."""
    scored = []
    for candidate in candidates:
        if candidate.idea_id == spec.idea_id or normalize_ref(candidate.code_ref) == RETUNE \
                or candidate.selects_causally:
            continue
        distance = _distance(spec, candidate)
        if distance is not None:
            # A candidate on other data or costs cannot be the other half of a pair
            # (`_setup_mismatch`): one that can is nearer, whatever the numbers.
            scored.append(((_setup_mismatch(spec, candidate) is not None, distance), candidate))
    if not scored:
        return BaseChoice(None, f"none of the {len(candidates)} candidate specs runs the same "
                                f"code as {spec.idea_id}")
    best = min(key for key, _ in scored)
    nearest = [c for key, c in scored if key == best]
    if len(nearest) > 1:
        names = ", ".join(c.idea_id for c in nearest)
        return BaseChoice(None, f"{names} are equally near to {spec.idea_id} "
                                f"({best[1]} differing parameter(s)): the base is not clear")
    return BaseChoice(nearest[0])


# --------------------------------------------------------------------------
# The file


def twin_yaml(twin: StrategySpec, origin: str) -> str:
    """The twin as a spec file's text (no None fields; `params` keeps its nulls)."""
    raw = twin.model_dump(mode="json")
    raw = {k: v for k, v in raw.items() if v is not None
           and not (k in ("differs_from_source", "sources", "regime_claim", "review_answers")
                    and not v)}
    raw["data"] = {k: v for k, v in raw["data"].items() if v is not None}
    header = (f"# Causal twin of {origin} (qlab twin, src/qlab/twin.py): its numbers are chosen\n"
              "# by code, from the past only, on Binance's history (docs/RETUNE.md). Written by\n"
              "# code, not by hand.\n")
    return header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)


def twin_path(spec_path: Path) -> Path:
    return spec_path.with_name(f"{spec_path.stem}{TWIN_SUFFIX}.yaml")


def write_twin(spec_path: Path, twin: StrategySpec, origin: str) -> Path:
    """Write `<name>-causal.yaml` next to the spec. An existing identical file is left as it
    is; a different one is never overwritten."""
    target = twin_path(spec_path)
    text = twin_yaml(twin, origin)
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"{target} exists and differs: not overwritten")
    target.write_text(text, encoding="utf-8")
    return target


__all__ = ["BINANCE_ARCHIVE_START", "BaseChoice", "TwinResult", "causal_twin", "find_base",
           "first_choice", "twin_path", "twin_yaml", "warmup_days", "write_twin"]
