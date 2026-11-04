"""Doubles of an idea: caught before a run by a fingerprint, after it by identical results
(docs/DUPLICATES.md).

Owner, 2026-11-04, after seven Bv2 variants were retired as duplicates on 2026-11-03 (on Binance's
history five of them gave the same numbers to the digit, because the code picked production's
numbers every month, and none of the agents' checks caught it): «может перед тем как создавать
новую стратегию проверять на наличие уже такой - так мы утонем в двойниках» -> «да, делай».

Two checks, both code, neither calls an agent:

1. **Before a run: a fingerprint of the spec** (`fingerprint`). Whatever is written differently in
   two specs but makes the stand run the same thing reads the same:

   - the INNER strategy's code, wrappers (`Retune`, `SplitCadence`, `RegimeGate`, `Overlaid`) taken
     off; a wrapper that changes what is held (cadence, gate, overlays) stays as its own part;
   - the strategy's numbers AFTER the strategy's own defaults (`DEFAULT_PARAMS` of the strategy, or
     the one `*Params` dataclass of its module): a key written out at its default value reads the
     same as a key left out;
   - what code chooses among: the idea's `choice` lists, or the grid of a `Retune` form, both read
     as the set of full variants (names do not count), with the rule of the choice;
   - the data: venue, spot venue, how spot is priced, interval, spot columns, instruments and the
     universe rule, the first day;
   - the costs and the smallest leg; the strategy's money.

   Ids, titles, comments, plain texts and dates of fixation are not in it: variants that differ
   only there are doubles. An idea whose fingerprint equals that of an OPEN idea's latest spec is
   not run (`refusal`); the report and the memo say «дубль идеи X» with the parts that matched.

2. **After a run: identical results** (`mark_by_results`). The new run's daily returns on the
   strategy's money against the latest run of every older open idea on the same venue and window:
   correlation at least `MIN_CORRELATION` and yearly return within `MAX_ANNUAL_DIFF`. The idea is
   marked «дубль X по результатам». The mark is a stored reading: it decides nothing by itself,
   it is shown on the idea's page, in step 5 of its ladder and in the morning report under
   «предлагается вывести (решает владелец)».

The module writes only its ledger (`data/duplicates/ledger.json`) and, in `mark_by_results`, never a
status, a verdict or a trial.
"""

from __future__ import annotations

import dataclasses
import importlib
import itertools
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qlab.pipeline.spec import StrategySpec

LEDGER_ENV = "QLAB_DUPLICATES"
LEDGER = Path("data/duplicates/ledger.json")
SPEC_ROOT = Path("specs")
CLOSED_STATUSES = frozenset({"retired", "rejected", "decayed"})
"""An idea in any other status is open (candidate, bench, paper, live, validated, ...)."""

SECTION_TITLE = "Предлагается вывести (решает владелец)"
"""The morning report's section for ideas marked as doubles by their results."""

MIN_CORRELATION = 0.999
MAX_ANNUAL_DIFF = 0.001
"""A tenth of a percentage point of yearly return, as a fraction (owner, 2026-11-04)."""
MIN_COMMON_DAYS = 30
MIN_OVERLAP = 0.9
"""Two runs share a window when their common days are this part of the shorter one."""

RETUNE = "qlab.strategies.retune:Retune"
CHOICE_RULE = ("M", "expanding", "sharpe_net")
"""How code chooses among the lists an idea declares (`qlab.one_check`, `qlab.twin`): every
month, over the whole past, by the Sharpe after costs."""

PART_NAMES = {
    "code": "код",
    "wrappers": "обёртки (каденция, фильтр режима, надстройки)",
    "params": "числа после умолчаний",
    "choice": "что выбирает код",
    "data": "данные",
    "costs": "издержки",
    "money": "деньги стратегии",
}


# --------------------------------------------------------------------------
# The fingerprint


def normalize_ref(code_ref: str) -> str:
    """`module:attr` for both spellings a spec may use."""
    if ":" in code_ref:
        return code_ref
    module, _, attr = code_ref.rpartition(".")
    return f"{module}:{attr}"


def _plain(value: Any) -> Any:
    """A value as it is compared: numbers that are equal read the same (3 and 3.0), dicts by
    sorted keys, tuples as lists."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        number = float(value)
        return int(number) if number.is_integer() else round(number, 12)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple | set | frozenset):
        items = [_plain(v) for v in value]
        return sorted(items, key=_dump) if isinstance(value, set | frozenset) else items
    return str(value)


def _dump(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str)


def _strategy_class(code: str) -> Any | None:
    module, _, attr = normalize_ref(code).partition(":")
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:  # noqa: BLE001 - a strategy that cannot be imported has no known defaults
        return None


def defaults_for(code: str) -> dict[str, Any]:
    """The numbers the strategy takes when a spec leaves them out, best effort.

    First the strategy's own `DEFAULT_PARAMS`; else the one `*Params` dataclass of its module
    that builds with no arguments. Nothing found (the module cannot be imported, nothing declares
    defaults): empty, and a key left out then reads as a key left out. That can only miss a double,
    never invent one: both sides of every comparison are filled the same way."""
    cls = _strategy_class(code)
    if cls is None:
        return {}
    declared = getattr(cls, "DEFAULT_PARAMS", None)
    if isinstance(declared, Mapping):
        return dict(declared)
    module = importlib.import_module(cls.__module__)
    found = []
    for name, obj in vars(module).items():
        if name.endswith("Params") and isinstance(obj, type) and dataclasses.is_dataclass(obj):
            try:
                found.append(dataclasses.asdict(obj()))
            except Exception:  # noqa: BLE001 - a dataclass with required fields has no defaults
                continue
    return found[0] if len(found) == 1 else {}


def effective(code: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """The strategy's numbers with its own defaults filled in."""
    return {**defaults_for(code), **dict(params)}


def _set_path(params: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    node = params
    for name in parents:
        node = node.setdefault(name, {})
    node[leaf] = value


def _deep(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def declared_variants(spec: StrategySpec) -> list[dict[str, Any]] | None:
    """Every full parameter set the idea's `choice` lists name, or None when it names none."""
    choice = spec.choice
    if choice is None or not choice.lists:
        return None
    out = []
    if choice.axes:
        paths = list(choice.axes)
        for combo in itertools.product(*[choice.axes[p] for p in paths]):
            params = _deep(spec.params)
            for path, value in zip(paths, combo, strict=True):
                _set_path(params, path, value)
            out.append(params)
    else:
        for overrides in choice.candidates.values():
            params = _deep(spec.params)
            for path, value in dict(overrides).items():
                _set_path(params, path, value)
            out.append(params)
    return out


def retune_variants(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The full parameter sets a `Retune` form expands to (its own grid expansion)."""
    from qlab.strategies.retune import expand_grid

    return list(expand_grid(dict(params["base_params"]), dict(params["grid"])).values())


def _unwrap(code: str, params: Mapping[str, Any]) -> tuple[list[list[Any]], str, dict[str, Any]]:
    """Take the non-choosing wrappers off: (their layers, inner code, inner numbers). A layer is
    the wrapper's code and its own numbers, because those change what is held."""
    layers: list[list[Any]] = []
    code, params = normalize_ref(code), dict(params)
    while "inner_code_ref" in params and (code != RETUNE):
        inner_key = "inner_params" if "inner_params" in params else None
        if inner_key is None:
            break
        own = {k: v for k, v in params.items() if k not in ("inner_code_ref", inner_key)}
        layers.append([code, _plain(own)])
        code, params = normalize_ref(str(params["inner_code_ref"])), dict(params[inner_key])
    return layers, code, params


@dataclass(frozen=True)
class Fingerprint:
    """What makes a spec the same run. `parts` are printable; equal parts, equal fingerprints."""

    parts: Mapping[str, str]
    code: str

    def key(self) -> str:
        return _dump(dict(self.parts))

    def same_parts(self, other: Fingerprint) -> list[str]:
        return [name for name in self.parts if self.parts[name] == other.parts.get(name)]


def fingerprint(spec: StrategySpec, *, money: float | None = None,
                min_leg: float | None = None) -> Fingerprint:
    """The spec's fingerprint (see the module docstring). `money` and `min_leg` override the
    spec's own, for a spec read from the registry that does not store them."""
    code = normalize_ref(spec.code_ref)
    choice_rule: Any = None
    variants = declared_variants(spec)
    if variants is not None:
        choice_rule = list(CHOICE_RULE)
        params_list = variants
    elif code == RETUNE and "grid" in spec.params:
        choice_rule = _plain([spec.params.get(k) for k in ("refit_every", "window", "objective")]
                             + ([spec.params["choose_at"]] if spec.params.get("choose_at") else []))
        params_list = retune_variants(spec.params)
        code = normalize_ref(str(spec.params["inner_code_ref"]))
    else:
        params_list = [dict(spec.params)]
    layers_seen: list[list[Any]] = []
    inner_code = code
    kept = []
    for params in params_list:
        layers, inner_code, inner = _unwrap(code, params)
        layers_seen = layers
        kept.append(_dump(effective(inner_code, inner)))
    data = spec.data
    data_part = {
        "source": data.source, "spot_source": data.spot_source, "spot_prices": data.spot_prices,
        "interval": data.interval, "include_spot": data.include_spot,
        "instruments": sorted(data.instruments) if data.instruments else None,
        "min_daily_volume_usd": data.min_daily_volume_usd,
        "required": sorted(spec.required_instruments) if spec.required_instruments else None,
        "execution_venue": spec.execution_venue or data.source, "start": data.start.isoformat(),
    }
    leg = spec.min_leg_notional if min_leg is None else min_leg
    cost_part = [spec.costs.taker_fee_bps, spec.costs.slippage_bps, leg]
    cap = spec.strategy_capital_usd if money is None else money
    parts = {
        "code": inner_code,
        "wrappers": _dump(layers_seen),
        "params": _dump(sorted(kept)),
        "choice": _dump(choice_rule),
        "data": _dump(data_part),
        "costs": _dump(cost_part),
        "money": _dump(cap),
    }
    return Fingerprint(parts=parts, code=inner_code)


def describe(fp: Fingerprint, spec: StrategySpec | None = None) -> str:
    """The matched parts in plain words, for the report and the memo."""
    bits = [f"код {fp.code}"]
    if fp.parts["wrappers"] != "[]":
        bits.append("те же обёртки")
    variants = len(json.loads(fp.parts["params"]))
    bits.append("числа после умолчаний совпали"
                + (f", вариантов выбора {variants}" if variants > 1 else ""))
    data = json.loads(fp.parts["data"])
    bits.append(f"данные {data['source']} {data['interval']} с {data['start']}"
                + (f", спот {data['spot_source']}" if data["spot_source"] else "")
                + (", цена спота своя" if data["spot_prices"] == "own" else ""))
    costs = json.loads(fp.parts["costs"])
    bits.append(f"издержки {costs[0]:g}+{costs[1]:g} б.п., минимальная нога {costs[2]:g}")
    money = json.loads(fp.parts["money"])
    bits.append("деньги стратегии " + ("не заданы" if money is None else f"{money:g}"))
    return "; ".join(bits)


# --------------------------------------------------------------------------
# The open ideas


@dataclass(frozen=True)
class OpenIdea:
    idea_id: str
    title: str
    plain_title: str
    status: str
    fingerprint: Fingerprint
    source: str
    """Where the spec was read: its file, or `registry` for the latest stored row."""


def _spec_files(root: Path) -> dict[str, list[tuple[Path, StrategySpec]]]:
    from qlab.pipeline.spec import load_spec

    out: dict[str, list[tuple[Path, StrategySpec]]] = {}
    if not root.is_dir():
        return out
    pattern = re.compile(r"^idea_id:\s*(\S+)\s*$", re.M)
    for path in sorted(root.rglob("*.yaml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found = pattern.search(text)
        if found is None:
            continue
        try:
            out.setdefault(found.group(1).strip("'\""), []).append((path, load_spec(path)))
        except Exception:  # noqa: BLE001 - a spec that does not load cannot be compared
            continue
    return out


def _followed(watchlist: Path) -> set[str]:
    try:
        data = yaml.safe_load(watchlist.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    return {Path(str(x)).as_posix() for x in data.get("specs") or []}


def _pick(entries: list[tuple[Path, StrategySpec]], followed: set[str]
          ) -> tuple[Path, StrategySpec]:
    """The spec the night follows, then one that states `params_fixed_at`, then the last by name
    (as `qlab.pool_allocation.spec_path_for_idea`)."""
    for tier in ([e for e in entries if e[0].as_posix() in followed],
                 [e for e in entries if e[1].params_fixed_at is not None], entries):
        if tier:
            return tier[-1]
    return entries[-1]


def _row_spec(row: Any, idea: Any) -> StrategySpec | None:
    """A stored spec row as a spec, for an idea that has no spec file. The money and the smallest
    leg are not stored, so they read as unknown."""
    try:
        data = dict(row.data_requirements)
        return StrategySpec.model_validate({
            "idea_id": idea.id, "title": idea.title, "code_ref": row.code_ref,
            "params": row.params, "data": data, "costs": row.costs_model,
            "min_leg_notional": 1.0, "unexpressed_mechanisms": []})
    except Exception:  # noqa: BLE001
        return None


def _ideas_with_trials(session: Any) -> set[str]:
    from sqlalchemy import exists

    from qlab.registry.models import Spec, Trial

    return {i for (i,) in session.query(Spec.idea_id).filter(
        exists().where(Trial.spec_id == Spec.id)).distinct()}


def is_new(session: Any, idea_id: str) -> bool:
    """Whether the stand has never run the idea: no trial under any of its specs. An idea may
    already stand in the registry (the programmer's review row needs it) and still be new."""
    from sqlalchemy import exists

    from qlab.registry.models import Spec, Trial

    return not session.query(
        exists().where(Spec.idea_id == idea_id).where(Trial.spec_id == Spec.id)).scalar()


def open_index(session: Any, *, spec_root: Path = SPEC_ROOT,
               watchlist: Path = Path("night/watchlist.yaml")) -> list[OpenIdea]:
    """Every open idea the stand has run, with the fingerprint of its latest spec: its spec file
    when it has one (the form with `choice`, the money and the smallest leg), else its latest
    stored row. An idea with no trial yet is not on the stand and is not compared with."""
    from qlab.registry.models import Idea, Spec

    files = _spec_files(spec_root)
    followed = _followed(watchlist)
    run = _ideas_with_trials(session)
    out: list[OpenIdea] = []
    for idea in session.query(Idea).order_by(Idea.id):
        status = getattr(idea.status, "value", str(idea.status))
        if status in CLOSED_STATUSES or idea.id not in run:
            continue
        spec: StrategySpec | None = None
        source = "registry"
        if idea.id in files:
            path, spec = _pick(files[idea.id], followed)
            source = path.as_posix()
        else:
            row = (session.query(Spec).filter(Spec.idea_id == idea.id)
                   .order_by(Spec.version.desc()).first())
            if row is not None:
                spec = _row_spec(row, idea)
        if spec is None:
            continue
        try:
            fp = fingerprint(spec, min_leg=None if source != "registry" else -1.0)
        except Exception:  # noqa: BLE001 - one odd spec must not stop the others
            continue
        out.append(OpenIdea(idea.id, idea.title, spec.plain_title or idea.title, status, fp,
                            source))
    return out


# --------------------------------------------------------------------------
# Before a run


@dataclass(frozen=True)
class Duplicate:
    of: str
    title: str
    parts: list[str]
    words: str

    def text(self) -> str:
        return f"дубль идеи {self.of} («{self.title}»): {self.words}"


def refusal(spec: StrategySpec, index: Iterable[OpenIdea]) -> Duplicate | None:
    """The open idea whose fingerprint is this spec's, or None. An idea is never its own double.
    A spec whose fingerprint cannot be made is not refused: the check never blocks a run by
    failing."""
    try:
        mine = fingerprint(spec)
    except Exception:  # noqa: BLE001
        return None
    for other in index:
        if other.idea_id == spec.idea_id:
            continue
        if other.fingerprint.key() == mine.key():
            return Duplicate(other.idea_id, other.plain_title,
                             [PART_NAMES[p] for p in mine.same_parts(other.fingerprint)],
                             describe(mine, spec))
    return None


def refusal_for_new_idea(spec: StrategySpec, session: Any,
                         index: Iterable[OpenIdea] | None = None) -> Duplicate | None:
    """`refusal` for an idea the stand has never run (`is_new`); None for one it has (a re-run is
    never refused). Reads the open ideas itself unless an index is given."""
    if not is_new(session, spec.idea_id):
        return None
    return refusal(spec, open_index(session) if index is None else index)


def as_open(spec: StrategySpec) -> OpenIdea:
    """A spec about to run as one more open idea, so that the next spec of the same batch is
    compared with it as well."""
    return OpenIdea(spec.idea_id, spec.title, spec.plain_title or spec.title, "candidate",
                    fingerprint(spec), "batch")


def prompt_block(index: Iterable[OpenIdea]) -> str:
    """For an agent's prompt: the open ideas in plain words, and the rule."""
    lines = ["## Open ideas (a spec equal to one of them is refused as a double and not run)", "",
             "Doubles are refused by code before any run: the same strategy code, the same "
             "numbers after the strategy's defaults, the same lists to choose among, the same "
             "data, costs and strategy money. Different ids, titles or comments make no "
             "difference. Bring an idea that is not on this list.", ""]
    lines += [f"- {o.idea_id}: {o.plain_title}" for o in index]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The ledger


def ledger_path() -> Path:
    return Path(os.environ.get(LEDGER_ENV) or LEDGER)


def load_ledger(path: Path | None = None) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((path or ledger_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"results": dict(data.get("results") or {}), "refused": dict(data.get("refused") or {}),
            "checked": dict(data.get("checked") or {})}


def save_ledger(ledger: Mapping[str, Any], path: Path | None = None) -> None:
    target = path or ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(target)


def note_refusal(spec: StrategySpec, duplicate: Duplicate, where: str,
                 path: Path | None = None) -> None:
    """Remember that an idea was refused before its run, so the report can say so."""
    ledger = load_ledger(path)
    ledger["refused"][spec.idea_id] = {
        "of": duplicate.of, "words": duplicate.text(), "where": where,
        "at": datetime.now(UTC).isoformat(timespec="seconds")}
    save_ledger(ledger, path)


# --------------------------------------------------------------------------
# After a run


def daily_returns(payload: Mapping[str, Any]) -> pd.Series:
    """The stored forward series (`qlab.forward_series`) as the strategy's money earned each
    day: the cumulative curve from zero turned back into days (a ratio for a reinvesting book,
    a difference for one that is not)."""
    points = payload["points"]
    days = pd.to_datetime([p[0] for p in points])
    cum = pd.Series([float(p[1]) for p in points], index=days)
    if payload.get("reinvested", True):
        daily = (1.0 + cum) / (1.0 + cum.shift(1)) - 1.0
    else:
        daily = cum.diff()
    return daily.iloc[1:]


@dataclass(frozen=True)
class SameResults:
    correlation: float
    annual_diff: float
    days: int
    annual: float


def compare_series(a: Mapping[str, Any], b: Mapping[str, Any]) -> SameResults | None:
    """Whether two stored runs are one and the same result: same venue, a common window, daily
    returns correlated at least `MIN_CORRELATION`, yearly returns within `MAX_ANNUAL_DIFF`.
    None when they are not, or when there is too little to compare."""
    if a.get("venue") != b.get("venue"):
        return None
    ra, rb = daily_returns(a), daily_returns(b)
    common = ra.index.intersection(rb.index)
    if len(common) < MIN_COMMON_DAYS or len(common) < MIN_OVERLAP * min(len(ra), len(rb)):
        return None
    x, y = ra.loc[common].to_numpy(), rb.loc[common].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    reinvested = bool(a.get("reinvested", True)) and bool(b.get("reinvested", True))

    def annual(r: np.ndarray) -> float:
        n = len(r)
        total = float(np.prod(1.0 + r) - 1.0) if reinvested else float(r.sum())
        if not reinvested:
            return total * 365.25 / n
        return -1.0 if 1.0 + total <= 0 else (1.0 + total) ** (365.25 / n) - 1.0

    ann_x, ann_y = annual(x), annual(y)
    diff = abs(ann_x - ann_y)
    if corr < MIN_CORRELATION or diff > MAX_ANNUAL_DIFF:
        return None
    return SameResults(corr, diff, int(len(common)), ann_x)


def latest_series_trials(session: Any, store: Path | None = None) -> dict[str, int]:
    """Idea -> the id of its latest trial that has a stored series."""
    from sqlalchemy import func

    from qlab import forward_series
    from qlab.registry.models import Spec, Trial

    folder = store if store is not None else forward_series.store_dir()
    ids = sorted(int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()) \
        if folder.is_dir() else []
    out: dict[str, int] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        rows = (session.query(Spec.idea_id, func.max(Trial.id))
                .join(Trial, Trial.spec_id == Spec.id).filter(Trial.id.in_(chunk))
                .group_by(Spec.idea_id).all())
        for idea_id, trial_id in rows:
            out[idea_id] = max(trial_id, out.get(idea_id, 0))
    return out


def older_open_ideas(session: Any, idea_id: str) -> list[str]:
    """Open ideas created before this one (ties by id): a double is marked against the older."""
    from qlab.registry.models import Idea

    me = session.get(Idea, idea_id)
    if me is None:
        return []
    out = []
    for other in session.query(Idea).filter(Idea.id != idea_id):
        status = getattr(other.status, "value", str(other.status))
        if status in CLOSED_STATUSES:
            continue
        if (other.created_at, other.id) < (me.created_at, me.id):
            out.append(other.id)
    return out


def find_same_results(session: Any, idea_id: str, trial_id: int, *, store: Path | None = None,
                      latest: Mapping[str, int] | None = None,
                      candidates: Iterable[str] | None = None) -> dict[str, Any] | None:
    """The older open idea whose latest run this run equals, as a ledger record; None if none."""
    from qlab import forward_series

    mine = forward_series.load(trial_id, store)
    if mine is None:
        return None
    if latest is None:
        latest = latest_series_trials(session, store)
    pool = list(older_open_ideas(session, idea_id) if candidates is None else candidates)
    for other in sorted(pool):
        other_trial = latest.get(other)
        if other == idea_id or other_trial is None:
            continue
        theirs = forward_series.load(other_trial, store)
        if theirs is None:
            continue
        same = compare_series(mine, theirs)
        if same is not None:
            return {
                "of": other, "trial_id": trial_id, "of_trial_id": other_trial,
                "venue": mine.get("venue"), "correlation": round(same.correlation, 6),
                "annual_diff_pp": round(same.annual_diff * 100, 4), "days": same.days,
                "annual": round(same.annual, 4),
                "marked_at": datetime.now(UTC).isoformat(timespec="seconds")}
    return None


def mark_by_results(session: Any, idea_id: str, trial_id: int, *, store: Path | None = None,
                    path: Path | None = None) -> dict[str, Any] | None:
    """After a run: mark the idea «дубль X по результатам» when its daily returns equal an older
    open idea's latest run, or take the mark off when this run no longer does. A stored reading;
    it changes no status. Returns the record, or None."""
    record = find_same_results(session, idea_id, trial_id, store=store)
    ledger = load_ledger(path)
    # `checked`: the run the check was made on, so a page can tell «not a double» from «not
    # checked yet» (docs/IDEA_STANDARD.md, item 14)
    ledger["checked"][idea_id] = {"trial_id": trial_id,
                                  "at": datetime.now(UTC).isoformat(timespec="seconds")}
    if record is None:
        ledger["results"].pop(idea_id, None)
        save_ledger(ledger, path)
        return None
    ledger["results"][idea_id] = record
    save_ledger(ledger, path)
    return record


def reading_ru(record: Mapping[str, Any]) -> str:
    """The mark in plain words, for the ladder's step 5 and the idea's page."""
    return (f"дубль {record['of']} по результатам: дневная доходность на деньгах стратегии "
            f"совпала (корреляция {record['correlation']:.4f}, годовая доходность различается на "
            f"{record['annual_diff_pp']:.3f} п.п., {record['days']} дн., прогон "
            f"{record['trial_id']} и прогон {record['of_trial_id']}); запись, сама ничего "
            "не решает")


def report_lines(session: Any, ledger: Mapping[str, Any] | None = None) -> list[str]:
    """The morning report's section: ideas to propose for retirement (the owner decides), only
    while both the double and the idea it doubles are open."""
    from qlab.registry.models import Idea

    ledger = load_ledger() if ledger is None else ledger
    lines: list[str] = []
    for idea_id, record in sorted(ledger.get("results", {}).items()):
        mine, theirs = session.get(Idea, idea_id), session.get(Idea, record["of"])
        if mine is None or theirs is None:
            continue
        if any(getattr(i.status, "value", str(i.status)) in CLOSED_STATUSES
               for i in (mine, theirs)):
            continue
        lines.append(f"- `{idea_id}` — {reading_ru(record)}. Вывести: "
                     f"`qlab retire {idea_id} --reason 'дубль {record['of']}'`")
    if lines:
        lines = ["Это показание, не решение: код статусов не меняет (docs/DUPLICATES.md).", "",
                 *lines]
    refused = ledger.get("refused", {})
    recent = [(i, r) for i, r in sorted(refused.items())
              if (datetime.now(UTC) - datetime.fromisoformat(r["at"])).days < 3]
    if recent:
        lines += ["", "Не запущено до прогона, потому что это дубль:", "",
                  *[f"- `{i}` — {r['words']} ({r['where']})" for i, r in recent]]
    return lines


# --------------------------------------------------------------------------
# The registry as it is now (scripts/find_duplicates.py)


@dataclass
class Group:
    kind: str
    ideas: list[str]
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def groups_by_fingerprint(index: Iterable[OpenIdea]) -> list[Group]:
    by_key: dict[str, list[OpenIdea]] = {}
    for item in index:
        by_key.setdefault(item.fingerprint.key(), []).append(item)
    return [Group("fingerprint", sorted(i.idea_id for i in items),
                  describe(items[0].fingerprint))
            for items in by_key.values() if len(items) > 1]


def groups_by_results(session: Any, index: Iterable[OpenIdea], *, store: Path | None = None
                      ) -> list[Group]:
    """Open ideas whose latest runs equal each other, grouped (union of the pairs)."""
    from qlab import forward_series

    latest = latest_series_trials(session, store)
    open_ids = [i.idea_id for i in index if i.idea_id in latest]
    series = {i: forward_series.load(latest[i], store) for i in open_ids}
    parent = {i: i for i in open_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: dict[tuple[str, str], SameResults] = {}
    for a, b in itertools.combinations(sorted(open_ids), 2):
        if series[a] is None or series[b] is None:
            continue
        same = compare_series(series[a], series[b])  # type: ignore[arg-type]
        if same is not None:
            pairs[(a, b)] = same
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for i in open_ids:
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        shown = [(p, s) for p, s in pairs.items() if p[0] in members]
        worst = min(s.correlation for _, s in shown)
        widest = max(s.annual_diff for _, s in shown) * 100
        out.append(Group("results", sorted(members),
                         f"venue {series[members[0]].get('venue')}; "  # type: ignore[union-attr]
                         f"correlation >= {worst:.5f}; yearly return differs by <= {widest:.3f} pp",
                         {"annual": shown[0][1].annual}))
    return out
