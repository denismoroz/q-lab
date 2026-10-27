"""The night's answer on the history that already exists (docs/NIGHT.md, «Двойники и вторая
ступень»).

Owner, 2026-10-27: «чем быстрее результат - тем понятнее куда дальше двигаться. а не ждать по
пол года». An idea whose forward test is too short (`needs-forward`: its numbers were chosen
recently, so little data came after them) is not left to wait for months. This stage gives it
two chances to get an answer the same night:

1. **The causal twin** (`qlab.twin`, docs/RETUNE.md): the same idea whose numbers are chosen by
   code every month from the past only, on Binance's history from 2019. Every day after the
   first choice is a forward test, so years of them exist at once. The twin is written next to
   the spec (`<name>-causal.yaml`), run on the stand like any idea, and, if it survives stage 1,
   goes to the second stage on Hyperliquid (`qlab.pipeline.stage2`, a consistency check with
   stage 1's long history, rules 2026-10-27.1).
2. **The second stage of a researcher's experiment** is run by `qlab.agents.researcher` and by
   the waiting stage (`qlab.waiting`); this module only decides which experiments count
   (`goes_to_stage2`).

Which ideas get a twin: an idea in work (`candidate`, `bench`, `implemented`, `speccing`) whose
latest routed trial is `needs-forward`, with exactly one spec file, that does not choose its
numbers causally already (a twin of a twin adds nothing) and is not a twin or a second stage
itself. The base of an experiment (`qlab.twin.find_base`) is looked for among the specs that
`night/research.yaml` lists for the strategy whose folder (`specs/research/<strategy>/`) the
experiment lives in, and among the other specs of that folder; an idea outside such a folder has
no base, only the declared lists.

**Bounded.** At most `max_new_twins_per_night` (night/twins.yaml) new twins a night; the rest
wait for the next night and the report names them. The night's stop time is checked before each
idea (`qlab.night.past_stop`): a started idea always finishes. A run longer than
`slow_twin_minutes` is reported as slow. **Idempotent.** An idea is handled once a night (the
night's `done` markers); an existing `<name>-causal.yaml` is never rebuilt -- the twin is then an
ordinary idea that `qlab.waiting` re-runs by its own rules; and an idea with no twin has its
reason recorded once in the ledger (`data/twins/ledger.json`) and is not tried again until its
spec file (or the base's) changes. Code only: no agent, no tokens.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qlab.pipeline.spec import StrategySpec
from qlab.twin import COIN_LIST_PARAMS, SECONDS_PER_INSTRUMENT, fetch_estimate, needs_listing

CONFIG = Path("night/twins.yaml")
LEDGER = Path("data/twins/ledger.json")
RESEARCH = Path("night/research.yaml")
WATCHLIST = Path("night/watchlist.yaml")
SPEC_DIR = Path("specs")
SECTION_TITLE = "Двойники и вторая ступень: ответ на имеющейся истории"
IN_WORK = ("candidate", "bench", "implemented", "speccing")
TWIN_METRICS = ("ann_return_net", "sharpe_net", "max_dd", "ann_return_net_ex_best_1pct",
                "noise_return_percentile", "forward_days", "forward_days_needed",
                "min_capital_usd")
BINANCE = "binance"


# --------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class Config:
    enabled: bool
    max_new_twins_per_night: int
    slow_twin_minutes: float
    max_hourly_instruments: int = 40
    """An hourly twin may fetch at most this many columns (night/twins.yaml)."""
    seconds_per_instrument: float = SECONDS_PER_INSTRUMENT
    coin_list_params: tuple[str, ...] = COIN_LIST_PARAMS


def load_config(path: Path = CONFIG) -> Config | None:
    """`night/twins.yaml`; None when the file is missing (the stage is then off: a working
    directory without the project's night folder has nothing to configure)."""
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Config(
            enabled=bool(raw["enabled"]),
            max_new_twins_per_night=int(raw["max_new_twins_per_night"]),
            slow_twin_minutes=float(raw["slow_twin_minutes"]),
            max_hourly_instruments=int(raw.get("max_hourly_instruments", 40)),
            seconds_per_instrument=float(raw.get("seconds_per_instrument",
                                                 SECONDS_PER_INSTRUMENT)),
            coin_list_params=tuple(raw.get("coin_list_params") or COIN_LIST_PARAMS))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None


# --------------------------------------------------------------------------
# Which experiments go to the second stage


def goes_to_stage2(spec: StrategySpec) -> bool:
    """Whether an experiment that survived stage 1 goes on to the second stage on Hyperliquid.
    The second stage re-builds the idea on Hyperliquid's own data; an experiment that already
    runs on Hyperliquid has nothing new to look at there, so only an experiment on Binance's
    long history (stage 1 in the sense of docs/TWO_STAGE.md) goes."""
    return spec.data.source == BINANCE


HEAVY_REASON = (
    "the twin needs Binance's hourly candles for every listed perpetual since 2019; the first "
    "measurement (2026-10-27, the twin of frab-binance-spot-research-sw24) fetched them at about "
    "25 seconds per instrument for roughly 800 instruments, hours against the night's "
    "{minutes:g}-minute bound for one twin (night/twins.yaml); it needs a short instrument list "
    "in the spec (data.instruments, required_instruments or a coin list in its params), or a "
    "cached panel, or a hand run with --allow-heavy")
BIG_LIST_REASON = (
    "the twin carries {columns} instruments and its hourly candles since 2019 take about "
    "{minutes:.0f} minutes to fetch at {seconds:g} seconds each, more than the {limit} "
    "instruments night/twins.yaml allows a night (max_hourly_instruments) against its "
    "{bound:g}-minute bound for one twin; a hand run with --allow-heavy")


def heavy_data(spec: StrategySpec, minutes: float, max_instruments: int | None = None,
               seconds_per_instrument: float = SECONDS_PER_INSTRUMENT) -> str | None:
    """Why the twin's data is too heavy to fetch within a night, or None. Binance's candles
    are fetched one instrument at a time (data/raw/binance); a daily panel of the whole
    universe is cached and quick, an hourly one of the whole universe is not, and an hourly
    one of a named list costs `qlab.twin.fetch_estimate` -- heavy past `max_instruments`
    columns (a spot column counts as one)."""
    data = spec.data
    if data.source != BINANCE or data.interval == "1d":
        return None
    if not data.instruments:
        return HEAVY_REASON.format(minutes=minutes)
    if max_instruments is not None and len(data.instruments) > max_instruments:
        return BIG_LIST_REASON.format(
            columns=len(data.instruments), limit=max_instruments, bound=minutes,
            minutes=fetch_estimate(len(data.instruments), seconds_per_instrument) / 60,
            seconds=seconds_per_instrument)
    return None


# --------------------------------------------------------------------------
# The ledger of ideas that have no twin (pure file, no database)


def _sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _read(path: Path | None) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path is not None else None
    except OSError:
        return None


def load_ledger(path: Path = LEDGER) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ledger(ledger: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


LEDGER_VERSION = 2
"""Bumped when the rules that decide "no twin" change, so a reason recorded under the old
rules is not kept for ever (version 2, 2026-10-27: a bounded hourly idea now gets a twin with
an instrument list instead of being refused as heavy)."""


def fingerprint(spec_path: Path, base_path: Path | None) -> str:
    """What a recorded "no twin" depends on: the idea's spec file and its base's, and the
    version of the rules that refused it."""
    return f"v{LEDGER_VERSION}:{_sha(_read(spec_path))}:{_sha(_read(base_path))}"


# --------------------------------------------------------------------------
# Candidates and bases


@dataclass
class Candidate:
    idea_id: str
    path: Path
    spec: StrategySpec
    status: str
    route: str | None


def _is_experiment(path: Path) -> bool:
    return "research" in path.parts


def find_candidates(session, specs: Mapping[str, list[Path]], *, only: Collection[str] | None = None
                    ) -> tuple[list[Candidate], dict[str, str]]:
    """(the ideas that can have a twin, ideas named in `only` that cannot with the reason).

    Without `only`: every idea in work whose latest routed trial is `needs-forward`. With
    `only`: those ideas, whatever their status, if their route is `needs-forward`. A twin, a
    second stage and an idea that chooses its numbers causally already are never candidates."""
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Idea
    from qlab.registry.stages import is_stage2_child, latest_routed
    from qlab.twin import RETUNE, TWIN_SUFFIX, normalize_ref
    wanted = list(only) if only is not None else [
        i.id for i in session.query(Idea).order_by(Idea.id)
        if i.status.value in IN_WORK and not is_stage2_child(i)]
    found: list[Candidate] = []
    refused: dict[str, str] = {}
    for idea_id in wanted:
        idea = session.get(Idea, idea_id)
        if idea is None:
            refused[idea_id] = "the idea is not in the registry"
            continue
        routed = latest_routed(session, idea_id)
        route = None if routed is None else getattr(routed[0].route, "value", routed[0].route)
        if is_stage2_child(idea) or idea_id.endswith(TWIN_SUFFIX):
            refused[idea_id] = "a twin or a second stage"
            continue
        if route != "needs-forward":
            refused[idea_id] = f"its latest route is {route}, not needs-forward"
            continue
        paths = specs.get(idea_id) or []
        if len(paths) != 1:
            refused[idea_id] = "no spec file" if not paths else "several spec files"
            continue
        try:
            spec = load_spec(paths[0])
        except Exception as exc:  # noqa: BLE001 - a broken file fails in its own runs
            refused[idea_id] = f"the spec does not load: {exc}"
            continue
        if spec.selects_causally or normalize_ref(spec.code_ref) == RETUNE:
            refused[idea_id] = "its numbers are chosen by code from the past already"
            continue
        found.append(Candidate(idea_id, paths[0], spec, idea.status.value, route))
    found.sort(key=lambda c: (not _is_experiment(c.path), c.idea_id))
    return found, refused


def base_candidates(path: Path, research_plan: Mapping, *, watch: Collection[str] = (),
                    root: Path = Path(".")) -> list[tuple[Path, StrategySpec]]:
    """The specs an idea at `path` may modify, with their files: those night/research.yaml lists
    for the strategy whose folder (`.../research/<strategy>/`) holds the experiment, the
    folder's other specs, and the specs on the watch list (the strategies the night follows:
    `watch`, as paths). `find_base` then picks the one that is the experiment's base."""
    from qlab.pipeline.spec import load_spec
    from qlab.twin import TWIN_SUFFIX

    files: list[Path] = []
    strategy = None
    if _is_experiment(path) and path.parent.parent.name == "research":
        strategy = (research_plan.get("strategies") or {}).get(path.parent.name)
    if strategy is not None:
        files += [root / p for p in strategy.get("specs") or []]
        files += sorted(path.parent.glob("*.yaml"))
    files += [root / p for p in watch]
    out: list[tuple[Path, StrategySpec]] = []
    seen: set[Path] = set()
    for file in files:
        if file.resolve() == path.resolve() or file.resolve() in seen:
            continue
        seen.add(file.resolve())
        if file.stem.endswith(TWIN_SUFFIX):
            continue
        try:
            out.append((file, load_spec(file)))
        except Exception:  # noqa: BLE001 - an unloadable spec is not a base
            continue
    return out


# --------------------------------------------------------------------------
# The stage


def _metrics(session, trial_id: int | None) -> dict:
    from qlab.registry.models import Trial

    trial = session.get(Trial, trial_id) if trial_id else None
    if trial is None or not trial.metrics:
        return {}
    from qlab.year_view import year_keys

    keep = (*TWIN_METRICS, *year_keys(trial.metrics))
    return {k: trial.metrics[k] for k in keep if k in trial.metrics}


def _ensure_twin_idea(twin: StrategySpec, origin: str) -> None:
    """The twin as an idea of the registry, a variant of its original (one branch, not one more
    place among the strategy's variants: night/limits.yaml)."""
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    with session_scope() as session:
        if session.get(Idea, twin.idea_id) is None:
            repo.upsert_idea(
                session, id=twin.idea_id, title=twin.title, source_type=SourceType.INTERNAL,
                asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                notes=f"Causal twin of {origin} (qlab.twin_stage, docs/RETUNE.md): its numbers "
                      "are chosen by code from the past only, on Binance's history.")
            session.flush()
        if session.get(Idea, origin) is not None:
            session.get(Idea, twin.idea_id).parent_id = origin


def _has_run(idea_id: str) -> bool:
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea
    from qlab.registry.stages import latest_routed

    with session_scope() as session:
        return session.get(Idea, idea_id) is not None and latest_routed(
            session, idea_id) is not None


def _binance_listing() -> list[str]:
    from qlab.data.sources import binance

    names = binance.discover_universe()
    return sorted({*names, *(f"{n}-SPOT" for n in names)})


def _second_stage(twin: StrategySpec, entry: dict, *, end: date, noise_trials: int,
                  deployable_capital_usd: float, config) -> None:
    """The twin's second stage when its stage 1 survived: the same call the generator makes."""
    from qlab import night
    from qlab.pipeline import stage2

    if config is None or not stage2.survives(entry.get("route"), config):
        return
    started = time.monotonic()
    outcome = stage2.run_stage2(night.with_end(twin, end), stage1_route=entry["route"],
                                stage1_trial_id=entry.get("trial"), noise_trials=noise_trials,
                                deployable_capital_usd=deployable_capital_usd, config=config)
    entry["stage2"] = asdict(outcome)
    entry["stage2_seconds"] = round(time.monotonic() - started, 1)


def _run_twin(twin: StrategySpec, origin: str, entry: dict, *, end: date, noise_trials: int,
              deployable_capital_usd: float, slow_minutes: float, stage2_config) -> None:
    """Stage 1 of the twin on the stand, then its second stage if it survived."""
    from qlab import night
    from qlab.pipeline.sources import source_problems
    from qlab.registry.db import session_scope

    problems = source_problems(twin)
    if problems:
        entry.update(action="error", error="numbers without a source, not run: "
                     + "; ".join(problems))
        return
    _ensure_twin_idea(twin, origin)
    started = time.monotonic()
    result = night.evaluate_item("twins", twin, end=end, noise_trials=noise_trials,
                                 deployable_capital_usd=deployable_capital_usd)
    elapsed = time.monotonic() - started
    entry["seconds"] = round(elapsed, 1)
    entry["slow"] = elapsed > slow_minutes * 60
    entry.update(trial=result.trial_id, route=result.route, route_reason=result.reason)
    if result.error or result.route == "error":
        entry.update(action="error", error=result.error or result.reason)
        return
    with session_scope() as session:
        entry["metrics"] = _metrics(session, result.trial_id)
    _second_stage(twin, entry, end=end, noise_trials=noise_trials,
                  deployable_capital_usd=deployable_capital_usd, config=stage2_config)


def twins_stage(day: date, end: date, state: dict, *, noise_trials: int = 200,
                deployable_capital_usd: float = 3000.0, stop_at: datetime | None = None,
                only: Collection[str] | None = None, max_new: int | None = None,
                config_path: Path = CONFIG, research_config: Path = RESEARCH,
                spec_dir: Path = SPEC_DIR, ledger_path: Path = LEDGER,
                watch_config: Path = WATCHLIST, label: str = "twins", dry_run: bool = False,
                allow_heavy: bool = False) -> None:
    """Build, run and judge the causal twins of the ideas that wait for forward data; write what
    happened to `state["twins"]` for the morning report. See the module docstring.

    `only` names the ideas to handle (then neither the nightly cap nor a recorded "no twin"
    holds them back, unless `max_new` is given); `dry_run` writes and runs nothing and only
    reports what would be done. A twin whose data is too heavy to fetch in a night
    (`heavy_data`) is not run unless `allow_heavy`."""
    from qlab import night
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope
    from qlab.waiting import spec_files

    config = load_config(config_path)
    if config is None or not config.enabled:
        return
    block = state.setdefault("twins", {"entries": [], "new": 0})
    block["deferred"], block["recorded"] = [], []
    try:
        stage2_config = stage2.load_config()
    except (OSError, ValueError) as exc:  # a broken config switches the second stage off here
        stage2_config = None
        block.setdefault("notes", []).append(f"night/stage2.yaml не прочитан: {exc}")
    run = _Run(day=day, end=end, state=state, block=block, config=config,
               stage2_config=stage2_config, noise_trials=noise_trials,
               deployable_capital_usd=deployable_capital_usd, stop_at=stop_at, label=label,
               plan=night.load_research_plan(research_config), ledger_path=ledger_path,
               ledger=load_ledger(ledger_path), forced=bool(only), dry_run=dry_run,
               allow_heavy=allow_heavy,
               watch=_watch_paths(watch_config),
               cap=max_new if max_new is not None else (
                   None if only else config.max_new_twins_per_night))
    with session_scope() as session:
        candidates, refused = find_candidates(session, spec_files(spec_dir), only=only)
    if only:
        block["refused"] = refused
    for cand in candidates:
        if f"twins:{cand.idea_id}" in state["done"]:
            continue
        try:
            if run.handle(cand) == "stop":
                break
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            run.finish(cand, {"idea": cand.idea_id, "spec": cand.path.as_posix(),
                              "action": "error", "error": f"{type(exc).__name__}: {exc}"},
                       counted=False)
    if not dry_run:
        night.save_state(day, state)


def _watch_paths(config: Path) -> list[str]:
    try:
        return list((yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("specs") or [])
    except (OSError, ValueError):
        return []


@dataclass
class _Run:
    """One call of the stage: what it needs while it walks the candidates."""

    day: date
    end: date
    state: dict
    block: dict
    config: Config
    stage2_config: object
    noise_trials: int
    deployable_capital_usd: float
    stop_at: datetime | None
    label: str
    plan: dict
    watch: list[str]
    ledger_path: Path
    ledger: dict
    forced: bool
    dry_run: bool
    allow_heavy: bool
    cap: int | None
    venue_names: list[str] | None = None

    def full(self) -> bool:
        return self.cap is not None and self.block["new"] >= self.cap

    def finish(self, cand: Candidate, entry: dict, *, counted: bool) -> None:
        from qlab import night

        if counted:
            self.block["new"] += 1
        self.block["entries"].append(entry)
        if self.dry_run:
            return  # a dry run leaves no trace: no marker, no saved state
        if entry["action"] != "error":
            self.state["done"][f"twins:{cand.idea_id}"] = datetime.now(UTC).isoformat()
        night.save_state(self.day, self.state)

    def run_twin(self, cand: Candidate, twin: StrategySpec, entry: dict) -> None:
        _run_twin(twin, cand.idea_id, entry, end=self.end, noise_trials=self.noise_trials,
                  deployable_capital_usd=self.deployable_capital_usd,
                  slow_minutes=self.config.slow_twin_minutes, stage2_config=self.stage2_config)

    def handle(self, cand: Candidate) -> str | None:
        """Handle one candidate; "stop" when the night's stop time has passed."""
        from qlab import night
        from qlab.pipeline.spec import load_spec
        from qlab.twin import find_base, twin_path

        existing = twin_path(cand.path)
        if existing.is_file():  # an ordinary idea from now on: qlab.waiting re-runs it
            twin = load_spec(existing)
            if _has_run(twin.idea_id) or self.dry_run or self.full():
                return None
            if night.past_stop(self.stop_at, self.state, f"{self.label}:{cand.idea_id}"):
                return "stop"
            entry = {"idea": cand.idea_id, "spec": cand.path.as_posix(), "action": "existing",
                     "twin": twin.idea_id, "twin_spec": existing.as_posix()}
            self.run_twin(cand, twin, entry)
            self.finish(cand, entry, counted=True)
            return None
        pool = base_candidates(cand.path, self.plan, watch=self.watch)
        choice = find_base(cand.spec, [s for _, s in pool]) if pool else None
        base = None if choice is None else choice.base
        base_file = next((f for f, s in pool if base is not None and s.idea_id == base.idea_id),
                         None)
        stamp = fingerprint(cand.path, base_file)
        if (self.ledger.get(cand.idea_id) or {}).get("fingerprint") == stamp and not self.forced:
            if cand.idea_id not in self.block.setdefault("recorded", []):
                self.block["recorded"].append(cand.idea_id)
            return None  # the reason was recorded once; nothing changed since
        if self.full():
            self.block["deferred"].append(cand.idea_id)
            return None
        if night.past_stop(self.stop_at, self.state, f"{self.label}:{cand.idea_id}"):
            return "stop"
        entry = {"idea": cand.idea_id, "spec": cand.path.as_posix(),
                 "base": None if base is None else base.idea_id}
        if choice is not None and base is None:
            entry["base_reason"] = choice.reason
        names = None
        if needs_listing(cand.spec, base, coin_params=self.config.coin_list_params):
            if self.venue_names is None:
                try:
                    self.venue_names = _binance_listing()
                except Exception as exc:  # noqa: BLE001 - transient: no ledger entry, tried again
                    entry.update(action="error", error="Binance listing not read: "
                                 f"{type(exc).__name__}: {exc}")
                    self.finish(cand, entry, counted=False)
                    return None
            names = self.venue_names
        result = _causal_twin(cand, base, base_file, names, self.config,
                              [(s, f.as_posix()) for f, s in pool])
        if result.spec is None:
            entry.update(action="none", reason=result.reason)
            if not self.dry_run:
                self.ledger[cand.idea_id] = {"reason": result.reason, "fingerprint": stamp,
                                             "day": self.day.isoformat(),
                                             "spec": cand.path.as_posix()}
                save_ledger(self.ledger, self.ledger_path)
            self.finish(cand, entry, counted=False)
            return None
        entry.update(action="made", kind=result.kind, candidates=list(result.candidates),
                     twin=result.spec.idea_id, twin_spec=existing.as_posix())
        if result.instruments:
            entry.update(instruments=list(result.instruments), list_source=result.list_source,
                         fetch_minutes=round((result.fetch_seconds or 0) / 60, 1))
        heavy = None if self.allow_heavy else heavy_data(
            result.spec, self.config.slow_twin_minutes, self.config.max_hourly_instruments,
            self.config.seconds_per_instrument)
        if heavy:
            entry.update(action="heavy", reason=heavy)
            if not self.dry_run:
                self.ledger[cand.idea_id] = {"reason": heavy, "fingerprint": stamp,
                                             "day": self.day.isoformat(),
                                             "spec": cand.path.as_posix()}
                save_ledger(self.ledger, self.ledger_path)
            self.finish(cand, entry, counted=False)
            return None
        if self.dry_run:
            entry["action"] = "would-make"
            self.finish(cand, entry, counted=False)
            return None
        from qlab.twin import write_twin

        write_twin(cand.path, result.spec, cand.idea_id)
        if self.ledger.pop(cand.idea_id, None) is not None:
            save_ledger(self.ledger, self.ledger_path)
        self.run_twin(cand, result.spec, entry)
        self.finish(cand, entry, counted=True)
        return None


def _causal_twin(cand: Candidate, base: StrategySpec | None, base_file: Path | None, names,
                 config: Config, peers=()):
    """`peers`: the specs of the same strategy (the pool the base was found in), which may date
    the twin's hand-picked list (`qlab.twin.chosen_list`)."""
    from qlab.twin import causal_twin

    return causal_twin(cand.spec, base=base, spec_path=cand.path, base_path=base_file,
                       venue_names=names, coin_params=config.coin_list_params,
                       seconds_per_instrument=config.seconds_per_instrument, peers=peers)


# --------------------------------------------------------------------------
# The morning report


def _pct(value: object) -> str:
    return f"{value:+.1%}" if isinstance(value, int | float) and value == value else "—"


def _numbers(m: Mapping[str, object]) -> str:
    sharpe, beat = m.get("sharpe_net"), m.get("noise_return_percentile")
    parts = [f"в год {_pct(m.get('ann_return_net'))}",
             f"Шарп {f'{sharpe:.2f}' if isinstance(sharpe, int | float) else '—'}",
             f"просадка {_pct(m.get('max_dd'))}"]
    if "ann_return_net_ex_best_1pct" in m:
        parts.append(f"без лучшего 1% дней {_pct(m['ann_return_net_ex_best_1pct'])}")
    if isinstance(beat, int | float):
        parts.append(f"обгоняет шум {beat:.0%}")
    days = m.get("forward_days")
    if isinstance(days, int | float) and days:
        parts.append(f"проверка вперёд {days:,.0f} дн.")
    return ", ".join(parts)


def _route_ru(route: str | None) -> str:
    from qlab.funnel import ROUTE_TITLE

    return ROUTE_TITLE.get(route or "", route or "—")


def _minutes(seconds: object) -> str:
    if not isinstance(seconds, int | float):
        return ""
    return f"{seconds / 60:.1f} мин" if seconds >= 90 else f"{seconds:.0f} с"


def entry_lines(e: Mapping) -> list[str]:
    """One idea's block of the report."""
    from qlab.pipeline.stage2 import report_lines as stage2_lines
    from qlab.year_view import years_line

    head = f"**{e['idea']}**"
    base = f" (база: `{e['base']}`)" if e.get("base") else ""
    if e["action"] == "none":
        why = f" База не выбрана: {e['base_reason']}." if e.get("base_reason") else ""
        return [f"- {head}: двойника нет — причина (слова кода): {e.get('reason')}.{why} "
                "Больше не пробуем, пока не изменится файл спека."]
    if e["action"] == "heavy":
        if e.get("instruments"):
            return [f"- {head}: двойник сделать можно{base}, но не прогнан: его список из "
                    f"{len(e['instruments'])} столбцов ({e.get('list_source')}) качается около "
                    f"{e.get('fetch_minutes')} мин, это больше предела ночи "
                    "(`max_hourly_instruments` в `night/twins.yaml`); вручную — `qlab twins run "
                    f"--only {e['idea']} --allow-heavy`. Больше не пробуем, пока не изменится "
                    "файл спека."]
        return [f"- {head}: двойник сделать можно{base}, но не прогнан: ему нужны часовые свечи "
                "всех контрактов Binance с 2019 года, а они качаются по одному инструменту "
                "(около 25 секунд на каждый, около 800 инструментов — часы вместо отведённых "
                "двадцати минут на двойника). В спеке нет ограниченного списка монет (ни "
                "`instruments`, ни `required_instruments`, ни списка монет в параметрах), поэтому "
                "двойнику нужна готовая выгрузка; вручную — `qlab twins run --only "
                f"{e['idea']} --allow-heavy`. Больше не пробуем, пока не изменится файл спека."]
    if e["action"] == "would-make":
        listed = (f"; только {len(e['instruments'])} столбцов ({e.get('list_source')}), "
                  f"выгрузка около {e.get('fetch_minutes')} мин" if e.get("instruments") else "")
        return [f"- {head}: двойник можно сделать{base}; между чем выбирает код: "
                f"{', '.join(e.get('candidates') or [])}{listed}"]
    if e["action"] == "error":
        return [f"- {head}: **сбой** — {str(e.get('error'))[:200]}; повторится в следующую ночь"]
    made = "двойник уже был, прогнан впервые" if e["action"] == "existing" else "двойник сделан"
    line = (f"- {head}: {made}{base}, файл `{e.get('twin_spec')}`; код выбирает между "
            f"{len(e.get('candidates') or [])} наборами чисел раз в месяц только по прошлому. "
            if e.get("candidates") else f"- {head}: {made}{base}, файл `{e.get('twin_spec')}`. ")
    if e.get("instruments"):
        line += (f"Двойник торгует только {len(e['instruments'])} столбцами "
                 f"({', '.join(e['instruments'])}; {e.get('list_source')}): список подобран "
                 "вручную по монетам, которые дожили до тех лет, поэтому двойник судится только "
                 "со дня, когда список уже был известен; история до него служит разгоном для "
                 "выбора кода и не судится. ")
    line += (f"Первый этап (история Binance с 2019 года), прогон {e.get('trial')}: "
             f"{_route_ru(e.get('route'))} — {_numbers(e.get('metrics') or {})}; "
             f"считалось {_minutes(e.get('seconds'))}")
    if e.get("slow"):
        line += " (**дольше нормы**, см. `night/twins.yaml`)"
    out = [line]
    years = years_line(e.get("metrics"))
    if years:
        out.append(f"    - {years}")
    if e.get("stage2") is not None:
        out += [f"    - {ln}" for ln in stage2_lines({"route": e.get("route")}, e["stage2"])]
    elif e.get("route") is not None:
        out.append(f"    - второй этап не проводился: на первом этапе исход "
                   f"«{_route_ru(e.get('route'))}»")
    return out


def report_lines(block: Mapping | None) -> list[str]:
    """The section's body for the morning report."""
    if not block:
        return []
    entries = block.get("entries") or []
    if not (entries or block.get("notes") or block.get("deferred") or block.get("refused")):
        return []  # the stage found nothing to do tonight: no section
    lines = [
        "Идея, которой мало данных после выбора её чисел, не ждёт месяцы. Её «двойник» — тот же "
        "рецепт, но числа каждый месяц выбирает код и только по прошлому; данные берутся с "
        "Binance, у которой история с 2019 года, поэтому проверка вперёд набирается за годы в одну "
        "ночь (`docs/RETUNE.md`). Двойник, прошедший первый этап, идёт на второй — на "
        "Hyperliquid (`docs/TWO_STAGE.md`). Токены не тратятся.", ""]
    lines += [f"- {n}" for n in block.get("notes") or []]
    made = [e for e in entries if e["action"] in ("made", "existing")]
    lines.append(f"Двойников прогнано: {len(made)}, без двойника: "
                 f"{sum(e['action'] in ('none', 'heavy') for e in entries)}, сбоев: "
                 f"{sum(e['action'] == 'error' for e in entries)}.")
    if block.get("recorded"):
        lines.append(f"Без двойника по записи прошлых ночей (причины в `data/twins/ledger.json`, "
                     f"заново только если изменился файл спека): {len(block['recorded'])}.")
    if block.get("deferred"):
        lines.append(f"Не уместились в ночной предел и ждут следующей ночи: "
                     f"{', '.join(block['deferred'])}.")
    for idea, why in (block.get("refused") or {}).items():
        lines.append(f"- {idea}: не подходит для двойника — {why}")
    lines.append("")
    for e in entries:
        lines += entry_lines(e)
    return lines


__all__ = ["CONFIG", "Candidate", "Config", "SECTION_TITLE", "base_candidates", "entry_lines",
           "find_candidates", "fingerprint", "goes_to_stage2", "load_config", "load_ledger",
           "report_lines", "twins_stage"]
