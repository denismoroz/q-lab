"""The night's "Пересчёт старых результатов" stage: ideas whose latest run was measured the old way
are measured again (docs/RESTAMP.md, docs/NIGHT.md).

Owner, 2026-10-28, after several changes to how results are measured landed in one day (the year
signs, the percent of the strategy's own money, stage 2's liquidity floor, the anomaly detector):
«да, делай» to a mechanism that corrects the old numbers by itself. The night only re-ran ideas
waiting for forward data; an idea that had passed (validated, on the shelf, a paper route) kept the
numbers it was measured with, so two ideas on the owner's screen could be measured two different
ways.

**Which ideas.** Every idea that is open (`live`, `paper`, `validated`, `bench`, `candidate`,
`implemented`, `speccing`; calibration noise, retired, rejected and decayed ideas are left out) with
a routed run whose

* stand version (`trial.stand_version`, `qlab.harness.version`) is older than the current one (no
  version at all is older than every one), or
* rules version (the version stamped on its verdicts) is older than the latest rules **and the two
  differ in what decides** (`rules_changed`: a version that only adds informational readings, as
  2026-10-29.1 added the protection readings, moves no verdict and is no reason to run anything
  again), or
* noise books: the run's noise bar was measured on fewer than the 200 books asked for, because some
  books gave no number and nobody drew replacements yet (`noise_books_short`; owner 2026-10-29,
  «да, чини учёт»; docs/T28_SHAPE_AWARE_BAR.md). This is a reason of its own and not a bump of the
  stand version: only the runs whose number can differ are measured again,

or whose last recomputation failed (a ledger, below). A second stage (`<idea>-hl`) is checked
the same way and is re-run with its parent; a second stage whose first stage is closed is not
re-run on its own (stage 2 moves its parent's status), and the report names it. Causal twins, an
experiment on Binance and an overlay variant are ordinary ideas with a spec file of their own; for
a variant the stored comparison with its base is rebuilt as well.

**How.** Through the very steps the night's waiting stage uses (`qlab.waiting.rerun_idea`,
`rerun_stage2`): the spec of the idea unchanged, on the last day its latest run saw, so the
snapshots on disk answer the request and nothing is fetched unless the data request itself changed
(stage 2's liquidity floor makes a new request for a strategy that picks from the whole listing);
stage 1 on the stand with its random books, then, if stage 1 still goes on, stage 2 through
`qlab.pipeline.stage2.run_stage2`. Every run is a counted trial that carries the present versions;
old trials and verdicts stay as they were, each with its own rules version and data range. Then the
idea's forward series is rebuilt (`qlab.forward_series`); the night's anomaly stage, which follows,
scans the new runs. The status moves only by the lifecycle's own rules.

**Bounded.** At most `max_ideas_per_night` ideas a night (night/restamp.yaml), the passed and live
ones first; the rest wait and the report names them. The stop time is checked before each idea
(`qlab.night.past_stop`): a started idea always finishes. **Idempotent.** An idea recomputed has
the present versions, so it is not selected again; an idea that failed is tried again on the next
nights, up to `max_attempts` times under the same versions. Code only: no agent, no tokens.
"""

from __future__ import annotations

import json
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path("night/restamp.yaml")
LEDGER = Path("data/restamp/ledger.json")
WATCHLIST = Path("night/watchlist.yaml")
SPEC_DIR = Path("specs")
SECTION_TITLE = "Пересчёт старых результатов: заново, по нынешнему способу измерения"
OPEN_STATUSES = ("live", "paper", "validated", "bench", "candidate", "implemented", "speccing")
"""Ideas that are not closed (retired, rejected, decayed)."""
MOVED_BY = 0.01
"""A yearly return or a worst fall that moved by this much (a point of the strategy's money) is
named as moved in the report."""


# --------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class Config:
    enabled: bool
    max_ideas_per_night: int
    slow_idea_minutes: float
    max_attempts: int
    passed_statuses: tuple[str, ...]
    passed_routes: tuple[str, ...]
    max_minutes_per_night: float


def load_config(path: Path = CONFIG) -> Config | None:
    """`night/restamp.yaml`; None when the file is missing (the stage is then off: a working
    directory without the project's night folder has nothing to configure)."""
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Config(
            enabled=bool(raw["enabled"]),
            max_ideas_per_night=int(raw["max_ideas_per_night"]),
            slow_idea_minutes=float(raw["slow_idea_minutes"]),
            max_attempts=int(raw["max_attempts"]),
            passed_statuses=tuple(str(s) for s in raw["passed_statuses"]),
            passed_routes=tuple(str(r) for r in raw["passed_routes"]),
            max_minutes_per_night=float(raw["max_minutes_per_night"]))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None


# --------------------------------------------------------------------------
# The ledger of failures (a pure file, no database)


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


# --------------------------------------------------------------------------
# Reading a run


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and value == value else None


def run_view(session, idea_id: str | None) -> dict | None:
    """The idea's latest routed run as the report shows it: route, yearly return and worst fall
    (on the strategy's own money when the run has them, else the stand's), the versions it was
    made and judged under, the last day of its data. None: the idea has no routed run."""
    from qlab.forward_series import _rules_version
    from qlab.registry.models import Idea
    from qlab.registry.stages import latest_routed

    if idea_id is None:
        return None
    routed = latest_routed(session, idea_id)
    idea = session.get(Idea, idea_id)
    if routed is None or idea is None:
        return None
    trial, spec = routed
    m = trial.metrics or {}
    capital = _number(m.get("cap_ann_return")) is not None
    return {
        "idea": idea_id, "trial": trial.id, "status": idea.status.value,
        "route": getattr(trial.route, "value", trial.route),
        "ann": _number(m.get("cap_ann_return" if capital else "ann_return_net")),
        "dd": _number(m.get("cap_max_dd" if capital else "max_dd")),
        "on_capital": capital,
        # The place among the noise books the rules read: on the strategy's own money when the
        # run has it (rules 2026-10-29.2), else the stand's.
        "noise": _number(m.get("noise_cap_return_percentile" if capital
                               else "noise_return_percentile")),
        "noise_short": noise_books_short(m), "noise_percentile": _number(
            m.get("noise_return_percentile")),
        "stand": trial.stand_version, "rules": _rules_version(session, trial.id),
        "end": str((spec.data_requirements or {}).get("end") or "")[:10] or None}


NOISE_BOOKS_ASKED = 200
"""The number of random books a run's noise bar is measured on."""


def decisive_part(ruleset) -> dict:
    """What of a ruleset can move a verdict: everything but its version, its parent's name and the
    informational readings (which are shown beside verdicts and decide nothing)."""
    return ruleset.model_dump(mode="json", exclude={"version", "based_on", "informational"})


def rules_changed(was: str, now: str) -> bool:
    """Whether rules version `now` decides anything differently from `was`. A version that cannot
    be loaded (a sentinel, a missing file) counts as changed: a stale run must not hide."""
    from qlab.rules import loader

    if was == now:
        return False
    try:
        return decisive_part(loader.load(was)) != decisive_part(loader.load(now))
    except Exception:  # noqa: BLE001 - when in doubt the run is measured again
        return True


def noise_books_short(metrics: Mapping | None) -> bool:
    """Whether a run's noise bar stands on fewer random books than were asked for and nobody drew
    replacements.

    A run made since the replacement exists records `noise_n_usable` and `noise_n_requested`; its
    count is what it is (a shortage that survived the bounded replacement is the honest answer)
    and is never a reason to run it again. An older run records no counts; the bar's denominator
    shows in the percentile itself: with 200 books it is a multiple of 1/200, with 199 or 150 it
    generally is not."""
    if not metrics or "noise_n_usable" in metrics:
        return False
    percentile = _number(metrics.get("noise_return_percentile"))
    if percentile is None:
        return False
    return abs(percentile * NOISE_BOOKS_ASKED - round(percentile * NOISE_BOOKS_ASKED)) > 1e-6


def stale_reasons(view: Mapping | None, stand: str, rules: str | None) -> list[dict]:
    """Why a run is out of date: the stand version it was made under and the rules version it
    was judged under, when either is older than the present one. A run with no verdicts
    (not evaluable) has no rules version and is stale only by the stand."""
    from qlab.harness.version import is_older

    if view is None:
        return []
    out = []
    if is_older(view.get("stand"), stand):
        out.append({"kind": "stand", "was": view.get("stand"), "now": stand})
    if rules is not None and view.get("rules") is not None and is_older(view["rules"], rules) \
            and rules_changed(view["rules"], rules):
        out.append({"kind": "rules", "was": view["rules"], "now": rules})
    if view.get("noise_short"):
        out.append({"kind": "noise_books", "was": view.get("noise_percentile"), "now": None})
    return out


# --------------------------------------------------------------------------
# Which ideas


@dataclass
class Item:
    idea_id: str
    spec_path: Path
    status: str
    route: str
    trial_id: int
    passed: bool
    reasons: list[dict]
    steps: list[str]
    """"idea": stage 1 on the stand (then stage 2 if it goes on); "stage2": only the second
    stage, because the first stage's run is up to date."""
    child_id: str | None = None
    variant: bool = False


def _pick_spec(session, idea, specs: Mapping[str, list[Path]], watch: Collection[str]
               ) -> tuple[Path | None, str | None]:
    """The spec file the idea's runs come from. One file: it. Several: the file whose content is
    the stored spec of the latest run, else the one file the watch list names; else none (the
    code does not guess which to run)."""
    from qlab.forward_series import spec_file_for
    from qlab.registry.stages import latest_routed

    found = specs.get(idea.id) or []
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, "нет файла спека"
    routed = latest_routed(session, idea.id)
    stored = spec_file_for(session, routed[1].id) if routed is not None else None
    if stored is not None and any(p.resolve() == stored.resolve() for p in found):
        return stored, None
    watched = [p for p in found if p.as_posix() in watch]
    if len(watched) == 1:
        return watched[0], None
    return None, ("спеков несколько (" + ", ".join(p.as_posix() for p in found)
                  + "): какой гонять, код не знает")


def find_items(session, *, specs: Mapping[str, list[Path]], watch: Collection[str],
               cfg: Config, ledger: Mapping, stand: str, rules: str | None,
               stage2_config=None, only: Collection[str] | None = None
               ) -> tuple[list[Item], dict[str, str]]:
    """(the ideas to recompute in the order they are taken, ideas that could not be with the
    reason). Without `only`: the stale ones. With `only`: those ideas, stale or not (an explicit
    request); a second stage named there stands for its first stage."""
    from qlab.overlay_variants import is_variant_id
    from qlab.pipeline.stage2 import survives
    from qlab.registry.models import Idea
    from qlab.registry.stages import is_stage2_child, stage2_child

    wanted: set[str] | None = None
    if only is not None:
        wanted = set()
        for name in only:
            idea = session.get(Idea, name)
            wanted.add(idea.parent_id if idea is not None and is_stage2_child(idea)
                       and idea.parent_id else name)
    items: list[Item] = []
    skipped: dict[str, str] = {}
    for idea in session.query(Idea).order_by(Idea.id):
        if idea.id.startswith("noise-") or idea.status.value not in OPEN_STATUSES:
            continue
        if wanted is not None and idea.id not in wanted:
            continue
        if is_stage2_child(idea):
            parent = session.get(Idea, idea.parent_id) if idea.parent_id else None
            if parent is None or parent.status.value not in OPEN_STATUSES:
                skipped[idea.id] = ("второй этап идеи, чей первый этап закрыт "
                                    f"(`{idea.parent_id}`): второй этап двигает статус "
                                    "первого, сам по себе не пересчитывается")
            continue  # an open parent runs it
        own = run_view(session, idea.id)
        if own is None:
            if wanted is not None:
                skipped[idea.id] = "у идеи нет ни одного прогона"
            continue
        child = stage2_child(session, idea.id)
        child_view = run_view(session, child.id) if child is not None else None
        reasons = stale_reasons(own, stand, rules)
        child_reasons = stale_reasons(child_view, stand, rules)
        tried = ledger.get(idea.id) or {}
        failed_before = tried.get("status") == "error"
        same_versions = tried.get("stand") == stand and tried.get("rules") == rules
        if failed_before and same_versions and tried.get("attempts", 0) >= cfg.max_attempts \
                and wanted is None:
            skipped[idea.id] = (f"пересчёт не удался {tried['attempts']} раз подряд "
                                f"({str(tried.get('error'))[:120]}); ждёт смены версии "
                                "измерения или правил, либо ручного запуска")
            continue
        if failed_before and not reasons and not child_reasons:
            reasons = [{"kind": "retry", "was": None, "now": None}]
        goes_on = stage2_config is not None and survives(own["route"], stage2_config)
        steps = []
        if reasons or wanted is not None:
            steps = ["idea"]
        elif child_reasons and goes_on:
            steps = ["stage2"]
        if not steps:
            continue
        path, why = _pick_spec(session, idea, specs, watch)
        if path is None:
            skipped[idea.id] = why or "нет файла спека"
            continue
        passed = (idea.status.value in cfg.passed_statuses or own["route"] in cfg.passed_routes
                  or (child_view is not None and child_view["route"] in cfg.passed_routes))
        items.append(Item(
            idea_id=idea.id, spec_path=path, status=idea.status.value, route=own["route"],
            trial_id=own["trial"], passed=passed,
            reasons=reasons or child_reasons, steps=steps,
            child_id=child.id if child is not None else None, variant=is_variant_id(idea.id)))
    items.sort(key=lambda i: (not i.passed, -len(i.reasons), -i.trial_id, i.idea_id))
    return items, skipped


# --------------------------------------------------------------------------
# The stage


def restamp_stage(day: date, end: date, state: dict, *, noise_trials: int = 200,
                  stop_at: datetime | None = None,
                  only: Collection[str] | None = None, no_cap: bool = False,
                  config_path: Path = CONFIG, spec_dir: Path = SPEC_DIR,
                  ledger_path: Path = LEDGER, watch_config: Path = WATCHLIST,
                  label: str = "restamp", dry_run: bool = False,
                  max_ideas: int | None = None, allow_heavy: bool = False,
                  fetch_config: Path | None = None) -> None:
    """Recompute the ideas whose latest run was measured an older way; write what happened to
    `state["restamp"]` for the morning report. See the module docstring.

    `only` names the ideas (stale or not; the nightly cap does not hold them back unless
    `max_ideas` is given); `no_cap` lifts the cap for the stale ones (`--all`); `dry_run` writes
    and runs nothing and only lists what would be done. `end` is the night's data end, kept for the
    signature of the night's stages: every idea is run to the last day its own latest run saw.

    Bounded in time as well: `max_minutes_per_night` for the stage, checked between ideas (not for
    named ideas), and an idea whose Hyperliquid data is not cached and would take longer to fetch
    than that is not run (`qlab.hl_fetch`): it is reported «данные слишком тяжелы за ночь» and
    `allow_heavy` runs it by hand."""
    from qlab import night, waiting
    from qlab.harness.version import STAND_VERSION
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    cfg = load_config(config_path)
    if cfg is None or not cfg.enabled:
        return
    block = state.setdefault("restamp", {"entries": [], "skipped": {}, "deferred": [],
                                         "notes": []})
    block.update(deferred=[], skipped={}, stand=STAND_VERSION)
    try:
        stage2_config = stage2.load_config()
    except (OSError, ValueError) as exc:  # a broken config switches the second stage off here
        stage2_config = None
        block["notes"].append(f"night/stage2.yaml не прочитан: {exc}")
    rules = load_latest().version
    block["rules"] = rules
    ledger = load_ledger(ledger_path)
    watch = _watch_paths(watch_config)
    specs = waiting.spec_files(spec_dir)
    done = state.setdefault("done", {})
    with session_scope() as session:
        items, skipped = find_items(
            session, specs=specs, watch=watch, cfg=cfg, ledger=ledger,
            stand=STAND_VERSION, rules=rules, stage2_config=stage2_config, only=only)
    block["skipped"] = skipped
    cap = max_ideas if max_ideas is not None else (
        None if (only or no_cap) else cfg.max_ideas_per_night)
    run = _Run(day=day, state=state, block=block, cfg=cfg, stage2_config=stage2_config,
               noise_trials=noise_trials,
               ledger=ledger, ledger_path=ledger_path, specs=specs,
               dry_run=dry_run, stand=STAND_VERSION, rules=rules)
    handled = 0
    started = time.monotonic()
    limit_seconds = None if only else cfg.max_minutes_per_night * 60
    block["limit_minutes"] = cfg.max_minutes_per_night
    block["heavy"] = {}
    block.pop("time_up", None)
    from qlab import hl_fetch

    cost = hl_fetch.load_config(fetch_config) if fetch_config else hl_fetch.load_config()
    for item in items:
        if f"{label}:{item.idea_id}" in done and not dry_run:
            continue
        if cap is not None and handled >= cap:
            block["deferred"].append(item.idea_id)
            continue
        if night.past_stop(stop_at, state, f"{label}:{item.idea_id}"):
            break
        if limit_seconds is not None and time.monotonic() - started >= limit_seconds:
            block["time_up"] = True
            block["deferred"].append(item.idea_id)
            continue
        if not allow_heavy:
            heavy = run.heavy(item, cost, spent=time.monotonic() - started,
                              limit_minutes=cfg.max_minutes_per_night)
            if heavy is not None:
                block["heavy"][item.idea_id] = heavy  # no slot of the night used: next idea
                continue
        handled += 1
        if dry_run:
            block["entries"].append({"idea": item.idea_id, "action": "would-restamp",
                                     "steps": item.steps, "reasons": item.reasons,
                                     "status_before": item.status, "route_before": item.route})
            continue
        entry = run.handle(item)
        block["entries"].append(entry)
        if entry["action"] != "error":
            done[f"{label}:{item.idea_id}"] = datetime.now(UTC).isoformat()
        night.save_state(day, state)
    if not dry_run:
        night.save_state(day, state)


def _watch_paths(config: Path) -> list[str]:
    try:
        return list((yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("specs") or [])
    except (OSError, ValueError):
        return []


@dataclass
class _Run:
    """One call of the stage: what it needs while it walks the ideas."""

    day: date
    state: dict
    block: dict
    cfg: Config
    stage2_config: object
    noise_trials: int
    ledger: dict
    ledger_path: Path
    specs: Mapping[str, list[Path]]
    dry_run: bool
    stand: str
    rules: str | None

    def heavy(self, item: Item, cost, *, spent: float, limit_minutes: float) -> str | None:
        """Why the idea's Hyperliquid data is too heavy to fetch tonight, or None (cached, quick,
        or an idea with no second stage). Asked before anything is run, so an idea is never
        half recomputed."""
        from qlab import hl_fetch, night
        from qlab.pipeline import stage2
        from qlab.pipeline.spec import load_spec
        from qlab.registry.db import session_scope

        if self.stage2_config is None or not (item.child_id or "stage2" in item.steps):
            return None
        try:
            with session_scope() as session:
                own = run_view(session, item.idea_id)
                child = run_view(session, item.child_id)
                # The same day `recompute` will run to.
                own_end = _date(((own if "idea" in item.steps else child or own) or {}).get("end"))
                spec = night.with_end(load_spec(item.spec_path), own_end)
                names = (stage2.venue_listing()
                         if spec.data.instruments is not None
                         or spec.required_instruments is not None else None)
                translation = stage2.translate(spec, self.stage2_config, venue_names=names)
                if translation.spec is None:
                    return None
                est = hl_fetch.estimate(translation.spec, session, cost)
        except Exception:  # noqa: BLE001 - an estimate is advice; the run reports its own errors
            return None
        reason = hl_fetch.heavy_reason(est, limit_minutes)
        if reason is None and not est.cached and est.minutes * 60 > limit_minutes * 60 - spent:
            return (f"{hl_fetch.HEAVY_RU}: не уместится в остаток времени этапа ({est.why}); "
                    "ночью не запускается")
        return reason

    def handle(self, item: Item) -> dict:
        """Recompute one idea (stage 1, then stage 2) and describe it; a failure is the entry's
        `error`, never an exception."""
        from qlab.registry.db import session_scope

        started = time.monotonic()
        with session_scope() as session:
            before = {"idea": run_view(session, item.idea_id),
                      "stage2": run_view(session, item.child_id)}
        entry: dict[str, Any] = {
            "idea": item.idea_id, "action": "restamped", "steps": item.steps,
            "reasons": item.reasons, "variant": item.variant, "before": before, "notes": []}
        try:
            self.recompute(item, entry, before)
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            entry.update(action="error", error=f"{type(exc).__name__}: {exc}")
        elapsed = time.monotonic() - started
        entry["seconds"] = round(elapsed, 1)
        entry["slow"] = elapsed > self.cfg.slow_idea_minutes * 60
        with session_scope() as session:
            entry["after"] = {"idea": run_view(session, item.idea_id),
                              "stage2": run_view(session, item.child_id)}
        if entry["action"] != "error":
            self.series(item, entry)
        self.record(item, entry)
        return entry

    def recompute(self, item: Item, entry: dict, before: dict) -> None:
        from qlab import waiting
        from qlab.pipeline.stage2 import survives
        from qlab.registry.db import session_scope
        from qlab.registry.stages import stage2_child

        own, child = before["idea"], before["stage2"]
        path = item.spec_path.as_posix()
        end = _date((own if "idea" in item.steps else child or own)["end"])
        stage1 = None
        plan = waiting.Plan(item.idea_id, "idea", path, item.route, True,
                            "пересчёт: " + _reasons_ru(item.reasons))
        if "idea" in item.steps:
            stage1 = waiting.rerun_idea(plan, end, noise_trials=self.noise_trials)
            entry["stage1"] = _brief(stage1)
            if stage1["action"] == "error":
                entry.update(action="error", error=f"первый этап: {stage1.get('error')}")
                return
            plan2 = waiting.new_survivor(plan, stage1, self.stage2_config, set())
            if plan2 is None:
                with session_scope() as session:
                    owes = stage2_child(session, item.idea_id) is not None
                if owes:
                    entry["notes"].append(
                        f"второй этап не пересчитан: первый этап теперь «{stage1['route_after']}», "
                        "а дальше идут только выжившие")
                self.variant_comparison(item, entry, end)
                return
        else:
            plan2 = waiting.Plan(item.idea_id, "stage2", path, item.route, True,
                                 "пересчёт: " + _reasons_ru(item.reasons))
        entry["steps"] = [*item.steps, *(["stage2"] if "stage2" not in item.steps else [])]
        second = waiting.rerun_stage2(
            plan2, end, stage1=stage1, noise_trials=self.noise_trials, config=self.stage2_config)
        entry["stage2"] = _brief(second)
        if second["action"] == "error":
            entry.update(action="error", error=f"второй этап: {second.get('error')}")
        elif second["action"] == "wait":
            entry["notes"].append(f"второй этап не проведён: {second.get('why')}")
        if self.stage2_config is not None and stage1 is not None \
                and not survives(stage1.get("route_after"), self.stage2_config):
            entry["notes"].append("первый этап больше не пропускает дальше")
        self.variant_comparison(item, entry, end)

    def variant_comparison(self, item: Item, entry: dict, end: date) -> None:
        """An overlay variant's stored comparison with its base, rebuilt on the new run
        (qlab.overlay_compare; no stand trial of its own)."""
        if not item.variant or "stage1" not in entry:
            return
        from qlab import night, overlay_compare, overlay_variants
        from qlab.pipeline.spec import load_spec
        from qlab.registry.db import session_scope

        try:
            base_files = self.specs.get(overlay_variants.base_of(item.idea_id) or "") or []
            if len(base_files) != 1:
                raise ValueError("the base has no single spec file")
            variant = night.with_end(load_spec(item.spec_path), end)
            shared = load_spec(base_files[0]).model_copy(update={"data": variant.data})
            with session_scope() as session:
                record = overlay_compare.compare_specs(shared, variant, session)
            record["trial_id"] = entry["stage1"].get("trial")
            overlay_compare.save_record(record)
            entry["notes"].append("сравнение с базой пересчитано")
        except Exception as exc:  # noqa: BLE001 - the run stands; the comparison is retried by hand
            entry["notes"].append(f"сравнение с базой не пересчитано: {type(exc).__name__}: "
                                  f"{str(exc)[:120]}")

    def series(self, item: Item, entry: dict) -> None:
        """The forward series of the idea's stages, built where a stage has none (a passed run
        stores its own when `evaluate_spec` writes it)."""
        from qlab import forward_series
        from qlab.registry.db import session_scope

        try:
            with session_scope() as session:
                done = forward_series.rebuild(session, [item.idea_id])
            entry["series"] = [outcome for _, _, outcome in done]
        except Exception as exc:  # noqa: BLE001 - a chart is not worth failing the recomputation
            entry["series_error"] = f"{type(exc).__name__}: {exc}"

    def record(self, item: Item, entry: dict) -> None:
        if self.dry_run:
            return
        if entry["action"] == "error":
            earlier = self.ledger.get(item.idea_id) or {}
            same = earlier.get("stand") == self.stand and earlier.get("rules") == self.rules
            attempts = (earlier.get("attempts", 0) if same and earlier.get("status") == "error"
                        else 0) + 1
            self.ledger[item.idea_id] = {
                "status": "error", "attempts": attempts, "error": entry.get("error"),
                "stand": self.stand, "rules": self.rules, "day": self.day.isoformat()}
        else:
            self.ledger[item.idea_id] = {"status": "done", "stand": self.stand,
                                         "rules": self.rules, "day": self.day.isoformat()}
        save_ledger(self.ledger, self.ledger_path)


def _date(value: str | None) -> date:
    if not value:
        raise ValueError("the idea's latest run records no end of data")
    return date.fromisoformat(value[:10])


def _brief(step: Mapping) -> dict:
    """What the report keeps of a step's result."""
    return {k: step.get(k) for k in ("action", "route_before", "route_after", "trial", "error",
                                     "why", "combined", "status") if k in step}


# --------------------------------------------------------------------------
# The morning report


def pct(value: object) -> str:
    return f"{value:+.1%}".replace("-", "−") if isinstance(value, int | float) \
        and value == value else "—"


def _reasons_ru(reasons: list[dict]) -> str:
    parts = []
    for r in reasons:
        if r["kind"] == "stand":
            was = r.get("was") or "версии не было записано"
            parts.append(f"способ измерения сменился ({was} → {r['now']})")
        elif r["kind"] == "rules":
            parts.append(f"правила сменились ({r['was']} → {r['now']})")
        elif r["kind"] == "noise_books":
            parts.append("шумовых книг в замере меньше 200: часть книг не дала числа, и замену "
                         "им тогда не тянули")
        else:
            parts.append("прошлый пересчёт не удался")
    return "; ".join(parts)


def _route_ru(route: str | None) -> str:
    from qlab.funnel import ROUTE_TITLE

    return ROUTE_TITLE.get(route or "", route or "—")


def _moved(before: Mapping | None, after: Mapping | None) -> bool:
    if before is None or after is None:
        return False
    if before["route"] != after["route"] or before["status"] != after["status"]:
        return True
    return any(isinstance(before[k], float) and isinstance(after[k], float)
               and abs(before[k] - after[k]) >= MOVED_BY for k in ("ann", "dd"))


def stage_line(title: str, before: Mapping | None, after: Mapping | None) -> str | None:
    if before is None and after is None:
        return None
    if after is None or before is None or before["trial"] == after["trial"]:
        return f"{title}: нового прогона нет"
    route = (_route_ru(after["route"]) if before["route"] == after["route"]
             else f"{_route_ru(before['route'])} → {_route_ru(after['route'])}")
    base = "на деньгах стратегии" if after["on_capital"] else "по счёту стенда"
    old = "" if before["on_capital"] == after["on_capital"] else (
        " (раньше считалось по счёту стенда, не по деньгам стратегии)" if after["on_capital"]
        else "")
    noise = ""
    if isinstance(before.get("noise"), float) and isinstance(after.get("noise"), float) \
            and abs(before["noise"] - after["noise"]) >= 0.005:
        noise = (f"; место среди случайных книг {before['noise']:.1%} → {after['noise']:.1%} "
                 "(планка 99%)")
    return (f"{title}, прогон {after['trial']}: {route}; в год {base} "
            f"{pct(before['ann'])} → {pct(after['ann'])}, худшее падение "
            f"{pct(before['dd'])} → {pct(after['dd'])}{noise}{old}")


def entry_lines(e: Mapping) -> list[str]:
    head = f"**{e['idea']}**"
    why = _reasons_ru(e.get("reasons") or [])
    if e["action"] == "would-restamp":
        steps = " и ".join({"idea": "первый этап", "stage2": "второй этап"}[s]
                           for s in e.get("steps") or [])
        return [f"- {head}: был бы пересчитан ({steps}); причина: {why}"]
    before, after = e.get("before") or {}, e.get("after") or {}
    status = (f"статус {before['idea']['status']}" if before.get("idea") else "")
    if before.get("idea") and after.get("idea") and before["idea"]["status"] != after["idea"][
            "status"]:
        status = f"**статус {before['idea']['status']} → {after['idea']['status']}**"
    if e["action"] == "error":
        lines = [f"- {head}: **сбой** — {str(e.get('error'))[:200]}; причина пересчёта: {why}. "
                 "Повторится в следующую ночь."]
    else:
        took = f"{e['seconds'] / 60:.1f} мин" if e.get("seconds", 0) >= 90 \
            else f"{e.get('seconds', 0):.0f} с"
        slow = " (**дольше нормы**, см. `night/restamp.yaml`)" if e.get("slow") else ""
        lines = [f"- {head}: пересчитана ({took}{slow}); причина: {why}; {status}."]
    for title, key in (("Первый этап", "idea"), ("Второй этап (Hyperliquid)", "stage2")):
        line = stage_line(title, before.get(key), after.get(key))
        if line:
            lines.append(f"    - {line}")
    lines += [f"    - {n}" for n in e.get("notes") or []]
    return lines


def report_lines(block: Mapping | None, *, even_if_quiet: bool = False) -> list[str]:
    """The section's body for the morning report. A night on which nothing was recomputed or
    waiting has no section, whatever stays skipped for good (a second stage whose first stage is
    closed would be named every morning); a hand run asks for the whole picture."""
    if not block:
        return []
    entries = block.get("entries") or []
    if not (entries or block.get("deferred") or block.get("notes") or block.get("heavy")
            or (even_if_quiet and block.get("skipped"))):
        return []
    done = [e for e in entries if e["action"] == "restamped"]
    moved_route = [e for e in done if _moved_route(e)]
    moved_status = [e for e in done if _moved_status(e)]
    lines = [
        "Результат — это число вместе со способом, которым его измерили. Когда способ меняется "
        "(другая основа процентов, другие данные для второго этапа, новый набор правил), старые "
        "числа остаются как были и перестают быть сравнимы с новыми. Идея, чей последний прогон "
        "сделан по старому способу, прогоняется заново: то же описание на тех же днях, тем же "
        "путём, каким ночь прогоняет идеи такого рода; старые прогоны и вердикты не трогаются, "
        "каждый хранит свои правила и диапазон данных (`docs/RESTAMP.md`). Способ измерения "
        f"сейчас: `{block.get('stand')}`, правила `{block.get('rules')}`. Токены не тратятся.", "",
        f"Пересчитано: {len(done)}, из них исход изменился у {len(moved_route)}, "
        f"статус у {len(moved_status)}; сбоев: {sum(e['action'] == 'error' for e in entries)}.",
    ]
    if block.get("deferred"):
        lines.append("Не уместились в ночной предел и ждут следующей ночи: "
                     f"{', '.join(block['deferred'])}.")
    if block.get("time_up"):
        lines.append(f"Время этапа ({block.get('limit_minutes'):g} минут на ночь) вышло: остальные "
                     "ждут следующей ночи.")
    for idea, why in (block.get("heavy") or {}).items():
        lines.append(f"- {idea}: не запущена — {why}. Вручную: "
                     f"`qlab restamp run --idea {idea} --allow-heavy`")
    lines += [f"- {n}" for n in block.get("notes") or []]
    for idea, why in (block.get("skipped") or {}).items():
        lines.append(f"- {idea}: не пересчитана — {why}")
    lines.append("")
    for e in entries:
        lines += entry_lines(e)
    return lines


def _moved_route(e: Mapping) -> bool:
    b, a = (e.get("before") or {}), (e.get("after") or {})
    return any(b.get(k) and a.get(k) and b[k]["route"] != a[k]["route"]
               for k in ("idea", "stage2"))


def _moved_status(e: Mapping) -> bool:
    b, a = (e.get("before") or {}).get("idea"), (e.get("after") or {}).get("idea")
    return bool(b and a and b["status"] != a["status"])


__all__ = ["CONFIG", "Config", "Item", "SECTION_TITLE", "entry_lines", "find_items", "load_config",
           "load_ledger", "noise_books_short", "report_lines", "restamp_stage", "rules_changed",
           "run_view", "stale_reasons"]
