"""Ideas that wait for data get their forward test as the data arrives (docs/NIGHT.md).

The night's watch list re-runs the strategies somebody put on it. Everything else
that waits on data -- a bench idea the researcher declared, a generated idea, the
Hyperliquid second stage of a generated idea (docs/TWO_STAGE.md) -- was run once
and never again: "the same gap as for any unwatched bench idea" (T45, 2026-10-25).
This stage closes it. It is recheck, the cheap and mandatory first part of the
night (CLAUDE.md): pure code, no agent, no tokens, outside the budget guard.

**What is re-run.** An idea in `candidate`/`bench` whose latest routed trial says
`needs-forward` or `needs-more-data`, and that has exactly one spec file; the spec
is run again unchanged on data extended to the last closed day. For a generated
idea (`gen-*`, or any idea with a second stage) whose stage 1 survived, the second
stage is re-run too, through `qlab.pipeline.stage2.run_stage2` -- which already
combines the two stages, moves the parent's status and sends a stage 2 reject to
the graveyard with its Hyperliquid data range. Decided ideas (`paper`,
`needs-infrastructure`, and `shelf` in runs recorded before rules 2026-10-29.3: they wait for
the owner or an adapter, not for data) are not waiting for data, but their numbers go stale
all the same: since 2026-10-29 (owner: «да, по всем рекомендациям, делай», question В6 of
docs/EVALUATION_PATHS.md) each is re-run on the fresh tail at least every `decided_recheck_days`
days (night/waiting.yaml, 7), at most `decided_max_per_night` a night (4), the oldest run
first. A decided idea whose spec is on the watch list is re-run there every night.

**What is not re-run twice.** An idea whose spec is on the watch list runs there.
A second stage is run with its parent, never as an idea of its own. A name is
re-run once a night (the night's `done` markers), and an idea whose latest trial
already reaches the night's last closed day has nothing new to look at.

**When a re-run is due.** Not every night: a re-run costs a stand run (200 random
books) and writes a trial that deflation counts (`qlab.registry.deflation`), and
until the forward test is as long as the rules ask, the rules decide nothing in
EITHER direction (`qlab.pipeline.evaluate._decide_fit_forward`, docs/FIT_VS_FORWARD.md):
the run would say `needs-forward` again. So the next run is due on the first day
the route CAN change, derived from the idea's own last run, no new constant:

* at least one new closed day (the bar of the data) since the last run;
* for a run with a forward test, not before every market regime has
  `regime_coverage_days` days -- each new day adds to one regime only, so the
  shortfalls added up are the fewest days that can still be needed -- and, for a
  run split into a selection period and a forward test, not before the forward
  test reaches `forward_days_needed`;
* any other run (no forward test yet; a forward test that only lacks the noise
  bar's power, which no day count predicts) is due after one new day;
* `needs-more-data` twice in a row, for the same reason, on different data is
  stuck: a day more does not supply the missing metric, so it waits for a new
  rule set instead;
* at once, when the last run gave no outcome or the rule set changed since it.

Re-running later loses nothing: the stand is deterministic on the data, so one run
on the whole tail equals the daily runs it replaces. The days in between are
counted from the calendar in the report. The unchanged spec is the point: nothing
is refitted, so the days that arrive are forward days by construction.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SECTION_TITLE = "Идеи, ждущие данных: проверка вперёд по мере их прихода"
SPEC_DIR = Path("specs")
WAITING_ROUTES = ("needs-forward", "needs-more-data", "error")
"""Latest routes of an idea that wait for data. `paper` (the owner) and `needs-infrastructure`
(an adapter) are decided; they are not waiting for data. A crashed run (`error`) decided
nothing: it is tried again (2026-10-27: two experiments whose run crashed dropped out of the
plan and nothing would have run them again)."""
DECIDED_ROUTES = ("paper", "needs-infrastructure", "shelf")
"""Latest routes of a decided idea; `shelf` is only in runs made before rules 2026-10-29.3."""
RECHECK_STATUSES = ("candidate", "bench", "implemented", "speccing", "validated", "paper", "live")
"""Statuses of a decided idea that is re-run on the fresh tail (the open ones)."""
CONFIG = Path("night/waiting.yaml")


@dataclass(frozen=True)
class Recheck:
    """`night/waiting.yaml`: how often decided ideas are measured again."""

    enabled: bool
    decided_recheck_days: int
    decided_max_per_night: int


def load_recheck(path: Path = CONFIG) -> Recheck:
    """The re-run policy of decided ideas. A missing key is an error: nothing here has a
    default, so a typo cannot silently switch the policy off."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        config = Recheck(bool(raw["enabled"]), int(raw["decided_recheck_days"]),
                         int(raw["decided_max_per_night"]))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    if config.decided_recheck_days < 1 or config.decided_max_per_night < 0:
        raise ValueError(f"{path}: decided_recheck_days must be >= 1, "
                         "decided_max_per_night >= 0")
    return config
STAGE2_WAITING = ("needs-forward", "not-evaluable", "needs-more-data", "error")
"""Routes of a second stage that decided nothing yet (`qlab.pipeline.stage2.UNDECIDED`
and `needs-forward`)."""
IN_WORK = ("candidate", "bench", "implemented", "speccing")
"""Statuses of an idea that may wait for data. `validated` joins only for a second stage
that never finished."""


# --------------------------------------------------------------------------
# When a re-run can change anything (pure)


@dataclass(frozen=True)
class Basis:
    """What the last run of an idea (or of its second stage) leaves to decide on."""

    data_end: date | None
    """The last day of the data the run saw (its snapshot); None: no run on record."""
    route: str | None
    metrics: dict = field(default_factory=dict)
    failed: bool = False
    """The latest run gave no outcome (it crashed): try again at once."""
    rules_version: str | None = None
    stuck: bool = False
    """`needs-more-data` for the same reason on two runs whose data differed: more days
    did not supply the missing metric, so it waits for a change of rules, not of data."""


@dataclass(frozen=True)
class Due:
    due: bool
    due_end: date | None
    """The first data end at which a re-run can change the route; None: now."""
    why: str


def days_before_decision(metrics: dict, route: str | None, coverage_days: int | None) -> int:
    """The fewest new days after which a re-run could route anything but `needs-forward`.

    While the forward test is shorter than the rules ask (`forward_days_needed`, only
    for a run split into a selection period and a forward test) it decides nothing
    either way, and while some market regime has fewer than `coverage_days` days it
    waits too (`decide_fit_forward_route`). Both are necessary for a decision, so the
    larger of the two is a lower bound; each new day adds to one regime only, so the
    regimes' shortfalls added up are the fewest days coverage can still need. Without
    a forward test yet, or for any other route, there is no bound: 0."""
    forward = metrics.get("forward_days")
    if route != "needs-forward" or not forward:
        return 0
    needed = metrics.get("forward_days_needed") if "fit_window_days" in metrics else None
    length = 0 if needed is None else max(0, math.ceil(needed - forward))
    coverage = 0
    if coverage_days:
        from qlab.regimes import REGIMES

        seen = [metrics.get(f"regime_{r}_days") for r in REGIMES]
        if all(isinstance(d, int | float) for d in seen):
            coverage = sum(max(0, math.ceil(coverage_days - d)) for d in seen)
    return max(length, coverage)


def decide_due(basis: Basis, end: date, *, coverage_days: int | None = None,
               current_rules: str | None = None) -> Due:
    """Whether the idea is re-run on a night whose data ends at `end`."""
    if basis.data_end is None:
        return Due(True, None, "в реестре нет прогона с данными")
    if basis.failed:
        return Due(True, None, "прошлый прогон не дал исхода")
    if end <= basis.data_end:
        return Due(False, basis.data_end + timedelta(days=1),
                   f"новых закрытых дней нет (данные по {basis.data_end})")
    if current_rules and basis.rules_version and basis.rules_version != current_rules:
        return Due(True, None, f"набор правил сменился ({basis.rules_version} → {current_rules})")
    if basis.stuck:
        return Due(False, None, "два прогона на разных данных дали ту же причину "
                   "(нечего вычислить): ждёт смены правил, новые дни не помогут")
    wait = days_before_decision(basis.metrics, basis.route, coverage_days)
    due_end = basis.data_end + timedelta(days=max(1, wait))
    if end >= due_end:
        return Due(True, due_end, "с прошлого прогона пришли новые дни, исход уже может "
                   "измениться" if wait else "пришёл новый закрытый день")
    return Due(False, due_end, f"до дня, когда исход может измениться, ещё "
               f"{(due_end - end).days} дн.")


def decide_recheck_due(basis: Basis, end: date, days: int) -> Due:
    """Whether a DECIDED idea is re-run on a night whose data ends at `end`: when its last run's
    data is `days` days old or more (a route that waits for nothing has no day on which it can
    change, so the age of the run is the only reason), or when there is no run or the last one
    gave no outcome."""
    if basis.data_end is None:
        return Due(True, None, "в реестре нет прогона с данными")
    if basis.failed:
        return Due(True, None, "прошлый прогон не дал исхода")
    due_end = basis.data_end + timedelta(days=days)
    age = (end - basis.data_end).days
    if end >= due_end:
        return Due(True, due_end, f"решённая идея: данные прогона по {basis.data_end}, "
                   f"прошло {age} дн. (пересчёт не реже раза в {days})")
    return Due(False, due_end, f"решённая идея: данные прогона по {basis.data_end}, "
               f"пересчёт через {(due_end - end).days} дн.")


# --------------------------------------------------------------------------
# Reading the registry


def read_basis(session, idea_id: str) -> Basis:
    """The last run of `idea_id`, from the registry."""
    from qlab.registry.models import DataSnapshot, Spec, Trial, Verdict
    from qlab.registry.stages import latest_routed

    last = (session.query(Trial).join(Spec, Spec.id == Trial.spec_id)
            .filter(Spec.idea_id == idea_id).order_by(Trial.id.desc()).first())
    if last is None:
        return Basis(None, None)
    routed = latest_routed(session, idea_id)
    shown = routed[0] if routed else last

    def end_of(trial) -> date | None:
        snapshot = session.get(DataSnapshot, trial.snapshot_id) if trial.snapshot_id else None
        return snapshot.range_end if snapshot is not None else None

    version = (session.query(Verdict.rules_version).filter(Verdict.trial_id == shown.id)
               .order_by(Verdict.id.desc()).first())
    route = getattr(shown.route, "value", shown.route)
    stuck = False
    if route == "needs-more-data":
        before = (session.query(Trial).join(Spec, Spec.id == Trial.spec_id)
                  .filter(Spec.idea_id == idea_id, Trial.route.isnot(None), Trial.id < shown.id)
                  .order_by(Trial.id.desc()).first())
        stuck = (before is not None and before.route == shown.route
                 and before.route_reason == shown.route_reason
                 and None not in (end_of(before), end_of(shown))
                 and end_of(before) < end_of(shown))
    return Basis(
        data_end=end_of(shown), metrics=dict(shown.metrics or {}), route=route,
        failed=(last.id != shown.id and last.route is None) or route == "error",
        rules_version=version[0] if version else None, stuck=stuck)


def spec_files(spec_dir: Path = SPEC_DIR) -> dict[str, list[Path]]:
    """idea id -> the spec files that name it (a file that does not load is skipped:
    it fails loudly in its own run)."""
    from qlab.pipeline.spec import load_spec

    out: dict[str, list[Path]] = {}
    for path in sorted(spec_dir.rglob("*.yaml")):
        try:
            out.setdefault(load_spec(path).idea_id, []).append(path)
        except Exception:  # noqa: BLE001
            continue
    return out


def watched_idea_ids(watch: list[str]) -> set[str]:
    """The ideas the watch list runs every night."""
    from qlab.pipeline.spec import load_spec

    out: set[str] = set()
    for path in watch:
        try:
            out.add(load_spec(Path(path)).idea_id)
        except Exception:  # noqa: BLE001 - an unloadable spec fails in its own run
            continue
    return out


def _stage2_experiment(path: Path | None) -> bool:
    """Whether the spec file is an experiment (a `research` folder) on Binance's history."""
    from qlab.pipeline.spec import load_spec
    from qlab.twin_stage import goes_to_stage2

    if path is None or "research" not in path.parts:
        return False
    try:
        return goes_to_stage2(load_spec(path))
    except Exception:  # noqa: BLE001 - an unloadable spec fails loudly in its own run
        return False


# --------------------------------------------------------------------------
# The plan


@dataclass
class Plan:
    idea_id: str
    step: str
    """"idea": the spec of the idea, unchanged, on the fresh tail; "stage2": its second
    stage on Hyperliquid (`qlab.pipeline.stage2`)."""
    spec_path: str | None
    route: str | None
    due: bool
    why: str
    last_end: date | None = None
    due_end: date | None = None
    forward_days: float | None = None
    forward_days_needed: float | None = None
    skip: str | None = None
    """Why nothing can be run (no spec file, several of them)."""
    kind: str = "waiting"
    """"waiting": the idea waits for data; "recheck": it is decided and measured again on a
    schedule (`decide_recheck_due`)."""


def _pick_spec(specs: dict[str, list[Path]], idea_id: str) -> tuple[Path | None, str | None]:
    found = specs.get(idea_id) or []
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, "нет файла спека"
    return None, ("спеков несколько (" + ", ".join(p.as_posix() for p in found)
                  + "): какой гонять, код не знает")


def _plan_for(idea_id: str, step: str, path: Path | None, skip: str | None, basis: Basis,
              end: date, coverage_days: int | None, current_rules: str | None,
              recheck_days: int | None = None) -> Plan:
    if recheck_days is not None:
        due, kind = decide_recheck_due(basis, end, recheck_days), "recheck"
    else:
        due = decide_due(basis, end, coverage_days=coverage_days, current_rules=current_rules)
        kind = "waiting"
    return Plan(idea_id=idea_id, step=step, spec_path=path.as_posix() if path else None,
                route=basis.route, due=due.due and skip is None, why=skip or due.why,
                last_end=basis.data_end, due_end=due.due_end,
                forward_days=basis.metrics.get("forward_days"),
                forward_days_needed=basis.metrics.get("forward_days_needed"), skip=skip,
                kind=kind)


def cap_rechecks(plans: list[Plan], limit: int) -> list[Plan]:
    """At most `limit` decided ideas are re-run on one night, the oldest run first; the others
    wait for the next night and the plan says so. Waiting plans are not touched."""
    due = sorted((p for p in plans if p.kind == "recheck" and p.due),
                 key=lambda p: (p.last_end or date.min, p.idea_id))
    for plan in due[max(0, limit):]:
        plan.due = False
        plan.why = (f"очередь: за ночь пересчитывается не больше {limit} решённых идей "
                    f"(night/waiting.yaml); {plan.why}")
    return plans


def plan_waiting(session, *, end: date, watched: set[str], specs: dict[str, list[Path]],
                 stage2_config=None, coverage_days: int | None = None,
                 current_rules: str | None = None, recheck: Recheck | None = None) -> list[Plan]:
    """Every idea waiting for data and whether tonight re-runs it; with `recheck`, every decided
    open idea too, on its schedule. Reads only."""
    from qlab.night import is_generator_idea
    from qlab.pipeline.stage2 import survives
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.registry.stages import (
        child_id_taken,
        is_stage2_child,
        latest_routed,
        stage2_child,
    )

    plans: list[Plan] = []
    statuses = [IdeaStatus(s) for s in (*IN_WORK, "validated")]
    recheck_days = recheck.decided_recheck_days if recheck is not None and recheck.enabled \
        else None
    if recheck_days is not None:
        statuses += [IdeaStatus.PAPER, IdeaStatus.LIVE]
    for idea in session.query(Idea).filter(Idea.status.in_(statuses)).order_by(Idea.id):
        if idea.id in watched or is_stage2_child(idea):
            continue  # the watch list runs it / it runs with its parent
        routed = latest_routed(session, idea.id)
        route = None if routed is None else getattr(routed[0].route, "value", routed[0].route)
        path, skip = _pick_spec(specs, idea.id)
        if idea.status in (IdeaStatus.PAPER, IdeaStatus.LIVE):  # production: only the schedule
            if route in DECIDED_ROUTES:
                plans.append(_plan_for(idea.id, "idea", path, skip, read_basis(session, idea.id),
                                       end, coverage_days, current_rules, recheck_days))
            continue
        in_work = idea.status.value in IN_WORK
        if recheck_days is not None and route in DECIDED_ROUTES \
                and not (skip == "нет файла спека" and idea.status != IdeaStatus.BENCH):
            plans.append(_plan_for(idea.id, "idea", path, skip, read_basis(session, idea.id),
                                   end, coverage_days, current_rules, recheck_days))
        if in_work and route in WAITING_ROUTES:
            if skip == "нет файла спека" and idea.status != IdeaStatus.BENCH:
                continue  # a noise book or a bare card: not an idea anybody can re-run
            plans.append(_plan_for(idea.id, "idea", path, skip, read_basis(session, idea.id),
                                   end, coverage_days, current_rules))
        child = stage2_child(session, idea.id)
        # `<idea>-hl` may be another idea (bv2-research-shared-pool-hl): nothing is written there
        child_taken = child is None and child_id_taken(session, idea.id)
        survivor = (stage2_config is not None and stage2_config.enabled and route is not None
                    and not child_taken and survives(route, stage2_config))
        # A researcher's experiment on Binance's history goes to the second stage as a
        # generated idea does (it is run right after the experiment; this catches the ones
        # that never were, and the twins): `qlab.twin_stage.goes_to_stage2`.
        if not survivor or not (is_generator_idea(idea.id) or child is not None
                                or _stage2_experiment(path)):
            continue
        child_routed = latest_routed(session, child.id) if child is not None else None
        if child_routed is None:  # the second stage never finished
            plans.append(Plan(idea.id, "stage2", path.as_posix() if path else None, None,
                              skip is None, skip or "второй этап ещё не проведён", skip=skip))
        elif in_work and getattr(child_routed[0].route, "value",
                                 child_routed[0].route) in STAGE2_WAITING:
            plans.append(_plan_for(idea.id, "stage2", path, skip, read_basis(session, child.id),
                                   end, coverage_days, current_rules))
    return cap_rechecks(plans, recheck.decided_max_per_night) if recheck_days is not None \
        else plans


# --------------------------------------------------------------------------
# The stage


def _status(idea_id: str) -> str | None:
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea

    with session_scope() as session:
        idea = session.get(Idea, idea_id)
        return None if idea is None else idea.status.value


def _entry(plan: Plan, end: date) -> dict:
    """The report row of a plan that is not run tonight (or cannot be)."""
    entry = {"idea": plan.idea_id, "step": plan.step, "action": "skip" if plan.skip else "wait",
             "kind": plan.kind, "route_before": plan.route, "route_after": plan.route,
             "why": plan.why,
             "forward_days_needed": plan.forward_days_needed,
             "due_end": plan.due_end.isoformat() if plan.due_end else None}
    if plan.forward_days and plan.last_end is not None:
        entry["forward_days"] = plan.forward_days + max(0, (end - plan.last_end).days)
        entry["forward_days_counted"] = "calendar"
    return entry


def _rerun_idea(plan: Plan, end: date, *, noise_trials: int) -> dict:
    from qlab import night
    from qlab.pipeline.spec import load_spec

    result = night.evaluate_item("waiting", load_spec(Path(plan.spec_path)), end=end,
                                 noise_trials=noise_trials)
    # The stand catches a strategy's exception and routes the trial "error": still a failure,
    # tried again on the next run (2026-10-27 counted two as re-run and done).
    failed = bool(result.error) or result.route == "error"
    return {"idea": plan.idea_id, "step": "idea", "action": "error" if failed else "rerun",
            "kind": plan.kind,
            "route_before": result.previous_route or plan.route, "route_after": result.route,
            "reason": result.reason, "trial": result.trial_id,
            "error": result.error or (result.reason if failed else None),
            "forward_days": result.metrics.get("forward_days"),
            "forward_days_needed": result.metrics.get("forward_days_needed"),
            "status": _status(plan.idea_id), "why": plan.why}


def _rerun_stage2(plan: Plan, end: date, *, stage1: dict | None, noise_trials: int, config) -> dict:
    """The second stage on the fresh tail. `stage1` is tonight's re-run of the first
    stage when there was one; otherwise stage 1's latest outcome in the registry."""
    from qlab import night, own_numbers
    from qlab.pipeline import stage2
    from qlab.pipeline.spec import load_spec
    from qlab.registry.db import session_scope
    from qlab.registry.stages import latest_routed

    if stage1 is not None:
        route, trial_id = stage1["route_after"], stage1["trial"]
    else:
        with session_scope() as session:
            routed = latest_routed(session, plan.idea_id)
            route = getattr(routed[0].route, "value", routed[0].route) if routed else None
            trial_id = routed[0].id if routed else None
    spec = load_spec(Path(plan.spec_path))
    if own_numbers.open_params(spec):  # the values chosen at stage 1's cut, as the generator
        retune = night.rerun_spec(spec)
        with session_scope() as session:
            chosen = own_numbers.chosen_params(night.with_end(retune, end), session)
        if chosen is None:
            return {**_entry(plan, end), "action": "wait", "due_end": None,
                    "why": "на отрезе не выбрано ни одного значения: второму этапу нечего нести"}
        spec = own_numbers.fixed_at_choice(spec, chosen, retune.params_fixed_at)
    outcome = stage2.run_stage2(night.with_end(spec, end),
                         stage1_route=route, stage1_trial_id=trial_id, noise_trials=noise_trials,
                             config=config)
    m = outcome.metrics or {}
    if outcome.skipped:  # stage 1 no longer lets the idea through, or stage 2 is switched off
        return {**_entry(plan, end), "action": "wait", "why": outcome.skipped, "due_end": None}
    return {"idea": plan.idea_id, "step": "stage2",
            "action": "error" if outcome.error else "rerun",
            "route_before": plan.route, "route_after": outcome.route,
            "reason": outcome.reason, "trial": outcome.trial_id,
            "error": outcome.error,
            "forward_days": m.get("forward_days"),
            "forward_days_needed": m.get("forward_days_needed"),
            "combined": outcome.combined_route, "status": outcome.parent_status
            or _status(plan.idea_id), "data_end": outcome.data_end, "why": plan.why,
            "lines": stage2.report_lines({"route": route}, dataclasses.asdict(outcome))}


def waiting_stage(day: date, end: date, state: dict, *, watch: list[str], noise_trials: int = 200,
                  spec_dir: Path = SPEC_DIR,
                  only: Collection[str] | None = None, force: bool = False) -> None:
    """Re-run the ideas that wait for data and are due tonight; write what happened (and
    what waits until when) to `state["waiting"]` for the morning report. Resumable: a
    finished re-run has a done-marker, a failed one is tried again on the next run. `only`
    (by hand, `qlab twins run --only`) limits the plan to those ideas, and `force` runs them
    whether or not a day has come that can change the route (a spec file edited since the
    last run, for one)."""
    from qlab import night
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    try:
        stage2_config = stage2.load_config()
    except (OSError, ValueError) as exc:  # a broken config switches the second stage off here
        stage2_config = None
        state.setdefault("waiting_notes", []).append(f"night/stage2.yaml не прочитан: {exc}")
    try:
        recheck = load_recheck()
    except (OSError, ValueError) as exc:  # a broken config switches the schedule off, loudly
        recheck = None
        state.setdefault("waiting_notes", []).append(
            f"night/waiting.yaml не прочитан, решённые идеи не пересчитываются: {exc}")
    try:
        ruleset = load_latest()
        with session_scope() as session:
            plans = plan_waiting(session, end=end, watched=watched_idea_ids(watch),
                                 specs=spec_files(spec_dir), stage2_config=stage2_config,
                                 coverage_days=ruleset.regime_coverage_days,
                                 current_rules=ruleset.version, recheck=recheck)
    except Exception as exc:  # noqa: BLE001 - a report line, not a failed night
        state.setdefault("waiting_notes", []).append(
            f"список ждущих идей не составлен: {type(exc).__name__}: {exc}")
        night.save_state(day, state)
        return
    if only is not None:
        plans = [p for p in plans if p.idea_id in only]
    entries: list[dict] = state.setdefault("waiting", [])
    ran: dict[str, dict] = {}
    queue = list(plans)
    planned = {(p.idea_id, p.step) for p in plans}
    at = 0
    while at < len(queue):
        plan = queue[at]
        at += 1
        key = f"waiting:{plan.idea_id}:{plan.step}"
        if key in state["done"]:
            continue  # re-run earlier tonight: its row stays
        entries[:] = [e for e in entries if not (e["idea"] == plan.idea_id
                                                  and e["step"] == plan.step)]
        if plan.skip is not None or not (plan.due or force):
            entries.append(_entry(plan, end))
            continue
        stage1 = ran.get(plan.idea_id)
        try:
            if plan.step == "idea":
                entry = _rerun_idea(plan, end, noise_trials=noise_trials)
            elif stage1 is not None and stage1["action"] == "error":
                entry = {**_entry(plan, end), "why": "первый этап не дал исхода: второй ждёт"}
            else:
                entry = _rerun_stage2(plan, end, stage1=stage1, noise_trials=noise_trials,
                                      config=stage2_config)
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            entry = {"idea": plan.idea_id, "step": plan.step, "action": "error",
                     "route_before": plan.route, "error": f"{type(exc).__name__}: {exc}"}
        if plan.step == "idea":
            ran[plan.idea_id] = entry
            if follow := _new_survivor(plan, entry, stage2_config, planned):
                queue.insert(at, follow)  # its second stage runs right after, tonight
                planned.add((follow.idea_id, follow.step))
        entries.append(entry)
        if entry["action"] != "error":
            state["done"][key] = datetime.now(UTC).isoformat()
        night.save_state(day, state)
    night.save_state(day, state)


def _new_survivor(plan: Plan, entry: dict, stage2_config, planned: set) -> Plan | None:
    """The second-stage plan of an idea whose re-run tonight made it a survivor of stage 1
    (the plan was made before the re-run, from the old route): its second stage does not
    wait for another night."""
    from qlab.night import is_generator_idea
    from qlab.pipeline.stage2 import survives
    from qlab.registry.db import session_scope
    from qlab.registry.stages import child_id_taken, stage2_child

    if (stage2_config is None or not stage2_config.enabled or entry["action"] != "rerun"
            or (plan.idea_id, "stage2") in planned or plan.spec_path is None
            or not survives(entry.get("route_after"), stage2_config)):
        return None
    with session_scope() as session:
        has_child = stage2_child(session, plan.idea_id) is not None
        # `<idea>-hl` may be another idea, as in `plan_waiting` (2026-10-29: the report showed
        # bv2-research-shared-pool's second stage refused every night for that reason)
        if not has_child and child_id_taken(session, plan.idea_id):
            return None
    if not (is_generator_idea(plan.idea_id) or has_child
            or _stage2_experiment(Path(plan.spec_path))):
        return None
    return Plan(plan.idea_id, "stage2", plan.spec_path, entry.get("route_after"), True,
                "первый этап сегодня стал выжившим: второй ещё не проведён")


# The same steps for other stages that run an idea the way the night's waiting stage does
# (`qlab.restamp`): one idea's first stage on the stand, its second stage on Hyperliquid, and the
# plan of the second stage owed by a first stage that has just become a survivor.
rerun_idea = _rerun_idea
rerun_stage2 = _rerun_stage2
new_survivor = _new_survivor


def followed_ideas(entries: list[dict] | None) -> set[str]:
    """Ideas this stage follows (the report's "on the bench, but not followed" leaves
    them out); one with no spec file is not followed."""
    return {e["idea"] for e in entries or [] if e["action"] != "skip"}


# --------------------------------------------------------------------------
# The morning report


def _days(entry: dict) -> str:
    now, needed = entry.get("forward_days"), entry.get("forward_days_needed")
    if not isinstance(now, int | float):
        return "—"
    shown = f"{now:,.0f}" + (" (по календарю)" if entry.get("forward_days_counted") else "")
    if isinstance(needed, int | float):
        return f"{shown} из {needed:,.0f}"
    return f"{shown} (нужная длина не задана)"


def report_lines(entries: list[dict], notes: list[str] | None = None) -> list[str]:
    """The section for the morning report: which waiting ideas were re-run, how many
    forward days each has against how many it needs, and any change of outcome."""
    lines = [
        "Идея, которой не хватает данных для вывода, прогоняется заново на свежих данных с тем "
        "же описанием, ничего не подгоняется. Прогон нужен только в день, когда исход может "
        "измениться: пока проверка вперёд короче нужной, правила не решают ни в какую сторону "
        "(`docs/FIT_VS_FORWARD.md`), и прогон выдал бы то же самое и добавил бы лишнее "
        "испытание. Дни до этого срока считаются по календарю.", "",
        f"Прогнано: {sum(e['action'] == 'rerun' for e in entries)}, ждут своего дня: "
        f"{sum(e['action'] == 'wait' for e in entries)}, не прогнать: "
        f"{sum(e['action'] == 'skip' for e in entries)}, сбоев: "
        f"{sum(e['action'] == 'error' for e in entries)}.", ""]
    lines += [f"- {n}" for n in notes or []]
    if not entries:
        return lines
    lines += ["| идея | что | вперёд, дней: есть из нужных | исход: было → стало | статус | "
              "следующий прогон / причина |", "|---|---|---:|---|---|---|"]
    step = {"idea": "описание на свежем хвосте", "stage2": "второй этап (Hyperliquid)"}
    kind = {"recheck": " — решённая, по расписанию"}
    for e in sorted(entries, key=lambda e: (e["action"] == "wait", e["idea"], e["step"])):
        if e["action"] == "error":
            lines.append(f"| {e['idea']} | {step[e['step']]} | | **ошибка** | | "
                         f"{str(e.get('error'))[:140]} |")
            continue
        what = step[e["step"]] + kind.get(e.get("kind"), "")
        before, after = e.get("route_before") or "—", e.get("route_after") or "—"
        changed = (e["action"] == "rerun"
                   and e.get("route_before") not in (None, e.get("route_after")))
        route = f"{before} → {after}" + (" **(изменился)**" if changed else "")
        if e["action"] == "rerun" and e.get("combined") and e.get("combined") != after:
            route += f"; по двум этапам {e['combined']}"
        when = {"rerun": "прогнано сегодня",
                "wait": f"{e.get('due_end') or '—'}: {e['why']}",
                "skip": e["why"]}[e["action"]]
        lines.append(f"| {e['idea']} | {what} | {_days(e)} | {route} | "
                     f"{e.get('status') or '—'} | {when} |")
    for e in sorted(entries, key=lambda e: e["idea"]):
        if e["step"] == "stage2" and e["action"] == "rerun" and e.get("lines"):
            lines += ["", f"Второй этап `{e['idea']}`, подробно:", *[f"- {x}" for x in e["lines"]]]
    if any("**(изменился)**" in ln for ln in lines):
        lines += ["", "Исход изменился там, где стоит пометка; статус идеи меняет код по правилам "
                  "(`qlab.registry.lifecycle`), а отказ второго этапа уводит идею на кладбище с "
                  "диапазоном данных Hyperliquid."]
    return lines


__all__ = ["Basis", "DECIDED_ROUTES", "Due", "Plan", "Recheck", "SECTION_TITLE", "WAITING_ROUTES",
           "cap_rechecks", "days_before_decision", "decide_due", "decide_recheck_due",
           "followed_ideas", "load_recheck", "new_survivor", "plan_waiting", "read_basis",
           "report_lines", "rerun_idea", "rerun_stage2", "spec_files", "waiting_stage",
           "watched_idea_ids"]
