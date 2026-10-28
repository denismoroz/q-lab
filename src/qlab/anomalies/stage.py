"""The night's anomaly stage (docs/ANOMALIES.md): `anomalies_stage(day, end, state, ...)`.

Written like `qlab.twin_stage.twins_stage` so the night can call it in one line (the wiring is a
separate step, docs/ANOMALIES.md, «Что осталось подключить»). Two parts:

1. **The code part, always.** The latest routed run of every idea whose route is one of
   `scan.routes` (the runs that passed) that has no scan under the current version of
   `night/anomalies.yaml` is re-run from its stored spec and snapshot, the seven flags are
   computed and stored (`anomaly_scan`). No agent, no tokens. At most `max_scans_per_night`.
2. **The agent part, bounded and guarded.** For scans where at least one flag fired and the
   investigator has not yet looked under the same flags (a ledger, `data/anomalies/ledger.json`,
   keyed by trial, holds the fingerprint of the fired flags and the lines' version), up to
   `max_per_night` investigations in priority order -- passed ideas first -- each through the
   budget guard (stage «перепроверка», its own ceiling). A budget stop saves what is done and
   leaves the rest for the next night; it is the normal way to end, not a failure.

Idempotent: a scan is stored once per trial and version; a trial is investigated once per set of
fired flags; both are also marked in the night's `state["done"]` so a restarted night does not
repeat them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from qlab.anomalies import flags as fl
from qlab.anomalies import report, store
from qlab.anomalies.config import CONFIG, Config, load_config
from qlab.anomalies.scan import scan_trial
from qlab.pipeline.sources import QLAB_ROOT

LEDGER = Path("data/anomalies/ledger.json")
MAX_ATTEMPTS_KEY = "attempts"


# --------------------------------------------------------------------------
# The ledger (pure file, no database)


def load_ledger(path: Path = LEDGER) -> dict:
    target = path if path.is_absolute() else QLAB_ROOT / path
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ledger(ledger: dict, path: Path = LEDGER) -> None:
    target = path if path.is_absolute() else QLAB_ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(target)


# --------------------------------------------------------------------------
# Which runs


def scan_candidates(session, cfg: Config, *, only: Collection[str] | None = None,
                    rescan: bool = False) -> list[tuple[int, str]]:
    """(trial id, idea id) of the runs to scan: for each idea (calibration noise excluded) its
    latest routed run when the route is one of `scan.routes` and the run has no scan under the
    current version (`rescan`: also those that have one). Passed (`paper`) runs first, newest
    first."""
    from qlab.registry.models import Idea
    from qlab.registry.stages import latest_routed

    found: list[tuple[int, str, str]] = []
    for idea in session.query(Idea).order_by(Idea.id):
        if idea.id.startswith("noise-") or (only is not None and idea.id not in only):
            continue
        routed = latest_routed(session, idea.id)
        if routed is None:
            continue
        trial = routed[0]
        route = str(trial.route.value)
        if route not in cfg.scan.routes or trial.metrics is None:
            continue
        if not rescan and store.scan_exists(session, trial.id, cfg.version):
            continue
        found.append((trial.id, idea.id, route))
    found.sort(key=lambda t: (t[2] not in cfg.investigator.passed_routes, -t[0]))
    return [(trial_id, idea_id) for trial_id, idea_id, _ in found]


def scan_entry(result) -> dict[str, Any]:
    sentences = [report.flag_sentence(f) for f in result.flags if f.fired]
    return {"idea": result.idea_id, "trial": result.trial_id, "route": result.route,
            "fired": result.fired, "sentences": sentences,
            "not_defined": sum(f.state == fl.UNDEFINED for f in result.flags)}


def scan_one(session, trial_id: int, cfg: Config, *, force: bool = False) -> str:
    """Scan one trial and store it; a line for the CLI."""
    if store.scan_exists(session, trial_id, cfg.version) and not force:
        return f"already scanned under {cfg.version} (use --force)"
    result = scan_trial(session, trial_id, cfg)
    store.save_scan(session, result, cfg.version)
    return (f"{result.idea_id}: fired {', '.join(result.fired) or 'none'}; "
            f"not defined {scan_entry(result)['not_defined']}")


def priority(row, cfg: Config) -> tuple:
    """Order of investigation: passed ideas on their way to the owner first, then the run with
    more flags, then the newer."""
    return (row.route not in cfg.investigator.passed_routes, -len(row.fired), -row.trial_id)


def investigation_candidates(session, cfg: Config, ledger: dict, *,
                             only: Collection[str] | None = None) -> list:
    """The latest scan of each trial where a flag fired and the investigator has not looked
    under the same flags (or looked and failed too often)."""
    out = []
    for row in store.latest_scans(session, all_scans=False):
        if row.config_version != cfg.version or not row.fired:
            continue
        if only is not None and row.idea_id not in only:
            continue
        stamp = store.fingerprint(row.fired, row.config_version)
        seen = ledger.get(str(row.trial_id)) or {}
        if only is None and seen.get("fingerprint") == stamp:
            if seen.get("status") == "done" or (
                    seen.get("status") == "error"
                    and seen.get(MAX_ATTEMPTS_KEY, 0) >= cfg.investigator.max_attempts):
                continue
        done = store.latest_investigation(session, trial_id=row.trial_id)
        if only is None and done is not None and done.flags_fingerprint == stamp:
            continue
        out.append(row)
    out.sort(key=lambda r: priority(r, cfg))
    return out


# --------------------------------------------------------------------------
# The stage


def default_budget(session):
    """Tonight's budget from a fresh probe of the shared windows (docs/BUDGET.md)."""
    from qlab.budget import NightBudget, probe

    return NightBudget.open(session, probe(session).report.reading)


def anomalies_stage(day: date, end: date, state: dict, *, stop_at: datetime | None = None,
                    only: Collection[str] | None = None, max_scans: int | None = None,
                    max_investigations: int | None = None, config_path: Path = CONFIG,
                    ledger_path: Path = LEDGER, queue_path: Path | None = None,
                    budget_factory: Callable[[Any], Any] | None = None, agent: bool = True,
                    label: str = "anomalies", dry_run: bool = False) -> None:
    """Scan tonight's runs and investigate the flagged ones; write what happened to
    `state["anomalies"]` for the morning report (`qlab.anomalies.report.report_lines`).

    `only` names the ideas to handle; `agent=False` runs the code part only; `dry_run` writes
    and runs nothing and only lists what would be done. `budget_factory(session)` returns the
    `NightBudget` (default: a fresh probe); `end` is the night's data end, kept for the
    signature of the night's stages."""
    from qlab import night
    from qlab.budget import BudgetExhausted, NightExhausted, Stage
    from qlab.registry.db import session_scope

    cfg = load_config(config_path)
    if cfg is None or not cfg.enabled:
        return
    block = state.setdefault("anomalies", {"scans": [], "investigations": [], "notes": []})
    block["deferred"] = []
    done = state.setdefault("done", {})

    # 1. The code part.
    cap = max_scans if max_scans is not None else cfg.scan.max_scans_per_night
    with session_scope() as session:
        candidates = scan_candidates(session, cfg, only=only)
    made = 0
    for trial_id, idea_id in candidates:
        if f"{label}:scan:{trial_id}" in done:
            continue
        if made >= cap:
            block["deferred"].append(trial_id)
            continue
        if night.past_stop(stop_at, state, f"{label}:scan:{trial_id}"):
            break
        if dry_run:
            block["scans"].append({"idea": idea_id, "trial": trial_id, "fired": [],
                                   "action": "would-scan"})
            continue
        try:
            with session_scope() as session:
                result = scan_trial(session, trial_id, cfg)
                store.save_scan(session, result, cfg.version)
            block["scans"].append(scan_entry(result))
            done[f"{label}:scan:{trial_id}"] = datetime.now(UTC).isoformat()
            made += 1
        except Exception as exc:  # noqa: BLE001 - one run's failure must not stop the others
            block["scans"].append({"idea": idea_id, "trial": trial_id, "route": None,
                                   "error": f"{type(exc).__name__}: {exc}"})
        night.save_state(day, state)

    # 2. The agent part.
    if not agent or dry_run:
        return
    ledger = load_ledger(ledger_path)
    with session_scope() as session:
        todo = [(r.id, r.trial_id, r.idea_id, list(r.fired)) for r in investigation_candidates(
            session, cfg, ledger, only=only) if f"{label}:inv:{r.trial_id}" not in done]
    limit = max_investigations if max_investigations is not None else cfg.investigator.max_per_night
    todo = todo[:limit]
    if not todo:
        return
    from qlab.anomalies.investigator import investigate
    from qlab.registry.models import AnomalyScan

    try:
        with session_scope() as session:
            night_budget = (budget_factory or default_budget)(session)
            stage = night_budget.stage(Stage.RECHECK, candidates=len(todo))
            for scan_id, trial_id, idea_id, fired in todo:
                entry: dict[str, Any] = {"idea": idea_id, "trial": trial_id}
                stamp = store.fingerprint(fired, cfg.version)
                key = str(trial_id)
                try:
                    candidate = stage.candidate(idea_id)
                    scan = session.get(AnomalyScan, scan_id)
                    new = investigate(session, scan, cfg, budget=candidate, day=day,
                                      queue_path=queue_path)
                    entry.update(cause=new.cause, summary=new.summary, caught_by=new.caught_by,
                                 new_check=new.new_check,
                                 programmer_requests=new.programmer_requests,
                                 tokens=new.tokens)
                    ledger[key] = {"fingerprint": stamp, "day": day.isoformat(), "status": "done",
                                   "cause": new.cause, MAX_ATTEMPTS_KEY: 0}
                    done[f"{label}:inv:{trial_id}"] = datetime.now(UTC).isoformat()
                except NightExhausted as stop:
                    entry["stopped"] = str(stop)
                    block["investigations"].append(entry)
                    session.commit()
                    break
                except BudgetExhausted as stop:  # this candidate's own cap: the others go on
                    entry["stopped"] = str(stop)
                except Exception as exc:  # noqa: BLE001 - recorded; tried again, up to a bound
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    earlier = ledger.get(key) or {}
                    attempts = (earlier.get(MAX_ATTEMPTS_KEY, 0)
                                if earlier.get("fingerprint") == stamp else 0) + 1
                    ledger[key] = {"fingerprint": stamp, "day": day.isoformat(),
                                   "status": "error", "error": entry["error"],
                                   MAX_ATTEMPTS_KEY: attempts}
                block["investigations"].append(entry)
                session.commit()  # the call's token row stays whatever happened after it
                save_ledger(ledger, ledger_path)
                night.save_state(day, state)
    except NightExhausted as stop:  # the account, the windows or the night's target
        block.setdefault("notes", []).append(f"разбор агентом не начат: {stop}")
    save_ledger(ledger, ledger_path)
    night.save_state(day, state)


def investigate_by_hand(trial_id: int, *, force: bool = False) -> str:
    """`qlab anomalies investigate`: one investigation now, through the guard (stage
    recheck). Scans the trial first if it has no scan. Returns the dossier."""
    from qlab.anomalies.investigator import investigate
    from qlab.budget import Stage
    from qlab.budget.agent import open_candidate
    from qlab.registry.db import session_scope

    cfg = load_config()
    if cfg is None:
        raise ValueError("night/anomalies.yaml is missing")
    with session_scope() as session:
        scan = store.latest_scan(session, trial_id)
        if scan is None or scan.config_version != cfg.version:
            scan_one(session, trial_id, cfg, force=True)
            scan = store.latest_scan(session, trial_id)
        if not scan.fired and not force:
            raise ValueError(f"no flag fired for trial {trial_id}; nothing to investigate "
                             "(--force to look anyway)")
        stamp = store.fingerprint(scan.fired, scan.config_version)
        done = store.latest_investigation(session, trial_id=trial_id)
        if done is not None and done.flags_fingerprint == stamp and not force:
            raise ValueError(f"trial {trial_id} was investigated under the same flags "
                             f"(investigation {done.id}); --force to do it again")
        candidate = open_candidate(session, scan.idea_id, Stage.RECHECK)
        new = investigate(session, scan, cfg, budget=candidate, day=date.today())
        return new.dossier


__all__ = ["LEDGER", "anomalies_stage", "default_budget", "investigate_by_hand",
           "investigation_candidates", "load_ledger", "priority", "save_ledger",
           "scan_candidates", "scan_entry", "scan_one"]

