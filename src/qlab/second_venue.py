"""The night's queue «вторая площадка не проводилась» (docs/VENUE_SPLIT.md,
`night/second_venue.yaml`).

Owner, 2026-10-30: «смотрим обе площадки независимо. проходить если где-то». Until rules
2026-10-30.2 a reject on Binance's history was final and the Hyperliquid run never happened, so the
graveyard holds ideas that were never looked at on the second venue. CLAUDE.md allows a revival only
on data the verdict did not see, and Hyperliquid's data is such data: the rejection was rendered on
Binance's candles and Binance's coins.

Each night this stage takes the oldest of those ideas (at most `max_per_night`, a config value) and
runs their Hyperliquid side through the same step the generator uses (`qlab.pipeline.stage2`):

* a **pass** moves the idea by the normal rules (rejected -> validated, with the venue's own data
  dating the revival, `qlab.registry.lifecycle.next_status(other_venue=True)`), and the
  `stage_transition` says why: the idea was rejected on Binance on a day, Hyperliquid had not been
  run, and what Hyperliquid said;
* anything else leaves the closed idea closed: a reject is the end it already had, and a result
  that cannot decide yet (the window is short) does not reopen it. The run is kept as a trial and
  the idea is looked at again after `retry_after_days`, on the data that has come since.

Which ideas: status `rejected`, not a second stage and not noise, whose latest run is a reject on
Binance's data, with exactly one spec file the Hyperliquid side can express (a spec on Binance's
data whose coins and spot legs Hyperliquid lists, `qlab.pipeline.stage2.translate`), and no
Hyperliquid run judged under independent venues yet. Code only: no agent, no tokens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

CONFIG = Path("night/second_venue.yaml")
LEDGER = Path("data/second_venue/ledger.json")
SPEC_DIR = Path("specs")
FIRST_RULES = "2026-10-30.2"
"""The first rules version that judges the two venues independently."""
SECTION_TITLE = "Вторая площадка для закрытых идей"
FINAL_OUTCOMES = ("revived", "rejected", "not-evaluable")
"""Ledger outcomes after which an idea is not queued again."""


@dataclass(frozen=True)
class Config:
    enabled: bool
    max_per_night: int
    retry_after_days: int


def load_config(path: Path = CONFIG) -> Config:
    """`night/second_venue.yaml`. A missing key is an error: no default can silently switch the
    queue off or make it unbounded."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        config = Config(bool(raw["enabled"]), int(raw["max_per_night"]),
                        int(raw["retry_after_days"]))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    if config.max_per_night < 0 or config.retry_after_days < 1:
        raise ValueError(f"{path}: max_per_night must be >= 0 and retry_after_days >= 1")
    return config


@dataclass
class Candidate:
    idea_id: str
    spec_path: Path
    stage1_trial: int
    rejected_on: date
    """The day of the rejection (the idea's last transition into `rejected`)."""
    last_outcome: str | None = None
    last_day: str | None = None


def load_ledger(path: Path = LEDGER) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_ledger(ledger: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1, sort_keys=True),
                    encoding="utf-8")


def outcome_of(result: dict) -> str:
    """The ledger word for a Stage2Result (as a dict): what the run found."""
    if result.get("error"):
        return "error"
    if result.get("skipped"):
        return "skipped"
    route = result.get("route")
    if route in ("paper", "shelf", "needs-infrastructure"):
        return "revived" if result.get("parent_status") not in (None, "rejected") else "passed"
    if route == "reject":
        return "rejected"
    if route == "not-evaluable" or result.get("status") == "not-evaluable":
        return "not-evaluable"
    return "undecided"


def _rejected_on(session, idea) -> date:
    from qlab.registry.models import IdeaStatus, StageTransition

    row = (session.query(StageTransition)
           .filter(StageTransition.idea_id == idea.id,
                   StageTransition.to_status == IdeaStatus.REJECTED)
           .order_by(StageTransition.id.desc()).first())
    if row is not None and row.at is not None:
        return row.at.date()
    return idea.updated_at.date()


def _listing_names(session) -> list[str] | None:
    """What Hyperliquid lists, from the widest Hyperliquid snapshot in the registry (no network);
    None when the registry holds none."""
    from qlab.data.snapshot import load_snapshot
    from qlab.registry.models import DataSnapshot

    rows = session.query(DataSnapshot).filter(DataSnapshot.source == "hyperliquid").all()
    if not rows:
        return None
    best = max(rows, key=lambda r: (len(r.instruments or {}), str(r.range_end)))
    return [str(c) for c in load_snapshot(best.id, session=session).prices.columns]


def find_candidates(session, specs: dict, *, ledger: dict, today: date, config: Config,
                    stage2_config=None, names: list[str] | None = None
                    ) -> tuple[list[Candidate], dict[str, str]]:
    """(the rejected ideas whose Hyperliquid side is owed, oldest rejection first; the closed
    ideas that cannot be queued with the reason each). Reads only."""
    from qlab.harness.version import is_older
    from qlab.pipeline import stage2
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.registry.stages import is_stage2_child, latest_routed, stage2_child
    from qlab.twin_stage import goes_to_stage2
    from qlab.waiting import _pick_spec

    if stage2_config is None:
        stage2_config = stage2.load_config()
    if names is None:
        names = _listing_names(session)
    found: list[Candidate] = []
    refused: dict[str, str] = {}
    for idea in session.query(Idea).filter(Idea.status == IdeaStatus.REJECTED).order_by(Idea.id):
        if idea.id.startswith("noise-") or is_stage2_child(idea):
            continue
        routed = latest_routed(session, idea.id)
        if routed is None:
            continue  # a dossier from before the lab: nothing to run
        trial, spec_row = routed
        route = getattr(trial.route, "value", trial.route)
        if route != "reject" or (spec_row.data_requirements or {}).get("source") != "binance":
            continue  # a reject on Hyperliquid's own data has no Binance side to wait for here
        path, skip = _pick_spec(specs, idea.id)
        if path is None:
            refused[idea.id] = skip or "нет файла спека"
            continue
        try:
            spec = load_spec(path)
        except Exception as exc:  # noqa: BLE001 - a broken file is its own reason
            refused[idea.id] = f"спек не читается: {type(exc).__name__}"
            continue
        if not goes_to_stage2(spec):
            refused[idea.id] = "спек не на данных Binance: переносить на Hyperliquid нечего"
            continue
        child = stage2_child(session, idea.id)
        child_run = None if child is None else latest_routed(session, child.id)
        if child_run is not None:
            from qlab.forward_series import _rules_version

            if not is_older(_rules_version(session, child_run[0].id), FIRST_RULES):
                continue  # already judged on Hyperliquid by independent venues
        entry = ledger.get(idea.id) or {}
        last = entry.get("outcome")
        if last in FINAL_OUTCOMES:
            continue
        if last is not None and entry.get("day"):
            # A result that could not decide waits `retry_after_days` for data to arrive; a run
            # that crashed or was skipped is tried again the next night.
            wait = 1 if last in ("error", "skipped") else config.retry_after_days
            try:
                due = date.fromisoformat(entry["day"]) + timedelta(days=wait)
            except ValueError:
                due = today
            if due > today:
                continue
        if spec.data.instruments is not None or spec.required_instruments is not None:
            if names is None:
                refused[idea.id] = "в реестре нет снимка Hyperliquid со списком монет"
                continue
            translation = stage2.translate(spec, stage2_config, venue_names=names)
            if translation.spec is None or translation.unavailable:
                refused[idea.id] = ("Hyperliquid не может выразить спек: "
                                    + "; ".join([*translation.problems, *translation.unavailable]))
                continue
        found.append(Candidate(idea.id, path, trial.id, _rejected_on(session, idea),
                               entry.get("outcome"), entry.get("day")))
    found.sort(key=lambda c: (c.rejected_on, c.idea_id))
    return found, refused


def counts(candidates: list[Candidate], config: Config) -> dict:
    """How many are owed and how many nights the queue takes."""
    per = max(config.max_per_night, 1)
    return {"owed": len(candidates), "max_per_night": config.max_per_night,
            "nights": math.ceil(len(candidates) / per) if config.max_per_night else None}


def settle_note(c: Candidate) -> str:
    return (f"second-venue queue (night/second_venue.yaml): the idea was rejected on Binance on "
            f"{c.rejected_on} (trial {c.stage1_trial}), before the venues were judged "
            "independently, and Hyperliquid had not been run; Hyperliquid's data is data that "
            "rejection did not see")


def second_venue_stage(day: date, end: date, state: dict, *, noise_trials: int = 200,
                       stop_at: datetime | None = None, config_path: Path = CONFIG,
                       spec_dir: Path = SPEC_DIR, ledger_path: Path = LEDGER,
                       only: list[str] | None = None, max_new: int | None = None,
                       dry_run: bool = False) -> None:
    """Run the Hyperliquid side of the oldest rejected ideas that never had one; write what
    happened to `state["second_venue"]` for the morning report. `only` names ideas (by hand);
    `dry_run` runs nothing and only counts."""
    from qlab import night, waiting
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope

    config = load_config(config_path)
    if not config.enabled:
        return
    block = state.setdefault("second_venue", {"entries": [], "owed": 0, "nights": None})
    try:
        stage2_config = stage2.load_config()
    except (OSError, ValueError) as exc:
        block["note"] = f"night/stage2.yaml не прочитан: {exc}"
        return
    if not stage2_config.enabled:
        block["note"] = "второй этап выключен в night/stage2.yaml"
        return
    ledger = load_ledger(ledger_path)
    with session_scope() as session:
        candidates, refused = find_candidates(
            session, waiting.spec_files(spec_dir), ledger=ledger, today=day, config=config,
            stage2_config=stage2_config)
    if only:
        candidates = [c for c in candidates if c.idea_id in only]
    block.update(counts(candidates, config))
    block["refused"] = refused
    cap = max_new if max_new is not None else config.max_per_night
    done = state.setdefault("done", {})
    handled = 0
    for cand in candidates:
        key = f"second_venue:{cand.idea_id}"
        if key in done or dry_run:
            continue
        if handled >= cap:
            break
        if night.past_stop(stop_at, state, key):
            break
        handled += 1
        plan = waiting.Plan(cand.idea_id, "stage2", cand.spec_path.as_posix(), "reject", True,
                            "вторая площадка не проводилась")
        entry = {"idea": cand.idea_id, "rejected_on": cand.rejected_on.isoformat(),
                 "stage1_trial": cand.stage1_trial}
        try:
            result = waiting.rerun_stage2(
                plan, end, stage1=None, noise_trials=noise_trials, config=stage2_config,
                settle_note=settle_note(cand), only_a_pass_moves=True)
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            result = {"action": "error", "error": f"{type(exc).__name__}: {exc}"}
        entry.update(
            action=result.get("action"), route=result.get("route_after"),
            trial=result.get("trial"), status=result.get("status"), error=result.get("error"),
            why=result.get("why"), lines=result.get("lines") or [])
        word = outcome_of({
            "error": result.get("error"), "route": result.get("route_after"),
            "parent_status": result.get("status"),
            "skipped": result.get("why") if result.get("action") == "wait" else None})
        entry["outcome"] = word
        ledger[cand.idea_id] = {"day": day.isoformat(), "outcome": word,
                                "trial": result.get("trial")}
        block["entries"].append(entry)
        if word != "error":
            done[key] = datetime.now(UTC).isoformat()
        save_ledger(ledger, ledger_path)
        night.save_state(day, state)
    block["owed_after"] = max(0, len(candidates) - handled)
    if not dry_run:
        night.save_state(day, state)


def report_lines(block: dict | None) -> list[str]:
    """The section of the morning report."""
    if not block:
        return []
    lines = [
        "Идеи, отвергнутые на Binance до 2026-10-30.2, не проходили Hyperliquid: тогда отказ на "
        "Binance был окончательным. Данные Hyperliquid — данные, которых отказ не видел, поэтому "
        "каждую ночь код берёт самые старые такие идеи и считает их на Hyperliquid "
        "(`night/second_venue.yaml`, `docs/VENUE_SPLIT.md`). Прошла — идея оживает по обычным "
        "правилам, переход назван в журнале; не прошла или рано судить — закрытая идея остаётся "
        "закрытой.", ""]
    owed, nights = block.get("owed"), block.get("nights")
    if owed is not None:
        lines.append(f"В очереди: {owed}, ночей {nights} при {block.get('max_per_night')} "
                     "в ночь.")
    if block.get("note"):
        lines.append(f"- {block['note']}")
    words = {"revived": "оживлена: прошла на Hyperliquid",
             "rejected": "отвергнута и на Hyperliquid",
             "undecided": "рано судить: закрыта, посмотрим снова через "
                          "`retry_after_days`", "not-evaluable": "проверить нечем",
             "error": "прогон упал, повторится", "skipped": "не проводилась", "passed": "прошла"}
    for e in block.get("entries") or []:
        lines.append(f"- `{e['idea']}` (отвергнута {e['rejected_on']}): "
                     f"{words.get(e.get('outcome'), e.get('outcome'))}"
                     + (f"; статус «{e['status']}»" if e.get("status") else ""))
        lines += [f"  - {x}" for x in e.get("lines") or []]
    return lines


__all__ = ["Candidate", "Config", "FIRST_RULES", "SECTION_TITLE", "counts", "find_candidates",
           "load_config", "load_ledger", "outcome_of", "report_lines", "second_venue_stage",
           "settle_note"]
