"""Where the detector's scans and the investigator's dossiers are kept (the `anomaly_scan` and
`anomaly_investigation` tables, docs/ANOMALIES.md), and the statistics over them.

Why tables and not `trial.metrics`: a flag is not a fact the stand measured while it ran. It
needs a second look at the trial (a re-run, the idea's other stage, the paper book, the
family), it changes when a line moves, and the scan of a trial that was run long ago can be
made today. The trial row is append-only and read by the rules engine; a flag must not be able
to reach it. The year signs avoid a table because they are recomputed from stored metrics in
microseconds; a scan costs seconds, so it is stored, with the version of the lines it was
made under.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from qlab.anomalies.config import FLAGS
from qlab.anomalies.flags import FlagResult
from qlab.anomalies.scan import ScanResult
from qlab.registry.models import AnomalyInvestigation, AnomalyScan, Idea, Spec, Trial

CAUSES = ("data-artifact", "stand-bug", "unrealistic-tradability", "genuine", "unclear")
CAUSES_RU = {
    "data-artifact": "ошибка или странность в данных",
    "stand-bug": "ошибка стенда (нашего расчёта)",
    "unrealistic-tradability": "результат был бы недостижим в торговле",
    "genuine": "настоящий механизм, подозрение не подтвердилось",
    "unclear": "причину установить не удалось",
}
CHECK_STATUSES = ("none", "proposed", "adopted", "declined")


def fingerprint(fired: list[str], config_version: str) -> str:
    """What an investigation is a response to: the set of flags that fired and the version of
    their lines. A trial is investigated again only when this changes."""
    text = config_version + "|" + ",".join(sorted(fired))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Scans


def save_scan(session: Session, result: ScanResult, config_version: str) -> AnomalyScan:
    """Write a scan, replacing the same trial's scan made under the same version."""
    row = (session.query(AnomalyScan)
           .filter(AnomalyScan.trial_id == result.trial_id,
                   AnomalyScan.config_version == config_version).one_or_none())
    if row is None:
        row = AnomalyScan(trial_id=result.trial_id, config_version=config_version)
        session.add(row)
    row.idea_id = result.idea_id
    row.scanned_at = datetime.now(UTC)
    row.route = result.route
    row.fired = result.fired
    row.flags = [f.to_json() for f in result.flags]
    row.facts = result.facts
    session.flush()
    return row


def latest_scan(session: Session, trial_id: int) -> AnomalyScan | None:
    return (session.query(AnomalyScan).filter(AnomalyScan.trial_id == trial_id)
            .order_by(AnomalyScan.scanned_at.desc(), AnomalyScan.id.desc()).first())


def scan_exists(session: Session, trial_id: int, config_version: str) -> bool:
    return session.query(AnomalyScan.id).filter(
        AnomalyScan.trial_id == trial_id,
        AnomalyScan.config_version == config_version).first() is not None


def flag_results(row: AnomalyScan) -> list[FlagResult]:
    return [FlagResult.from_json(f) for f in row.flags]


def latest_scans(session: Session, *, all_scans: bool = False) -> list[AnomalyScan]:
    """The newest scan of each trial, then (unless `all_scans`) only the newest trial of each
    idea, as `qlab year-signs` counts: one row per idea."""
    rows = session.query(AnomalyScan).order_by(AnomalyScan.idea_id, AnomalyScan.trial_id,
                                               AnomalyScan.scanned_at).all()
    per_trial: dict[int, AnomalyScan] = {}
    for row in rows:
        per_trial[row.trial_id] = row
    if all_scans:
        return list(per_trial.values())
    per_idea: dict[str, AnomalyScan] = {}
    for row in sorted(per_trial.values(), key=lambda r: r.trial_id):
        per_idea[row.idea_id] = row
    return list(per_idea.values())


def summarize(rows: list[AnomalyScan]) -> dict[str, Any]:
    """Per flag: how many fire, are quiet, are not defined; and the split by the run's route."""
    totals = {name: Counter({"fires": 0, "quiet": 0, "not defined": 0}) for name in FLAGS}
    by_route: dict[str, dict[str, Counter]] = {}
    for row in rows:
        for flag in row.flags:
            totals[flag["name"]][flag["state"]] += 1
            by_route.setdefault(row.route or "none", {}).setdefault(
                flag["name"], Counter())[flag["state"]] += 1
    return {"trials": len(rows), "ideas": len({r.idea_id for r in rows}),
            "totals": {k: dict(v) for k, v in totals.items()},
            "by_route": {r: {k: dict(c) for k, c in v.items()} for r, v in by_route.items()},
            "flagged": sum(1 for r in rows if r.fired)}


# --------------------------------------------------------------------------
# Investigations


@dataclass(frozen=True)
class NewInvestigation:
    idea_id: str
    trial_id: int
    model: str
    flags_fingerprint: str
    cause: str
    summary: str
    dossier: str
    evidence: list
    rejected: list
    proposals: list
    caught_by: list
    new_check: str | None
    programmer_requests: list
    tokens: int
    calls: int


def save_investigation(session: Session, new: NewInvestigation) -> AnomalyInvestigation:
    row = AnomalyInvestigation(
        idea_id=new.idea_id, trial_id=new.trial_id, at=datetime.now(UTC), model=new.model,
        flags_fingerprint=new.flags_fingerprint, cause=new.cause, summary=new.summary,
        dossier=new.dossier, evidence=new.evidence, rejected=new.rejected,
        proposals=new.proposals, caught_by=new.caught_by, new_check=new.new_check,
        new_check_status="proposed" if new.new_check else "none",
        programmer_requests=new.programmer_requests, tokens=new.tokens, calls=new.calls)
    session.add(row)
    session.flush()
    return row


def latest_investigation(session: Session, *, trial_id: int | None = None,
                         idea_id: str | None = None) -> AnomalyInvestigation | None:
    query = session.query(AnomalyInvestigation)
    if trial_id is not None:
        query = query.filter(AnomalyInvestigation.trial_id == trial_id)
    if idea_id is not None:
        query = query.filter(AnomalyInvestigation.idea_id == idea_id)
    return query.order_by(AnomalyInvestigation.at.desc(), AnomalyInvestigation.id.desc()).first()


def set_check_status(session: Session, investigation_id: int, status: str) -> AnomalyInvestigation:
    """The owner's decision on an investigation's proposed new check."""
    if status not in CHECK_STATUSES:
        raise ValueError(f"status must be one of {CHECK_STATUSES}")
    row = session.get(AnomalyInvestigation, investigation_id)
    if row is None:
        raise ValueError(f"no investigation {investigation_id}")
    if not row.new_check:
        raise ValueError(f"investigation {investigation_id} proposed no new check")
    row.new_check_status = status
    session.flush()
    return row


def pending_checks(session: Session) -> list[AnomalyInvestigation]:
    """Investigations that propose a check nobody has turned into a detector yet."""
    return (session.query(AnomalyInvestigation)
            .filter(AnomalyInvestigation.new_check_status == "proposed")
            .order_by(AnomalyInvestigation.id).all())


def learning(session: Session) -> dict[str, Any]:
    """The learning loop in numbers: of the investigations whose cause was confirmed (anything
    but `unclear` and `genuine`), which detector flags caught it and how many none did."""
    rows = session.query(AnomalyInvestigation).all()
    confirmed = [r for r in rows if r.cause not in ("unclear", "genuine")]
    caught: Counter[str] = Counter()
    missed = 0
    for r in confirmed:
        if r.caught_by:
            caught.update(r.caught_by)
        else:
            missed += 1
    return {"investigations": len(rows), "confirmed": len(confirmed),
            "causes": dict(Counter(r.cause for r in rows)), "caught_by": dict(caught),
            "caught_by_none": missed}


def idea_of(session: Session, trial_id: int) -> str | None:
    trial = session.get(Trial, trial_id)
    spec = None if trial is None else session.get(Spec, trial.spec_id)
    return None if spec is None else spec.idea_id


def status_of(session: Session, idea_id: str) -> str | None:
    idea = session.get(Idea, idea_id)
    return None if idea is None else idea.status.value


__all__ = ["CAUSES", "CAUSES_RU", "CHECK_STATUSES", "NewInvestigation", "fingerprint",
           "flag_results", "idea_of", "latest_investigation", "latest_scan", "latest_scans",
           "learning", "pending_checks", "save_investigation", "save_scan", "scan_exists",
           "set_check_status", "status_of", "summarize"]
