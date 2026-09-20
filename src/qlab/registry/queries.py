"""The three registry queries the whole M1 milestone exists to answer
(docs/REGISTRY.md, "Три запроса, ради которых всё строится"), plus a
funnel-stats query that feeds the future funnel screen.

Every function here takes an already-open `Session` and does no I/O of its
own (no file reads, no session/engine construction) — that is the caller's
job (see `qlab.registry.db.session_scope`), so these can be composed freely
and tested against an in-memory database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from qlab.registry.models import Idea, Verdict
from qlab.rules import RuleSet, is_near


def _enum_value(x: object) -> str:
    """Normalize an Enum-or-str group-by key to its plain string value."""
    return x.value if hasattr(x, "value") else str(x)


def _ideas_by_id(session: Session, idea_ids: set[str]) -> dict[str, Idea]:
    if not idea_ids:
        return {}
    rows = session.query(Idea).filter(Idea.id.in_(idea_ids)).all()
    return {row.id: row for row in rows}


# --------------------------------------------------------------------------
# 1. killed_by_retired_rules
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RetiredRuleKill:
    """One idea killed (at least in part) by a rule that has since been retired."""

    idea: Idea
    verdicts: list[Verdict]


def killed_by_retired_rules(session: Session, ruleset: RuleSet) -> list[RetiredRuleKill]:
    """Ideas with a FAILED verdict whose `rule_id` is in `ruleset.retired`.

    These died on a criterion that no longer exists in the current
    screening rules — the whole point of tracking `retired` separately
    from `rules` (docs/REGISTRY.md) is to be able to ask this question
    without re-running anything.

    Only "decision" rows (see docs/REGISTRY.md's three verdict kinds) can
    ever match here: `passed IS FALSE` already excludes "measurement" rows
    (which always have `passed IS NULL` and `rule_id = "unidentified"`,
    never a real, possibly-retired, rule id) and "unknown" rows (`passed
    IS NULL` too). Nothing extra needs to be excluded explicitly.
    """
    retired_ids = {retired.id for retired in ruleset.retired}
    if not retired_ids:
        return []

    verdicts = (
        session.query(Verdict)
        .filter(Verdict.passed.is_(False), Verdict.rule_id.in_(retired_ids))
        .order_by(Verdict.idea_id, Verdict.id)
        .all()
    )
    if not verdicts:
        return []

    by_idea: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_idea.setdefault(verdict.idea_id, []).append(verdict)

    ideas_by_id = _ideas_by_id(session, set(by_idea.keys()))
    return [
        RetiredRuleKill(idea=ideas_by_id[idea_id], verdicts=idea_verdicts)
        for idea_id, idea_verdicts in by_idea.items()
        if idea_id in ideas_by_id
    ]


# --------------------------------------------------------------------------
# 2. near_threshold
# --------------------------------------------------------------------------


@dataclass(slots=True)
class NearMiss:
    """One FAILED verdict that came close to passing."""

    idea: Idea
    verdict: Verdict
    margin_used: float


def near_threshold(
    session: Session, margin: float, ruleset: RuleSet | None = None
) -> list[NearMiss]:
    """FAILED verdicts that came close to passing (`qlab.rules.nearness.is_near`).

    A verdict with a null `value` cannot be "near" anything — there is no
    number to compare — so those are excluded up front by the SQL filter,
    not by `is_near` (which would also return `False` for them, but
    filtering in SQL avoids pulling rows that can never match).

    **Measurement** rows (docs/REGISTRY.md's third verdict kind: a value
    with no `comparator`/`threshold` behind it, imported from graveyard
    documents where a human decided rather than a formal rule) are
    excluded for the same reason as a null value, not despite having one:
    they have `passed IS NULL`, so the `passed.is_(False)` filter already
    drops them, and the extra `threshold.isnot(None)` filter makes that
    exclusion explicit rather than incidental. "Near" a threshold that was
    never fixed is not a real number — reporting one would pass off an
    invented near-miss as a genuine one, which docs/REGISTRY.md explicitly
    warns against for the sibling case of an invented threshold.

    When a rule in `ruleset` declares its own `near_margin`, that value is
    used for verdicts against that rule instead of `margin`
    (docs/REGISTRY.md: `sharpe_floor` example). `ruleset` is optional
    because a graveyard's older verdicts may reference `rule_id`s that no
    longer exist in any ruleset at all (frab-legacy imports, or rules that
    were later retired) — those simply fall back to `margin`.
    """
    rule_margins: dict[str, float] = {}
    if ruleset is not None:
        for rule in ruleset.rules:
            if rule.near_margin is not None:
                rule_margins[rule.id] = rule.near_margin

    failed = (
        session.query(Verdict)
        .filter(
            Verdict.passed.is_(False),
            Verdict.value.isnot(None),
            Verdict.threshold.isnot(None),
        )
        .order_by(Verdict.idea_id, Verdict.id)
        .all()
    )
    if not failed:
        return []

    ideas_by_id = _ideas_by_id(session, {verdict.idea_id for verdict in failed})

    results: list[NearMiss] = []
    for verdict in failed:
        idea = ideas_by_id.get(verdict.idea_id)
        if idea is None:
            continue
        effective_margin = rule_margins.get(verdict.rule_id, margin)
        if is_near(verdict.value, verdict.threshold, verdict.comparator, effective_margin):
            results.append(NearMiss(idea=idea, verdict=verdict, margin_used=effective_margin))
    return results


# --------------------------------------------------------------------------
# 3. ripe_for_revival
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RipeIdea:
    """A rejected idea old enough (by data range) to be worth re-checking."""

    idea: Idea
    verdict: Verdict
    days_since_rejection: int


@dataclass(slots=True)
class UnknownDataRange:
    """A rejection-shaped verdict with no recorded data range.

    Cannot participate in revival at all (see `ripe_for_revival`), but
    must still be surfaced rather than dropped — not knowing when a
    rejection was measured is itself a fact about the graveyard.
    """

    idea: Idea
    verdict: Verdict


@dataclass(slots=True)
class RevivalReport:
    ripe: list[RipeIdea]
    unknown_data_range: list[UnknownDataRange]


def ripe_for_revival(
    session: Session, min_new_days: int, today: date | None = None
) -> RevivalReport:
    """Ideas whose rejection is old enough that genuinely new data exists.

    "Rejection" here means either of the two verdict kinds
    (docs/REGISTRY.md) that represent code or a human turning an idea
    down, and are therefore worth reconsidering once enough new data has
    accumulated:

    - a **decision** that FAILED (`passed IS FALSE`) — a rule was applied
      and the idea didn't clear it;
    - a **measurement** (`value` set, `passed`/`comparator`/`threshold`
      all null) — an old graveyard document recorded a number and a human
      rejected the idea on it without ever formalizing a rule. Many
      ideas in the historical graveyard have *only* measurements and no
      formal decision at all, and dropping them here would make them
      vanish from the registry's one query for "is anything worth
      reviving" — exactly the kind of silent loss docs/REGISTRY.md warns
      against.

    A genuine **unknown** row (`value`/`passed` both null, a real
    `comparator`/`threshold` exists) is excluded: it isn't a rejection at
    all, just an unevaluated rule, so there is nothing to revive.

    A candidate rejection is "ripe" when `data_range_end` is known and
    `today - data_range_end >= min_new_days`: that many days of data the
    original rejection could not possibly have seen have now accumulated.

    A verdict with a **null** `data_range_end` cannot be judged ripe or
    not — without knowing when the rejection's data ended, there is no
    reference point from which to count "new" data; any cutoff would have
    to be invented, and docs/REGISTRY.md is explicit that dates must never
    be invented for revival logic. Such verdicts are therefore never
    silently dropped: they come back in `RevivalReport.unknown_data_range`
    so the registry owner can see exactly how much of the graveyard is
    unrevivable-by-default and, if warranted, go find the missing date by
    hand.
    """
    as_of = today if today is not None else date.today()

    is_measurement = and_(Verdict.passed.is_(None), Verdict.value.isnot(None))
    rejections = (
        session.query(Verdict)
        .filter(or_(Verdict.passed.is_(False), is_measurement))
        .order_by(Verdict.idea_id, Verdict.id)
        .all()
    )
    if not rejections:
        return RevivalReport(ripe=[], unknown_data_range=[])

    ideas_by_id = _ideas_by_id(session, {verdict.idea_id for verdict in rejections})

    ripe: list[RipeIdea] = []
    unknown: list[UnknownDataRange] = []
    for verdict in rejections:
        idea = ideas_by_id.get(verdict.idea_id)
        if idea is None:
            continue
        if verdict.data_range_end is None:
            unknown.append(UnknownDataRange(idea=idea, verdict=verdict))
            continue
        days_elapsed = (as_of - verdict.data_range_end).days
        if days_elapsed >= min_new_days:
            ripe.append(RipeIdea(idea=idea, verdict=verdict, days_since_rejection=days_elapsed))

    return RevivalReport(ripe=ripe, unknown_data_range=unknown)


# --------------------------------------------------------------------------
# funnel_stats
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FunnelStats:
    """Raw counts feeding the funnel screen (docs/PLAN.md M7)."""

    ideas_by_status: dict[str, int]
    verdicts_by_stage: dict[str, int]
    # "passed" / "failed" (decisions) / "unknown" (rule exists, metric
    # missing) / "measurement" (a number with no formal rule behind it —
    # docs/REGISTRY.md's third verdict kind).
    verdicts_by_outcome: dict[str, int]


def funnel_stats(session: Session) -> FunnelStats:
    """Counts of ideas by status, verdicts by stage, and verdicts by outcome.

    `stage_transition` is the documented source of truth for the funnel's
    *flow* over time (docs/REGISTRY.md); this query instead reports the
    current snapshot from `idea.status` and `verdict`, which is what a
    first funnel screen needs before any transition-history view exists.

    Outcomes are counted as the three legitimate verdict kinds from
    docs/REGISTRY.md, not just pass/fail: a "measurement" (value present,
    no comparator/threshold, `passed` null) is neither a pass, a fail, nor
    an "unknown" rule waiting to be computed — it's a historical number a
    human judged without a formal rule. What fraction of the graveyard is
    measurement-only is itself a finding: it says how much of the old
    screening process never had a written-down criterion at all.
    """
    ideas_rows = session.query(Idea.status, func.count(Idea.id)).group_by(Idea.status).all()
    ideas_by_status = {_enum_value(status): count for status, count in ideas_rows}

    stage_rows = (
        session.query(Verdict.stage, func.count(Verdict.id)).group_by(Verdict.stage).all()
    )
    verdicts_by_stage = {_enum_value(stage): count for stage, count in stage_rows}

    passed = session.query(func.count(Verdict.id)).filter(Verdict.passed.is_(True)).scalar() or 0
    failed = (
        session.query(func.count(Verdict.id)).filter(Verdict.passed.is_(False)).scalar() or 0
    )
    unknown = (
        session.query(func.count(Verdict.id))
        .filter(Verdict.passed.is_(None), Verdict.value.is_(None))
        .scalar()
        or 0
    )
    measurement = (
        session.query(func.count(Verdict.id))
        .filter(Verdict.passed.is_(None), Verdict.value.isnot(None))
        .scalar()
        or 0
    )

    return FunnelStats(
        ideas_by_status=ideas_by_status,
        verdicts_by_stage=verdicts_by_stage,
        verdicts_by_outcome={
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "measurement": measurement,
        },
    )


__all__ = [
    "FunnelStats",
    "NearMiss",
    "RetiredRuleKill",
    "RevivalReport",
    "RipeIdea",
    "UnknownDataRange",
    "funnel_stats",
    "killed_by_retired_rules",
    "near_threshold",
    "ripe_for_revival",
]
