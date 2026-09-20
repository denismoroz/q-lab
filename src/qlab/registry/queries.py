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

from qlab.registry.models import Idea, IdeaStatus, Verdict
from qlab.rules import NearnessVerdict, RuleSet, classify_nearness, nearness


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


@dataclass(slots=True)
class UndefinedNearness:
    """A FAILED verdict whose closeness to its threshold cannot be judged.

    In practice this is almost always `threshold == 0` (the single most
    common rule in the graveyard, "net edge positive") with no
    `near_margin_abs` configured on the rule: relative nearness — "value is
    within X% of the threshold" — is meaningless when the threshold itself
    is zero (a Sharpe of -0.22 is *not* "almost 0", not by any sane
    definition), and no rule has an absolute margin defined yet because
    that number isn't in any source document and must not be invented
    (see `qlab.rules.nearness.classify_nearness`). `raw_distance` —
    `|value - threshold|`, in the metric's own units — is included so a
    human can still judge it by eye instead of q-lab making up an opinion.
    """

    idea: Idea
    verdict: Verdict
    raw_distance: float


@dataclass(slots=True)
class NearThresholdReport:
    near: list[NearMiss]
    undefined_nearness: list[UndefinedNearness]


def near_threshold(
    session: Session, margin: float, ruleset: RuleSet | None = None
) -> NearThresholdReport:
    """FAILED verdicts that came close to passing.

    Classifies every FAILED decision with `qlab.rules.nearness.classify_nearness`
    into two groups instead of one boolean:

    - `.near` — genuinely close, in a well-defined sense (a real margin, in
      the right units, was available to compare against).
    - `.undefined_nearness` — closeness literally cannot be computed with
      what's configured (typically: zero threshold, no `near_margin_abs`
      on the rule). These are **not** "not near" — collapsing "clearly far"
      and "we have no way to tell" into one bucket is exactly what
      produced Sharpe -0.22 being reported as "almost passed" against a
      threshold of 0.0 before this function used `classify_nearness`.
      A verdict that *is* comfortably far from its threshold (genuine
      `NOT_NEAR`) is silently dropped, same as before — it belongs in
      neither list.

    Only FAILED **decisions** (docs/REGISTRY.md's first verdict kind) are
    considered: `passed IS NOT FALSE` already excludes both "unknown" rows
    (no `value`) and "measurement" rows (no `comparator`/`threshold`) —
    there is no threshold to be near for either.

    When the verdict's `rule_id` names a rule in `ruleset`, that rule's own
    `near_margin`/`near_margin_abs` are used (a rule with no `near_margin`
    set still falls back to `margin`, but its `near_margin_abs` — even if
    null — is used as-is, since there is no argument-level fallback for the
    absolute margin). When `rule_id` doesn't match any rule in `ruleset`
    (an unrecognized or retired id, or `ruleset=None`), only the plain
    `margin` argument applies, so a zero-threshold verdict in that case has
    no `margin_abs` at all and lands in `undefined_nearness`.
    """
    rules_by_id = {rule.id: rule for rule in ruleset.rules} if ruleset is not None else {}

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
        return NearThresholdReport(near=[], undefined_nearness=[])

    ideas_by_id = _ideas_by_id(session, {verdict.idea_id for verdict in failed})

    near: list[NearMiss] = []
    undefined: list[UndefinedNearness] = []
    for verdict in failed:
        idea = ideas_by_id.get(verdict.idea_id)
        if idea is None:
            continue

        rule = rules_by_id.get(verdict.rule_id)
        effective_margin = margin if rule is None or rule.near_margin is None else rule.near_margin
        effective_margin_abs = rule.near_margin_abs if rule is not None else None

        outcome = classify_nearness(
            verdict.value,
            verdict.threshold,
            verdict.comparator,
            margin=effective_margin,
            margin_abs=effective_margin_abs,
        )
        if outcome is NearnessVerdict.NEAR:
            near.append(NearMiss(idea=idea, verdict=verdict, margin_used=effective_margin))
        elif outcome is NearnessVerdict.UNDEFINED:
            undefined.append(
                UndefinedNearness(
                    idea=idea,
                    verdict=verdict,
                    raw_distance=abs(verdict.value - verdict.threshold),
                )
            )
        # NearnessVerdict.NOT_NEAR: comfortably far, reported in neither list.

    # Closest first in both lists: the top of an undefined-nearness list is
    # where the owner looks to decide whether an absolute margin is worth
    # setting, and a list sorted by idea name buries it.
    near.sort(key=lambda m: nearness(m.verdict.value, m.verdict.threshold))
    undefined.sort(key=lambda u: u.raw_distance)
    return NearThresholdReport(near=near, undefined_nearness=undefined)


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
    # Breakdown of DECAYED ideas by `shutdown_cause`. Only "edge-decayed"
    # belongs in the project's "how long does a working strategy last"
    # statistic — "false-discovery" means the screening was wrong, not
    # that the market changed, and mixing the two would corrupt that
    # number. Every DECAYED idea has a cause (enforced at the DB level by
    # `ck_idea_decayed_requires_shutdown_cause`), so this dict's total
    # always equals `ideas_by_status["decayed"]`.
    decayed_by_shutdown_cause: dict[str, int]


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

    shutdown_cause_rows = (
        session.query(Idea.shutdown_cause, func.count(Idea.id))
        .filter(Idea.status == IdeaStatus.DECAYED)
        .group_by(Idea.shutdown_cause)
        .all()
    )
    decayed_by_shutdown_cause = {
        _enum_value(cause): count for cause, count in shutdown_cause_rows if cause is not None
    }

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
        decayed_by_shutdown_cause=decayed_by_shutdown_cause,
    )


__all__ = [
    "FunnelStats",
    "NearMiss",
    "NearThresholdReport",
    "RetiredRuleKill",
    "RevivalReport",
    "RipeIdea",
    "UndefinedNearness",
    "UnknownDataRange",
    "funnel_stats",
    "killed_by_retired_rules",
    "near_threshold",
    "ripe_for_revival",
]
