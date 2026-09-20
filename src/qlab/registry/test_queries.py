"""Tests for the three registry queries the M1 milestone exists to answer,
plus funnel_stats. Fixtures are hand-built directly through `registry.repo`
(no YAML, no seed/graveyard.yaml) against in-memory SQLite, covering all
three legitimate verdict kinds from docs/REGISTRY.md: decision, unknown and
measurement.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    IdeaStatus,
    Profile,
    ShutdownCause,
    SourceType,
    TrialSource,
    VerdictStage,
)
from qlab.registry.queries import (
    funnel_stats,
    killed_by_retired_rules,
    near_threshold,
    ripe_for_revival,
)
from qlab.rules import Comparator, RetiredRule, Rule, RuleSet, Stage

TODAY = date(2026, 9, 20)

RULESET = RuleSet(
    version="2026-09-20.1",
    rules=[
        Rule(
            id="sharpe_floor",
            stage=Stage.EDGE,
            metric="sharpe_net",
            comparator=Comparator.GE,
            threshold=0.8,
            near_margin=0.1,
        ),
        Rule(
            id="net_edge_positive",
            stage=Stage.EDGE,
            metric="ann_return_net",
            comparator=Comparator.GT,
            threshold=0.0,
        ),
    ],
    retired=[
        RetiredRule(
            id="decorrelation_required",
            retired_in="2026-09-12.1",
            reason="correlation affects sizing, not admission",
        )
    ],
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()


def _add_idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.REJECTED) -> None:
    repo.upsert_idea(
        session,
        id=idea_id,
        title=idea_id,
        source_type=SourceType.GRAVEYARD,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.CARRY,
        status=status,
    )


def _add_verdict(session, **kwargs) -> None:
    defaults = dict(
        rules_version="frab-legacy",
        source=TrialSource.IMPORTED,
        decided_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    defaults.update(kwargs)
    repo.add_verdicts(session, [defaults])


@pytest.fixture()
def populated(session):
    """Seven ideas, one verdict each, one per interesting case:

    - idea-retired:        FAILED, rule_id is retired in RULESET
    - idea-near:            FAILED, close to its threshold (rule-specific near_margin)
    - idea-ripe:            FAILED, data_range_end old enough to revive
    - idea-unknown-range:   FAILED, data_range_end is null
    - idea-measurement:     measurement (value, no comparator/threshold/rule),
                            old enough to revive
    - idea-passed:          decision that PASSED
    - idea-unknown-metric:  unknown (rule exists, metric not computed)
    """
    _add_idea(session, "idea-retired")
    _add_verdict(
        session,
        idea_id="idea-retired",
        stage=VerdictStage.CORRELATION,
        rule_id="decorrelation_required",
        metric="corr_to_frab_proxy",
        value=0.86,
        comparator="<",
        threshold=0.30,
        passed=False,
        data_range_start=None,
        data_range_end=None,
    )

    _add_idea(session, "idea-near")
    _add_verdict(
        session,
        idea_id="idea-near",
        stage=VerdictStage.EDGE,
        rule_id="sharpe_floor",
        metric="sharpe_net",
        value=0.75,
        comparator=">=",
        threshold=0.8,
        passed=False,
        data_range_start=date(2024, 1, 1),
        data_range_end=date(2024, 6, 1),
    )

    _add_idea(session, "idea-ripe")
    _add_verdict(
        session,
        idea_id="idea-ripe",
        stage=VerdictStage.EDGE,
        rule_id="net_edge_positive",
        metric="ann_return_net",
        value=-0.15,
        comparator=">",
        threshold=0.0,
        passed=False,
        data_range_start=date(2024, 1, 1),
        data_range_end=TODAY - timedelta(days=200),
    )

    _add_idea(session, "idea-unknown-range")
    _add_verdict(
        session,
        idea_id="idea-unknown-range",
        stage=VerdictStage.EDGE,
        rule_id="net_edge_positive",
        metric="ann_return_net",
        value=-0.15,
        comparator=">",
        threshold=0.0,
        passed=False,
        data_range_start=None,
        data_range_end=None,
    )

    _add_idea(session, "idea-measurement")
    _add_verdict(
        session,
        idea_id="idea-measurement",
        stage=VerdictStage.EDGE,
        rule_id="unidentified",
        metric="sharpe_net",
        value=0.77,
        comparator=None,
        threshold=None,
        passed=None,
        data_range_start=None,
        data_range_end=TODAY - timedelta(days=200),
    )

    _add_idea(session, "idea-passed", status=IdeaStatus.VALIDATED)
    _add_verdict(
        session,
        idea_id="idea-passed",
        stage=VerdictStage.EDGE,
        rule_id="net_edge_positive",
        metric="ann_return_net",
        value=0.1,
        comparator=">",
        threshold=0.0,
        passed=True,
        data_range_start=date(2024, 1, 1),
        data_range_end=date(2024, 6, 1),
    )

    _add_idea(session, "idea-unknown-metric")
    _add_verdict(
        session,
        idea_id="idea-unknown-metric",
        stage=VerdictStage.EDGE,
        rule_id="net_edge_positive",
        metric="ann_return_net",
        value=None,
        comparator=">",
        threshold=0.0,
        passed=None,
        data_range_start=date(2024, 1, 1),
        data_range_end=date(2024, 6, 1),
    )

    session.commit()
    return session


# --------------------------------------------------------------------------
# killed_by_retired_rules
# --------------------------------------------------------------------------


def test_killed_by_retired_rules(populated):
    result = killed_by_retired_rules(populated, RULESET)

    assert {r.idea.id for r in result} == {"idea-retired"}
    kill = result[0]
    assert len(kill.verdicts) == 1
    assert kill.verdicts[0].rule_id == "decorrelation_required"


def test_killed_by_retired_rules_empty_ruleset_retired_section(populated):
    empty = RuleSet(version="2026-09-20.1", rules=[], retired=[])
    assert killed_by_retired_rules(populated, empty) == []


def test_killed_by_retired_rules_never_includes_measurements(populated):
    """A measurement's rule_id is always 'unidentified', never a real
    (possibly retired) rule id — even if 'unidentified' were retired, a
    measurement's `passed` is null, so `passed IS FALSE` still excludes it."""
    ruleset_with_unidentified_retired = RuleSet(
        version="2026-09-20.1",
        rules=[],
        retired=[
            RetiredRule(id="unidentified", retired_in="2026-09-12.1", reason="hypothetical")
        ],
    )
    result = killed_by_retired_rules(populated, ruleset_with_unidentified_retired)
    assert result == []


# --------------------------------------------------------------------------
# near_threshold
# --------------------------------------------------------------------------


def test_near_threshold_uses_rule_specific_margin_over_the_argument(populated):
    # idea-near: sharpe 0.75 vs threshold 0.8 -> nearness 0.0625, within the
    # rule's own near_margin=0.1 even though the call-site margin is tiny.
    report = near_threshold(populated, margin=0.01, ruleset=RULESET)
    assert {m.idea.id for m in report.near} == {"idea-near"}
    assert report.near[0].margin_used == 0.1


def test_near_threshold_zero_threshold_without_margin_abs_is_undefined_not_near(populated):
    """The regression this function exists to fix: idea-ripe/idea-unknown-range
    both FAIL `net_edge_positive` (threshold 0.0) with a negative value. Since
    RULESET's `net_edge_positive` sets no `near_margin_abs`, "closeness to
    zero" is undefined — it must NOT be reported as a near miss just because
    a relative margin happened to look small (that's the Sharpe -0.22 bug)."""
    report = near_threshold(populated, margin=0.01, ruleset=RULESET)
    near_ids = {m.idea.id for m in report.near}
    undefined_ids = {u.idea.id for u in report.undefined_nearness}

    assert "idea-ripe" not in near_ids
    assert "idea-unknown-range" not in near_ids
    assert {"idea-ripe", "idea-unknown-range"} <= undefined_ids

    by_id = {u.idea.id: u for u in report.undefined_nearness}
    assert by_id["idea-ripe"].raw_distance == pytest.approx(0.15)


def test_near_threshold_falls_back_to_argument_without_a_ruleset(populated):
    report = near_threshold(populated, margin=0.1)
    assert {m.idea.id for m in report.near} == {"idea-near"}
    # without a ruleset there is no per-rule near_margin_abs to use, so the
    # zero-threshold failures are still undefined, not silently dropped
    assert {"idea-ripe", "idea-unknown-range"} <= {
        u.idea.id for u in report.undefined_nearness
    }


def test_near_threshold_excludes_measurement_and_unknown(populated):
    """Measurements (no threshold) and unknowns (no value) must never
    appear in either group, no matter how generous the margin — there is
    nothing to be "near" to, and nothing undefined about "not applicable"."""
    report = near_threshold(populated, margin=10.0, ruleset=RULESET)
    all_ids = {m.idea.id for m in report.near} | {u.idea.id for u in report.undefined_nearness}
    assert "idea-measurement" not in all_ids
    assert "idea-unknown-metric" not in all_ids
    assert "idea-passed" not in all_ids  # passed, not a failure, regardless of margin


def test_near_threshold_zero_threshold_with_margin_abs_is_classified_properly(session):
    """Once the ruleset owner sets an explicit `near_margin_abs`, a
    zero-threshold failure can be judged NEAR or genuinely not-near again —
    `classify_nearness` isn't permanently stuck at UNDEFINED, only until a
    real number is configured."""
    _add_idea(session, "idea-zero-abs")
    _add_verdict(
        session,
        idea_id="idea-zero-abs",
        stage=VerdictStage.EDGE,
        rule_id="net_edge_positive",
        metric="ann_return_net",
        value=-0.03,
        comparator=">",
        threshold=0.0,
        passed=False,
        data_range_start=date(2024, 1, 1),
        data_range_end=date(2024, 6, 1),
    )
    session.commit()

    rule_without_abs = Rule(
        id="net_edge_positive",
        stage=Stage.EDGE,
        metric="ann_return_net",
        comparator=Comparator.GT,
        threshold=0.0,
    )
    report = near_threshold(session, margin=0.5, ruleset=RuleSet(
        version="2026-09-20.1", rules=[rule_without_abs]
    ))
    assert report.near == []
    assert len(report.undefined_nearness) == 1
    assert report.undefined_nearness[0].raw_distance == pytest.approx(0.03)

    rule_with_abs = Rule(
        id="net_edge_positive",
        stage=Stage.EDGE,
        metric="ann_return_net",
        comparator=Comparator.GT,
        threshold=0.0,
        near_margin_abs=0.05,
    )
    report2 = near_threshold(
        session, margin=0.5, ruleset=RuleSet(version="2026-09-20.1", rules=[rule_with_abs])
    )
    assert {m.idea.id for m in report2.near} == {"idea-zero-abs"}
    assert report2.undefined_nearness == []


# --------------------------------------------------------------------------
# ripe_for_revival
# --------------------------------------------------------------------------


def test_ripe_for_revival_includes_measurements(populated):
    report = ripe_for_revival(populated, min_new_days=180, today=TODAY)

    ripe_ids = {r.idea.id for r in report.ripe}
    unknown_ids = {u.idea.id for u in report.unknown_data_range}

    # idea-near is also long-rejected enough (data_range_end 2024-06-01) to
    # be ripe here; its interesting property (nearness) is covered by the
    # near_threshold tests, not this one.
    assert ripe_ids == {"idea-ripe", "idea-measurement", "idea-near"}
    assert unknown_ids == {"idea-retired", "idea-unknown-range"}
    # a true "unknown" (no value at all) is not a rejection -> absent from both
    assert "idea-unknown-metric" not in ripe_ids
    assert "idea-unknown-metric" not in unknown_ids
    # a pass is not a rejection either
    assert "idea-passed" not in ripe_ids
    assert "idea-passed" not in unknown_ids


def test_ripe_for_revival_respects_min_new_days(populated):
    report = ripe_for_revival(populated, min_new_days=1000, today=TODAY)
    assert report.ripe == []
    # verdicts with a null data_range_end stay in unknown_data_range
    # regardless of min_new_days -- they were never eligible to begin with
    assert {u.idea.id for u in report.unknown_data_range} == {
        "idea-retired",
        "idea-unknown-range",
    }


def test_ripe_for_revival_days_since_rejection(populated):
    report = ripe_for_revival(populated, min_new_days=180, today=TODAY)
    by_id = {r.idea.id: r for r in report.ripe}
    assert by_id["idea-ripe"].days_since_rejection == 200
    assert by_id["idea-measurement"].days_since_rejection == 200


# --------------------------------------------------------------------------
# funnel_stats
# --------------------------------------------------------------------------


def test_funnel_stats_counts_three_verdict_kinds_separately(populated):
    stats = funnel_stats(populated)

    assert stats.verdicts_by_outcome["passed"] == 1  # idea-passed
    assert stats.verdicts_by_outcome["failed"] == 4  # retired/near/ripe/unknown-range
    assert stats.verdicts_by_outcome["unknown"] == 1  # idea-unknown-metric
    assert stats.verdicts_by_outcome["measurement"] == 1  # idea-measurement

    assert sum(stats.verdicts_by_outcome.values()) == 7
    assert sum(stats.ideas_by_status.values()) == 7


def test_funnel_stats_ideas_by_status(populated):
    stats = funnel_stats(populated)
    assert stats.ideas_by_status["rejected"] == 6
    assert stats.ideas_by_status["validated"] == 1


def test_funnel_stats_verdicts_by_stage(populated):
    stats = funnel_stats(populated)
    assert stats.verdicts_by_stage["edge"] == 6
    assert stats.verdicts_by_stage["correlation"] == 1


def test_funnel_stats_decayed_by_shutdown_cause(session):
    """Two decayed ideas with different causes, plus a live idea that must
    not appear in the breakdown at all — only DECAYED ideas are counted."""
    repo.upsert_idea(
        session,
        id="decayed-edge",
        title="Decayed on edge loss",
        source_type=SourceType.GRAVEYARD,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.MOMENTUM,
        status=IdeaStatus.DECAYED,
        shutdown_cause=ShutdownCause.EDGE_DECAYED,
    )
    repo.upsert_idea(
        session,
        id="decayed-false-discovery",
        title="Decayed on false discovery",
        source_type=SourceType.GRAVEYARD,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.MOMENTUM,
        status=IdeaStatus.DECAYED,
        shutdown_cause=ShutdownCause.FALSE_DISCOVERY,
    )
    repo.upsert_idea(
        session,
        id="still-live",
        title="Still live",
        source_type=SourceType.GRAVEYARD,
        asset_class=AssetClass.CRYPTO_PERP,
        profile=Profile.CARRY,
        status=IdeaStatus.LIVE,
    )
    session.commit()

    stats = funnel_stats(session)

    assert stats.decayed_by_shutdown_cause == {
        "edge-decayed": 1,
        "false-discovery": 1,
    }
    assert sum(stats.decayed_by_shutdown_cause.values()) == stats.ideas_by_status["decayed"]
