"""Tests for `qlab.funnel`: the agents' funnel and the price of one survivor."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab import funnel
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    IdeaStatus,
    Profile,
    SourceType,
    TokenSpend,
    TrialRoute,
    TrialStatus,
)

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
NO_NIGHTS = Path("/nonexistent-night-dir")


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id, *, notes=None, status=IdeaStatus.CANDIDATE, age_days=1.0,
          routes=()):
    """An idea created `age_days` before NOW, with one routed trial per route
    (oldest first)."""
    idea = repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                            asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                            status=status, notes=notes)
    idea.created_at = NOW - timedelta(days=age_days)
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                         rebalance="1d", costs_model={}, code_ref="a:b")
    if session.get(DataSnapshot, "snap") is None:
        repo.add_data_snapshot(session, id="snap", source="hyperliquid", instruments={},
                               range_start=date(2025, 1, 1), range_end=date(2026, 10, 24),
                               path="x", rows=1)
    for i, route in enumerate(routes):
        repo.add_trial(session, spec_id=spec.id, config_hash=f"{idea_id}-{i}", code_sha="x",
                       params={}, status=TrialStatus.OK, route=route, snapshot_id="snap")
    session.flush()
    return idea


def _spend(session, agent, stage, tokens, *, age_days=0.5, cache=0):
    session.add(TokenSpend(at=NOW - timedelta(days=age_days), stage=stage, agent=agent,
                           tokens_in=tokens, tokens_out=0, tokens_cache_read=cache,
                           usd_est=0.0))
    session.flush()


def test_each_agent_leaves_its_own_mark_on_the_ideas_it_creates() -> None:
    assert funnel.idea_source("gen-funding-skew", "Generated from docs/MECHANISMS.md (m1)") == \
        "generator"
    assert funnel.idea_source("gen-x", None) == "generator"
    assert funnel.idea_source("bv2-research-headroom", "Declared by the researcher (bv2).") == \
        "researcher"
    assert funnel.idea_source("trend-research-x", None) == "researcher"
    assert funnel.idea_source("card-idea", "Implemented by the implementer agent from c.yaml.") \
        == "implementer"
    # The owner's, imported, and interactive-session ideas are nobody's agent.
    assert funnel.idea_source("trend-t39-meta", "T39: two-level meta-labelling.") is None
    assert funnel.idea_source("frab", None) is None


def test_the_second_stage_of_a_generated_idea_is_not_a_new_idea(session) -> None:
    """`gen-x-hl` holds the second look at `gen-x` (docs/TWO_STAGE.md): counting it would
    make every survivor two ideas and two survivors."""
    assert funnel.idea_source(
        "gen-x-hl", "Second stage of gen-x (docs/TWO_STAGE.md): the same mechanism") is None
    _idea(session, "gen-x", notes="Generated from docs/MECHANISMS.md (m1)",
          status=IdeaStatus.VALIDATED, routes=[TrialRoute.PAPER])
    _idea(session, "gen-x-hl", notes="Second stage of gen-x (docs/TWO_STAGE.md): same",
          status=IdeaStatus.VALIDATED, routes=[TrialRoute.PAPER])
    period = funnel.period_funnel(session, NOW - timedelta(days=7), NOW, label="w",
                                  night_dir=NO_NIGHTS)
    row = {r.source: r for r in period.by_source}["generator"]
    assert (row.created, row.evaluated, row.survivors) == (1, 1, 1)
    assert row.survivor_ids == ["gen-x"]


def test_counts_by_source_evaluation_routes_and_survivors(session) -> None:
    _idea(session, "gen-a", notes="Generated from docs/MECHANISMS.md (m1)",
          routes=[TrialRoute.REJECT])
    _idea(session, "gen-b", notes="Generated from docs/MECHANISMS.md (m2)")  # never run
    _idea(session, "x-research-a", notes="Declared by the researcher (x).",
          routes=[TrialRoute.NEEDS_MORE_DATA, TrialRoute.PAPER], status=IdeaStatus.VALIDATED)
    _idea(session, "x-research-b", notes="Declared by the researcher (x).",
          routes=[TrialRoute.NEEDS_FORWARD], status=IdeaStatus.BENCH)
    _idea(session, "impl-a", notes="Implemented by the implementer agent from c.yaml.",
          routes=[TrialRoute.REJECT, TrialRoute.NEEDS_FORWARD])
    _idea(session, "owner-made", notes="T39: by hand.", routes=[TrialRoute.PAPER])
    _idea(session, "noise-0001", notes="Declared by the researcher (calibration).")

    period = funnel.period_funnel(session, NOW - timedelta(days=7), NOW, label="last-7-days",
                                  night_dir=NO_NIGHTS)
    by = {row.source: row for row in period.by_source}

    assert (by["generator"].created, by["generator"].evaluated) == (2, 1)
    assert by["generator"].routes == {"reject": 1}
    assert (by["researcher"].created, by["researcher"].evaluated) == (2, 2)
    # The latest routed trial decides, not the best one.
    assert by["researcher"].routes == {"paper": 1, "needs-forward": 1}
    assert by["implementer"].routes == {"needs-forward": 1}
    assert period.created == 5 and period.evaluated == 4
    assert period.survivors == 1 and period.survivor_ids == ["x-research-a"]
    assert period.routes["needs-forward"] == 2


def test_the_period_bounds_the_ideas_and_the_tokens(session) -> None:
    _idea(session, "old-research-a", notes="Declared by the researcher (x).", age_days=20,
          routes=[TrialRoute.REJECT])
    _idea(session, "new-research-b", notes="Declared by the researcher (x).", age_days=2,
          routes=[TrialRoute.REJECT])
    _spend(session, "researcher", "recheck", 1_000, age_days=20)
    _spend(session, "researcher", "recheck", 2_000, age_days=2)

    week = funnel.period_funnel(session, NOW - timedelta(days=7), NOW, label="w",
                                night_dir=NO_NIGHTS)
    since_start = funnel.period_funnel(session, NOW - timedelta(days=30), NOW, label="s",
                                       night_dir=NO_NIGHTS)

    assert (week.created, week.tokens_total) == (1, 2_000)
    assert (since_start.created, since_start.tokens_total) == (2, 3_000)


def test_tokens_are_counted_the_way_the_budget_guard_counts_them(session) -> None:
    _spend(session, "researcher", "implement", 100, cache=900)
    _spend(session, "scout", "search", 50)
    _spend(session, "researcher", "recheck", 10)
    _spend(session, "probe", "no-budget", 7_000)  # the guard's own meter reading

    period = funnel.period_funnel(session, NOW - timedelta(days=1), NOW, label="d",
                                  night_dir=NO_NIGHTS)

    assert period.tokens_by_stage == {"implement": 1_000, "search": 50, "recheck": 10}
    assert period.tokens_by_agent == {"researcher": 1_010, "scout": 50}
    assert period.tokens_total == 1_060
    assert period.probe_tokens == 7_000  # shown apart, not charged to the survivors
    assert period.scout_calls == 1


def test_price_of_one_survivor_is_the_agents_tokens_over_the_survivors(session) -> None:
    _idea(session, "x-research-a", notes="Declared by the researcher (x).",
          routes=[TrialRoute.PAPER], status=IdeaStatus.VALIDATED)
    _idea(session, "x-research-b", notes="Declared by the researcher (x).",
          routes=[TrialRoute.PAPER], status=IdeaStatus.VALIDATED)
    _spend(session, "researcher", "implement", 1_000_001)

    period = funnel.period_funnel(session, NOW - timedelta(days=7), NOW, label="d",
                                  night_dir=NO_NIGHTS)

    assert period.survivors == 2 and period.tokens_per_survivor == 500_000
    assert "цена одного выжившего — 500,000 токенов" in "\n".join(funnel.report_lines(period))


def test_with_no_survivors_the_price_is_stated_not_divided(session) -> None:
    _idea(session, "x-research-a", notes="Declared by the researcher (x).",
          routes=[TrialRoute.REJECT], status=IdeaStatus.REJECTED)
    _spend(session, "researcher", "implement", 5_000)

    period = funnel.period_funnel(session, NOW - timedelta(days=7), NOW, label="last-7-days",
                                  night_dir=NO_NIGHTS)

    assert period.survivors == 0 and period.tokens_per_survivor is None
    text = "\n".join(funnel.report_lines(period))
    assert "цена одного выжившего — выживших пока нет" in text
    assert "Дошли до владельца (прошли все правила и ждут его решения): 0" in text


def test_an_empty_registry_gives_an_empty_funnel(session) -> None:
    result = funnel.agent_funnel(session, NOW, night_dir=NO_NIGHTS)

    assert [p.label for p in result.periods] == ["last-7-days", "since-agents-started"]
    assert all(p.created == 0 and p.tokens_per_survivor is None for p in result.periods)
    assert result.periods[1].since == funnel.AGENT_WORK_START
    assert result.to_dict()["periods"][0]["tokens_total"] == 0


def test_scout_drafts_come_from_the_nights_saved_state(tmp_path) -> None:
    for day, scout in (("2026-10-20", {"reply": "NO CANDIDATE: x", "draft": ""}),
                       ("2026-10-22", {"reply": "DRAFT: cards/drafts/a.yaml",
                                       "draft": "cards/drafts/a.yaml"}),
                       ("2026-10-04", {"draft": "cards/drafts/older.yaml"}),
                       ("2026-10-01", {"draft": "cards/drafts/before-agents.yaml"}),
                       ("2026-10-23", {"stopped": "night budget"})):
        (tmp_path / f"{day}.json").write_text(json.dumps({"scout": scout}))
    (tmp_path / "logs.json").write_text("{}")  # not a night

    assert funnel.scout_drafts(tmp_path, NOW - timedelta(days=7), NOW) == 1
    assert funnel.scout_drafts(tmp_path, funnel.AGENT_WORK_START, NOW) == 2


def test_report_lines_are_plain_russian(session) -> None:
    _idea(session, "x-research-a", notes="Declared by the researcher (x).",
          routes=[TrialRoute.NEEDS_FORWARD], status=IdeaStatus.BENCH)

    lines = funnel.report_lines(funnel.period_funnel(
        session, NOW - timedelta(days=7), NOW, label="last-7-days",
        night_dir=NO_NIGHTS))

    assert lines[0] == "**За последние 7 дней**"
    assert "Агенты придумали идей: 1 (генератор 0, исследователь 1, реализатор 0)" in lines[1]
    assert "Прогнано на стенде: 1 из 1; исходы: ждёт проверки на новых данных — 1" in lines[2]
