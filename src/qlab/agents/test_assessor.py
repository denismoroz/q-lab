"""Tests for the assessor agent: its claims are checked against the run's
facts by code, a dispute needs a verified claim, and the verdict is never
touched."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from qlab.agents.assessor import Brief, build_brief, parse_answer, run_assessment, verify
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.registry import repo
from qlab.registry.models import (
    Assessment,
    AssetClass,
    Base,
    Profile,
    SourceType,
    Spec,
    Trial,
    TrialRoute,
    TrialSource,
    TrialStatus,
)

BRIEF = Brief(idea_id="a", trial_id=1,
              facts={"trial.noise_power": 0.01, "trial.ann_return_net": 0.238},
              context={"paper": "paper  +4.11%   replay +4.11%"})


def test_only_claims_that_match_the_facts_are_kept() -> None:
    answer = {"view": "too_harsh", "summary": "s", "claims": [
        {"text": "the bar cannot tell", "fact": "trial.noise_power", "value": 0.01},
        {"text": "rounded", "fact": "trial.ann_return_net", "value": 0.24},
        {"text": "wrong", "fact": "trial.ann_return_net", "value": 0.30},
        {"text": "invented", "fact": "trial.sharpe_net", "value": 2.0},
        {"text": "paper", "quote": "paper +4.11%  replay +4.11%"},  # whitespace normalised
        {"text": "made up", "quote": "paper +9%"},
        {"text": "nothing"},
    ], "checks": ["one", "two", "three", "four"]}
    out = verify(answer, BRIEF)
    assert [c.text for c in out.accepted] == ["the bar cannot tell", "rounded", "paper"]
    reasons = {c.text: why for c, why in out.rejected}
    assert "not 0.3" in reasons["wrong"]
    assert "not among the facts" in reasons["invented"]
    assert "not verbatim" in reasons["made up"]
    assert "no evidence" in reasons["nothing"]
    assert out.view == "too_harsh" and out.dispute
    assert out.checks == ["one", "two", "three"]


def test_a_dispute_without_a_verified_claim_is_only_unsure() -> None:
    out = verify({"view": "too_lenient", "summary": "s",
                  "claims": [{"text": "x", "fact": "trial.sharpe_net", "value": 1.0}]}, BRIEF)
    assert out.view == "unsure" and not out.dispute
    assert verify({"view": "maybe", "claims": []}, BRIEF).view == "unsure"


def test_parse_answer_tolerates_text_around_the_json() -> None:
    assert parse_answer('ok\n{"view": "agree"}\n') == {"view": "agree"}
    assert parse_answer("no json") is None


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for idea in ("a", "a-variant"):
        repo.upsert_idea(s, id=idea, title=idea, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    yield s
    s.close()


def _trial(session, idea: str, metrics: dict, route: TrialRoute) -> Trial:
    spec = Spec(idea_id=idea, version=1, code_ref="qlab.strategies.live.bv2:LiveBv2",
                params={}, data_requirements={}, rebalance="1h", costs_model={})
    session.add(spec)
    session.flush()
    trial = Trial(spec_id=spec.id, config_hash=f"h-{idea}", code_sha="x", params={},
                  started_at=datetime.now(UTC), metrics=metrics, status=TrialStatus.OK,
                  route=route, route_reason="too early", source=TrialSource.IMPORTED)
    session.add(trial)
    session.flush()
    return trial


def test_assessment_is_recorded_and_the_verdict_untouched(session, tmp_path, monkeypatch) -> None:
    variant = _trial(session, "a-variant", {"ann_return_net": 0.02}, TrialRoute.NEEDS_FORWARD)
    trial = _trial(session, "a", {"noise_power": 0.01, "ann_return_net": 0.238},
                   TrialRoute.NEEDS_FORWARD)
    brief = build_brief(session, trial.id, paper=["**Bv2 (book 3)**", "    paper  +4.11%",
                                                  "**trend**", "    paper -1.01%"])
    assert brief.facts["a-variant.ann_return_net"] == 0.02
    assert "+4.11%" in brief.context["production paper book vs the stand's replay"]
    assert "-1.01%" not in brief.context["production paper book vs the stand's replay"]
    assert variant.id != trial.id

    answer = {"view": "too_harsh", "summary": "рано", "claims": [
        {"text": "проверка бессильна", "fact": "trial.noise_power", "value": 0.01}]}
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    events = [
        {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
            "seven_day": {"utilization": 0.2, "resetsAt": reset}}}},
        {"type": "result", "is_error": False, "result": json.dumps(answer), "total_cost_usd": 0.1,
         "usage": {"input_tokens": 5000, "output_tokens": 300}, "modelUsage": {"claude-opus": {}}},
    ]
    out = tmp_path / "out.jsonl"
    out.write_text("\n".join(json.dumps(e) for e in events))
    script = tmp_path / "claude"
    script.write_text(f'#!/bin/sh\ncat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))

    outcome = run_assessment(brief=brief, budget=ExplicitNoBudget("test"), session=session)
    assert outcome.dispute
    row = session.query(Assessment).one()
    assert (row.trial_id, row.view, row.summary) == (trial.id, "too_harsh", "рано")
    assert session.get(Trial, trial.id).route == TrialRoute.NEEDS_FORWARD


def test_a_family_members_run_without_metrics_is_skipped_for_its_last_good_one(session) -> None:
    # The night of 2026-10-27: a variant's latest run failed (metrics None, once stored as the
    # JSON text 'null' that `IS NOT NULL` let through) and the brief crashed the night.
    good = _trial(session, "a-variant", {"ann_return_net": 0.02}, TrialRoute.NEEDS_FORWARD)
    failed = Trial(spec_id=good.spec_id, config_hash="h-failed", code_sha="x", params={},
                   started_at=datetime.now(UTC), metrics=None, status=TrialStatus.NOT_EVALUABLE,
                   route=TrialRoute.NOT_EVALUABLE, route_reason="error",
                   source=TrialSource.IMPORTED)
    session.add(failed)
    session.flush()
    trial = _trial(session, "a", {"ann_return_net": 0.238}, TrialRoute.NEEDS_FORWARD)
    raw = session.execute(text("select metrics from trial where id = :i"), {"i": failed.id})
    assert raw.scalar() is None  # SQL NULL, not 'null'
    brief = build_brief(session, trial.id)
    assert brief.facts["a-variant.ann_return_net"] == 0.02
