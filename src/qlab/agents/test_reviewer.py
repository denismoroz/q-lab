"""Tests for the reviewer agent: evidence is verified by code, and an
accepted blocking finding makes a run not evaluable."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.agents.reviewer import (
    code_files,
    parse_findings,
    review_key,
    run_review,
    unanswered_blocking,
    verify,
)
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
from qlab.pipeline.evaluate import review_reasons
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Profile, Review, SourceType

SOURCES = {"card.md": "Exit when the signal turns.\n"
                      "Keep a margin pool and top it up when it bleeds."}
CODE = {"src/x.py": "def weights():\n    return signal\n"}


def _spec(**extra) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "rev", "title": "t", "code_ref": "qlab.strategies.trend:TrendTSMOMEnsemble",
        "params": {}, "data": {"source": "hyperliquid", "interval": "1d", "start": "2026-01-01",
                               "end": "2026-02-01"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], **extra})


def test_only_findings_with_verified_evidence_are_accepted() -> None:
    raw = [
        {"kind": "unexpressed_mechanism", "summary": "no margin pool",
         "source_quote": "Keep a margin pool and   top it up"},  # whitespace normalised
        {"kind": "unexpressed_mechanism", "summary": "invented", "source_quote": "a stop loss"},
        {"kind": "lookahead_risk", "summary": "peek", "code_ref": "src/x.py:2"},
        {"kind": "lookahead_risk", "summary": "bad line", "code_ref": "src/x.py:99"},
        {"kind": "parameter_mismatch", "summary": "no ref", "source_quote": "Exit when"},
    ]
    out = verify(raw, SOURCES, CODE)
    assert [f.summary for f in out.accepted] == ["no margin pool", "peek"]
    reasons = dict((f.summary, why) for f, why in out.rejected)
    assert "not verbatim" in reasons["invented"]
    assert "outside" in reasons["bad line"]
    assert "no code_ref" in reasons["no ref"]


def test_parse_findings_tolerates_text_around_the_json() -> None:
    assert parse_findings('Here:\n{"findings": []}\nDone') == []
    assert parse_findings("no json") is None


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    repo.upsert_idea(s, id="rev", title="t", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    yield s
    s.close()


def test_review_is_recorded_and_blocks_until_answered(session, tmp_path, monkeypatch) -> None:
    answer = {"findings": [{"kind": "unexpressed_mechanism", "summary": "no margin pool",
                            "source_quote": "Keep a margin pool"}]}
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

    spec = _spec()
    code = code_files(spec)
    outcome = run_review(spec=spec, spec_text="spec", sources=SOURCES, code=code,
                         budget=ExplicitNoBudget("test"), session=session)
    assert [f.summary for f in outcome.accepted] == ["no margin pool"]
    row = session.query(Review).one()
    assert row.review_key == review_key(spec, code)

    [reason] = review_reasons(session, spec)
    assert "no margin pool" in reason
    answered = _spec(review_answers={"no margin pool": "the book has no margin: spot only"})
    assert review_reasons(session, answered) == []
    assert unanswered_blocking(answered, row.accepted) == []
