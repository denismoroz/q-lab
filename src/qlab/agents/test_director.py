"""Tests for the research director: the evidence code builds (aggregates only, no
per-variant performance), the agent call through the budget guard with a fake
`claude`, and the memo's block becoming `night/directions.yaml`."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import qlab.agents.director as director
import qlab.pipeline.sources as sources
from qlab import directions as dirs
from qlab.budget import ExplicitNoBudget
from qlab.budget.agent import CLAUDE_BIN_ENV
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
CATALOGUE = """# Каталог (фикстура)

### xs-mom

**Суть:** Покупают выросшие монеты.

**Блоки:** universe all; signal momentum

**Параметры:**

| параметр блока | значения из источника | источник |
|---|---|---|
| signal.lookback_days | 7, 14 | [A 2020](https://e.org/a) |
| rebalance_days | 7 | [A 2020](https://e.org/a) |

**Чего блоки не выражают:** (1) Вес по капитализации.

### xs-rev

**Суть:** Откат после недели роста.

**Блоки:** universe all; signal reversal

**Параметры:**

| параметр блока | значения из источника | источник |
|---|---|---|
| signal.lookback_days | 7 | [B 2021](https://e.org/b) |

**Чего блоки не выражают:** (0) Нужны малооборотные монеты.

### carry-1

**Суть:** Перенос ставки.

**Блоки:** signal funding

**Параметры:**

| параметр блока | значения из источника | источник |
|---|---|---|

**Чего блоки не выражают:** (1) Нога в самой монете.

## Каких параметров не хватает

| механизм | чего не хватает | можно ли использовать |
|---|---|---|
| `xs-mom` | ничего | да: `signal.lookback_days` = 7, 14 |
| `xs-rev` | `portfolio.quantiles` | нет |
| `carry-1` | `signal.window_days`; **нужен новый блок** связки | нет |
"""

CONFIG = dirs.DirectorConfig(enabled=True, cadence_days=2, max_turns=10, families={
    "momentum": ("xs-mom",), "reversal": ("xs-rev",), "funding": ("carry-1",)})
PLAN = {"strategies": {
    "bv2": {"ideas": ["strategy-b-v2"], "owner_ideas": ["включать хеджирование почаще"]},
    "trend": {"ideas": ["trend-live"]},
}}

# Values that are per-variant performance. None of them may reach the evidence.
SECRET_METRICS = {"sharpe_net": 1.2345, "ann_return_net": 0.4321, "max_dd": -0.0777,
                  "noise_return_percentile": 0.9876}


@pytest.fixture()
def root(tmp_path, monkeypatch) -> Path:
    r = tmp_path / "repo"
    (r / "docs" / "research").mkdir(parents=True)
    (r / "night").mkdir()
    (r / "specs" / "generated").mkdir(parents=True)
    (r / "docs" / "MECHANISMS.md").write_text(CATALOGUE, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                   cwd=r, check=True)
    monkeypatch.setattr(director, "QLAB_ROOT", r)
    monkeypatch.setattr("qlab.agents.implementer.QLAB_ROOT", r)
    monkeypatch.setattr(sources, "QLAB_ROOT", r)
    return r


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id, *, notes, route=None, failed=(), status=IdeaStatus.BENCH,
          age_days=2.0, parent=None):
    """An agent-created idea with one routed trial whose metrics are the secret ones."""
    idea = repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                            asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                            status=status, notes=notes)
    idea.created_at = NOW - timedelta(days=age_days)
    idea.parent_id = parent
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                         rebalance="1d", costs_model={}, code_ref="a:b")
    if route is None:
        session.flush()
        return idea
    if session.get(DataSnapshot, "snap") is None:
        repo.add_data_snapshot(session, id="snap", source="binance", instruments={},
                               range_start=date(2019, 9, 8), range_end=date(2026, 10, 24),
                               path="x", rows=1)
    trial = repo.add_trial(session, spec_id=spec.id, config_hash=idea_id, code_sha="x",
                           params={}, status=TrialStatus.OK, route=route, snapshot_id="snap",
                           metrics=SECRET_METRICS,
                           route_reason="fails with Sharpe 1.2345 and return 0.4321")
    repo.add_verdicts(session, [{
        "idea_id": idea_id, "trial_id": trial.id, "spec_id": spec.id, "stage": "edge",
        "rule_id": rule, "rules_version": "2026-10-24.2", "metric": "m", "value": 1.2345,
        "comparator": ">=", "threshold": 2.0, "passed": passed,
        "data_range_start": date(2019, 9, 8), "data_range_end": date(2026, 10, 24)}
        for rule, passed in [*[(r, False) for r in failed], ("a_rule_that_passed", True)]])
    session.flush()
    return idea


def _spec_header(root, idea_id, mechanism):
    (root / "specs" / "generated" / f"{idea_id}.yaml").write_text(
        f"# mechanism: {mechanism}\nidea_id: {idea_id}\n", encoding="utf-8")


def _populate(session, root):
    gen = "Generated from docs/MECHANISMS.md ({})"
    _idea(session, "gen-a", notes=gen.format("xs-mom"), route=TrialRoute.REJECT,
          failed=("venue_supported", "net_edge_positive"), status=IdeaStatus.REJECTED)
    _spec_header(root, "gen-a", "xs-mom")
    _idea(session, "gen-b", notes=gen.format("xs-mom"), route=TrialRoute.REJECT,
          failed=("venue_supported",), status=IdeaStatus.REJECTED)
    _spec_header(root, "gen-b", "xs-mom")
    # No spec file: the mechanism falls back to the note.
    _idea(session, "gen-c", notes=gen.format("xs-rev"), route=TrialRoute.NEEDS_FORWARD)
    _idea(session, "gen-d", notes=gen.format("xs-rev"))  # never run
    _idea(session, "strategy-b-v2", notes="owner's")
    _idea(session, "bv2-research-x", notes="Declared by the researcher (bv2).",
          route=TrialRoute.PAPER, status=IdeaStatus.VALIDATED, parent="strategy-b-v2")
    _idea(session, "bv2-research-y", notes="Declared by the researcher (bv2).",
          route=TrialRoute.NEEDS_FORWARD, failed=("shape_aware_edge",), parent="strategy-b-v2")
    _idea(session, "trend-live", notes="owner's")
    _idea(session, "trend-research-z", notes=None, route=TrialRoute.NEEDS_FORWARD,
          failed=("shape_aware_edge",), parent="trend-live")  # no note: the id says researcher
    _idea(session, "noise-0001", notes="Declared by the researcher (calibration).",
          route=TrialRoute.REJECT)
    _idea(session, "owner-made", notes="by hand", route=TrialRoute.PAPER)
    session.add(TokenSpend(at=NOW - timedelta(days=1), stage="implement", agent="researcher",
                           tokens_in=1000, tokens_out=0, usd_est=0.0))
    session.add(TokenSpend(at=NOW - timedelta(days=1), stage="search", agent="generator",
                           tokens_in=500, tokens_out=0, usd_est=0.0))
    session.flush()


def _evidence(session, root, **kw) -> str:
    return director.build_evidence(session, root=root, config=CONFIG, plan=PLAN, now=NOW, **kw)


def _section(evidence: str, title: str) -> str:
    start = evidence.index(f"### {title}")
    end = evidence.find("\n### ", start + 1)
    return evidence[start:end if end > 0 else len(evidence)]


# ---- the evidence ---------------------------------------------------------------------


def test_verdicts_are_aggregated_by_family_and_by_failed_rule(session, root) -> None:
    _populate(session, root)
    evidence = _evidence(session, root)

    momentum = _section(evidence, "generator / momentum")
    assert "ideas: 2" in momentum and "route of the latest trial: reject 2" in momentum
    # venue_supported is a fact about OUR setup (kind: infrastructure): counted apart
    # from the strategy rules, so it never reads as the reason an idea failed.
    assert "rules failed by the latest trial: net_edge_positive 1" in momentum
    assert "never a reason to close a direction): venue_supported 2" in momentum
    assert "status today: rejected 2" in momentum
    reversal = _section(evidence, "generator / reversal")  # the mechanism came from the note
    assert "ideas: 2" in reversal and "never evaluated: 1" in reversal
    assert "needs-forward 1" in reversal
    bv2 = _section(evidence, "researcher / bv2")
    assert "ideas: 2" in bv2 and "paper 1" in bv2 and "shape_aware_edge 1" in bv2
    assert "survivors: bv2-research-x" in bv2
    assert "researcher / trend" in evidence  # found by its id, without a note
    # Calibration noise and the owner's own ideas are nobody's agent.
    assert "noise-0001" not in evidence and "owner-made" not in evidence
    assert "a_rule_that_passed" not in evidence  # only failures are reasons


def test_no_per_variant_performance_reaches_the_evidence(session, root) -> None:
    _populate(session, root)
    evidence = _evidence(session, root)
    for value in ("1.2345", "0.4321", "0.0777", "0.9876", "-0.0777"):
        assert value not in evidence
    low = evidence.lower()
    for word in ("sharpe_net", "ann_return", "max_dd", "route_reason", "fails with"):
        assert word not in low
    assert "NO per-parameter or per-variant" in evidence
    # The module says so too.
    assert "no per-parameter or per-variant" in director.__doc__.lower()


def test_the_funnel_and_the_tokens_by_direction_are_in_the_evidence(session, root, tmp_path
                                                                    ) -> None:
    _populate(session, root)
    nights = root / "data" / "night"
    nights.mkdir(parents=True)
    (nights / "2026-10-24.json").write_text(json.dumps({"research": [
        {"strategy": "bv2", "tokens": 1200}, {"strategy": "trend", "tokens": 300},
        {"strategy": "trend", "tokens": 999, "stopped": "budget"}]}), encoding="utf-8")
    evidence = _evidence(session, root)
    assert "### last-14-days" in evidence and "### since-agents-started" in evidence
    assert "researcher tokens by strategy (from the nights' saved state): bv2 1,200, trend 300" \
        in evidence
    assert "by agent: researcher 1,000, generator 500" in evidence
    assert "price of one survivor:" in evidence


def test_catalogue_coverage_says_what_is_buildable_and_what_each_lacks(session, root) -> None:
    _populate(session, root)
    evidence = _evidence(session, root)
    assert "### family momentum: 1 mechanism(s), 1 buildable now" in evidence
    assert "- xs-mom: buildable now yes; parameter rows 2; ideas made 2" in evidence
    assert "- xs-rev: buildable now NO; parameter rows 1; ideas made 2" in evidence
    assert "lacks: `portfolio.quantiles`" in evidence
    assert "not expressed: (1) Вес по капитализации." in evidence
    assert "- carry-1: buildable now NO; parameter rows 0; ideas made 0" in evidence
    names = _section(evidence, "family funding") if "### family funding" in evidence else ""
    assert "carry-1" in names
    # Names the agent may use come from code.
    assert "mechanism ids (the catalogue docs/MECHANISMS.md): xs-mom, xs-rev, carry-1" in evidence
    assert "researcher strategies (night/research.yaml): bv2, trend" in evidence


def test_a_mechanism_outside_every_family_counts_as_its_own(session, root) -> None:
    config = dirs.DirectorConfig(True, 2, 10, {"momentum": ("xs-mom",)})
    evidence = director.build_evidence(session, root=root, config=config, plan=PLAN, now=NOW)
    assert "### family xs-rev: 1 mechanism(s)" in evidence
    assert "no family in night/director.yaml, each counted as its own: carry-1, xs-rev" in evidence


def test_drop_proposals_are_masked_and_the_owners_ideas_are_verbatim(session, root) -> None:
    (root / "docs" / "research" / "bv2-2026-10-25.md").write_text(
        "# memo\nDROP: bv2-research-y — Sharpe 0.41 is worse than 1.2345 of the other\n",
        encoding="utf-8")
    evidence = _evidence(session, root)
    bv2 = _section(evidence, "strategy bv2")
    assert "DROP proposal in bv2-2026-10-25.md: bv2-research-y -- Sharpe # is worse than # of " \
        "the other (numbers masked)" in bv2
    assert "0.41" not in evidence and "1.2345" not in evidence
    assert "the owner's idea, verbatim: «включать хеджирование почаще»" in bv2
    assert "no DROP proposal in its newest memo" in _section(evidence, "strategy trend")


def test_decay_lines_are_the_newest_nights(session, root) -> None:
    nights = root / "data" / "night"
    nights.mkdir(parents=True)
    (nights / "2026-10-23.json").write_text(json.dumps({"decay": {"lines": ["old line"]}}))
    (nights / "2026-10-24.json").write_text(json.dumps({"decay": {"lines": ["trend: inside"]}}))
    (nights / "2026-10-25.json").write_text(json.dumps({"research": []}))  # no decay yet
    evidence = _evidence(session, root)
    assert "night 2026-10-24:\ntrend: inside" in evidence and "old line" not in evidence
    (nights / "2026-10-24.json").unlink()
    (nights / "2026-10-23.json").unlink()
    assert "no night has decay-detector lines yet" in _evidence(session, root)


def test_the_previous_directions_are_shown_with_their_status(session, root) -> None:
    assert "none: this is the first run" in _evidence(session, root)
    block = "generator:\n  prefer_mechanisms: [xs-mom]\n"
    parsed = dirs.parse_block(block, mechanism_ids={"xs-mom"}, strategy_names=set(),
                              family_names=set(), day=date(2026, 10, 23),
                              pending_owner_review=True)
    dirs.write_directions(parsed.directions, root)
    evidence = _evidence(session, root)
    assert "status: pending" in evidence and "prefer_mechanisms:" in evidence


def test_the_gap_table_is_read_from_the_catalogues_last_section() -> None:
    gaps = director.parse_gap_table(CATALOGUE)
    assert gaps["xs-mom"] == ("ничего", "да: `signal.lookback_days` = 7, 14")
    assert gaps["xs-rev"][1] == "нет"
    assert director.parse_gap_table("# no table") == {}


def test_the_real_catalogue_and_config_make_an_evidence_file(session) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = dirs.load_config(repo_root / "night/director.yaml")
    evidence = director.build_evidence(
        session, root=repo_root, config=config,
        plan=yaml.safe_load((repo_root / "night/research.yaml").read_text(encoding="utf-8")),
        now=NOW, night_dir=repo_root / "no-such-night-dir")
    assert "crypto-xs-momentum: buildable now yes" in evidence
    assert "perp-funding-carry: buildable now NO" in evidence
    assert "no agent-created idea in the registry" in evidence


# ---- the agent call ---------------------------------------------------------------------


def _fake_claude(tmp_path: Path, monkeypatch, body: str) -> Path:
    reset = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"type": "rate_limit_event", "rate_limit_info": {
            "status": "allowed", "unifiedWindows": {"seven_day": {
                "utilization": 0.2, "resetsAt": reset}}}}) + "\n"
        + json.dumps({"type": "result", "is_error": False, "result": "docs/research/x.md",
                      "total_cost_usd": 0.01, "usage": {"input_tokens": 100, "output_tokens": 10},
                      "modelUsage": {"claude-opus": {}}}) + "\n")
    args_file = tmp_path / "args.txt"
    script = tmp_path / "claude"
    script.write_text(f'#!/bin/sh\necho "$@" > "{args_file}"\n{body}\ncat "{out}"\n')
    script.chmod(0o755)
    monkeypatch.setenv(CLAUDE_BIN_ENV, str(script))
    return args_file


MEMO = """# Записка руководителя

Всё хорошо.

```directions
generator:
  prefer_mechanisms: [xs-mom, nonexistent]
  pause_mechanisms:
    - {id: xs-rev, why: "нужны малооборотные монеты, которых блоки не выражают"}
researcher:
  bv2: {priority: high, note: "искать, где хедж включается поздно"}
propose_close_family:
  - {family: funding, why: "блоков для него нет"}
```
"""


def test_the_director_without_a_budget_raises_instead_of_spending(session, root) -> None:
    with pytest.raises(TypeError, match="run_agent needs a CandidateBudget"):
        director.direct(budget=None, session=session, config=CONFIG, plan=PLAN)  # type: ignore[arg-type]


def test_the_agent_runs_on_opus_with_read_only_tools_and_writes_only_its_memo(
        session, root, tmp_path, monkeypatch) -> None:
    today = datetime.now(UTC).date()
    memo_src = tmp_path / "memo.md"
    memo_src.write_text(MEMO, encoding="utf-8")
    args_file = _fake_claude(tmp_path, monkeypatch,
                             f'cp "{memo_src}" docs/research/director-{today}.md')
    outcome = director.direct(budget=ExplicitNoBudget("test"), session=session, config=CONFIG,
                              plan=PLAN)
    assert outcome.memo == Path(f"docs/research/director-{today}.md")
    assert outcome.problems == [] and outcome.stray == []
    args = args_file.read_text()
    assert "--model opus" in args and "Read,Glob,Grep,Write" in args
    assert "--disallowedTools Bash,Edit" in args and "--max-turns 10" in args
    # The prompt names the evidence file and the memo, and carries the owner's words.
    assert f"data/research/director-{today}.md" in args and "раз в 2 дня" in args
    assert (root / "data" / "research" / f"director-{today}.md").read_text().startswith(
        "# Evidence for the research director")
    spend = session.execute(text("select agent, stage, status from token_spend")).all()
    assert spend == [("director", "no-budget", "ok; no budget: test")]


def test_a_stray_file_is_reported_not_reverted_and_the_directions_file_is_restored(
        session, root, tmp_path, monkeypatch) -> None:
    today = datetime.now(UTC).date()
    memo_src = tmp_path / "memo.md"
    memo_src.write_text(MEMO, encoding="utf-8")
    dirs.write_directions(dirs.parse_block(
        "generator: {}", mechanism_ids=set(), strategy_names=set(), family_names=set(),
        day=date(2026, 10, 20)).directions, root)
    before = (root / dirs.DIRECTIONS).read_bytes()
    _fake_claude(tmp_path, monkeypatch, f"""
cp "{memo_src}" docs/research/director-{today}.md
echo stray > STRAY.txt
echo "pending_owner_review: false" > night/directions.yaml
""")
    outcome = director.direct(budget=ExplicitNoBudget("test"), session=session, config=CONFIG,
                              plan=PLAN)
    assert (root / "STRAY.txt").exists() and outcome.stray == ["STRAY.txt"]  # not reverted
    assert any("outside its place" in p and "STRAY.txt" in p for p in outcome.problems)
    assert (root / dirs.DIRECTIONS).read_bytes() == before  # the agent's edit is undone
    assert any("restored" in p for p in outcome.problems)


def test_an_agent_that_writes_nothing_leaves_no_memo(session, root, tmp_path, monkeypatch) -> None:
    _fake_claude(tmp_path, monkeypatch, "true")
    outcome = director.direct(budget=ExplicitNoBudget("test"), session=session, config=CONFIG,
                              plan=PLAN)
    assert outcome.memo is None and "no memo written" in outcome.problems


# ---- from the memo to the file ------------------------------------------------------------


def _apply(root, memo_rel="docs/research/director-2026-10-26.md", day=date(2026, 10, 26)):
    return director.apply_memo(Path(memo_rel), root=root, mechanism_ids={"xs-mom", "xs-rev",
                                                                          "carry-1"},
                               strategy_names={"bv2", "trend"},
                               family_names={"momentum", "reversal", "funding"}, day=day)


def test_the_first_run_writes_directions_waiting_for_the_owner(root) -> None:
    memo = root / "docs" / "research" / "director-2026-10-26.md"
    memo.write_text(MEMO, encoding="utf-8")
    result = _apply(root)
    d = result.directions
    assert d.pending_owner_review is True
    assert d.prefer_mechanisms == ("xs-mom",) and d.paused().keys() == {"xs-rev"}
    assert d.researcher["bv2"].priority == "high"
    assert [c.family for c in d.propose_close_family] == ["funding"]
    assert any("nonexistent" in p for p in result.problems)  # an unknown id was dropped
    # Until the owner clears the flag, the generator and the researcher ignore it.
    assert dirs.active_directions(root) is None
    assert (root / dirs.HISTORY_DIR / "2026-10-26.yaml").is_file()
    # The memo tells the owner what the code accepted and that it waits.
    text_ = memo.read_text(encoding="utf-8")
    assert director.APPLIED_HEADING in text_ and "ждёт проверки владельца" in text_
    assert "nonexistent" in text_ and "решает владелец" in text_


def test_a_later_run_applies_at_once_once_the_owner_has_reviewed(root) -> None:
    memo = root / "docs" / "research" / "director-2026-10-26.md"
    memo.write_text(MEMO, encoding="utf-8")
    _apply(root)
    dirs.approve(root)
    memo2 = root / "docs" / "research" / "director-2026-10-28.md"
    memo2.write_text(MEMO, encoding="utf-8")
    second = _apply(root, "docs/research/director-2026-10-28.md", date(2026, 10, 28))
    assert second.directions.pending_owner_review is False
    assert dirs.active_directions(root).date == date(2026, 10, 28)


def test_applying_twice_does_not_stack_sections(root) -> None:
    memo = root / "docs" / "research" / "director-2026-10-26.md"
    memo.write_text(MEMO, encoding="utf-8")
    _apply(root)
    _apply(root)
    assert memo.read_text(encoding="utf-8").count(director.APPLIED_HEADING) == 1


def test_a_memo_without_a_block_changes_nothing(root) -> None:
    memo = root / "docs" / "research" / "director-2026-10-26.md"
    memo.write_text("# Записка без блока\n", encoding="utf-8")
    result = _apply(root)
    assert result.directions is None and result.written is None
    assert not (root / dirs.DIRECTIONS).exists()
    assert "не принят" in memo.read_text(encoding="utf-8")
