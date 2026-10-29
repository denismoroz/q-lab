"""The four year signs and their statistics (rules 2026-10-28.2, docs/YEARS.md)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

import qlab.registry.db as db_module
from qlab import year_signs
from qlab.cli import app
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
)
from qlab.registry.stages import NOTES_PREFIX
from qlab.rules.loader import DEFAULT_RULES_DIR, load

RULES = load("2026-10-28.2", DEFAULT_RULES_DIR)

# Two counted years, both profitable, one of them lucky: A holds (the other year still earns
# 8% a year), B holds (all years positive), V holds, G fails (the best year gives 70%).
LUCKY = {
    "year_2024_return": 0.08, "year_2024_days": 366,
    "year_2025_return": 0.20, "year_2025_days": 365,
    "years_counted": 2, "years_positive": 2, "years_positive_share": 1.0,
    "ann_return_net_ex_best_year": 0.08, "best_year_share_of_profit": 0.7,
    "ann_return_net_ex_best_1pct": 0.05,
}
# Half the years lost money, and the profit is all in one: A fails, B fails, V fails, G fails.
BAD = {
    "year_2021_return": 0.5, "year_2021_days": 365,
    "year_2022_return": -0.2, "year_2022_days": 365,
    "year_2023_return": -0.1, "year_2023_days": 365,
    "year_2024_return": -0.1, "year_2024_days": 366,
    "years_counted": 4, "years_positive": 1, "years_positive_share": 0.25,
    "ann_return_net_ex_best_year": -0.14, "best_year_share_of_profit": 1.5,
    "ann_return_net_ex_best_1pct": -0.01,
}
# One counted year: A and G are not defined, B and V are.
ONE_YEAR = {
    "year_2026_return": 0.05, "year_2026_days": 200,
    "years_counted": 1, "years_positive": 1, "years_positive_share": 1.0,
    "ann_return_net_ex_best_1pct": 0.1,
}


def _states(metrics):
    return {r.letter: r.state for r in year_signs.read_signs(metrics, RULES)}


# ---- one run -------------------------------------------------------------------------------


def test_each_sign_holds_fails_or_is_not_defined():
    assert _states(LUCKY) == {"А": "holds", "Б": "holds", "В": "holds", "Г": "fails"}
    assert _states(BAD) == {"А": "fails", "Б": "fails", "В": "fails", "Г": "fails"}
    assert _states(ONE_YEAR) == {
        "А": "not defined", "Б": "holds", "В": "holds", "Г": "not defined"}


def test_the_lines_are_the_ones_in_the_ruleset():
    by_letter = {r.letter: r for r in year_signs.read_signs(LUCKY, RULES)}
    assert (by_letter["А"].comparator, by_letter["А"].threshold) == (">=", 0.04)
    assert (by_letter["Б"].comparator, by_letter["Б"].threshold) == (">=", 0.667)
    assert (by_letter["В"].comparator, by_letter["В"].threshold) == (">", 0.0)
    assert (by_letter["Г"].comparator, by_letter["Г"].threshold) == ("<=", 0.5)
    # The plain Russian titles name the same numbers as the lines.
    assert "4%" in by_letter["А"].title and "двух третей" in by_letter["Б"].title
    assert "половины" in by_letter["Г"].title


@pytest.mark.parametrize("value, holds", [(0.0399, False), (0.04, True)])
def test_sign_a_line_is_four_percent(value, holds):
    reading = year_signs.read_signs({"ann_return_net_ex_best_year": value}, RULES)[0]
    assert reading.letter == "А" and reading.holds is holds


@pytest.mark.parametrize("value, holds", [(0.666, False), (0.667, True)])
def test_sign_b_line_is_two_thirds(value, holds):
    reading = year_signs.read_signs({"years_positive_share": value}, RULES)[1]
    assert reading.letter == "Б" and reading.holds is holds


@pytest.mark.parametrize("value, holds", [(0.5, True), (0.5001, False)])
def test_sign_g_line_is_half_of_the_profit(value, holds):
    reading = year_signs.read_signs({"best_year_share_of_profit": value}, RULES)[3]
    assert reading.letter == "Г" and reading.holds is holds


def test_not_defined_says_why_in_plain_russian():
    one = {r.letter: r for r in year_signs.read_signs(ONE_YEAR, RULES)}
    assert one["А"].reason == "засчитанных лет меньше двух"
    assert one["Г"].reason == "засчитанных лет меньше двух"
    assert one["Б"].reason is None
    loss = {**LUCKY, "years_counted": 2}
    loss.pop("best_year_share_of_profit")
    assert {r.letter: r for r in year_signs.read_signs(loss, RULES)}["Г"].reason == \
        "прибыль за весь период не положительна"
    old = {r.letter: r for r in year_signs.read_signs({"ann_return_net": 0.2}, RULES)}
    assert old["А"].reason == "у прогона нет разбивки по годам"
    assert old["Б"].reason == "у прогона нет разбивки по годам"


def test_a_sign_is_worded_with_its_state_and_number():
    reading = year_signs.read_signs(LUCKY, RULES)[0]
    assert year_signs.sign_line(reading) == (
        "А: без лучшего года не меньше 4% в год — держится (8.0%)")
    undefined = year_signs.read_signs(ONE_YEAR, RULES)[0]
    assert year_signs.sign_line(undefined) == (
        "А: без лучшего года не меньше 4% в год — не определено (засчитанных лет меньше двух)")
    assert year_signs.signs_sentence(ONE_YEAR, RULES) == (
        "признаки: А не определено (засчитанных лет меньше двух), Б держится, "
        "В держится, Г не определено (засчитанных лет меньше двух)")


def test_an_older_ruleset_without_the_signs_reads_none():
    assert year_signs.read_signs(LUCKY, load("2026-10-27.1", DEFAULT_RULES_DIR)) == []
    assert year_signs.signs_sentence(LUCKY, load("2026-10-27.1", DEFAULT_RULES_DIR)) is None


# ---- the registry --------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _idea(session, idea_id, *, source="binance", parent=None):
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     notes=NOTES_PREFIX + (parent or "") if parent else None)
    if parent:
        session.get(Idea, idea_id).parent_id = parent
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d",
                         costs_model={}, code_ref="x:y")
    return spec


def _trial(session, spec, metrics, route, tag="t"):
    return repo.add_trial(
        session, spec_id=spec.id, config_hash=tag, code_sha="x", params={},
        status=TrialStatus.OK, started_at=datetime.now(UTC), metrics=metrics, route=route,
        route_reason="r", source=TrialSource.IMPORTED)


@pytest.fixture()
def registry(session):
    alpha = _idea(session, "alpha")
    # alpha: an early run that held, then a later one that failed -- the latest counts.
    _trial(session, alpha, LUCKY, TrialRoute.NEEDS_FORWARD, "a1")
    _trial(session, alpha, BAD, TrialRoute.REJECT, "a2")
    beta = _idea(session, "beta")
    _trial(session, beta, LUCKY, TrialRoute.PAPER, "b1")
    beta_hl = _idea(session, "beta-hl", source="hyperliquid", parent="beta")
    _trial(session, beta_hl, ONE_YEAR, TrialRoute.PAPER, "b2")
    gamma = _idea(session, "gamma", source="hyperliquid")  # a spec that runs on Hyperliquid itself
    _trial(session, gamma, ONE_YEAR, TrialRoute.NEEDS_FORWARD, "g1")
    # Left out: a run without a year view, a calibration noise book, a run with no route.
    delta = _idea(session, "delta")
    _trial(session, delta, {"ann_return_net": 0.2}, TrialRoute.PAPER, "d1")
    noise = _idea(session, "noise-007")
    _trial(session, noise, LUCKY, TrialRoute.REJECT, "n1")
    _trial(session, beta, LUCKY, None, "b3")
    session.commit()
    return session


def test_the_latest_judged_trial_of_each_idea_is_counted(registry):
    rows = year_signs.collect(registry, RULES)
    assert [r.idea_id for r in rows] == ["alpha", "beta", "beta-hl", "gamma"]
    alpha = rows[0]
    assert alpha.route == "reject" and alpha.signs["А"] is False  # BAD, not the earlier LUCKY
    assert alpha.stage == "binance-long" and alpha.years_counted == 4


def test_every_judged_trial_counts_with_all_trials(registry):
    rows = year_signs.collect(registry, RULES, all_trials=True)
    assert [(r.idea_id, r.route) for r in rows] == [
        ("alpha", "needs-forward"), ("alpha", "reject"), ("beta", "paper"),
        ("beta-hl", "paper"), ("gamma", "needs-forward")]


def test_a_trial_is_marked_with_its_stage(registry):
    stages = {r.idea_id: r.stage for r in year_signs.collect(registry, RULES)}
    assert stages == {"alpha": "binance-long", "beta": "binance-long",
                      "beta-hl": "hyperliquid-stage2", "gamma": "other"}


def test_the_totals_count_holds_fails_and_not_defined(registry):
    summary = year_signs.summarize(year_signs.collect(registry, RULES))
    assert summary["ideas"] == 4 and summary["trials"] == 4
    # alpha=BAD, beta=LUCKY, beta-hl=ONE_YEAR, gamma=ONE_YEAR
    assert summary["totals"]["А"] == {"holds": 1, "fails": 1, "not defined": 2}
    assert summary["totals"]["Б"] == {"holds": 3, "fails": 1, "not defined": 0}
    assert summary["totals"]["В"] == {"holds": 3, "fails": 1, "not defined": 0}
    assert summary["totals"]["Г"] == {"holds": 0, "fails": 2, "not defined": 2}


def test_the_cross_tables_split_each_sign_by_outcome_and_stage(registry):
    summary = year_signs.summarize(year_signs.collect(registry, RULES))
    outcome = summary["by_outcome"]["А"]
    assert outcome["rejected"] == {"holds": 0, "fails": 1, "not defined": 0}   # alpha
    assert outcome["passed"] == {"holds": 1, "fails": 0, "not defined": 1}     # beta, beta-hl
    assert outcome["other"] == {"holds": 0, "fails": 0, "not defined": 1}      # gamma
    stage = summary["by_stage"]["Г"]
    assert stage["binance-long"] == {"holds": 0, "fails": 2, "not defined": 0}
    assert stage["hyperliquid-stage2"] == {"holds": 0, "fails": 0, "not defined": 1}
    every = year_signs.summarize(year_signs.collect(registry, RULES, all_trials=True))
    assert every["trials"] == 5 and every["ideas"] == 4
    assert every["totals"]["А"] == {"holds": 2, "fails": 1, "not defined": 2}


def test_passed_means_no_strategy_rule_failed():
    for route in ("paper", "shelf", "needs-infrastructure"):
        row = year_signs.SignRow("i", "bench", 1, "other", route, 1)
        assert row.outcome == "passed"
    assert year_signs.SignRow("i", "bench", 1, "other", "reject", 1).outcome == "rejected"
    for route in ("needs-forward", "needs-more-data", "not-evaluable", "error"):
        assert year_signs.SignRow("i", "bench", 1, "other", route, 1).outcome == "other"


def test_the_morning_report_lines_give_totals_and_what_they_rest_on(registry):
    lines = year_signs.report_lines(registry, RULES)
    text = "\n".join(lines)
    assert "4 идей" in text and "всех таких прогонов в реестре 5" in text
    assert "- А (без лучшего года не меньше 4% в год): держится 1, не держится 1, " \
           "не определено 2" in text
    assert "qlab year-signs" in text
    assert "ничего не решают" in text


def test_the_morning_report_has_no_section_without_a_judged_run(session):
    assert year_signs.report_lines(session, RULES) == []


# ---- the command ---------------------------------------------------------------------------


@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    """A file registry holding the same rows, for the command."""
    path = tmp_path / "qlab.db"
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    alpha = _idea(s, "alpha")
    _trial(s, alpha, LUCKY, TrialRoute.NEEDS_FORWARD, "a1")
    _trial(s, alpha, BAD, TrialRoute.REJECT, "a2")
    beta = _idea(s, "beta")
    _trial(s, beta, LUCKY, TrialRoute.PAPER, "b1")
    s.commit()
    s.close()
    monkeypatch.setenv("QLAB_DB", str(path))
    db_module._engine = None
    db_module._SessionLocal = None
    yield path
    db_module._engine = None
    db_module._SessionLocal = None


def test_the_command_prints_a_row_per_idea_and_the_totals(cli_db):
    result = CliRunner().invoke(app, ["year-signs", "--stand", "--rules", "2026-10-28.2"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "rules: 2026-10-28.2; latest judged trial of each idea" in out
    alpha = next(line for line in out.splitlines() if line.startswith("alpha"))
    assert "reject" in alpha and "binance-long" in alpha
    assert alpha.split().count("fails") == 4
    beta = next(line for line in out.splitlines() if line.startswith("beta"))
    assert beta.split().count("holds") == 3 and beta.split().count("fails") == 1
    assert "totals over 2 trial(s) of 2 idea(s):" in out
    assert "against the run's outcome" in out and "against the stage" in out


def test_the_command_with_all_trials_counts_every_trial(cli_db):
    result = CliRunner().invoke(app, ["year-signs", "--stand", "--all-trials", "--json",
                                      "--rules", "2026-10-28.2"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["all_trials"] is True and body["rules_version"] == "2026-10-28.2"
    assert [r["trial_id"] for r in body["rows"]] == [1, 2, 3]
    assert body["rows"][0]["signs"] == {"А": "holds", "Б": "holds", "В": "holds", "Г": "fails"}
    assert body["rows"][1]["outcome"] == "rejected"
    assert body["summary"]["trials"] == 3 and body["summary"]["ideas"] == 2
    latest = json.loads(CliRunner().invoke(
        app, ["year-signs", "--stand", "--json", "--rules", "2026-10-28.2"]).output)
    assert [r["trial_id"] for r in latest["rows"]] == [2, 3]


def test_the_command_on_an_empty_registry_says_so(tmp_path, monkeypatch):
    path = tmp_path / "empty.db"
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    monkeypatch.setenv("QLAB_DB", str(path))
    db_module._engine = None
    db_module._SessionLocal = None
    try:
        result = CliRunner().invoke(app, ["year-signs"])
    finally:
        db_module._engine = None
        db_module._SessionLocal = None
    assert result.exit_code == 0
    assert "no judged trial with a year view" in result.output


def test_the_night_reports_the_signs_in_a_section_of_its_own(cli_db):
    from qlab import night

    lines = night._year_signs_lines()
    assert lines[0] == "## Четыре признака по годам: статистика"
    text = "\n".join(lines)
    assert "2 идей" in text and "qlab year-signs" in text
    assert "- Г (лучший год даёт не больше половины прибыли): держится 0, не держится 2, " \
           "не определено 0" in text


def test_the_night_has_no_signs_section_when_nothing_has_a_year_view(tmp_path, monkeypatch):
    from qlab import night

    path = tmp_path / "empty.db"
    Base.metadata.create_all(create_engine(f"sqlite+pysqlite:///{path}"))
    monkeypatch.setenv("QLAB_DB", str(path))
    db_module._engine = None
    db_module._SessionLocal = None
    try:
        assert night._year_signs_lines() == []
    finally:
        db_module._engine = None
        db_module._SessionLocal = None


def test_evaluate_prints_the_four_signs_beside_the_verdicts(cli_db, monkeypatch):
    from types import SimpleNamespace

    import qlab.cli as cli
    from qlab.rules.engine import VerdictRow
    from qlab.rules.schema import Stage

    row = VerdictRow(stage=Stage.EDGE, rule_id="net_edge_positive", rules_version="2026-10-28.2",
                     metric="ann_return_net", value=0.2, comparator=">", threshold=0.0,
                     passed=True)
    result = SimpleNamespace(
        error=None, trial_id=9, status_decision=None, metrics={"ann_return_net": 0.2, **ONE_YEAR},
        rules_result=SimpleNamespace(rows=[row]),
        routing=SimpleNamespace(route="paper", reason="passed", required_capital_usd=None))
    monkeypatch.setattr(cli, "load_spec", lambda path: SimpleNamespace(idea_id="x", title="X"))
    monkeypatch.setattr(cli, "evaluate_spec", lambda spec, **kw: result)

    out = CliRunner().invoke(
        app, ["evaluate", "spec.yaml", "--rules", "2026-10-28.2", "--noise-trials", "0"]).output

    assert "year signs (recorded, they decide nothing):" in out
    assert "  А: без лучшего года не меньше 4% в год — не определено " \
           "(засчитанных лет меньше двух)" in out
    assert "  Б: прибыльных лет не меньше двух третей — держится (100.0%)" in out
    assert "  В: без 1% лучших дней выше нуля — держится (10.0%)" in out
    assert out.index("year signs") < out.index("routing: paper")

    result.metrics = {"ann_return_net": 0.2}  # a run without a year view prints no signs
    assert "year signs" not in CliRunner().invoke(
        app, ["evaluate", "spec.yaml", "--rules", "2026-10-28.2", "--noise-trials", "0"]).output


def test_evaluate_runs_200_noise_books_by_default_and_calls_zero_a_sketch(cli_db, monkeypatch):
    """Owner, 2026-10-29 (docs/EVALUATION_PATHS.md Р19): a hand run is the night's measurement
    unless it says it is a sketch."""
    from types import SimpleNamespace

    import qlab.cli as cli
    from qlab.rules.engine import VerdictRow
    from qlab.rules.schema import Stage

    row = VerdictRow(stage=Stage.EDGE, rule_id="net_edge_positive", rules_version="2026-10-28.2",
                     metric="ann_return_net", value=0.2, comparator=">", threshold=0.0,
                     passed=True)
    result = SimpleNamespace(
        error=None, trial_id=9, status_decision=None, metrics={"ann_return_net": 0.2},
        rules_result=SimpleNamespace(rows=[row]),
        routing=SimpleNamespace(route="paper", reason="passed", required_capital_usd=None))
    asked = []
    monkeypatch.setattr(cli, "load_spec", lambda path: SimpleNamespace(idea_id="x", title="X"))
    monkeypatch.setattr(cli, "evaluate_spec", lambda spec, **kw: asked.append("sketch") or result)
    monkeypatch.setattr(cli, "evaluate_spec_with_shape_aware_bar",
                        lambda spec, **kw: asked.append(kw["n_trials"])
                        or SimpleNamespace(candidate=result))

    full = CliRunner().invoke(app, ["evaluate", "spec.yaml", "--rules", "2026-10-28.2"]).output
    sketch = CliRunner().invoke(
        app, ["evaluate", "spec.yaml", "--rules", "2026-10-28.2", "--noise-trials", "0"]).output

    assert asked == [200, "sketch"]
    assert "SKETCH" not in full
    assert "SKETCH, NOT A VERDICT" in sketch and "not a verdict" in sketch.splitlines()[-1]
