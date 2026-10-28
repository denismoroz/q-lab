"""The year-by-year view's wording and wiring (docs/YEARS.md): the report line, the
informational readings of rules 2026-10-27.2, and the places that print or quote them."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.agents.assessor import build_brief
from qlab.agents.researcher import build_context
from qlab.pipeline import stage2
from qlab.registry import repo
from qlab.registry.models import (
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
from qlab.rules.engine import evaluate, read_informational
from qlab.rules.loader import DEFAULT_RULES_DIR, load
from qlab.rules.schema import Comparator, InformationalRule, RuleSet
from qlab.year_view import best_year, is_year_key, year_keys, year_rows, year_view, years_line

# XSMOM-like: a few huge years, then nothing.
METRICS = {
    "year_2020_return": 0.41, "year_2020_days": 300, "year_2020_max_dd": -0.2,
    "year_2021_return": 0.88, "year_2021_days": 365, "year_2021_max_dd": -0.3,
    "year_2021_sharpe": 1.9,
    "year_2022_return": -0.12, "year_2022_days": 365, "year_2022_max_dd": -0.25,
    "year_2023_return": 0.05, "year_2023_days": 365,
    "year_2024_return": 0.03, "year_2024_days": 366,
    "year_2025_return": 0.02, "year_2025_days": 365,
    "year_2026_return": 0.01, "year_2026_days": 282,
    "years_counted": 7, "years_positive": 5, "years_positive_share": 5 / 7,
    "ann_return_net_ex_best_year": 0.04, "best_year_share_of_profit": 0.6,
}


# ---- the line ----------------------------------------------------------------------------


def test_the_line_lists_years_marks_partial_ones_and_names_the_summaries() -> None:
    assert years_line(METRICS) == (
        "по годам: 2020 (300 дн.) +41%, 2021 +88%, 2022 −12%, 2023 +5%, 2024 +3%, "
        "2025 +2%, 2026 (282 дн.) +1%; без лучшего года (2021): +4% в год; "
        "прибыльных лет 5 из 7")


def test_small_returns_keep_a_decimal_and_the_minus_is_a_minus() -> None:
    line = years_line({"year_2021_return": -0.004, "year_2021_days": 365,
                       "year_2022_return": 0.0004, "year_2022_days": 365})
    assert line == "по годам: 2021 −0.4%, 2022 +0.0%"


def test_without_the_summaries_the_line_is_just_the_years() -> None:
    m = {k: v for k, v in METRICS.items() if k.startswith("year_20")}
    assert years_line(m) == (
        "по годам: 2020 (300 дн.) +41%, 2021 +88%, 2022 −12%, 2023 +5%, 2024 +3%, "
        "2025 +2%, 2026 (282 дн.) +1%")


@pytest.mark.parametrize("metrics", [
    None, {}, {"ann_return_net": 0.1},
    # one stretch of 90 days is not "years"
    {"year_2026_return": 0.02, "year_2026_days": 90},
])
def test_no_line_for_a_run_without_years_or_with_one_short_stretch(metrics) -> None:
    assert years_line(metrics) is None


def test_one_long_year_is_still_a_line() -> None:
    assert years_line({"year_2025_return": 0.1, "year_2025_days": 365,
                       "years_counted": 1, "years_positive": 1}).endswith(
        "прибыльных лет 1 из 1; правила по годам (без лучшего года, доля лучшего года): "
        "не определено — засчитанных лет меньше двух")


# ---- rows and keys -----------------------------------------------------------------------


def test_rows_are_read_back_in_year_order_and_keys_are_recognised() -> None:
    rows = year_rows({**METRICS, "year_2019_return": float("nan"), "ann_return_net": 0.3})
    assert [r.year for r in rows] == [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    assert rows[1].sharpe == 1.9 and rows[0].sharpe is None
    assert rows[0].partial and not rows[1].partial and rows[0].counted
    assert best_year(rows) == 2021
    assert is_year_key("year_2021_max_dd") and is_year_key("years_positive_share")
    assert not is_year_key("ann_return_net") and not is_year_key("year_to_date")
    assert "ann_return_net" not in year_keys({**METRICS, "ann_return_net": 1})


# ---- the informational rules --------------------------------------------------------------


def test_rules_2026_10_27_2_add_readings_and_change_no_rule() -> None:
    before = load("2026-10-27.1", DEFAULT_RULES_DIR)
    after = load("2026-10-27.2", DEFAULT_RULES_DIR)
    assert after.based_on == "2026-10-27.1"
    assert after.rules == before.rules and after.retired == before.retired
    assert after.stage2_confirmation == before.stage2_confirmation
    assert after.holdout_share == before.holdout_share and after.noise_power == before.noise_power
    assert [r.id for r in after.informational] == [
        "ex_best_year_positive", "ex_best_1pct_positive", "years_positive_share",
        "best_year_share_of_profit"]
    assert all(r.rationale.strip() for r in after.informational)
    assert before.informational == []


def test_readings_say_holds_or_not_but_decide_nothing() -> None:
    ruleset = load("2026-10-27.2", DEFAULT_RULES_DIR)
    readings = {r.rule_id: r for r in read_informational(
        {"ann_return_net_ex_best_year": -0.05, "ann_return_net_ex_best_1pct": float("nan"),
         "years_positive_share": 0.5}, ruleset)}
    assert readings["ex_best_year_positive"].holds is False
    assert readings["ex_best_1pct_positive"].value is None      # NaN is unknown
    assert readings["ex_best_1pct_positive"].holds is None
    assert readings["years_positive_share"].holds is None        # no line drawn yet
    assert readings["years_positive_share"].value == 0.5
    assert readings["best_year_share_of_profit"].value is None   # absent

    # The engine does not see them: the same metrics give the same verdict either way.
    metrics = {"ann_return_net": 0.1, "ann_return_net_ex_best_year": -0.5}
    bare = RuleSet(version=ruleset.version, rules=list(ruleset.rules))
    assert evaluate(metrics, ruleset).rows == evaluate(metrics, bare).rows
    assert not any(r.metric == "ann_return_net_ex_best_year"
                   for r in evaluate(metrics, ruleset).rows)


def test_an_informational_rule_needs_comparator_and_threshold_together() -> None:
    with pytest.raises(ValueError, match="go together"):
        InformationalRule(id="x", metric="m", comparator=Comparator.GT, rationale="r")
    with pytest.raises(ValueError, match="duplicate informational"):
        RuleSet(version="2026-10-27.9", informational=[
            InformationalRule(id="x", metric="m", rationale="r"),
            InformationalRule(id="x", metric="n", rationale="r")])


def test_the_view_for_the_console_has_rows_summary_and_readings() -> None:
    ruleset = load("2026-10-27.2", DEFAULT_RULES_DIR)
    view = year_view(METRICS, ruleset)
    assert view["best_year"] == 2021
    assert view["years"][0] == {"year": 2020, "return": 0.41, "days": 300, "max_dd": -0.2,
                                "sharpe": None, "counted": True, "partial": True}
    assert view["summary"]["years_positive"] == 5
    assert view["line"] == years_line(METRICS)
    assert {r["id"]: r["holds"] for r in view["readings"]}["ex_best_year_positive"] is True
    assert year_view({"ann_return_net": 0.1}, ruleset) is None
    assert year_view(METRICS)["readings"] == []


# ---- the places that print or quote it -----------------------------------------------------


def test_stage2_report_prints_the_years_under_the_stage_line() -> None:
    result = {"status": "evaluated", "route": "paper", "trial_id": 7,
              "data_start": "2025-01-30", "data_end": "2026-10-24",
              "metrics": {**METRICS, "ann_return_net": 0.4}}
    lines = stage2.report_lines({"route": "paper"}, result)
    at = next(i for i, ln in enumerate(lines) if "прогон 7" in ln)
    assert lines[at + 1] == years_line(METRICS)

    old = {**result, "metrics": {"ann_return_net": 0.4}}  # a run from before the view
    assert not any(ln.startswith("по годам") for ln in stage2.report_lines({"route": "paper"}, old))


def test_the_night_report_prints_the_years_under_the_watch_table(tmp_path, monkeypatch) -> None:
    from datetime import date

    from qlab import night

    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "WATCHLIST", tmp_path / "watch.yaml")
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    (tmp_path / "watch.yaml").write_text("specs: []\n")
    state = {"results": [
        {"kind": "watch", "name": "with-years", "route": "paper", "previous_route": "paper",
         "metrics": {"ann_return_net": 0.3, "sharpe_net": 1.0, **METRICS}},
        {"kind": "watch", "name": "old-run", "route": "paper", "previous_route": "paper",
         "metrics": {"ann_return_net": 0.3, "sharpe_net": 1.0}},
    ]}

    report = night.write_report(date(2026, 10, 27), state).read_text(encoding="utf-8")

    assert f"- `with-years`: {years_line(METRICS)}" in report
    assert "`old-run`: по годам" not in report


def test_a_generated_idea_shows_its_stage_1_years_above_its_second_stage() -> None:
    from qlab.agents.generator import GeneratedResult, second_stage_lines

    result = GeneratedResult(idea_id="g", spec_path="g.yaml", mechanism="m", metrics=METRICS,
                             route="paper", stage2={
        "status": "evaluated", "route": "paper", "trial_id": 7, "metrics": {"ann_return_net": 0.1},
        "data_start": "2025-01-30", "data_end": "2026-10-24"})
    lines = second_stage_lines(result)
    assert lines[0] == years_line(METRICS) and "прогон 7" in lines[1]
    assert second_stage_lines(GeneratedResult(idea_id="g", spec_path="g.yaml", mechanism="m",
                                              route="paper")) == []


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    repo.upsert_idea(s, id="a", title="a", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    spec = Spec(idea_id="a", version=1, code_ref="qlab.strategies.live.bv2:LiveBv2",
                params={}, data_requirements={}, rebalance="1d", costs_model={})
    s.add(spec)
    s.flush()
    s.add(Trial(spec_id=spec.id, config_hash="h", code_sha="x", params={},
                started_at=datetime.now(UTC), metrics={"ann_return_net": 0.3, **METRICS},
                status=TrialStatus.OK, route=TrialRoute.NEEDS_FORWARD, route_reason="r",
                source=TrialSource.IMPORTED))
    s.flush()
    yield s
    s.close()


def test_the_assessor_gets_the_year_line_in_its_brief(session) -> None:
    trial = session.query(Trial).one()
    brief = build_brief(session, trial.id)
    assert brief.context["by calendar year"] == years_line(METRICS)
    # And the numbers stay checkable facts.
    assert brief.facts["trial.year_2021_return"] == 0.88


def test_the_researchers_evidence_has_the_line_instead_of_the_per_year_keys(session) -> None:
    text = build_context("a", ["a"], session)
    assert f"by calendar year (judged part): {years_line(METRICS)}" in text
    assert "year_2021_return" not in text and "year_2020_days" not in text
    assert "- ann_return_net_ex_best_year:" in text and "- ann_return_net:" in text
