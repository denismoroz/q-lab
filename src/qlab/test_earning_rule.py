"""«Заработок отличим от нуля» decides from rules 2026-11-04.2, and its failure is «рано судить».

Owner, 2026-11-04, «Да, решающим», after bv3-research-fast-release passed with two of three
forward years negative: «странно что … прошла 2 года в минусе 25 и 26. все вытягивает 24 год»."""

from __future__ import annotations

from datetime import date

import pytest

from qlab import earning_test
from qlab import ladder as lad
from qlab.ladder import Ledgers, Outcome
from qlab.pipeline import test_evaluate as te
from qlab.pipeline.evaluate import EARNING_WAIT, decide_fit_forward_route, evaluate_spec
from qlab.pipeline.stage2 import combine_venues
from qlab.registry.models import IdeaStatus, TrialRoute
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.loader import load
from qlab.rules.schema import Comparator, Rule, RuleSet, SplitByData, Stage
from qlab.test_ladder import idea, run

engine, session, _idea_row, _no_network = te.engine, te.session, te._idea_row, te._no_network

NET_EDGE = Rule(id="net_edge_positive", stage=Stage.EDGE, metric="cap_ann_return",
                comparator=Comparator.GT, threshold=0.0, fatal=True)


def _earning(threshold: float = 0.05) -> Rule:
    return Rule(id="earning_distinguishable", stage=Stage.EDGE, metric="cap_mean_positive_p",
                comparator=Comparator.LE, threshold=threshold, fit_period="informational",
                on_short_forward="wait", on_fail="wait")


def _rules(*rules: Rule) -> RuleSet:
    return RuleSet(version="2026-11-04.2", rules=list(rules), holdout_share=te.HOLDOUT,
                   split_by_data=SplitByData(rationale="test"))


def _route(metrics: dict, ruleset: RuleSet | None = None):
    ruleset = ruleset or _rules(NET_EDGE, _earning())
    metrics = {"forward_days": 300.0, "min_capital_usd": 10.0, **metrics}
    return decide_fit_forward_route(ruleset=ruleset, selection=None,
                                    forward=evaluate_rules(metrics, ruleset), metrics=metrics,
                                    params_fixed_at=None)


def test_an_earning_told_from_zero_passes_and_one_that_is_not_waits() -> None:
    assert _route({"cap_ann_return": 0.05, "cap_mean_positive_p": 0.05}).route == "paper"
    held = _route({"cap_ann_return": 0.05, "cap_mean_positive_p": 0.205})
    assert held.route == "needs-forward" and held.reason.startswith(EARNING_WAIT)
    assert "cap_mean_positive_p = 0.205" in held.reason


def test_another_failed_rule_still_rejects() -> None:
    both = _route({"cap_ann_return": -0.01, "cap_mean_positive_p": 0.6})
    assert both.route == "reject" and "net_edge_positive" in both.reason


def test_a_run_where_the_earning_cannot_be_computed_is_not_decided_never_a_pass() -> None:
    unknown = _route({"cap_ann_return": 0.05})
    assert unknown.route == "needs-more-data" and "cap_mean_positive_p" in unknown.reason


def test_without_the_rule_the_number_decides_nothing() -> None:
    assert _route({"cap_ann_return": 0.05, "cap_mean_positive_p": 0.9},
                  _rules(NET_EDGE)).route == "paper"


def test_the_rules_file_draws_the_owners_line() -> None:
    new, old = load("2026-11-04.2"), load("2026-11-04.1")
    rule = next(r for r in new.rules if r.id == "earning_distinguishable")
    assert (rule.metric, rule.comparator, rule.threshold) == ("cap_mean_positive_p",
                                                              Comparator.LE, 0.05)
    assert rule.on_fail == "wait" and not rule.fatal
    assert all(r.on_fail == "reject" for r in old.rules)
    assert "earning_distinguishable" not in {r.id for r in old.rules}


def test_a_held_run_is_marked_at_evaluation(session, tmp_path, monkeypatch) -> None:
    ruleset = te._by_data_ruleset([te.CAPITAL_FIT_GENEROUS, _earning(-1.0)])
    # The toy's forward third is ten days: too few for the reading, so the run is not decided.
    short = evaluate_spec(te._month(session, tmp_path), session=session, ruleset=ruleset,
                          check_lookahead=False, check_sources=False)
    assert short.routing.route == "needs-more-data"
    assert "cap_mean_positive_p" in short.routing.reason
    monkeypatch.setattr(earning_test, "MIN_DAYS", 5)
    # A line no run can meet (p <= -1) holds every run: the route waits and the trial says why.
    out = evaluate_spec(te._month(session, tmp_path), session=session, ruleset=ruleset,
                        check_lookahead=False, check_sources=False)
    assert out.error is None, out.error
    assert out.routing.route == "needs-forward", out.routing.reason
    assert out.metrics["earning_wait"] == 1.0
    ok = evaluate_spec(te._month(session, tmp_path), session=session,
                       ruleset=te._by_data_ruleset([te.CAPITAL_FIT_GENEROUS, _earning(1.0)]),
                       check_lookahead=False, check_sources=False)
    assert ok.routing.route == "paper" and "earning_wait" not in ok.metrics


def test_a_waiting_venue_keeps_the_idea_open() -> None:
    # qlab.pipeline.venue_status.settle combines the venues' routes: a wait is pending, never a
    # reject, even beside a reject on the other venue.
    assert combine_venues("reject", "needs-forward").route == "needs-forward"
    assert combine_venues("needs-forward", None).route == "needs-forward"


def test_the_ladder_and_the_report_say_it_in_plain_words(session) -> None:
    idea(session, "held", IdeaStatus.BENCH)
    run(session, "held", TrialRoute.NEEDS_FORWARD, "hyperliquid", rules="2026-11-04.2",
        metrics={"earning_wait": 1.0, "cap_mean_positive_p": 0.205, "cap_ann_ci_low": -0.05,
                 "cap_ann_ci_high": 0.12, "forward_days": 300.0})
    found = lad.build_ladder(session, "held", Ledgers())
    step3 = found.steps[2]
    assert step3.outcome == Outcome.UNDECIDED
    assert step3.detail.startswith("рано судить: заработок пока не отличим от нуля")
    assert "p = 0.205" in step3.detail and "нужно p не больше 0.05" in step3.detail
    reading = next(r for r in found.steps[4].readings if r.name == "earning_test")
    assert "(решает: нужно p не больше 0.05" in reading.text
    lines = lad.report_lines(session, Ledgers())
    assert any("заработок пока не отличим от нуля" in line and "held (Hyperliquid, p = 0.205)"
               in line for line in lines)


def test_an_old_run_keeps_its_reading_words() -> None:
    m = {"cap_mean_positive_p": 0.205}
    assert earning_test.line_ru(m).endswith("(ничего не решает)")
    assert "решает: нужно p не больше 0.05" in earning_test.line_ru(m, line=0.05)
    assert earning_test.line_ru({"cap_mean_positive_p": 0.01}, line=0.05).startswith(
        "заработок отличим от нуля")
    assert date(2026, 11, 4) and pytest
