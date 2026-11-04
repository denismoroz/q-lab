"""Tests for `qlab.money_fit`: the money step judged on the strategy's own money, in the «fewer
positions» form (rules 2026-11-04.2).

Owner, 2026-11-04: «опять вижу нужно 37000$ - мы же решили вчера проверять стратегию по доступному
бюджету а не на таких космических суммах»."""

from __future__ import annotations

from datetime import date

import pytest

from qlab import ladder as lad
from qlab import money_fit as mf
from qlab.ladder import Ledgers, Outcome, Reason
from qlab.pipeline import test_evaluate as te
from qlab.pipeline.evaluate import evaluate_spec
from qlab.registry.models import IdeaStatus, TrialRoute
from qlab.rules.loader import load, load_latest
from qlab.rules.schema import MoneyFit, RuleSet
from qlab.test_ladder import idea, run
from qlab.test_money_ladder import _toy_ladder

# The stand's fixtures (an in-memory registry with the toy idea, no network), used by name.
engine, session, _idea_row, _no_network = te.engine, te.session, te._idea_row, te._no_network
_month = te._month

RULE = MoneyFit(min_return_share=0.70, min_correlation=0.90, rationale="test")


def _m(**kw) -> dict[str, float]:
    base = {"own_usd": 3000.0, "reference_ann_return": 0.0365, "fewer_ann_return": 0.0412,
            "fewer_correlation": 0.997, "fewer_avg_positions": 17.0}
    base.update(kw)
    return {mf.PREFIX + k: v for k, v in base.items() if v is not None}


# --------------------------------------------------------------------------
# The rule (pure)


def test_the_owners_example_fits_and_reads_in_plain_words() -> None:
    j = mf.judge(_m(), RULE, 3000.0)
    assert j.state == mf.PASSED
    assert j.detail_ru == ("на своих $3 000 при меньшем числе позиций: доход +4.1% в год против "
                           "+3.6% без ограничений, день в день совпадает на 0.997 — помещается")


def test_the_thresholds_are_inclusive_and_each_one_alone_fails() -> None:
    at = mf.judge(_m(reference_ann_return=0.10, fewer_ann_return=0.07, fewer_correlation=0.90),
                  RULE, 3000.0)
    assert at.state == mf.PASSED  # exactly 70% and exactly 0.90
    low_return = mf.judge(_m(reference_ann_return=0.10, fewer_ann_return=0.0699), RULE, 3000.0)
    assert low_return.state == mf.FAILED and low_return.return_ok is False
    assert "(нужно не меньше 70% от него: +7.0%)" in low_return.detail_ru
    assert low_return.detail_ru.endswith("— не помещается")
    low_corr = mf.judge(_m(fewer_correlation=0.899), RULE, 3000.0)
    assert low_corr.state == mf.FAILED and low_corr.return_ok and not low_corr.correlation_ok
    assert "(нужно не меньше 0.90)" in low_corr.detail_ru
    assert "нужно не меньше 70%" not in low_corr.detail_ru


def test_the_thresholds_are_read_from_the_rule_not_from_code() -> None:
    strict = MoneyFit(min_return_share=0.95, min_correlation=0.999, rationale="test")
    assert mf.judge(_m(), strict, 3000.0).state == mf.FAILED
    loose = MoneyFit(min_return_share=0.5, min_correlation=0.5, rationale="test")
    assert mf.judge(_m(fewer_correlation=0.6, fewer_ann_return=0.02), loose, 3000.0).state \
        == mf.PASSED


def test_a_book_that_does_not_earn_unconstrained_is_not_decided_by_the_money_step() -> None:
    for ref in (0.0, -0.02):
        j = mf.judge(_m(reference_ann_return=ref, fewer_ann_return=-0.01), RULE, 3000.0)
        assert j.state == mf.NO_EDGE
        assert j.detail_ru.startswith("не решает: без ограничений стратегия не зарабатывает")


def test_what_was_not_measured_is_undecided_with_the_reason_never_a_pass() -> None:
    none = mf.judge({}, RULE, 3000.0)
    assert none.state == mf.NOT_MEASURED and "до пересчёта" in none.why_ru
    coded = mf.judge(mf.unmeasured(mf.Unmeasured.BOOK_WITHOUT_MONEY, 257.0), RULE, 257.0)
    assert coded.state == mf.NOT_MEASURED
    assert coded.detail_ru.startswith("на своих $257 при меньшем числе позиций не измерено: книга")
    moved = mf.judge(_m(), RULE, 5000.0)
    assert moved.state == mf.NOT_MEASURED and "деньги стратегии теперь $5,000" in moved.why_ru
    flat = mf.judge(_m(fewer_correlation=None), RULE, 3000.0)
    assert flat.state == mf.NOT_MEASURED and "совпадение по дням" in flat.why_ru


def test_no_position_fitting_the_money_fails() -> None:
    j = mf.judge(_m(fewer_avg_positions=0.0, fewer_ann_return=0.0, fewer_correlation=None), RULE,
                 3000.0)
    assert j.state == mf.FAILED and "не помещается ни одной позиции" in j.detail_ru


def test_the_strict_minimum_is_a_small_line_naming_the_position_that_sets_it() -> None:
    line = mf.strict_ru(37_591.0, _toy_ladder())
    assert line.startswith("чтобы держать все позиции как есть, нужно $37 591 (задаёт H на Binance")
    assert "2026-06-11" in line
    assert mf.strict_ru(37_591.0) == ("чтобы держать все позиции как есть, нужно $37 591; это "
                                      "показание, ступень решает строка выше")
    assert mf.strict_ru(None) is None
    # a reading of another number does not name the position
    assert "задаёт" not in mf.strict_ru(7_973.0, _toy_ladder())


# --------------------------------------------------------------------------
# The rules file


def test_the_new_version_carries_the_owners_numbers_and_inherits_the_rest() -> None:
    new, old = load("2026-11-04.2"), load("2026-11-04.1")
    assert new.money_fit is not None and old.money_fit is None
    assert (new.money_fit.min_return_share, new.money_fit.min_correlation) == (0.70, 0.90)
    # one rule added (the earning test decides), one retired (the $120,000 horizon)
    assert {r.id for r in new.rules} == ({r.id for r in old.rules} - {"capital_fit"}
                                         | {"earning_distinguishable"})
    assert "capital_fit" in {r.id for r in new.retired}
    assert "capital_fit" in {r.id for r in old.rules}  # older versions keep it
    assert new.splits_by_data and new.noise_bar == old.noise_bar
    assert load_latest().money_fit is not None


# --------------------------------------------------------------------------
# The ladder: the new rule for new runs, the old words for old ones


def _fit_run(session, idea_id: str, rules: str, **metrics) -> int:
    return run(session, idea_id, TrialRoute.PAPER, "hyperliquid", rules=rules,
               metrics={"min_capital_usd": 37_591.0, **_m(**metrics)})


def test_step_four_under_the_new_version_reads_fewer_positions_on_own_money(session) -> None:
    idea(session, "new", IdeaStatus.VALIDATED)
    _fit_run(session, "new", "2026-11-04.2")
    step = lad.build_ladder(session, "new", Ledgers(money={"new": 3000.0})).steps[3]
    hl = {m.venue: m for m in step.money}["hyperliquid"]
    assert step.outcome == Outcome.PASSED and hl.rule == "fewer_own"
    assert hl.detail.endswith("день в день совпадает на 0.997 — помещается")
    assert hl.strict_ru.startswith("чтобы держать все позиции как есть, нужно $37 591")
    body = hl.to_dict()
    assert body["rule"] == "fewer_own" and body["strict_ru"] == hl.strict_ru


def test_step_four_under_an_old_version_keeps_the_strict_words(session) -> None:
    idea(session, "old", IdeaStatus.VALIDATED)
    _fit_run(session, "old", "2026-11-04.1")
    step = lad.build_ladder(session, "old", Ledgers(money={"old": 3000.0})).steps[3]
    hl = {m.venue: m for m in step.money}["hyperliquid"]
    assert step.outcome == Outcome.FAILED and hl.rule == "strict" and hl.strict_ru is None
    assert hl.detail == "нужно $37,591, своих $3,000: не помещается"
    assert step.outcome_ru == "не пройден"


def test_step_four_unmeasured_and_no_edge_have_their_own_words(session) -> None:
    idea(session, "unmeasured", IdeaStatus.VALIDATED)
    run(session, "unmeasured", TrialRoute.PAPER, "hyperliquid", rules="2026-11-04.2",
        metrics={"min_capital_usd": 37_591.0})
    step = lad.build_ladder(session, "unmeasured", Ledgers(money={"unmeasured": 3000.0})).steps[3]
    assert step.outcome == Outcome.UNDECIDED and step.outcome_ru == "не решён"
    assert "не измерено" in step.detail and "рано судить" not in step.outcome_ru

    idea(session, "flat", IdeaStatus.BENCH)
    run(session, "flat", TrialRoute.REJECT, "hyperliquid", rules="2026-11-04.2",
        metrics=_m(reference_ann_return=-0.01))
    step = lad.build_ladder(session, "flat", Ledgers(money={"flat": 3000.0})).steps[3]
    hl = {m.venue: m for m in step.money}["hyperliquid"]
    assert hl.outcome == Outcome.NOT_RUN and hl.reason == Reason.NO_EDGE_TO_KEEP
    assert hl.outcome_ru == "не решает"
    assert step.reason == Reason.NO_EDGE_TO_KEEP
    assert step.short_ru() == "не решает: без ограничений стратегия не зарабатывает"


# --------------------------------------------------------------------------
# At evaluation time


def _ruleset(money: bool) -> RuleSet:
    extra = {"money_fit": RULE} if money else {}
    return te._by_data_ruleset([te.CAPITAL_FIT_GENEROUS], **extra)


def _evaluate(session, tmp_path, ruleset, **spec):
    if "strategy_capital_usd" in spec:
        spec["sources"] = {k: "owner 2026-01-01: test fixture"
                           for k in ("weight", "costs", "min_leg_notional", "strategy_capital_usd")}
    out = evaluate_spec(_month(session, tmp_path, **spec), session=session, ruleset=ruleset,
                        check_lookahead=False, check_sources=False)
    assert out.error is None, out.error
    return out.metrics


def test_every_run_under_the_new_version_measures_its_own_money(session, tmp_path) -> None:
    m = _evaluate(session, tmp_path, _ruleset(True), strategy_capital_usd=3000.0)
    # 0.4 of $3 000 is $1 200 a leg, far above the $12 minimum: the book is the unconstrained one
    assert m[mf.PREFIX + "own_usd"] == 3000.0
    assert m[mf.PREFIX + "fewer_ann_return"] == pytest.approx(m[mf.PREFIX + "reference_ann_return"])
    assert m[mf.PREFIX + "reference_ann_return"] == pytest.approx(m["cap_ann_return"])
    assert m[mf.PREFIX + "fewer_correlation"] == pytest.approx(1.0)
    assert mf.judge(m, RULE, 3000.0).state == mf.PASSED
    # on $20 a leg is $8 < $12: nothing fits
    tiny = _evaluate(session, tmp_path, _ruleset(True), strategy_capital_usd=20.0)
    assert tiny[mf.PREFIX + "fewer_avg_positions"] == 0.0
    assert mf.judge(tiny, RULE, 20.0).state == mf.FAILED


def test_a_run_that_cannot_be_measured_says_why(session, tmp_path) -> None:
    m = _evaluate(session, tmp_path, _ruleset(True))  # the spec names no money
    assert m[mf.PREFIX + "unmeasured"] == float(mf.Unmeasured.NO_MONEY)
    book = _evaluate(session, tmp_path, _ruleset(True), strategy_capital_usd=257.0,
                     code_ref="qlab.pipeline.test_evaluate:OrderedToy",
                     params={"weight": 0.4, "order": 0.05})
    assert book[mf.PREFIX + "unmeasured"] == float(mf.Unmeasured.BOOK_WITHOUT_MONEY)


def test_an_old_version_measures_nothing_new(session, tmp_path) -> None:
    m = _evaluate(session, tmp_path, _ruleset(False), strategy_capital_usd=3000.0)
    assert not any(k.startswith(mf.PREFIX) for k in m)


def test_the_night_money_ladder_does_not_erase_the_measurement() -> None:
    from types import SimpleNamespace

    from qlab import money_ladder as ml

    trial = SimpleNamespace(metrics={"ml_old": 1.0, **_m()})
    registry = SimpleNamespace(get=lambda _cls, _id: trial)
    ml.write_metrics(registry, _toy_ladder())
    assert all(k in trial.metrics for k in _m()) and "ml_old" not in trial.metrics


def test_the_measurement_is_not_asked_of_noise_books() -> None:
    import inspect

    from qlab.calibration import run as calibration_run

    assert "money_fit=False" in inspect.getsource(calibration_run._evaluate_noise)
    assert date(2026, 11, 4)  # the owner's day
