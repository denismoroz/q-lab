"""Stage 2's confirmation under rules 2026-10-29.2: the positivity rule reads the stage 2 total
on the strategy's own money (`cap_total_return`), the noise comparison and its power are the
ones on that money too (`noise_cap_return_percentile`, `noise_cap_power`), and a book whose base
of percent is undefined confirms nothing (docs/CAPITAL_BASE.md, owner 2026-10-29)."""

from __future__ import annotations

from qlab.pipeline import confirm
from qlab.pipeline.test_confirm import (
    ALERT,
    CAPITAL,
    _context,
    _median_total,
    _review,
    _stage1,
)
from qlab.rules.schema import (
    Comparator,
    FitPeriodUse,
    HoldoutShare,
    NoisePower,
    Rule,
    RuleSet,
    ShortForwardUse,
    Stage,
    Stage2Confirmation,
)

NOISE_CAP = Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="noise_cap_return_percentile",
                 comparator=Comparator.GE, threshold=0.99, fatal=True,
                 fit_period=FitPeriodUse.INFORMATIONAL, on_short_forward=ShortForwardUse.WAIT)
EDGE_CAP = Rule(id="net_edge_positive", stage=Stage.EDGE, metric="cap_ann_return",
                comparator=Comparator.GT, threshold=0.0, fatal=True)


def _ruleset() -> RuleSet:
    return RuleSet(
        version="2026-10-29.2", rules=[NOISE_CAP, EDGE_CAP, CAPITAL],
        holdout_share=HoldoutShare(share=1 / 3, rationale="t"),
        noise_power=NoisePower(
            good_return=0.07, min_power=0.80, return_metric="cap_ann_return",
            percentile_metric="noise_cap_return_percentile", power_metric="noise_cap_power",
            rationale="t"),
        stage2_confirmation=Stage2Confirmation(
            alert_percentile_from="night/decay.yaml", total_metric="cap_total_return",
            rationale="t"))


def _metrics(**over: float) -> dict[str, float]:
    base = {"cap_ann_return": 0.445, "cap_total_return": 0.50, "cap_reinvested": 1.0,
            "ann_return_net": 0.10, "total_return_net": 0.10, "n_periods": 100.0,
            "forward_days": 100.0, "min_capital_usd": 500.0,
            "noise_cap_return_percentile": 0.935, "noise_cap_power": 0.06,
            "noise_return_percentile": 0.50, "noise_power": 0.90}
    base.update(over)
    return base


def test_the_noise_bar_and_its_power_are_the_ones_on_the_strategys_money() -> None:
    history = _stage1()
    total = _median_total(history, 100)
    # The stand's power (0.90) would say "the bar decides"; the one the rules read (0.06) says
    # it cannot, so the consistency check replaces the comparison.
    ctx = _context(_metrics(cap_total_return=total), _ruleset())
    review, found = _review(ctx, history)
    assert found.status == "confirmed" and review.routing.route == "paper"
    # The other way round: a strong power on the strategy's money leaves the route alone.
    ctx = _context(_metrics(noise_cap_power=0.95, noise_power=0.01), _ruleset())
    assert _review(ctx, history) == (None, None)


def test_the_positivity_rule_reads_the_total_on_the_strategys_money() -> None:
    history = _stage1()
    total = _median_total(history, 100)
    # The stand's total is negative (a balance grown with set-aside cash), the money's is not.
    ctx = _context(_metrics(cap_total_return=total, total_return_net=-0.02), _ruleset())
    review, found = _review(ctx, history)
    assert found.status == "confirmed"
    rows = {r.rule_id: r for r in review.result.rows}
    assert rows[confirm.POSITIVE_RULE].metric == "cap_total_return"
    assert rows[confirm.POSITIVE_RULE].passed and rows[confirm.POSITIVE_RULE].value == total
    # And the other way: the stand's total is positive, the money's is not -> not confirmed.
    ctx = _context(_metrics(cap_total_return=-0.01, total_return_net=0.30), _ruleset())
    review, found = _review(ctx, history)
    assert found.status == "not-positive" and review.routing.route == "reject"
    assert "cap_total_return" in {r.metric for r in review.result.rows}


def test_the_segments_and_the_total_are_on_one_base() -> None:
    history = _stage1()
    ctx = _context(_metrics(cap_total_return=_median_total(history, 100)), _ruleset())
    _, found = _review(ctx, history)
    # stage 1's history carries no capital series in this fixture: the segments fall back to
    # the net returns, and the check says nothing false about the base
    assert found.on_capital is False


def test_a_book_without_a_base_of_percent_confirms_nothing() -> None:
    # Every `cap_` figure is missing: the other rules are unknown too, so something other than
    # the noise comparison keeps the run waiting and the check has nothing to replace.
    metrics = _metrics(cap_base_refused=1.0)
    del metrics["cap_total_return"], metrics["cap_ann_return"]
    assert _review(_context(metrics, _ruleset()), _stage1()) == (None, None)
    # Even with the other rules passing, a missing total is said to be the undefined base.
    metrics = _metrics(cap_base_refused=1.0)
    del metrics["cap_total_return"]
    review, found = _review(_context(metrics, _ruleset()), _stage1())
    assert found.status == "not-applied"
    assert "base of percent is undefined" in found.reason
    assert review.routing.route == "needs-forward"  # the run's own route stands
    assert ALERT == 7.0
