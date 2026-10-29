"""Rules 2026-10-29.2: every rule that read a money metric of the stand reads the one on the
strategy's own money (docs/CAPITAL_BASE.md; owner, 2026-10-29: «Да, и правила»)."""

from __future__ import annotations

from qlab.pipeline.evaluate import (
    BASE_REFUSED_NOTE,
    RoutingDecision,
    _power_note,
    _powerless,
    decide_route,
)
from qlab.rules.engine import evaluate, read_informational
from qlab.rules.loader import DEFAULT_RULES_DIR, list_versions, load, load_latest

NEW, OLD = "2026-10-29.2", "2026-10-29.1"

#: Facts every strategy rule needs besides the money.
FACTS = {"venue_supported": 1.0, "data_forward_available": 1.0, "atomic_execution": 1.0,
         "min_capital_usd": 150.0, "point_in_time_universe": 1.0, "accrual_applied": 1.0}


def _by_id(ruleset):
    return {r.id: r for r in ruleset.rules}


def test_the_capital_ruleset_is_on_the_list_and_not_older_than_the_newest() -> None:
    from qlab.rules.schema import parse_version

    assert NEW in list_versions(DEFAULT_RULES_DIR)
    assert parse_version(load_latest(DEFAULT_RULES_DIR).version) >= parse_version(NEW)


def test_the_decisive_rules_read_the_strategys_money_and_nothing_else_changes() -> None:
    old, new = load(OLD, DEFAULT_RULES_DIR), load(NEW, DEFAULT_RULES_DIR)
    assert new.based_on == OLD
    before, after = _by_id(old), _by_id(new)
    assert set(before) == set(after) and new.retired == old.retired
    changed = {i for i in before if before[i] != after[i]}
    assert changed == {"net_edge_positive", "shape_aware_edge"}
    assert after["net_edge_positive"].metric == "cap_ann_return"
    assert after["shape_aware_edge"].metric == "noise_cap_return_percentile"
    for rule_id in changed:  # the line, the fatality and the way of waiting are the old ones
        a, b = before[rule_id], after[rule_id]
        assert (a.comparator, a.threshold, a.fatal, a.kind, a.fit_period, a.on_short_forward,
                a.near_margin) == (b.comparator, b.threshold, b.fatal, b.kind, b.fit_period,
                                   b.on_short_forward, b.near_margin)
    assert all(after[i].rationale for i in changed)
    assert new.holdout_share == old.holdout_share
    assert new.regime_coverage_days == old.regime_coverage_days


def test_the_year_readings_and_the_power_of_the_bar_read_the_same_money() -> None:
    new = load(NEW, DEFAULT_RULES_DIR)
    metrics = {r.id: r.metric for r in new.informational}
    assert metrics["ex_best_year_positive"] == "cap_ann_return_ex_best_year"
    assert metrics["ex_best_year_4pct"] == "cap_ann_return_ex_best_year"
    assert metrics["ex_best_1pct_positive"] == "cap_ann_return_ex_best_1pct"
    assert metrics["years_positive_share"] == "cap_years_positive_share"
    assert metrics["best_year_share_of_profit"] == "cap_best_year_share_of_profit"
    # the protection readings were on the strategy's money already
    assert metrics["protection_fall_cut"] == "prot_fall_ratio"
    power = new.noise_power
    assert (power.return_metric, power.percentile_metric, power.power_metric) == (
        "cap_ann_return", "noise_cap_return_percentile", "noise_cap_power")
    assert (power.good_return, power.min_power) == (0.07, 0.80)
    assert new.stage2_confirmation.total_metric == "cap_total_return"
    # An older ruleset keeps reading the stand's keys, as it did.
    assert load(OLD, DEFAULT_RULES_DIR).noise_power.power_metric == "noise_power"
    assert load(OLD, DEFAULT_RULES_DIR).stage2_confirmation.total_metric == "total_return_net"


def test_a_run_passes_on_the_money_whatever_the_stand_says_and_the_other_way_round() -> None:
    new = load(NEW, DEFAULT_RULES_DIR)
    # Bv2 on Binance: the stand's share of a balance four times the money reads ~2%, the money +8%.
    metrics = {**FACTS, "ann_return_net": -0.01, "cap_ann_return": 0.084,
               "noise_return_percentile": 0.50, "noise_cap_return_percentile": 0.995}
    result = evaluate(metrics, new)
    assert result.overall_passed
    rows = {r.rule_id: r for r in result.rows}
    assert rows["net_edge_positive"].metric == "cap_ann_return" and rows["net_edge_positive"].passed
    assert rows["shape_aware_edge"].metric == "noise_cap_return_percentile"
    # The stand's figures are no longer read: the same run with the stand rosy and the money not.
    metrics = {**FACTS, "ann_return_net": 0.20, "cap_ann_return": -0.01,
               "noise_return_percentile": 1.0, "noise_cap_return_percentile": 0.95}
    result = evaluate(metrics, new)
    assert set(result.failed_rule_ids) == {"net_edge_positive", "shape_aware_edge"}


def test_the_xsmom_case_of_the_owners_table_fails_the_bar_on_the_money() -> None:
    """docs/CAPITAL_BASE.md, section 7: the stand compounded XSMOM's profit and ranked 1.00 among
    noise that was compounded too; on the money both are summed and the place is 0.95."""
    metrics = {**FACTS, "ann_return_net": 0.362, "cap_ann_return": 0.413,
               "noise_return_percentile": 1.0, "noise_cap_return_percentile": 0.95}
    assert evaluate(metrics, load(OLD, DEFAULT_RULES_DIR)).overall_passed
    assert not evaluate(metrics, load(NEW, DEFAULT_RULES_DIR)).overall_passed


def test_readings_are_asked_of_the_money_series() -> None:
    new = load(NEW, DEFAULT_RULES_DIR)
    readings = {r.rule_id: r for r in read_informational(
        {"cap_ann_return_ex_best_year": 0.05, "cap_years_positive_share": 1.0,
         "ann_return_net_ex_best_year": -0.5}, new)}
    assert readings["ex_best_year_4pct"].holds is True       # the stand's -50% is not read
    assert readings["years_positive_share"].holds is True
    assert readings["best_year_share_of_profit"].holds is None


def test_the_power_of_the_bar_is_read_under_the_name_the_ruleset_gives_it() -> None:
    new, old = load(NEW, DEFAULT_RULES_DIR), load(OLD, DEFAULT_RULES_DIR)
    metrics = {"noise_power": 0.95, "noise_cap_power": 0.10}
    assert _powerless(new, metrics) and not _powerless(old, metrics)
    assert "only 10%" in _power_note(new, metrics) and _power_note(old, metrics) == ""
    assert "not measured" in _power_note(new, {"noise_power": 0.95})


def test_a_book_with_no_base_of_percent_is_not_judged_on_the_stand_instead() -> None:
    """The base of percent of a book with deposits is undefined (`cap_base_refused`): its rules
    are unknown, it routes needs-more-data and the reason says why -- it does not silently fall
    back to the stand's share of a balance that the deposits grow."""
    new = load(NEW, DEFAULT_RULES_DIR)
    metrics = {**FACTS, "ann_return_net": 0.30, "noise_return_percentile": 1.0,
               "cap_base_refused": 1.0}
    result = evaluate(metrics, new)
    assert not result.decisive and "cap_ann_return" in result.unknown_metrics
    routing = decide_route(result, metrics)
    assert routing.route == "needs-more-data"
    assert "cap_ann_return" in routing.reason and BASE_REFUSED_NOTE in routing.reason
    # The older rules read the stand and decide as they did; the note is for the new ones.
    old = evaluate(metrics, load(OLD, DEFAULT_RULES_DIR))
    assert decide_route(old, metrics) == RoutingDecision(
        route="paper", reason="passed every rule", required_capital_usd=150.0)


def test_the_restamp_treats_the_switch_as_a_change_of_what_decides() -> None:
    from qlab.restamp import rules_changed

    assert rules_changed(OLD, NEW) is True          # a rule reads another metric
    assert rules_changed("2026-10-28.2", OLD) is False   # readings only: no verdict moves
