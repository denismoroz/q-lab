"""The topic rule: what it marks, why, and that it never reads claimed returns."""

from __future__ import annotations

import re

from qlab.publications.filter import FILTER_PATH, judge, load_rules

RULES = load_rules(FILTER_PATH)


def test_one_strong_term_is_enough() -> None:
    verdict = judge(RULES, "Perpetual Futures and Their Arbitrage on Decentralized Venues", "")
    assert verdict.candidate
    assert "perpetual futures" in verdict.matched["strong"]
    assert "perpetual futures" in verdict.why()


def test_crypto_alone_or_mechanism_alone_is_not_enough() -> None:
    assert not judge(RULES, "Bitcoin mining energy use", "We study electricity prices.").candidate
    assert not judge(RULES, "Momentum in equities", "Stocks with high past returns.").candidate
    # One mechanism term next to a crypto term is not enough either.
    assert not judge(RULES, "Momentum in bitcoin", "").candidate


def test_crypto_with_a_mechanism_is_a_candidate() -> None:
    verdict = judge(RULES, "Momentum and mean reversion in cryptocurrency returns", "")
    assert verdict.candidate
    assert set(verdict.matched) == {"crypto", "mechanism"}


def test_matching_ignores_case_hyphens_plurals_and_word_pieces() -> None:
    assert judge(RULES, "FUNDING-RATE dynamics", "").candidate
    assert judge(RULES, "Liquidation Cascades", "in Perps markets").candidate
    # "perp" must not fire inside "perpetuity".
    assert not judge(RULES, "Perpetuity bonds", "A perpetuity pays forever.").candidate


def test_a_term_found_inside_a_longer_matched_term_is_not_counted_twice() -> None:
    verdict = judge(RULES, "Perpetual futures", "")
    assert verdict.matched["strong"] == ["perpetual futures"]


def test_the_rule_has_no_term_about_claimed_returns() -> None:
    """The topic rule never reads what a paper claims to earn (the XSMOM lesson)."""
    claims = re.compile(r"sharpe|profit|alpha|outperform|annuali[sz]ed|excess return|abnormal"
                        r"|superior|beat the market|risk.adjusted", re.IGNORECASE)
    for group, terms in RULES.terms.items():
        for term in terms:
            assert not claims.search(term), (group, term)


def test_a_claimed_return_changes_nothing() -> None:
    plain = judge(RULES, "Funding rates on perpetual futures", "We describe the mechanism.")
    boasting = judge(RULES, "Funding rates on perpetual futures",
                     "We describe the mechanism and report a Sharpe ratio of 5 and large profits.")
    assert (plain.candidate, plain.score, plain.matched) == (
        boasting.candidate, boasting.score, boasting.matched)


def test_every_group_states_its_rationale_and_the_rules_are_versioned() -> None:
    import yaml

    raw = yaml.safe_load(FILTER_PATH.read_text(encoding="utf-8"))
    assert RULES.version == str(raw["version"]) and RULES.version
    for group in ("strong", "crypto", "mechanism"):
        assert raw["groups"][group]["rationale"].strip()
