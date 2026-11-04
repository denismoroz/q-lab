"""The agents are told how a spec that reads the coin's own spot closes is declared
(docs/SPOT_PRICES.md). Three ideas were once refused because the agent listed the perp-marked
spot as an unexpressed mechanism instead of declaring `data.spot_prices: own`."""

from __future__ import annotations

from qlab.agents import implementer, researcher, reviewer


def test_the_implementer_is_told_to_declare_own_and_not_to_list_the_perp_mark() -> None:
    text = implementer.INSTRUCTIONS
    assert "data.spot_prices: own" in text
    assert "never list" in text and "unexpressed_mechanisms" in text


def test_the_researcher_is_told_the_same() -> None:
    text = researcher.INSTRUCTIONS
    assert "data.spot_prices: own" in text
    assert "must NOT list the perp mark" in text


def test_the_reviewer_judges_the_code_against_the_declared_spot_price() -> None:
    text = reviewer.SYSTEM_PROMPT
    assert "`own`" in text and "`perp`" in text
    assert "spot-perp gap as missing" in text


def test_the_researchers_prompt_still_formats() -> None:
    """The template is filled with str.format: a stray brace in the new text would break it."""
    import inspect

    source = inspect.getsource(researcher)
    assert "SPOT PRICES (docs/SPOT_PRICES.md)" in source
    assert "{" not in researcher.INSTRUCTIONS.split("SPOT PRICES")[1].split("ONE IDEA")[0]
