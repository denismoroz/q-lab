"""The cataloguer agent (qlab.agents.cataloguer): a paper becomes a catalogue entry only when
every number stands in the paper's own text and the generator's parser reads it back. The agent
is faked; the checks, the catalogue writing and the parser are real."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import qlab.agents.cataloguer as cat
from qlab.agents.generator import parse_catalogue
from qlab.budget import ExplicitNoBudget

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = ROOT / "docs" / "MECHANISMS.md"
DAY = date(2026, 10, 27)
LINK = "https://arxiv.org/abs/2601.00001"

PAPER = """Funding Pressure and Perpetual Futures Returns

Abstract. We sort perpetual futures each week on their average funding rate over the past
two weeks and form quintile portfolios. The long-short portfolio buys the lowest-fun-
ding quintile and sells the highest-funding quintile.

3.2 Portfolio construction
Portfolios are rebalanced weekly. The average funding rate is computed over the past 14 days;
results with a 30-day window are not significant. We require 60 days of history.
"""

GOOD = {
    "verdict": "mechanism",
    "citation": "Doe, Roe 2026",
    "title": "Funding Pressure and Perpetual Futures Returns",
    "mechanism": {
        "id": "perp-funding-pressure",
        "summary_ru": "Каждую неделю покупают контракты с самой низкой ставкой финансирования "
                      "и продают в короткую контракты с самой высокой; платят те, кто "
                      "переплачивает за заём в покупку.",
        "blocks_ru": "`signal: funding`; `portfolio: long_short_quantile`; `sizing: equal`.",
        "checked_ru": "Бессрочные контракты, сортировка раз в неделю на пять групп.",
        "unexpressed_ru": "ничего",
        "params": [
            {"param": "signal.window_days", "values": [14], "note_ru": "две недели",
             "where": "раздел 3.2",
             "quotes": ["The average funding rate is computed over the past 14 days"]},
            {"param": "rebalance_days", "values": [7], "note_ru": "раз в неделю",
             "where": "раздел 3.2", "quotes": ["Portfolios are rebalanced weekly."]},
            {"param": "portfolio.quantiles", "values": [5], "note_ru": "пять групп",
             "where": "аннотация",
             "quotes": ["buys the lowest-funding quintile and sells the highest-funding "
                        "quintile", "form quintile portfolios"]},
        ],
    },
}


def _ids() -> set[str]:
    return {m.id for m in parse_catalogue(CATALOGUE.read_text(encoding="utf-8"))}


def _check(draft: dict) -> list[str]:
    return cat.check_draft(draft, PAPER, catalogue_ids=_ids(), params=cat.known_params())


def _copy(draft: dict, **mech) -> dict:
    out = yaml.safe_load(yaml.safe_dump(draft, allow_unicode=True))
    out["mechanism"].update(mech)
    return out


def test_a_quote_gives_its_numbers_and_spans_in_days() -> None:
    assert {14.0, 2.0}.issubset(cat.quote_numbers("over the past two weeks, i.e. 14 days"))
    assert 14.0 in cat.quote_numbers("over the past two weeks")
    assert {90.0, 365.0, 30.0}.issubset(cat.quote_numbers("3 months, one year, monthly"))
    assert {30.0, 0.3}.issubset(cat.quote_numbers("the top 30% of coins"))
    assert 7.0 in cat.quote_numbers("rebalanced weekly")
    assert cat.match_key("lowest-fun-\nding  Quintile") == cat.match_key("lowestfunding quintile")
    assert 5.0 in cat.quote_numbers("form quintile portfolios")


def test_a_draft_whose_numbers_stand_in_the_paper_passes() -> None:
    assert _check(GOOD) == []
    # A word broken at a line end in the PDF text still matches the quote.
    assert "lowest-funding quintile" in GOOD["mechanism"]["params"][2]["quotes"][0]


def test_a_quote_not_in_the_paper_or_a_number_it_does_not_give_is_refused() -> None:
    rows = GOOD["mechanism"]["params"]
    invented = _copy(GOOD, params=[{**rows[0], "quotes": ["computed over the past 21 days"]}])
    assert any("quote not in the paper" in p for p in _check(invented))
    rounded = _copy(GOOD, params=[{**rows[0], "values": [15]}])
    assert any("15 is no number of its quotes" in p for p in _check(rounded))
    bare = _copy(GOOD, params=[{**rows[0], "quotes": []}])
    assert any("values without a quote" in p for p in _check(bare))


def test_a_parameter_no_block_takes_and_jargon_are_refused() -> None:
    rows = GOOD["mechanism"]["params"]
    unknown = _copy(GOOD, params=[{**rows[0], "param": "signal.decay_halflife"}])
    assert any("no parameter of a block" in p for p in _check(unknown))
    jargon = _copy(GOOD, summary_ru="Шорт контрактов с высоким фандингом и лонг остальных.")
    problems = _check(jargon)
    assert any("«шорт»" in p for p in problems) and any("«фандинг»" in p for p in problems)
    english = _copy(GOOD, checked_ru="Perpetual futures sorted weekly into quintiles.")
    assert any("not in Russian" in p for p in _check(english))


def test_new_and_extends_are_told_apart_by_the_catalogue() -> None:
    taken = _copy(GOOD, id="crypto-xs-momentum")
    assert any("already in the catalogue" in p for p in _check(taken))
    absent = {**_copy(GOOD), "verdict": "extends"}
    assert any("not in the catalogue" in p for p in _check(absent))
    assert _check({"verdict": "none", "reason": "no trading rule"}) == []
    assert _check({"verdict": "none"}) == ["verdict none needs reason"]


def test_a_new_mechanism_is_written_where_the_generator_reads_it() -> None:
    text = CATALOGUE.read_text(encoding="utf-8")
    block = cat.render_mechanism(GOOD, LINK, day=DAY, text_file="data/cataloguer/papers/x.txt")
    new = cat.insert_mechanism(text, block)
    assert cat.written_back(new, GOOD) == []
    mech = next(m for m in parse_catalogue(new) if m.id == "perp-funding-pressure")
    assert {r.param: r.values for r in mech.params} == {
        "signal.window_days": (14.0,), "rebalance_days": (7.0,), "portfolio.quantiles": (5.0,)}
    assert mech.fully_expressed
    assert len(parse_catalogue(new)) == len(parse_catalogue(text)) + 1
    # The closing sections stay closing: the new entry sits before them.
    assert new.index("### perp-funding-pressure") < new.index("## Описано в источниках")


def test_numbers_for_a_catalogued_mechanism_are_added_to_its_table() -> None:
    text = CATALOGUE.read_text(encoding="utf-8")
    draft = {**_copy(GOOD, id="perp-carry-crash-timing", params=GOOD["mechanism"]["params"][:1],
                     checked_ru="Окно ставки финансирования в две недели."),
             "verdict": "extends"}
    assert _check(draft) == []
    new = cat.extend_mechanism(text, draft, LINK, day=DAY)
    assert cat.written_back(new, draft) == []
    before = next(m for m in parse_catalogue(text) if m.id == "perp-carry-crash-timing")
    after = next(m for m in parse_catalogue(new) if m.id == "perp-carry-crash-timing")
    assert len(after.params) == len(before.params) + 1
    assert "Окно ставки финансирования в две недели" in after.checked
    assert [m.id for m in parse_catalogue(new)] == [m.id for m in parse_catalogue(text)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    catalogue = tmp_path / "MECHANISMS.md"
    catalogue.write_text(CATALOGUE.read_text(encoding="utf-8"), encoding="utf-8")
    text = tmp_path / "paper.txt"
    text.write_text(PAPER * 20, encoding="utf-8")  # long enough to be a paper
    monkeypatch.setattr(cat, "fetch_text", lambda paper: text)
    monkeypatch.setattr(cat, "_tracked_changes", lambda: set())
    return SimpleNamespace(catalogue=catalogue, work=tmp_path / "work")


def _agent(monkeypatch, draft: dict | None, review: str = "VERDICT: OK") -> list[str]:
    calls = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        calls.append(agent)
        if agent == "cataloguer" and draft is not None:
            (Path(cwd) / "draft.yaml").write_text(yaml.safe_dump(draft, allow_unicode=True),
                                                  encoding="utf-8")
        return SimpleNamespace(text=review if agent.endswith("review") else "DRAFT: ok",
                               exit_code=0)

    monkeypatch.setattr(cat, "run_agent", fake)
    return calls


def _run(world, paper: cat.Paper) -> cat.Paper:
    return cat.run_paper(paper, {"model": "sonnet"}, budget=ExplicitNoBudget("test"),
                         session=None, day=DAY, catalogue=world.catalogue,
                         work_root=world.work, commit=False)


def test_a_paper_that_passes_lands_in_the_catalogue(world, monkeypatch) -> None:
    calls = _agent(monkeypatch, GOOD)
    paper = _run(world, cat.Paper(id="arxiv:2601.00001", link=LINK))
    assert paper.status == "catalogued", paper.problems
    assert calls == ["cataloguer", "cataloguer-review"]
    assert any(m.id == "perp-funding-pressure"
               for m in parse_catalogue(world.catalogue.read_text(encoding="utf-8")))


def test_a_refused_draft_or_a_gap_leaves_the_catalogue_untouched(world, monkeypatch) -> None:
    before = world.catalogue.read_text(encoding="utf-8")
    rows = GOOD["mechanism"]["params"]
    _agent(monkeypatch, _copy(GOOD, params=[{**rows[0], "values": [21]}]))
    paper = _run(world, cat.Paper(id="arxiv:2601.00001", link=LINK))
    assert paper.status == "failed" and any("21 is no number" in p for p in paper.problems)
    calls = _agent(monkeypatch, GOOD, review="GAP: row 1: 14 days is the sample\nVERDICT: GAP")
    paper = _run(world, paper)
    assert paper.status == "blocked" and paper.review == ["row 1: 14 days is the sample"]
    assert calls == ["cataloguer", "cataloguer-review"] and paper.attempts == 2
    assert world.catalogue.read_text(encoding="utf-8") == before


def test_a_retry_is_told_why_the_last_draft_was_refused(world, monkeypatch) -> None:
    prompts = []

    def fake(budget, prompt, *, session, agent, model, max_turns, cwd=None, extra_args=()):
        prompts.append(prompt)
        return SimpleNamespace(text="", exit_code=0)

    monkeypatch.setattr(cat, "run_agent", fake)
    paper = _run(world, cat.Paper(id="arxiv:2601.00001", link=LINK))
    assert paper.status == "failed" and "no draft written" in paper.problems[0]
    _run(world, paper)
    assert "previous attempt" not in prompts[0] and "no draft written" in prompts[1]


def test_a_paper_without_a_mechanism_is_closed_with_its_reason(world, monkeypatch) -> None:
    calls = _agent(monkeypatch, {"verdict": "none", "reason": "a survey without a rule"})
    paper = _run(world, cat.Paper(id="arxiv:2601.00001", link=LINK))
    assert (paper.status, paper.reason, calls) == ("none", "a survey without a rule",
                                                   ["cataloguer"])


def test_a_paper_whose_text_cannot_be_read_is_set_aside(world, monkeypatch) -> None:
    def no_pdf(paper):
        raise ValueError("no PDF at https://ssrn.com/abstract=1 (HTTP 403, text/html)")

    monkeypatch.setattr(cat, "fetch_text", no_pdf)
    calls = _agent(monkeypatch, GOOD)
    paper = _run(world, cat.Paper(id="doi:10.2139/ssrn.1", link="https://ssrn.com/abstract=1"))
    assert paper.status == "no_text" and calls == []


def test_the_owners_paper_comes_first_then_a_scout_draft_then_arxiv(tmp_path) -> None:
    from qlab.registry.models import Base, Publication

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    now = datetime(2026, 10, 27, tzinfo=UTC)

    def pub(pid, score, *, draft=None, note=None):
        session.add(Publication(id=pid, source=pid.split(":")[0], title=pid, title_key=pid,
                                authors=[], published=now, abstract="a", link=f"https://{pid}",
                                categories=[], first_seen=now, candidate=True, score=score,
                                matched={}, draft=draft, scout_note=note))

    pub("arxiv:1", 9)
    pub("arxiv:2", 5, draft="cards/drafts/x.yaml")
    pub("arxiv:3", 12, note="NO CANDIDATE: not a mechanism")
    pub("doi:10.2139/ssrn.4", 20)
    session.flush()
    ledger: list[cat.Paper] = []
    owner = cat.add_owner_paper(ledger, "https://arxiv.org/abs/2510.14435v2", day=DAY)
    assert owner.id == "arxiv:2510.14435"
    assert cat.add_owner_paper(ledger, "https://arxiv.org/abs/2510.14435v2", day=DAY) is None
    assert cat.next_paper(ledger, session, max_attempts=2) is owner
    owner.status = "catalogued"
    assert cat.next_paper(ledger, session, max_attempts=2).id == "arxiv:2"
    ledger[-1].status = "none"
    assert cat.next_paper(ledger, session, max_attempts=2).id == "arxiv:1"
    ledger[-1].status = "none"
    assert cat.next_paper(ledger, session, max_attempts=2) is None  # SSRN needs a person
