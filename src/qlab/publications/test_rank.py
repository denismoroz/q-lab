"""Tests for the paper queue ranking and the full text of a paper: no network, no agent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.publications import fulltext, rank, store
from qlab.publications.filter import FILTER_PATH, load_rules
from qlab.publications.items import Item
from qlab.registry.models import Base

NOW = datetime(2026, 11, 3, 5, 0, tzinfo=UTC)
WEIGHTS = yaml.safe_load((Path(__file__).resolve().parents[3] / "night/scout.yaml")
                         .read_text())["papers"]["rank"]
FUNDING = ("We study the funding rate of perpetual futures on Binance: who pays it, when it "
           "turns negative and how the basis closes after a liquidation. " * 3)
BOOK = ("Perpetual futures on a limit order book: we model order flow and tick data of a "
        "crypto exchange with deep learning. " * 3)


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _seed(session, rows: list[tuple[str, str, str, int]]) -> None:
    items = [Item(id=pid, source="arxiv" if pid.startswith("arxiv") else "ssrn", title=title,
                  authors=["A"], published=NOW - timedelta(days=days), abstract=abstract,
                  link=f"https://example.org/{pid}", categories=[])
             for pid, title, abstract, days in rows]
    store.add_new(session, items, now=NOW)
    store.apply_filter(session, load_rules(FILTER_PATH))


def test_a_paper_near_q_labs_data_outranks_one_that_needs_data_it_lacks() -> None:
    near, why_near = rank.usefulness("Funding rate of perpetual futures", FUNDING, {
        "strong": ["funding rate", "perpetual futures"]}, "arxiv:1", WEIGHTS)
    far, why_far = rank.usefulness("Perpetual futures on a limit order book", BOOK, {
        "strong": ["perpetual futures"]}, "arxiv:2", WEIGHTS)
    assert near > far
    assert any("near q-lab's data" in w for w in why_near)
    assert any("lacks" in w for w in why_far)


def test_a_paper_whose_text_can_be_read_gets_credit_and_the_score_is_pure() -> None:
    args = ("t", FUNDING, {"strong": ["funding rate"]})
    on_arxiv = rank.usefulness(*args, "arxiv:1", WEIGHTS)
    on_ssrn = rank.usefulness(*args, "doi:10.2139/ssrn.1", WEIGHTS)
    assert on_arxiv[0] - on_ssrn[0] == WEIGHTS["full_text_available"]
    assert on_arxiv == rank.usefulness(*args, "arxiv:1", WEIGHTS)


def test_the_queue_is_ranked_by_usefulness_not_by_arrival(session) -> None:
    _seed(session, [("arxiv:1", "Perpetual futures on a limit order book", BOOK, 1),
                    ("arxiv:2", "Funding rate of perpetual futures", FUNDING, 40),
                    ("doi:10.2139/ssrn.3", "Funding rate of perpetual futures", FUNDING + " ", 2)])
    ranked = rank.rank_candidates(session, WEIGHTS, min_abstract_chars=100)
    # The newest paper is last: arrival does not order the queue; the SSRN twin of the arXiv
    # paper is stored once by title, so two rows remain.
    assert [r.row.id for r in ranked][0] == "arxiv:2"
    assert ranked[-1].row.id == "arxiv:1"


def test_papers_another_agent_read_tonight_are_left_out(session) -> None:
    _seed(session, [("arxiv:1", "Funding rate of perpetual futures", FUNDING, 1),
                    ("arxiv:2", "Funding rate of perpetual swap", FUNDING + "x", 2)])
    ranked = rank.rank_candidates(session, WEIGHTS, skip_ids={"arxiv:1"})
    assert [r.row.id for r in ranked] == ["arxiv:2"]
    assert rank.rank_candidates(session, WEIGHTS, arxiv_only=True)


# --------------------------------------------------------------------------
# Full text


def test_references_are_dropped_from_the_end_of_a_text() -> None:
    body = "Introduction\n" + "text " * 400 + "\nReferences\n[1] Someone 2020\n"
    assert fulltext.strip_references(body).rstrip().endswith("text")
    early = "References\nare discussed here\n" + "text " * 400
    assert fulltext.strip_references(early) == early   # a heading in the first part stays


def test_html_becomes_text_without_scripts_or_tags() -> None:
    page = ("<html><head><style>p{}</style></head><body><script>x()</script>"
            "<h1>Title</h1><p>Funding &amp; basis</p></body></html>")
    assert fulltext.html_to_text(page) == "Title\nFunding & basis"


def test_ssrn_and_doi_papers_are_abstract_only_and_the_hand_off_says_so(tmp_path) -> None:
    text = fulltext.fetch("doi:10.2139/ssrn.1", "https://doi.org/10.2139/ssrn.1", "t",
                          cap_chars=1000, root=tmp_path)
    assert text.path is None and text.kind.startswith("abstract only")
    assert "SSRN" in text.kind


def test_an_arxiv_paper_is_read_as_text_cut_at_the_cap(tmp_path, monkeypatch) -> None:
    body = "A sentence about funding. " * 300 + "\nReferences\n[1] x\n"
    monkeypatch.setattr("qlab.agents.cataloguer.fetch_text",
                        lambda paper, client=None: _file(tmp_path, body))
    text = fulltext.fetch("arxiv:2601.00001", "https://arxiv.org/abs/2601.00001", "t",
                          cap_chars=2000, root=tmp_path)
    assert text.kind == "full text (arXiv PDF)" and text.cut and text.chars == 2000
    saved = (tmp_path / text.path).read_text()
    assert len(saved) == 2000 and "References" not in saved


def test_when_the_pdf_is_missing_the_html_version_is_tried_then_the_abstract(
        tmp_path, monkeypatch) -> None:
    def no_pdf(paper, client=None):
        raise ValueError("no PDF")

    monkeypatch.setattr("qlab.agents.cataloguer.fetch_text", no_pdf)
    html = "<html><body><p>" + "Funding is paid by longs. " * 200 + "</p></body></html>"

    class Client:
        def __init__(self, status):
            self.status = status

        def get(self, url):
            assert url == "https://arxiv.org/html/2601.00001"
            return SimpleNamespace(status_code=self.status, text=html)

    got = fulltext.fetch("arxiv:2601.00001", "l", "t", cap_chars=10_000, client=Client(200),
                         root=tmp_path)
    assert got.kind == "full text (arXiv HTML)" and got.path is not None
    none = fulltext.fetch("arxiv:2601.00001", "l", "t", cap_chars=10_000, client=Client(404),
                          root=tmp_path)
    assert none.path is None and "no readable text" in none.kind


def _file(tmp_path, body: str):
    path = tmp_path / "paper.txt"
    path.write_text(body, encoding="utf-8")
    return path
