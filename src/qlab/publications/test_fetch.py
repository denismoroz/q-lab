"""The API clients against recorded responses (no network in tests).

`fixtures/arxiv_qfin_tr.xml` and `fixtures/crossref_ssrn.json` are real
responses recorded on 2026-10-25 and cut to a few entries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from qlab.publications.fetch import (
    Throttle,
    fetch_arxiv,
    fetch_crossref,
    parse_arxiv,
    parse_crossref,
)

FIXTURES = Path(__file__).parent / "fixtures"
ARXIV_XML = (FIXTURES / "arxiv_qfin_tr.xml").read_text(encoding="utf-8")
CROSSREF = json.loads((FIXTURES / "crossref_ssrn.json").read_text(encoding="utf-8"))
LONG_AGO = datetime(2000, 1, 1, tzinfo=UTC)


def atom(entries: list[tuple[str, str, str]]) -> str:
    """An arXiv-shaped feed: (id, published, title) per entry."""
    body = "".join(
        f"<entry><id>http://arxiv.org/abs/{i}v2</id><title>{t}</title>"
        f"<summary>About {t}.</summary><published>{p}</published>"
        f'<category term="q-fin.TR" scheme="x"/><author><name>An Author</name></author></entry>'
        for i, p, t in entries)
    return ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            f"<title>arXiv Query</title>{body}</feed>")


def arxiv_cfg(**over) -> dict:
    cfg = {"endpoint": "https://export.arxiv.org/api/query", "min_interval_seconds": 3.0,
           "page_size": 2, "max_pages_per_query": 5,
           "queries": [{"name": "TR", "search_query": "cat:q-fin.TR"}]}
    return cfg | over


def crossref_cfg(**over) -> dict:
    cfg = {"endpoint": "https://api.crossref.org/works", "min_interval_seconds": 1.0, "rows": 2,
           "max_pages_per_query": 3, "prefix": "10.2139", "source_name": "ssrn",
           "queries": ["perpetual futures funding rate"]}
    return cfg | over


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeTime:
    def __init__(self) -> None:
        self.now, self.slept = 0.0, []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


# ---- parsers on recorded responses ------------------------------------------------------------


def test_the_recorded_arxiv_page_parses_into_items() -> None:
    items = parse_arxiv(ARXIV_XML)
    assert items
    for it in items:
        assert it.id.startswith("arxiv:") and "v" not in it.id.split(":")[1].split(".")[-1][-2:-1]
        assert it.title and it.abstract and it.authors and it.categories
        assert it.published.tzinfo is not None
        assert it.link == f"https://arxiv.org/abs/{it.id.removeprefix('arxiv:')}"
    assert [i.published for i in items] == sorted((i.published for i in items), reverse=True)


def test_the_recorded_crossref_page_parses_into_items() -> None:
    items = parse_crossref(CROSSREF, "ssrn")
    assert items
    for it in items:
        assert it.id.startswith("doi:10.2139/ssrn.") and it.source == "ssrn"
        assert it.title and it.link.startswith("https://doi.org/10.2139/ssrn.")
        assert "jats" not in it.abstract and "&amp;" not in it.title and "\xa0" not in it.title
        assert it.published.tzinfo is not None


def test_markup_in_crossref_text_is_cleaned() -> None:
    payload = {"message": {"items": [{
        "DOI": "10.2139/SSRN.1", "title": ["Limits to Coordination:&amp;nbsp;Funding Drag"],
        "created": {"date-time": "2026-09-29T12:59:55Z"}, "URL": "https://doi.org/10.2139/ssrn.1",
        "abstract": "<jats:p>Funding <jats:italic>rates</jats:italic> &amp; carry.</jats:p>",
        "author": [{"given": "Ada", "family": "Lovelace"}, {"name": "Group"}]}, {
        "DOI": "10.2139/ssrn.2", "title": [], "created": {"date-time": "2026-09-29T12:59:55Z"}}]}}
    [only] = parse_crossref(payload, "ssrn")  # the entry without a title is dropped
    assert only.id == "doi:10.2139/ssrn.1"
    assert only.title == "Limits to Coordination: Funding Drag"
    assert only.abstract == "Funding rates & carry."
    assert only.authors == ["Ada Lovelace", "Group"]


def test_a_less_than_sign_in_a_formula_is_not_a_tag() -> None:
    from qlab.publications.items import clean_text

    assert clean_text("p < 0.05 and q > 3 <b>bold</b>") == "p < 0.05 and q > 3 bold"


def test_an_arxiv_error_feed_raises() -> None:
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#x'
           "</id><title>Error</title><summary>incorrect query</summary></entry></feed>")
    with pytest.raises(ValueError, match="incorrect query"):
        parse_arxiv(xml)


def test_old_style_arxiv_ids_keep_their_archive() -> None:
    xml = atom([("q-fin/0501001", "2005-01-03T10:00:00Z", "Old paper")])
    assert parse_arxiv(xml)[0].id == "arxiv:q-fin/0501001"


# ---- arXiv: paging, the tail, pacing ----------------------------------------------------------


def test_arxiv_pages_newest_first_and_stops_at_the_page_that_reaches_since() -> None:
    entries = [("2610.00005", "2026-10-20T00:00:00Z", "E"),
               ("2610.00004", "2026-10-15T00:00:00Z", "D"),
               ("2610.00003", "2026-10-10T00:00:00Z", "C"),
               ("2610.00002", "2026-09-01T00:00:00Z", "B"),
               ("2610.00001", "2026-08-01T00:00:00Z", "A")]
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        asked.append((q["start"], q["sortBy"], q["sortOrder"]))
        start, size = int(q["start"]), int(q["max_results"])
        return httpx.Response(200, text=atom(entries[start:start + size]))

    fake = FakeTime()
    report = fetch_arxiv(client(handler), arxiv_cfg(), datetime(2026, 10, 1, tzinfo=UTC),
                         sleep=fake.sleep, clock=fake.clock)
    assert [i.id for i in report.items] == ["arxiv:2610.00005", "arxiv:2610.00004",
                                            "arxiv:2610.00003"]
    # Page 1 (E, D) and page 2 (C, B: B is older than since) -- page 3 is never asked.
    assert asked == [("0", "submittedDate", "descending"), ("2", "submittedDate", "descending")]
    assert report.requests == 2 and report.errors == []
    assert fake.slept == [3.0]  # one request every three seconds


def test_arxiv_stops_on_a_short_page() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text=atom([("2610.1", "2026-10-20T00:00:00Z", "Only")]))

    report = fetch_arxiv(client(handler), arxiv_cfg(), LONG_AGO, sleep=lambda s: None)
    assert len(report.items) == 1 and len(calls) == 1


def test_a_throttled_arxiv_is_asked_again_after_the_wait_it_names() -> None:
    answers = iter([httpx.Response(429, headers={"Retry-After": "7"}),
                    httpx.Response(200, text=atom([("2610.1", "2026-10-20T00:00:00Z", "P")]))])
    fake = FakeTime()
    report = fetch_arxiv(client(lambda r: next(answers)), arxiv_cfg(), LONG_AGO,
                         sleep=fake.sleep, clock=fake.clock)
    assert [i.id for i in report.items] == ["arxiv:2610.1"]
    assert 7.0 in fake.slept and report.requests == 2


def test_a_failing_query_is_reported_and_the_others_go_on() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "q-fin.PM" in request.url.params["search_query"]:
            return httpx.Response(500)
        return httpx.Response(200, text=atom([("2610.1", "2026-10-20T00:00:00Z", "Fine")]))

    cfg = arxiv_cfg(queries=[{"name": "PM", "search_query": "cat:q-fin.PM"},
                             {"name": "TR", "search_query": "cat:q-fin.TR"}])
    report = fetch_arxiv(client(handler), cfg, LONG_AGO, sleep=lambda s: None)
    assert [i.id for i in report.items] == ["arxiv:2610.1"]
    assert len(report.errors) == 1 and "PM" in report.errors[0] and "500" in report.errors[0]


def test_the_same_paper_from_two_queries_is_one_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=atom([("2610.1", "2026-10-20T00:00:00Z", "Same")]))

    cfg = arxiv_cfg(queries=[{"name": "a", "search_query": "x"},
                             {"name": "b", "search_query": "y"}])
    assert len(fetch_arxiv(client(handler), cfg, LONG_AGO, sleep=lambda s: None).items) == 1


def test_the_throttle_waits_only_for_the_missing_part_of_the_interval() -> None:
    fake = FakeTime()
    throttle = Throttle(3.0, clock=fake.clock, sleep=fake.sleep)
    throttle.wait()
    fake.now += 1.0
    throttle.wait()
    fake.now += 10.0
    throttle.wait()
    assert fake.slept == [2.0]


# ---- Crossref (SSRN) --------------------------------------------------------------------------


def test_crossref_asks_for_the_ssrn_prefix_and_the_tail_and_pages_by_offset() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if len(seen) == 1:  # a full page: another one is asked
            return httpx.Response(200, json=CROSSREF | {"message": {
                "items": CROSSREF["message"]["items"][:1] * 2}})
        return httpx.Response(200, json={"message": {"items": []}})

    report = fetch_crossref(client(handler), crossref_cfg(), LONG_AGO, sleep=lambda s: None)
    assert len(seen) == 2 and seen[1]["offset"] == "2" and seen[0]["offset"] == "0"
    assert seen[0]["filter"] == "prefix:10.2139,from-created-date:2000-01-01"
    assert seen[0]["query.bibliographic"] == "perpetual futures funding rate"
    assert seen[0]["sort"] == "score" and "mailto" not in seen[0]
    assert len(report.items) == 1  # the doubled entry is one paper


def test_crossref_sends_a_contact_only_when_the_owner_configured_one() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"message": {"items": []}})

    fetch_crossref(client(handler), crossref_cfg(), LONG_AGO, contact="me@example.org",
                   sleep=lambda s: None)
    assert seen[0]["mailto"] == "me@example.org"


def test_crossref_keeps_only_papers_at_or_after_since() -> None:
    newest = max(parse_crossref(CROSSREF, "ssrn"), key=lambda i: i.published)
    report = fetch_crossref(client(lambda r: httpx.Response(200, json=CROSSREF)),
                            crossref_cfg(rows=50), newest.published, sleep=lambda s: None)
    assert [i.id for i in report.items] == [newest.id]
