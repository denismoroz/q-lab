"""Clients for the public paper APIs. Pure code: no agent, no tokens.

- arXiv API (https://info.arxiv.org/help/api/tou.html): one request per three
  seconds, one connection at a time. Papers come newest first
  (`sortBy=submittedDate`), so a query stops at the first page that reaches
  back to `since`.
- Crossref REST API (https://www.crossref.org/documentation/retrieve-metadata/
  rest-api/): SSRN papers are the works under the DOI prefix 10.2139; SSRN
  itself has no public API. One request per second, one at a time.

Both clients take an `httpx.Client` so tests feed them recorded responses; the
pacing clock and sleep are injectable too. A failing query is reported, never
raised: one dead API must not take the night down.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from qlab.publications.items import Item, clean_text

USER_AGENT = "q-lab-research/0.1 (metadata feed for a private research lab)"
ATOM = "{http://www.w3.org/2005/Atom}"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_WAIT_SECONDS = 60.0


@dataclass
class FetchReport:
    items: list[Item] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests: int = 0


class Throttle:
    """Keeps at least `min_interval` seconds between requests to one API."""

    def __init__(self, min_interval: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self.min_interval - (self._clock() - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()


def _get(client: httpx.Client, url: str, params: dict, throttle: Throttle, report: FetchReport,
         *, sleep: Callable[[float], None], attempts: int = 3) -> httpx.Response:
    """GET with pacing; a throttled or failing server is retried after the wait
    it asks for (capped), then the last failure is raised."""
    for attempt in range(1, attempts + 1):
        throttle.wait()
        report.requests += 1
        try:
            response = client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        except httpx.TransportError:
            if attempt == attempts:
                raise
            sleep(min(MAX_RETRY_WAIT_SECONDS, 5.0 * attempt))
            continue
        if response.status_code in RETRY_STATUSES and attempt < attempts:
            try:
                asked = float(response.headers.get("Retry-After", ""))
            except ValueError:
                asked = 5.0 * attempt
            sleep(min(MAX_RETRY_WAIT_SECONDS, asked))
            continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)


# ---- arXiv ---------------------------------------------------------------


def _arxiv_id(url: str) -> str:
    tail = url.split("/abs/", 1)[1]
    head, _, version = tail.rpartition("v")
    return head if head and version.isdigit() else tail


def parse_arxiv(xml_text: str) -> list[Item]:
    """Items of one arXiv Atom page (a feed with an error entry raises)."""
    root = ET.fromstring(xml_text)
    items = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if "/api/errors" in entry_id:
            raise ValueError(f"arXiv error: {clean_text(entry.findtext(f'{ATOM}summary'))}")
        if "/abs/" not in entry_id:
            continue
        title = clean_text(entry.findtext(f"{ATOM}title"))
        published = entry.findtext(f"{ATOM}published")
        if not title or not published:
            continue
        arxiv_id = _arxiv_id(entry_id)
        items.append(Item(
            id=f"arxiv:{arxiv_id}",
            source="arxiv",
            title=title,
            authors=[clean_text(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
            published=_utc(published.strip()),
            abstract=clean_text(entry.findtext(f"{ATOM}summary")),
            link=f"https://arxiv.org/abs/{arxiv_id}",
            categories=[c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")],
        ))
    return items


def fetch_arxiv(client: httpx.Client, cfg: dict, since: datetime, *,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic) -> FetchReport:
    """Every paper of the configured queries submitted at or after `since`."""
    report = FetchReport()
    throttle = Throttle(float(cfg["min_interval_seconds"]), clock=clock, sleep=sleep)
    page_size = int(cfg["page_size"])
    for query in cfg["queries"]:
        try:
            for page in range(int(cfg["max_pages_per_query"])):
                response = _get(client, cfg["endpoint"], {
                    "search_query": " ".join(str(query["search_query"]).split()),
                    "sortBy": "submittedDate", "sortOrder": "descending",
                    "start": page * page_size, "max_results": page_size,
                }, throttle, report, sleep=sleep)
                items = parse_arxiv(response.text)
                report.items += [i for i in items if i.published >= since]
                if len(items) < page_size or min(i.published for i in items) < since:
                    break
        except Exception as exc:  # noqa: BLE001 - reported; the other queries go on
            report.errors.append(f"arxiv {query['name']}: {type(exc).__name__}: {exc}")
    report.items = merge(report.items)
    return report


# ---- Crossref (SSRN) -----------------------------------------------------


def _crossref_authors(entry: dict) -> list[str]:
    names = []
    for author in entry.get("author") or []:
        name = " ".join(p for p in (author.get("given"), author.get("family")) if p)
        names.append(clean_text(name or author.get("name")))
    return [n for n in names if n]


def parse_crossref(payload: dict, source_name: str) -> list[Item]:
    items = []
    for entry in payload["message"]["items"]:
        doi = (entry.get("DOI") or "").strip().lower()
        title = clean_text((entry.get("title") or [""])[0])
        created = (entry.get("created") or {}).get("date-time")
        if not doi or not title or not created:
            continue
        items.append(Item(
            id=f"doi:{doi}",
            source=source_name,
            title=title,
            authors=_crossref_authors(entry),
            published=_utc(created),
            abstract=clean_text(entry.get("abstract")),
            link=entry.get("URL") or f"https://doi.org/{doi}",
            categories=[],
        ))
    return items


def fetch_crossref(client: httpx.Client, cfg: dict, since: datetime, *, contact: str | None = None,
                   sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.monotonic) -> FetchReport:
    """Works under the configured DOI prefix registered at or after `since`
    that match any of the configured queries, best match first. (Crossref
    returns every work sharing any word with the query, so sorting by date
    would fill the pages with unrelated papers; the date filter bounds the
    tail and the relevance order keeps the best matches within the page cap.)"""
    report = FetchReport()
    throttle = Throttle(float(cfg["min_interval_seconds"]), clock=clock, sleep=sleep)
    rows = int(cfg["rows"])
    for query in cfg["queries"]:
        try:
            for page in range(int(cfg["max_pages_per_query"])):
                params = {
                    "filter": (f"prefix:{cfg['prefix']},"
                               f"from-created-date:{since.date().isoformat()}"),
                    "query.bibliographic": query, "sort": "score", "order": "desc",
                    "rows": rows, "offset": page * rows,
                    "select": "DOI,title,author,created,abstract,URL",
                }
                if contact:
                    params["mailto"] = contact
                response = _get(client, cfg["endpoint"], params, throttle, report, sleep=sleep)
                items = parse_crossref(response.json(), cfg["source_name"])
                report.items += [i for i in items if i.published >= since]
                if len(response.json()["message"]["items"]) < rows:
                    break
        except Exception as exc:  # noqa: BLE001 - reported; the other queries go on
            report.errors.append(f"crossref {query!r}: {type(exc).__name__}: {exc}")
    report.items = merge(report.items)
    return report


def merge(items: list[Item]) -> list[Item]:
    """One item per id, newest first."""
    unique = {i.id: i for i in items}
    return sorted(unique.values(), key=lambda i: i.published, reverse=True)


__all__ = ["USER_AGENT", "FetchReport", "Throttle", "fetch_arxiv", "fetch_crossref", "merge",
           "parse_arxiv", "parse_crossref"]
