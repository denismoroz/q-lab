"""One paper as the feed sees it, whatever API it came from."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime

_TAG = re.compile(r"</?[A-Za-z][^<>]*>")  # a tag, not a "<" of a formula
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Item:
    id: str  # "arxiv:<id without version>" or "doi:<lower-case doi>"
    source: str  # "arxiv" or the Crossref source name ("ssrn")
    title: str
    authors: list[str]
    published: datetime  # timezone-aware, UTC
    abstract: str
    link: str
    categories: list[str] = field(default_factory=list)


def clean_text(raw: str | None) -> str:
    """Plain text from API text: JATS/HTML tags dropped, entities (also the
    doubly escaped ones Crossref returns) decoded, whitespace collapsed."""
    # Tags are stripped before and after decoding: Crossref sends markup raw in
    # abstracts and escaped (sometimes twice) in titles.
    text = raw or ""
    for _ in range(2):
        text = html.unescape(_TAG.sub(" ", text))
    text = _TAG.sub(" ", text)
    return _SPACE.sub(" ", text.replace("\xa0", " ")).strip()


def title_key(title: str) -> str:
    """The key under which the same paper on two sources is recognised."""
    return _NON_WORD.sub("", title.lower())


__all__ = ["Item", "clean_text", "title_key"]
