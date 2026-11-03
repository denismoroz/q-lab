"""The text of a paper, where the source serves it (docs/SCOUT.md).

Owner, 2026-11-03: the scout used to see abstracts only, so a paper never gave it a number or a
rule. Here a paper's text is fetched where it is open without a person:

- arXiv: the PDF turned into text (`pdftotext`, the cataloguer's own cache, so a paper is
  downloaded once for both agents), else arXiv's HTML version stripped of its tags;
- SSRN and anything else (a DOI): the abstract only, and the hand-off says so -- the publisher
  answers scripts with a page that asks for a person (`night/publications.yaml`).

The text is cut for the scout: the reference list is dropped and a per-paper character cap
applies. The cut text is written to `data/publications/text/` and the scout reads that file.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from qlab.pipeline.sources import QLAB_ROOT

TEXT_DIR = Path("data/publications/text")
_REFERENCES = re.compile(r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*\n", re.IGNORECASE)


@dataclass(frozen=True)
class PaperText:
    path: Path | None      # relative to QLAB_ROOT; None when only the abstract is available
    kind: str              # "full text (arXiv PDF | HTML)" or "abstract only: <why>"
    chars: int = 0
    cut: bool = False      # the cap shortened it


def strip_references(text: str) -> str:
    """The text without its reference list: the last heading of that name in the last 40%."""
    found = [m for m in _REFERENCES.finditer(text) if m.start() > 0.6 * len(text)]
    return text[:found[-1].start()] if found else text


def html_to_text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|nav|header|footer)\b.*?</\1>", " ", page)
    page = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|section|figure|table)>|<br\s*/?>", "\n", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in html.unescape(page).splitlines())
    return "\n".join(line for line in lines if line)


def _arxiv_id(paper_id: str) -> str:
    return paper_id.removeprefix("arxiv:")


def fetch(paper_id: str, link: str, title: str, *, cap_chars: int, client=None,
          root: Path | None = None) -> PaperText:
    """The text of one paper for the scout, cut to `cap_chars`; never raises (a failure is an
    `abstract only` result that says why)."""
    root = root or QLAB_ROOT
    if not paper_id.startswith("arxiv:"):
        return PaperText(None, "abstract only: the publisher serves no text without a person "
                               "(SSRN and DOI links)")
    raw, kind, problem = None, "", ""
    try:
        from qlab.agents.cataloguer import Paper, fetch_text

        path = fetch_text(Paper(id=paper_id, link=link, title=title), client=client)
        raw, kind = path.read_text(encoding="utf-8", errors="replace"), "full text (arXiv PDF)"
    except Exception as exc:  # noqa: BLE001 - fall back to the HTML version
        problem = f"{type(exc).__name__}: {exc}"
    if raw is None:
        try:
            import httpx

            own = client is None
            client = client or httpx.Client(timeout=60.0, follow_redirects=True, headers={
                "User-Agent": "q-lab research (no orders)"})
            try:
                response = client.get(f"https://arxiv.org/html/{_arxiv_id(paper_id)}")
            finally:
                if own:
                    client.close()
            if response.status_code == 200 and "<html" in response.text[:2000].lower():
                raw, kind = html_to_text(response.text), "full text (arXiv HTML)"
            else:
                problem += f"; HTML version HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            problem += f"; HTML version: {type(exc).__name__}: {exc}"
    if not raw or len(raw) < 2000:
        return PaperText(None, f"abstract only: no readable text ({problem.strip('; ')[:200]})")
    body = strip_references(raw)
    cut = len(body) > cap_chars
    out = root / TEXT_DIR / (re.sub(r"[^\w.\-]+", "_", paper_id) + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body[:cap_chars], encoding="utf-8")
    return PaperText(out.relative_to(root), kind, min(len(body), cap_chars), cut)


__all__ = ["PaperText", "TEXT_DIR", "fetch", "html_to_text", "strip_references"]
