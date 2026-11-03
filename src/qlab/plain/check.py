"""What the researcher and the generator must give a spec so its page can open with a plain
description (docs/PLAIN_PAGE.md).

A variant of a known strategy needs nothing: code computes how it differs from the family's base.
A spec that is a NEW mechanism or a NEW family has no text to read, so it must carry its own
`plain_summary`: plain words, no jargon (the rule of the cards, `cards/SCHEMA.md`).
"""

from __future__ import annotations

from pathlib import Path

from qlab.plain.specs import view_from_raw
from qlab.plain.texts import TITLE_MAX, load_texts

SUMMARY_MIN = 120
"""A summary shorter than this (characters) cannot say what is bought and sold, when and why."""


def known_family(raw: dict, text: str = "", root: Path | None = None) -> bool:
    """Whether code can already describe the spec: its code is a known family's, or it is composed
    of blocks for a mechanism of the catalogue that has a plain text."""
    texts = load_texts(root)
    view = view_from_raw(raw, text=text)
    name = view.unwrapped.core_name
    if any(name in family.code_names for family in texts.families):
        return True
    return bool(view.mechanism and view.mechanism in texts.mechanisms)


def spec_problems(raw: dict, text: str = "", *, root: Path | None = None) -> list[str]:
    """Problems with a spec's plain description. `raw` is the spec file's mapping, `text` its text
    (the `# mechanism:` header is a comment). Empty when the page can open with words."""
    from qlab.agents.cataloguer import _russian_problems

    summary = raw.get("plain_summary")
    out: list[str] = []
    title = raw.get("plain_title")
    if title is not None:
        out += _russian_problems("plain_title", title)
        if isinstance(title, str) and len(title.strip()) > TITLE_MAX:
            out.append(f"plain_title is {len(title.strip())} characters, at most {TITLE_MAX}")
    if summary is not None:
        out += _russian_problems("plain_summary", summary)
        if isinstance(summary, str) and summary.strip() and len(summary.strip()) < SUMMARY_MIN:
            out.append(f"plain_summary is {len(summary.strip())} characters, at least "
                       f"{SUMMARY_MIN}: say what it buys and sells, when, why it should earn and "
                       "what it risks")
        return out
    own = view_from_raw(raw, text=text).mechanism == "own"
    if own or not known_family(raw, text, root):
        out.append("plain_summary is missing: this spec is a new mechanism or a new family, so "
                   "the idea's page has no plain text to open with. Write 3 to 6 short sentences "
                   "in Russian, no jargon (cards/SCHEMA.md): what it holds, what it buys and sells "
                   "and when, why it should earn, what it risks")
    return out
