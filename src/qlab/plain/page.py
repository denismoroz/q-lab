"""The top of an idea's page: «Что делает стратегия» in plain Russian, for every idea
(owner, 2026-11-03: «я открываю карту стратегии ... смотрю и не могу понять - а что она вообще
делает и как ... можно это как-то починить для всех стратегий?»; docs/PLAIN_PAGE.md).

What an idea shows, and where each part comes from:

1. **what it does**: the spec's own `plain_summary` (a new mechanism or family), else the idea's
   card, else the text of its strategy family (`strategies/plain/<family>.yaml`: by the strategy
   the idea descends from, then by the code its spec runs), else the card of the idea it
   descends from, else the text of the catalogue mechanism a generated spec names. A composed
   strategy gets, besides, the sentences its blocks give. Nothing is invented by a model: every
   sentence is a file a person wrote and checked, or a phrase code builds from the spec.
2. **how it differs**: code compares the spec with the family's reference spec;
3. **how the numbers were chosen**: from the spec (`qlab.plain.choice`);
4. **where it was checked**: from the spec's data (`qlab.plain.venue`);
5. **a short title**, from the same sources.

Read-only: nothing here writes, and nothing here fails the page (the caller catches).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from qlab.plain import choice as choice_mod
from qlab.plain import compose, venue
from qlab.plain.difference import Item, difference
from qlab.plain.specs import SpecView, load_view, view_for_idea
from qlab.plain.texts import TITLE_MAX, Family, Texts, load_texts
from qlab.plain.words import Words
from qlab.registry.models import Idea, Spec

NO_TEXT = ("Простого описания для этой стратегии пока нет: она не относится ни к одному из "
           "описанных семейств, а карточки у неё нет.")
TWIN_SUFFIX = "-causal"
"""`qlab.twin.TWIN_SUFFIX`; a test keeps the two equal."""


def clip(text: str, limit: int = TITLE_MAX) -> str:
    """`text` cut at a word boundary to at most `limit` characters, with an ellipsis."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:—-") + "…"


def first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    end = flat.find(". ")
    return flat if end < 0 else flat[: end + 1]


@dataclass(frozen=True)
class Resolved:
    """Where the idea's description comes from."""

    source: str
    """`spec`, `card`, `family`, `mechanism`, `ancestor_card`, `own_mechanism` or `none`."""
    family: Family | None
    paragraphs: tuple[str, ...]
    note: str | None = None
    card_title: str = ""
    """The title of the card the text comes from (a card's title is plain by its schema)."""


def _lineage(parents: Mapping[str, str | None], idea_id: str) -> list[str]:
    out = [idea_id]
    while (parent := parents.get(out[-1])) and parent not in out:
        out.append(parent)
    return out


def find_family(texts: Texts, *, root_id: str, views: list[SpecView]) -> Family | None:
    """The family of the strategy: by the strategy the idea descends from, then by the code its
    spec runs (the idea's own, then each ancestor's)."""
    for family in texts.families:
        if root_id in family.roots:
            return family
    for view in views:
        name = view.unwrapped.core_name
        for family in texts.families:
            if name in family.code_names:
                return family
    return None


def mechanism_text(texts: Texts, views: list[SpecView]) -> Family | None:
    for view in views:
        if view.mechanism and view.mechanism in texts.mechanisms:
            return texts.mechanisms[view.mechanism]
    return None


def _own_mechanism_paragraph(view: SpecView) -> str | None:
    own = view.extras.get("own_mechanism")
    if not isinstance(own, dict) or not own.get("idea"):
        return None
    why = f" Почему должна зарабатывать: {own['why']}" if own.get("why") else ""
    return f"{' '.join(str(own['idea']).split())}{why}"


def resolve(*, idea_id: str, root_id: str, view: SpecView | None, chain_views: list[SpecView],
            cards: Mapping[str, Mapping[str, Any]], lineage: list[str], texts: Texts
            ) -> Resolved:
    """The chain: own spec text -> own card -> family -> ancestor's card -> mechanism ->
    the idea's own mechanism words -> nothing (`NO_TEXT`)."""
    if view is not None:
        summary = view.extras.get("plain_summary")
        if isinstance(summary, str) and summary.strip():
            return Resolved("spec", None, (" ".join(summary.split()),
                                           compose.money_sentence(view, None)))
    card = cards.get(idea_id)
    if card and str(card.get("plain_summary") or "").strip():
        return Resolved("card", None, (" ".join(str(card["plain_summary"]).split()),),
                        card_title=str(card.get("title") or ""))
    family = find_family(texts, root_id=root_id, views=chain_views)
    if family is not None:
        # the numbers come from the idea's own spec, else from the nearest ancestor that has one
        source_view = view if view is not None else (chain_views[0] if chain_views else _EMPTY)
        sentences = compose.family_sentences(family, compose.fact_core(source_view))
        sentences.append(compose.money_sentence(source_view, family))
        return Resolved("family", family, (" ".join(sentences),))
    for ancestor in lineage[1:]:
        card = cards.get(ancestor)
        if card and str(card.get("plain_summary") or "").strip():
            return Resolved("ancestor_card", None, (" ".join(str(card["plain_summary"]).split()),),
                            note=f"Это описание идеи, от которой произошла эта: {ancestor}.",
                            card_title=str(card.get("title") or ""))
    mechanism = mechanism_text(texts, chain_views)
    if mechanism is not None and view is not None:
        paragraphs = [" ".join(mechanism.sentences)]
        here = compose.composed_sentences(view.unwrapped, texts)
        tail = [*here, compose.money_sentence(view, mechanism)]
        paragraphs.append("Конкретно в этой версии. " + " ".join(tail))
        return Resolved("mechanism", mechanism, tuple(paragraphs))
    if view is not None:
        own = _own_mechanism_paragraph(view)
        built = compose.composed_sentences(view.unwrapped, texts)
        if own:
            tail = ["Конкретно в этой версии. " + " ".join(built)] if built else []
            return Resolved("own_mechanism", None, (own, *tail, compose.money_sentence(view, None)))
        if built:
            tail = compose.money_sentence(view, None)
            return Resolved("mechanism", None, (" ".join([*built, tail]),))
    return Resolved("none", None, (NO_TEXT,))


_EMPTY = SpecView(idea_id="", title="", code_ref="", params={}, data={}, costs={})


def make_title(resolved: Resolved, *, idea: Idea, items: list[Item], second_stage: bool,
               family: Family | None, view: SpecView | None) -> str:
    """A plain title of at most `TITLE_MAX` characters, from the same source as the text: the
    spec's own `plain_title`, a card's title, the family's (or mechanism's) title with the idea's
    shortest tags, else the first sentence of the text."""
    stage = " (проверка на Hyperliquid)" if second_stage else ""
    own = view.extras.get("plain_title") if view is not None else None
    if isinstance(own, str) and own.strip():
        return clip(own)
    if resolved.card_title and len(resolved.card_title) <= TITLE_MAX:
        return resolved.card_title
    named = family or resolved.family
    if named is not None:
        tags = [i.tag for i in items if i.tag]
        for count in (2, 1):
            if tags[:count]:
                candidate = f"{named.title} — {'; '.join(tags[:count])}"
                if len(candidate + stage) <= TITLE_MAX:
                    return candidate + stage
        return clip(named.title + stage)
    if resolved.card_title:
        return clip(resolved.card_title)
    if resolved.source in ("spec", "own_mechanism"):
        return clip(first_sentence(resolved.paragraphs[0]))
    return clip(idea.title)


def _spec_row(session: Session, idea_id: str) -> Spec | None:
    return (session.query(Spec).filter(Spec.idea_id == idea_id)
            .order_by(Spec.version.desc()).first())


def plain_page(session: Session, idea: Idea, *, root: Path | None = None,
               cards: Mapping[str, Mapping[str, Any]] | None = None,
               texts: Texts | None = None) -> dict[str, Any]:
    """Everything the top of the idea's page shows (see the module docstring)."""
    from qlab.registry.stages import is_stage2_child, stage2_child

    texts = texts if texts is not None else load_texts(root)
    cards = cards or {}
    words = Words(texts.dictionary)
    parents = {i: p for i, p in session.query(Idea.id, Idea.parent_id)}
    lineage = _lineage(parents, idea.id)
    root_id = lineage[-1]

    def view_of(idea_id: str) -> SpecView | None:
        found = session.get(Idea, idea_id)
        return view_for_idea(idea_id, _spec_row(session, idea_id),
                             title=found.title if found else "", root=root)

    views = [(i, v) for i in lineage if (v := view_of(i)) is not None]
    own_view = views[0][1] if views and views[0][0] == idea.id else None
    chain = [v for _, v in views]
    resolved = resolve(idea_id=idea.id, root_id=root_id, view=own_view, chain_views=chain,
                       cards=cards, lineage=lineage, texts=texts)

    if resolved.source == "none" and getattr(idea, "claimed_edge", None):
        # an idea of the old list, rejected before cards existed: what its source claimed, as the
        # registry recorded it, is all there is to show
        resolved = Resolved("none", None, (
            NO_TEXT, "Что утверждал источник идеи (как записано в реестре): "
                     f"«{clip(str(idea.claimed_edge), 400)}»"))

    items: list[Item] = []
    base_view: SpecView | None = None
    # the idea's strategy family, even when its text came from a card: it names the reference
    # spec the difference is computed against
    family = family_any = find_family(texts, root_id=root_id, views=chain)
    if own_view is not None and family is not None:
        base_path = family.base_specs.get(own_view.unwrapped.core_name)
        base_view = load_view(Path(base_path), root) if base_path else None
        if base_view is not None:
            items = difference(own_view, base_view, words)
    second_stage = is_stage2_child(idea)
    child = None if second_stage else stage2_child(session, idea.id)

    base_name = family.base_name if family is not None else "как в основной версии"
    payload: dict[str, Any] = {
        "title": make_title(resolved, idea=idea, items=items, second_stage=second_stage,
                            family=family_any, view=own_view),
        "technical_title": idea.title,
        "what": {"source": resolved.source,
                 "family": resolved.family.id if resolved.family else None,
                 "family_title": resolved.family.title if resolved.family else None,
                 "paragraphs": list(resolved.paragraphs), "note": resolved.note},
        "difference": None,
        "choice": None,
        "venue": None,
    }
    if base_view is not None:
        payload["difference"] = {
            "base_id": base_view.idea_id, "base_file": base_view.path, "base_name": base_name,
            "is_base": not items and own_view is not None and own_view.path == base_view.path,
            "items": [{"text": i.text, "kind": i.kind} for i in items],
        }
    if own_view is not None:
        payload["choice"] = asdict(choice_mod.describe(own_view, base_name, words))
        payload["choice"]["lines"] = list(payload["choice"]["lines"])
        payload["venue"] = venue.describe(
            own_view, parent_id=idea.parent_id, child_id=child.id if child else None,
            is_second_stage=second_stage)
    return payload
