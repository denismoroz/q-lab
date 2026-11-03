"""The plain-Russian texts the idea page reads: strategy families, mechanisms, the dictionary of
parameter meanings and the phrases for blocks and wrappers (docs/PLAIN_PAGE.md).

Everything lives as data in `strategies/plain/`:

- `<family>.yaml`            one file per strategy family (bv2, trend, frab, ...);
- `mechanisms/<id>.yaml`      one file per mechanism of the catalogue (docs/MECHANISMS.md);
- `params.yaml`              what each parameter of a spec means, in words;
- `blocks.yaml`, `layers.yaml`  phrases for the blocks of a composed strategy and for wrappers.

The owner reads these texts, so they follow `cards/SCHEMA.md` (no jargon, a term explained at its
first use); `family_problems` checks the shape and runs the same jargon check the cards use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qlab.pipeline.sources import QLAB_ROOT

PLAIN_DIR = Path("strategies/plain")
TITLE_MAX = 80
"""The longest plain title, in characters (owner, 2026-11-03: a short title above the technical
one)."""
SENTENCES_MAX = 6
"""A family text is 3 to 6 short sentences; the money sentence is one of them."""
SENTENCES_MIN = 2
_NON_FAMILY = {"params.yaml", "blocks.yaml", "layers.yaml"}
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


@dataclass(frozen=True)
class Family:
    """One text: a strategy family or a mechanism of the catalogue."""

    id: str
    title: str
    sentences: tuple[str, ...]
    kind: str = "family"
    roots: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    base_specs: dict[str, str] = field(default_factory=dict)
    base_name: str = "как в основной версии"
    checked_against: tuple[str, ...] = ()
    facts_default: dict[str, Any] = field(default_factory=dict)
    money_fallback: str | None = None
    path: str = ""

    @property
    def code_names(self) -> tuple[str, ...]:
        return tuple(ref_name(ref) for ref in self.code_refs)


def ref_name(code_ref: str) -> str:
    """`qlab.strategies.live.bv2:LiveBv2` -> `LiveBv2`."""
    return code_ref.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


@dataclass(frozen=True)
class Dictionary:
    """The parameter dictionary and the block and wrapper phrases."""

    params: dict[str, dict[str, Any]]
    blocks: dict[str, Any]
    layers: dict[str, Any]


@dataclass(frozen=True)
class Texts:
    families: tuple[Family, ...]
    mechanisms: dict[str, Family]
    dictionary: Dictionary


def _read(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _family(raw: dict[str, Any], *, kind: str, path: Path) -> Family:
    return Family(
        id=str(raw["id"]), title=str(raw["title"]), kind=kind,
        sentences=tuple(" ".join(str(s).split()) for s in raw.get("sentences") or ()),
        roots=tuple(raw.get("roots") or ()), code_refs=tuple(raw.get("code_refs") or ()),
        base_specs=dict(raw.get("base_specs") or {}),
        base_name=str(raw.get("base_name") or "как в основной версии"),
        checked_against=tuple(raw.get("checked_against") or ()),
        facts_default=dict(raw.get("facts_default") or {}),
        money_fallback=raw.get("money_fallback"), path=str(path))


_CACHE: dict[tuple[str, tuple[tuple[str, int], ...]], Texts] = {}


def _signature(directory: Path) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(p), p.stat().st_mtime_ns) for p in directory.rglob("*.yaml")))


def load_texts(root: Path | None = None) -> Texts:
    """All the plain texts under `<root>/strategies/plain`. Cached until a file changes, so the
    page of an idea does not read thirty files on every request."""
    base = (root if root is not None else QLAB_ROOT) / PLAIN_DIR
    if not base.is_dir():
        return Texts((), {}, Dictionary({}, {}, {}))
    key = (str(base), _signature(base))
    if key in _CACHE:
        return _CACHE[key]
    families = tuple(
        _family(_read(p), kind="family", path=p)
        for p in sorted(base.glob("*.yaml")) if p.name not in _NON_FAMILY)
    mechanisms = {
        raw["id"]: _family(raw, kind="mechanism", path=p)
        for p in sorted((base / "mechanisms").glob("*.yaml"))
        for raw in [_read(p)]}
    params = (_read(base / "params.yaml").get("params") or {}) if (base / "params.yaml").is_file() \
        else {}
    blocks = _read(base / "blocks.yaml") if (base / "blocks.yaml").is_file() else {}
    layers = _read(base / "layers.yaml") if (base / "layers.yaml").is_file() else {}
    texts = Texts(families, mechanisms, Dictionary(params, blocks, layers))
    _CACHE.clear()
    _CACHE[key] = texts
    return texts


def placeholders(sentence: str) -> list[str]:
    return _PLACEHOLDER.findall(sentence)


def family_problems(family: Family, *, known_facts: frozenset[str] | None = None) -> list[str]:
    """What is wrong with a text as a text: its shape, its words, its placeholders. Empty when it
    can be shown to the owner."""
    from qlab.agents.cataloguer import _russian_problems

    out: list[str] = []
    name = f"{family.kind} {family.id}"
    if len(family.title) > TITLE_MAX:
        out.append(f"{name}: title is {len(family.title)} characters, at most {TITLE_MAX}")
    out += _russian_problems(f"{name} title", family.title)
    if not SENTENCES_MIN <= len(family.sentences) <= SENTENCES_MAX:
        out.append(f"{name}: {len(family.sentences)} sentences, "
                   f"{SENTENCES_MIN} to {SENTENCES_MAX} wanted")
    for i, sentence in enumerate(family.sentences, 1):
        # placeholders are code, not prose: the check reads the sentence with them removed
        out += _russian_problems(f"{name} sentence {i}", _PLACEHOLDER.sub("", sentence) or "-")
        if known_facts is not None:
            known = known_facts | set(family.facts_default)
            out += [f"{name} sentence {i}: unknown fact {{{p}}}"
                    for p in placeholders(sentence) if p not in known]
    if not family.checked_against:
        out.append(f"{name}: checked_against is empty: say which file each sentence was "
                   "checked against")
    if family.kind == "family" and not (family.roots or family.code_refs):
        out.append(f"{name}: neither roots nor code_refs: nothing would ever choose it")
    return out


def dictionary_problems(dictionary: Dictionary) -> list[str]:
    """The same checks for the parameter dictionary: every entry has words, no jargon."""
    from qlab.agents.cataloguer import _russian_problems

    out: list[str] = []
    for key, entry in dictionary.params.items():
        if not isinstance(entry, dict) or not str(entry.get("what", "")).strip():
            out.append(f"params.{key}: no `what`")
            continue
        for field_name in ("what", "when_true", "when_false", "short_true", "short_false", "tag",
                           "none_means"):
            if field_name in entry:
                out += _russian_problems(f"params.{key}.{field_name}", entry[field_name])
    return out
