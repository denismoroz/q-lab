"""A spec as the idea page needs it: the spec file's fields (or the registry row when there is no
file), and the strategy's layers taken apart (docs/PLAIN_PAGE.md).

A strategy is often wrapped: `Retune` picks numbers every month, `Overlaid` adds overlays on the
weights, `RegimeGate` trades in some market states only, `SplitCadence` splits the entry and exit
rhythm. The page reads the layers one by one and the strategy inside them (the CORE) separately:
the core says what the strategy holds, the layers say how its numbers are chosen and when it is
allowed to trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qlab.pipeline.sources import QLAB_ROOT
from qlab.plain.texts import ref_name

WRAPPERS: dict[str, str] = {
    "Retune": "base_params",
    "Overlaid": "inner_params",
    "RegimeGate": "inner_params",
    "SplitCadence": "inner_params",
}
"""Wrapper class -> the key of its params that holds the wrapped strategy's params. The key of the
wrapped strategy's `code_ref` is `inner_code_ref` in all of them."""

COMPOSED = "Composed"
"""The class of a strategy composed of blocks (`qlab.strategies.blocks`); its params are nested."""

_IDEA_LINE = re.compile(r"^idea_id\s*:\s*[\"']?([^\s\"'#]+)", re.M)
_MECHANISM_LINE = re.compile(r"^#\s*mechanism:\s*(\S+)\s*$", re.M)


@dataclass(frozen=True)
class Layer:
    """One wrapper around a strategy: its kind (the class name) and its own params."""

    kind: str
    own: dict[str, Any]


@dataclass(frozen=True)
class Unwrapped:
    layers: tuple[Layer, ...]
    core_ref: str
    core: dict[str, Any]

    @property
    def core_name(self) -> str:
        return ref_name(self.core_ref)

    def layer(self, kind: str) -> list[Layer]:
        return [layer for layer in self.layers if layer.kind == kind]


def unwrap(code_ref: str, params: dict[str, Any] | None) -> Unwrapped:
    """Take the wrappers off, outermost first. A wrapper whose params do not name the strategy
    inside it is the core itself (the page would rather show it than fail)."""
    layers: list[Layer] = []
    current = dict(params or {})
    ref = code_ref
    while True:
        name = ref_name(ref)
        key = WRAPPERS.get(name)
        if key is None or "inner_code_ref" not in current:
            return Unwrapped(tuple(layers), ref, current)
        layers.append(Layer(name, {k: v for k, v in current.items()
                                   if k not in ("inner_code_ref", key)}))
        ref = str(current["inner_code_ref"])
        inner = current.get(key)
        current = dict(inner) if isinstance(inner, dict) else {}


def core_items(unwrapped: Unwrapped) -> dict[str, Any]:
    """The core's params as `dotted path -> value`. A composed strategy's blocks are flattened
    (`signal.lookback_days`); everything else stays one level, a dict value being one value."""
    if unwrapped.core_name != COMPOSED:
        return dict(unwrapped.core)
    out: dict[str, Any] = {}
    for key, value in unwrapped.core.items():
        if isinstance(value, dict):
            for sub, item in value.items():
                out[f"{key}.{sub}"] = item
        else:
            out[key] = value
    return out


def _plain_date(value: Any) -> Any:
    return value.isoformat() if isinstance(value, date) else value


@dataclass(frozen=True)
class SpecView:
    """What the page reads of a spec."""

    idea_id: str
    title: str
    code_ref: str
    params: dict[str, Any]
    data: dict[str, Any]
    costs: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
    """The other top-level fields of the spec file (`choice`, `params_fixed_at`, `judged_from`,
    `strategy_capital_usd`, `plain_summary`, ...). Empty when the view is the registry's row."""
    mechanism: str | None = None
    """The `# mechanism:` header of a generated spec file."""
    path: str | None = None
    """The spec file, relative to the project root; None for a registry row."""

    @property
    def unwrapped(self) -> Unwrapped:
        return unwrap(self.code_ref, self.params)


def view_from_raw(raw: dict[str, Any], *, text: str = "", path: str | None = None) -> SpecView:
    data = {k: _plain_date(v) for k, v in (raw.get("data") or {}).items()}
    known = {"idea_id", "title", "code_ref", "params", "data", "costs"}
    extras = {k: _plain_date(v) for k, v in raw.items() if k not in known}
    header = _MECHANISM_LINE.search(text)
    return SpecView(
        idea_id=str(raw.get("idea_id", "")), title=str(raw.get("title", "")),
        code_ref=str(raw.get("code_ref", "")), params=dict(raw.get("params") or {}),
        data=data, costs=dict(raw.get("costs") or {}), extras=extras,
        mechanism=header.group(1) if header else None, path=path)


def view_from_row(row: Any, title: str = "") -> SpecView:
    """A registry `Spec` row: what ran, without the fields the file adds."""
    data = {k: _plain_date(v) for k, v in (row.data_requirements or {}).items()}
    return SpecView(idea_id=row.idea_id, title=title, code_ref=row.code_ref,
                    params=dict(row.params or {}), data=data, costs=dict(row.costs_model or {}))


def load_view(path: Path, root: Path | None = None) -> SpecView | None:
    """Read one spec file; None if it is not a spec (a broken file must not break a page)."""
    base = root if root is not None else QLAB_ROOT
    full = path if path.is_absolute() else base / path
    try:
        text = full.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict) or "idea_id" not in raw or "code_ref" not in raw:
        return None
    try:
        rel = str(full.relative_to(base))
    except ValueError:
        rel = str(full)
    return view_from_raw(raw, text=text, path=rel)


LAYER_KEYS = frozenset({"overlays", "trade_in"})
"""Override keys that name a layer (overlays, market states), not a parameter of the strategy."""


def candidate_values(view: SpecView) -> dict[str, list[Any]]:
    """Every parameter the candidates code chooses among change (the grids of the `Retune`
    layers, the spec's `choice`) -> the distinct values the candidates give it, in order. The
    parameters that name a layer (`LAYER_KEYS`) are left out."""
    found: dict[str, list[Any]] = {}

    def add(key: str, value: Any) -> None:
        values = found.setdefault(key, [])
        if value not in values:
            values.append(value)

    for layer in view.unwrapped.layer("Retune"):
        for candidates in (layer.own.get("grid") or {}).values():
            for overrides in (candidates or {}).values():
                for key, value in (overrides or {}).items():
                    if key not in LAYER_KEYS:
                        add(key, value)
    choice = view.extras.get("choice")
    if isinstance(choice, dict):
        for overrides in (choice.get("candidates") or {}).values():
            for key, value in (overrides or {}).items():
                add(key, value)
        for key, values in (choice.get("axes") or {}).items():
            for value in values:
                add(key, value)
    return found


_INDEX: dict[tuple[str, tuple[tuple[str, int], ...]], dict[str, list[Path]]] = {}


def spec_files(root: Path | None = None) -> dict[str, list[Path]]:
    """Idea id -> the spec files that name it, relative to the project root. The ids are read
    from the files' text (the page must not parse 230 specs); the index is rebuilt when a file
    changes."""
    base = root if root is not None else QLAB_ROOT
    folder = base / "specs"
    if not folder.is_dir():
        return {}
    paths = sorted(folder.rglob("*.yaml"))
    key = (str(base), tuple((str(p), p.stat().st_mtime_ns) for p in paths))
    if key in _INDEX:
        return _INDEX[key]
    index: dict[str, list[Path]] = {}
    for path in paths:
        try:
            head = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found = _IDEA_LINE.search(head)
        if found:
            index.setdefault(found.group(1), []).append(path.relative_to(base))
    _INDEX.clear()
    _INDEX[key] = index
    return index


def view_for_idea(idea_id: str, row: Any | None, *, title: str = "",
                  root: Path | None = None) -> SpecView | None:
    """The spec of an idea. Of several spec files the one whose code is the registry row's wins
    (the row is what ran), then the one that states `params_fixed_at`, then the last by name; no
    file: the registry row; no row either: None."""
    candidates = [v for p in spec_files(root).get(idea_id, ()) if (v := load_view(p, root))]
    if candidates:
        def rank(view: SpecView) -> tuple[bool, bool, bool, str]:
            same_row = row is not None and (view.code_ref == row.code_ref)
            same_params = same_row and view.params == (row.params or {})
            return (same_params, same_row, "params_fixed_at" in view.extras, view.path or "")

        return max(candidates, key=rank)
    return view_from_row(row, title) if row is not None else None
