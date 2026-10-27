"""The optional fields of a card that comes from the ideas inbox (cards/SCHEMA.md, the section
«Карточка из входящих идей»; docs/IMPLEMENTER.md, «Из идеи в блок и в проверку за одну ночь»).

A card written by hand has none of them and is read exactly as before. A card the carder agent
writes from an owner's idea carries more things, so that code can take the idea from words
to a verdict in one night without asking anybody:

- `baseline_spec` -- the strategy the idea is compared with (a path to a spec file);
- `grid` -- the variants the idea is judged over: `params` (a parameter path -> a list of
  values; the variants are the product) or `candidates` (named overrides). The first variant
  is the one the written spec carries;
- `new_block` -- the block of code the idea needs and does not exist: its name, its parameters
  with what each means, and the rule in plain words;
- `data` -- the venue, the interval, the coins (a list) or the rule that picks them, and
  `honest_from`, the first day the data honestly exists;
- `readings` (optional) -- when the idea's words allow several readings, one plain sentence per
  reading. Each reading is a variant of the grid; the block must serve all of them.

This module only reads and checks the shape of those fields. It writes nothing, calls no agent
and decides nothing: a card with a broken field is refused with every problem named, never
guessed at."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from qlab.pipeline.sources import QLAB_ROOT

BLOCK_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
COIN = re.compile(r"^[A-Za-z0-9]+(-SPOT)?$")


class CardFieldError(ValueError):
    """A card field is present and wrong; the message names every problem."""


@dataclass(frozen=True)
class BlockParam:
    name: str
    meaning: str


@dataclass(frozen=True)
class NewBlock:
    name: str
    params: tuple[BlockParam, ...]
    rule: str

    @property
    def module(self) -> str:
        return re.sub(r"[^a-z0-9_]", "_", self.name.lower())


@dataclass(frozen=True)
class CardGrid:
    axes: Mapping[str, tuple] = field(default_factory=dict)
    """A parameter path -> its list of values (the variants are the product of the lists)."""
    candidates: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    """A candidate's name -> {parameter path: value}."""

    @property
    def count(self) -> int:
        if self.candidates:
            return len(self.candidates)
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    @property
    def paths(self) -> list[str]:
        if self.candidates:
            return sorted({p for overrides in self.candidates.values() for p in overrides})
        return list(self.axes)

    def first_variant(self) -> dict[str, object]:
        """The values the written spec's params carry: the first of every list, or the first
        named candidate's overrides."""
        if self.candidates:
            return dict(next(iter(self.candidates.values())))
        return {path: values[0] for path, values in self.axes.items()}


@dataclass(frozen=True)
class CardData:
    venue: str
    interval: str | None = None
    coins: tuple[str, ...] = ()
    universe_rule: str | None = None
    honest_from: date | None = None


@dataclass(frozen=True)
class CardFields:
    card: Path
    idea_id: str
    title: str
    idea_text: str
    found_at: date | None = None
    baseline_spec: Path | None = None
    grid: CardGrid | None = None
    new_block: NewBlock | None = None
    data: CardData | None = None
    readings: tuple[str, ...] = ()

    @property
    def inbox(self) -> bool:
        """The card carries at least one of the inbox fields."""
        return any(x is not None for x in (self.baseline_spec, self.grid, self.new_block,
                                           self.data))


def param_at(params: Mapping, path: str) -> tuple[bool, object]:
    """(whether the dotted path exists in `params`, its value): `signal.window_days` reads
    `params["signal"]["window_days"]`."""
    node: object = params
    for name in path.split("."):
        if not isinstance(node, Mapping) or name not in node:
            return False, None
        node = node[name]
    return True, node


def _day(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value  # type: ignore[return-value]
    return date.fromisoformat(str(value))


def _grid(raw: object, problems: list[str]) -> CardGrid | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        problems.append("grid: must be a mapping with `params` or `candidates`")
        return None
    if "params" in raw or "candidates" in raw:
        axes_raw, cands_raw = raw.get("params"), raw.get("candidates")
        extra = set(raw) - {"params", "candidates"}
        if extra:
            problems.append(f"grid: unknown keys {sorted(extra)}")
    else:  # the short form: the lists directly under `grid`
        axes_raw, cands_raw = raw, None
    if (axes_raw is None) == (cands_raw is None):
        problems.append("grid: give either `params` (lists) or `candidates` (named), not both "
                        "and not neither")
        return None
    if axes_raw is not None:
        if not isinstance(axes_raw, Mapping) or not axes_raw:
            problems.append("grid.params: must be a mapping of a parameter path to a list")
            return None
        bad = [p for p, v in axes_raw.items() if not isinstance(v, list) or not v]
        if bad:
            problems.append(f"grid.params: {', '.join(map(str, bad))} must be non-empty lists")
            return None
        grid = CardGrid(axes={str(p): tuple(v) for p, v in axes_raw.items()})
    else:
        if not isinstance(cands_raw, Mapping) or not cands_raw or any(
                not isinstance(v, Mapping) or not v for v in cands_raw.values()):
            problems.append("grid.candidates: must be a mapping of a name to its overrides "
                            "(a parameter path -> a value)")
            return None
        grid = CardGrid(candidates={str(n): {str(p): v for p, v in o.items()}
                                    for n, o in cands_raw.items()})
    if grid.count < 2:
        problems.append("grid: needs at least two variants to choose between")
    return grid


def _block(raw: object, problems: list[str]) -> NewBlock | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        problems.append("new_block: must be a mapping {name, params, rule}")
        return None
    name = str(raw.get("name") or "").strip()
    if not BLOCK_NAME.match(name):
        problems.append(f"new_block.name: {name!r} is not a slug (a-z, 0-9, _ and -)")
    rule = str(raw.get("rule") or "").strip()
    if len(rule) < 20:
        problems.append("new_block.rule: say the rule in plain words (a sentence at least)")
    params_raw = raw.get("params")
    params: list[BlockParam] = []
    if isinstance(params_raw, Mapping):
        params_raw = [{"name": k, "meaning": v} for k, v in params_raw.items()]
    if not isinstance(params_raw, list) or not params_raw:
        problems.append("new_block.params: list every parameter with what it means")
    else:
        for item in params_raw:
            if not isinstance(item, Mapping) or not item.get("name") or not item.get("meaning"):
                problems.append(f"new_block.params: {item!r} needs a name and a meaning")
                continue
            params.append(BlockParam(str(item["name"]), str(item["meaning"]).strip()))
    return NewBlock(name, tuple(params), rule)


def _data(raw: object, problems: list[str]) -> CardData | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        problems.append("data: must be a mapping {venue, interval, coins, honest_from}")
        return None
    venue = str(raw.get("venue") or "").strip().lower()
    if not venue:
        problems.append("data.venue: required")
    coins = raw.get("coins") or []
    if not isinstance(coins, list) or any(not COIN.match(str(c)) for c in coins):
        problems.append("data.coins: must be a list of tickers (BTC, ETH, ...)")
        coins = []
    try:
        honest = _day(raw.get("honest_from"))
    except ValueError:
        problems.append(f"data.honest_from: {raw.get('honest_from')!r} is not a date")
        honest = None
    rule = raw.get("universe_rule")
    if not coins and not rule:
        problems.append("data: give the coins (a list) or the rule that picks them "
                        "(universe_rule)")
    interval = raw.get("interval")
    return CardData(venue=venue, interval=None if interval is None else str(interval),
                    coins=tuple(str(c) for c in coins),
                    universe_rule=None if rule is None else str(rule).strip(), honest_from=honest)


def read_card_fields(card: Path | str, root: Path | None = None) -> CardFields:
    """The card's identity and its inbox fields. A path relative to the project is read from
    `root` (the project's folder by default). Raises `CardFieldError` with every problem."""
    root = QLAB_ROOT if root is None else root
    card = Path(card)
    path = card if card.is_absolute() else root / card
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or "idea_id" not in raw:
        raise CardFieldError(f"{card}: no idea_id")
    problems: list[str] = []
    grid = _grid(raw.get("grid"), problems)
    block = _block(raw.get("new_block"), problems)
    data = _data(raw.get("data"), problems)
    readings_raw = raw.get("readings")
    readings: tuple[str, ...] = ()
    if readings_raw is not None:
        if not isinstance(readings_raw, list) or any(not str(r).strip() for r in readings_raw):
            problems.append("readings: must be a list of sentences, one per reading of the idea")
        else:
            readings = tuple(" ".join(str(r).split()) for r in readings_raw)
    baseline = raw.get("baseline_spec")
    baseline_path = None
    if baseline is not None:
        baseline_path = Path(str(baseline))
        if not (baseline_path if baseline_path.is_absolute() else root / baseline_path).is_file():
            problems.append(f"baseline_spec: {baseline} is not a file")
    try:
        found = _day(raw.get("found_at"))
    except ValueError:
        found = None
    if problems:
        raise CardFieldError(f"{card}: " + "; ".join(problems))
    text = raw.get("idea_text") or raw.get("plain_summary") or raw.get("title") or ""
    return CardFields(
        card=card, idea_id=str(raw["idea_id"]), title=str(raw.get("title") or raw["idea_id"]),
        idea_text=" ".join(str(text).split()), found_at=found, baseline_spec=baseline_path,
        grid=grid, new_block=block, data=data, readings=readings)


__all__ = ["BlockParam", "CardData", "CardFieldError", "CardFields", "CardGrid", "NewBlock",
           "param_at", "read_card_fields"]
