"""No silent narrowing of a strategy's coin list (docs/COIN_SCOPE.md).

Owner, 2026-11-04, «да, делай оба»: the researcher narrowed Bv2's four coins to BTC and ETH in the
first Bv3 experiment because it based the experiment on its own earlier two-coin experiment; the
owner had never asked for two coins. A new experiment whose coins are a strict part of its
parent's coins needs a source for that:

- the owner's word, quoted in `sources["coins"]`;
- a declared choice (`choice.candidates` overriding the coin list), or
- a rule of code that picks the coins (`params.universe`, for example the first K by traded value).

Otherwise the spec is refused with a reason in Russian that the console shows
(`problems`, `refuse`). The parent is the production spec of the experiment's family
(`night/coin_scope.yaml`). Pure but for reading that file and the parent spec; it is called where a
NEW agent spec is accepted (researcher, generator, implementer) and never touches a spec that
already exists in the registry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from qlab.pipeline.spec import StrategySpec

CONFIG = Path("night/coin_scope.yaml")
SPOT_SUFFIX = "-SPOT"
COIN_PARAMS = ("coins", "universe")
"""Parameters that name the coins a book trades (`qlab.twin.COIN_LIST_PARAMS`)."""
_QUOTE = re.compile(r"[«\"“][^»\"”]{6,}[»\"”]")
_OWNER = re.compile(r"владел|owner", re.IGNORECASE)


@dataclass(frozen=True)
class CoinList:
    """The coins a spec trades and where it says so."""

    coins: frozenset[str] | None
    """None when the spec names no list of coins (a rule chooses them, or none is named)."""
    where: str
    by_code: bool = False
    """The coins are chosen by a rule of code (`params.universe` as a mapping)."""


def _tickers(value: object) -> list[str] | None:
    if (isinstance(value, list) and value and all(
            isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9]+", v) for v in value)):
        return [v.upper() for v in value]
    return None


def coin_list(spec: StrategySpec) -> CoinList:
    """The coin list a spec trades, read the way `qlab.twin` reads a coin list: a list in the
    params first, then the instruments of its data, then its required instruments."""
    for name in COIN_PARAMS:
        value = spec.params.get(name)
        if isinstance(value, Mapping):
            return CoinList(None, f"params.{name}", by_code=value.get("rule") is not None)
        if (coins := _tickers(value)) is not None:
            return CoinList(frozenset(coins), f"params.{name}")
    for where, names in (("data.instruments", spec.data.instruments),
                         ("required_instruments", spec.required_instruments)):
        if names:
            return CoinList(frozenset(n.removesuffix(SPOT_SUFFIX).upper() for n in names), where)
    return CoinList(None, "")


def declared_choice(spec: StrategySpec) -> bool:
    """Whether the spec declares the coin list as one of the things code chooses among: a named
    variant of `choice.candidates` that overrides a coin list, or an axis of a coin rule."""
    choice = spec.choice
    if choice is None:
        return False
    if any(path.split(".")[0] in COIN_PARAMS for path in choice.axes):
        return True
    return any(key.split(".")[0] in COIN_PARAMS or key == "data.instruments"
               for overrides in choice.candidates.values() for key in overrides)


def owner_word(spec: StrategySpec) -> bool:
    """Whether `sources["coins"]` quotes the owner: a quotation in «» and the word owner/владелец.
    A source that quotes a paper, a report or the agent's own memo is not the owner's word."""
    cited = spec.sources.get("coins", "")
    return bool(_QUOTE.search(cited)) and bool(_OWNER.search(cited))


def load_config(path: Path = CONFIG, root: Path | None = None) -> dict:
    full = path if root is None or path.is_absolute() else root / path
    return yaml.safe_load(full.read_text(encoding="utf-8")) or {}


def parent_path(spec: StrategySpec, config: Mapping) -> str | None:
    """The production spec of the family the experiment belongs to, if it has one: found by the
    idea id's prefix, else by the strategy module the spec runs."""
    from qlab.registry.families import strategy_key

    families = config.get("families") or {}
    for row in families.values():
        if any(spec.idea_id.startswith(p) for p in row.get("prefixes", [])):
            return str(row["parent"])
    key = strategy_key(spec.code_ref, spec.params)
    row = families.get(key) if key else None
    return None if row is None else str(row["parent"])


def narrowing(spec: StrategySpec, parent: StrategySpec
              ) -> tuple[frozenset[str], frozenset[str]] | None:
    """(the parent's coins, the spec's coins) when the spec's list is a strict part of the
    parent's, else None."""
    own, theirs = coin_list(spec), coin_list(parent)
    if own.coins is None or theirs.coins is None:
        return None
    if own.coins < theirs.coins:
        return theirs.coins, own.coins
    return None


def reason(spec: StrategySpec, parent: StrategySpec, parent_where: str,
           kept: frozenset[str], whole: frozenset[str]) -> str:
    """The refusal, in Russian, for the console."""
    gone = ", ".join(sorted(whole - kept))
    return (
        f"список монет сужен без источника: у родителя ({parent_where}) монет {len(whole)} — "
        f"{', '.join(sorted(whole))}, а в этом опыте {len(kept)} — {', '.join(sorted(kept))} "
        f"(убраны: {gone}). Монеты стратегии молча не сужают: нужно слово владельца "
        "(цитата в «» и слово «владелец» в sources[\"coins\"]), или список монет как объявленный "
        "выбор (choice), или правило кода, например «каждый день первые K монет по обороту» "
        "(params.universe, K объявлен выбором).")


def problems(spec: StrategySpec, *, root: Path | None = None, config: Mapping | None = None,
             parent: StrategySpec | None = None) -> list[str]:
    """Why the spec's coin list may not be accepted; empty when it may. `parent` is given by
    tests; otherwise it is the family's production spec (`night/coin_scope.yaml`). A spec that
    is its own parent, has no family, or whose parent cannot be read has nothing to compare."""
    from qlab.pipeline.spec import load_spec

    where = ""
    if parent is None:
        try:
            cfg = config if config is not None else load_config(CONFIG, root)
        except OSError:  # a checkout without the file names no parents
            return []
        where = parent_path(spec, cfg) or ""
        if not where:
            return []
        full = Path(where) if root is None or Path(where).is_absolute() else root / where
        try:
            parent = load_spec(full)
        except Exception:  # noqa: BLE001 - no readable parent, nothing to compare with
            return []
    if parent.idea_id == spec.idea_id:
        return []
    found = narrowing(spec, parent)
    if found is None:
        return []
    if coin_list(spec).by_code or declared_choice(spec) or owner_word(spec):
        return []
    whole, kept = found
    return [reason(spec, parent, where or parent.idea_id, kept, whole)]


def refuse(spec: StrategySpec, spec_path: str, *, root: Path | None = None,
           ledger_path: Path | None = None) -> list[str]:
    """`problems`, and when there are some, the refusal is recorded in the ledger the ladder and
    the console read (`qlab.one_check.note_refusal`, data/twins/ledger.json)."""
    from qlab import one_check

    found = problems(spec, root=root)
    if found:
        one_check.note_refusal(spec.idea_id, "; ".join(found), spec_path, ledger_path=ledger_path)
    return found


def narrowed_without_source(specs: Mapping[str, StrategySpec], *, root: Path | None = None
                            ) -> dict[str, list[str]]:
    """The existing specs (path -> spec) that narrow their family's coin list with no source:
    a listing for the owner, nothing is changed by it."""
    return {path: found for path, spec in specs.items()
            if (found := problems(spec, root=root))}


__all__ = ["CONFIG", "CoinList", "coin_list", "declared_choice", "narrowed_without_source",
           "narrowing", "owner_word", "parent_path", "problems", "refuse"]
