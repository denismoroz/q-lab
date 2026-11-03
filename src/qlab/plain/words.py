"""Turning parameters and values into plain Russian, through the dictionary
(`strategies/plain/params.yaml`).

A parameter the dictionary does not know is never hidden: it reads «параметр X: было A, стало B»
(`Words.change`), and `unknown_keys` lets a test name every such parameter of the specs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qlab.plain.texts import Dictionary

MISSING = object()
"""A parameter the spec does not state (as opposed to one stated as null)."""

VENUE_NAMES = {
    "binance": "Binance",
    "hyperliquid": "Hyperliquid",
    "hyperliquid-xyz": "раздел xyz на Hyperliquid (акции, индексы, сырьё)",
}
_LIST_SHOWN = 8


def num(value: float | int) -> str:
    """`1.0` -> `1`, `0.0525` -> `0.0525`, `5.25` -> `5.25`."""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def percent(share: float) -> str:
    """`0.0525` -> `5.25%`: a share written as the percent the owner reads."""
    return f"{num(round(share * 100, 6))}%"


def money(value: float | int) -> str:
    return f"${num(value)}"


def venue_name(source: str | None) -> str:
    return VENUE_NAMES.get(source or "", source or "не названа")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) < 1e-12
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


same = _same


class Words:
    """The dictionary, with the phrases built on it."""

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def entry(self, key: str) -> dict[str, Any] | None:
        return self.dictionary.params.get(key)

    def known(self, key: str) -> bool:
        return key in self.dictionary.params

    def unknown_keys(self, keys: Iterable[str]) -> list[str]:
        return sorted({k for k in keys if not self.known(k)})

    def open_list(self, key: str, value: Any) -> bool:
        """A number the spec leaves open: a list of two or more numbers under a parameter that
        is a single number (`short_window_days: [3, 7, 14, 30]`, docs/OWN_NUMBERS.md). A
        parameter that is a list by nature (`lookbacks`, `coins`) is not."""
        if (self.entry(key) or {}).get("list_valued"):
            return False
        return (isinstance(value, list) and len(value) >= 2
                and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value))

    def default(self, key: str) -> Any:
        entry = self.entry(key)
        return entry["default"] if entry and "default" in entry else MISSING

    # ---- values -------------------------------------------------------------------------

    def states(self, value: Any) -> str:
        names = self.dictionary.layers.get("regime", {}).get("states", {})
        return ", ".join(names.get(str(v), str(v)) for v in value)

    def value(self, key: str, value: Any) -> str:
        """A value as the owner reads it. A bool with words in the dictionary reads as those
        words; a number keeps the unit the dictionary gives."""
        if value is MISSING:
            return "не задано"
        entry = self.entry(key) or {}
        if value is None:
            return str(entry.get("none_means", "не задано"))
        if isinstance(value, bool):
            return "да" if value else "нет"
        if isinstance(value, int | float):
            unit = entry.get("unit")
            return f"{num(value)} {unit}" if unit else num(value)
        if key.endswith("trade_in") and isinstance(value, list):
            return self.states(value)
        if isinstance(value, list | tuple):
            items = [self.value(key, v) if not isinstance(v, str) else v for v in value]
            if len(items) > _LIST_SHOWN:
                return f"{', '.join(items[:6])} … (всего {len(items)})"
            return ", ".join(items)
        if isinstance(value, dict):
            items = [f"{k} {self.value(key, v)}" for k, v in value.items()]
            if len(items) > _LIST_SHOWN:
                return f"{', '.join(items[:6])} … (всего {len(items)})"
            return ", ".join(items)
        return str(value)

    def choices(self, key: str, values: list[Any]) -> str:
        """Several values of one parameter, to choose among. A value that is itself a list is
        one choice and stands in parentheses: «(рост, боковик, падение); (рост, падение)»."""
        if any(isinstance(v, list | tuple) for v in values):
            return "; ".join(f"({self.value(key, v)})" for v in values)
        return ", ".join(self.value(key, v) for v in values)

    def label(self, key: str, value: Any) -> str | None:
        """The dictionary's short name of a value (`short_true`, or `tag` plus the number), None
        when it has none: then the variant is explained, not named."""
        entry = self.entry(key) or {}
        if isinstance(value, bool):
            return entry.get("short_true" if value else "short_false")
        if "tag" in entry and isinstance(value, int | float):
            return f"{entry['tag']} {num(value)}"
        return None

    def name(self, key: str, value: Any) -> str:
        """A short name for one value of a parameter, to call a variant by: «общий залог»,
        «запас залога под страховку 1.5»."""
        entry = self.entry(key) or {}
        if isinstance(value, bool):
            short = entry.get("short_true" if value else "short_false")
            if short:
                return str(short)
            clause = entry.get("when_true" if value else "when_false")
            if clause:
                return str(clause)
            return f"{entry.get('what', f'параметр {key}')}: {'да' if value else 'нет'}"
        what = entry.get("what", f"параметр {key}")
        return f"{what.split(' (')[0]} {self.value(key, value)}"

    def describe(self, key: str, value: Any) -> str:
        """One value of a parameter as a statement: the clause of a yes/no parameter, else
        «<what>: <value>»."""
        entry = self.entry(key) or {}
        if isinstance(value, bool):
            clause = entry.get("when_true" if value else "when_false")
            if clause:
                return str(clause)
        return f"{entry.get('what', f'параметр {key}')}: {self.value(key, value)}"

    def tag(self, key: str, value: Any) -> str | None:
        """The shortest name of a changed parameter, for the plain title (see `label`)."""
        return self.label(key, value)

    # ---- changes ------------------------------------------------------------------------

    def change(self, key: str, old: Any, new: Any) -> str:
        """How one parameter differs: a clause when the dictionary has one for the new value of
        a yes/no parameter, else «<what>: было A, стало B»."""
        entry = self.entry(key)
        if old is MISSING:
            old = self.default(key)
        if entry and isinstance(new, bool):
            clause = entry.get("when_true" if new else "when_false")
            if clause:
                return str(clause)
        if (isinstance(old, list) and isinstance(new, list)
                and all(isinstance(v, str) for v in (*old, *new))):
            added = [v for v in new if v not in old]
            removed = [v for v in old if v not in new]
            what = entry["what"] if entry else f"параметр {key}"
            parts = ([f"добавлено: {', '.join(added)}"] if added else []) \
                + ([f"убрано: {', '.join(removed)}"] if removed else [])
            return f"{what}: {'; '.join(parts)}"
        what = entry["what"] if entry else f"параметр {key}"
        if new is MISSING:
            return f"{what}: не задаётся (было {self.value(key, old)})"
        if old is MISSING:
            return f"{what}: теперь {self.value(key, new)} (раньше не задавалось)"
        return f"{what}: было {self.value(key, old)}, стало {self.value(key, new)}"
