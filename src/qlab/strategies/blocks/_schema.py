"""Block registry plumbing: what a block is and how a spec fragment is checked.

A block is a pure function plus the list of parameters it takes. Each kind of
block (universe, signal, portfolio, sizing) keeps a dict `{name: Block}`; adding
a block is one function plus one dict entry. A spec fragment such as
`{block: momentum, lookback_days: 30}` is checked here before anything runs: a
missing key, an unknown key or an unknown block name raises `ValueError` that
names the offender. There are no defaults, so nothing is guessed silently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real


def positive_number(label: str, value: object) -> float:
    """A real number greater than zero (booleans are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return float(value)


def non_negative_number(label: str, value: object) -> float:
    """A real number greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return float(value)


def positive_int(label: str, value: object) -> int:
    """An integer of at least 1 (a whole-valued float such as 5.0 is accepted)."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value != int(value):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Block:
    """A block function and the validators of its keyword parameters."""

    fn: Callable
    params: Mapping[str, Callable[[str, object], float | int]]


def parse_block(
    kind: str, raw: object, registry: Mapping[str, Block], selector: str
) -> tuple[Callable, dict[str, float | int]]:
    """Check one spec fragment against `registry`; return `(fn, validated kwargs)`.

    `selector` is the key that names the block (`block`, or `rule` for the
    universe). Raises `ValueError` on anything that is not exactly the block's
    parameter list.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{kind} must be a mapping with a '{selector}' key, got {raw!r}")
    if selector not in raw:
        raise ValueError(f"{kind}: missing key '{selector}'")
    name = raw[selector]
    if name not in registry:
        raise ValueError(f"{kind}: unknown {selector} {name!r}; known: {sorted(registry)}")
    block = registry[name]
    given = set(raw) - {selector}
    missing = sorted(set(block.params) - given)
    if missing:
        raise ValueError(f"{kind} '{name}': missing param(s) {missing}")
    unknown = sorted(given - set(block.params), key=str)
    if unknown:
        raise ValueError(f"{kind} '{name}': unknown param(s) {unknown}")
    kwargs = {key: check(f"{kind}.{key}", raw[key]) for key, check in block.params.items()}
    return block.fn, kwargs
