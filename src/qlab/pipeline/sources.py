"""Every number and choice in a spec cites where it comes from
(docs/PLAN.md stage 5, docs/SOURCES.md).

CLAUDE.md: never invent a number. Until now a spec's provenance lived in YAML
comments, which nothing checks; a spec written by an agent from a card could
carry a number from nowhere. A spec now has `sources`: dotted paths ->
citations. Before a run starts:

- **coverage**: every leaf of `params` (not None), both `costs` and
  `min_leg_notional` must be covered by a citation at its own path or at an
  ancestor's (`base_params` covers everything under it);
- **a citation must name something checkable**: an existing file (q-lab's or
  frab's), a URL, an owner's decision with its date ("владелец 2026-10-01" /
  "owner 2026-10-01"), or a formula `derived: <arithmetic>; <why>`;
- **a cited file must exist**;
- **a number must literally appear in the cited file(s)** -- the tripwire
  against an invented value; a `derived:` formula must evaluate to exactly
  the value it covers. Booleans and strings need a citation but are not
  matched literally (column names and switches rarely appear verbatim).

The literal check is a tripwire, not proof: a small integer like 3 appears in
almost any file. What it reliably catches is the number that exists nowhere
but in the spec.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path

from qlab.pipeline.spec import StrategySpec

QLAB_ROOT = Path(__file__).resolve().parents[3]
FRAB_ROOT = QLAB_ROOT.parent / "funding-rate-arbitrage"
FRAB_PREFIX = "funding-rate-arbitrage/"

_PATH_TOKEN = re.compile(r"[\w./-]+\.(?:py|md|yaml|yml|json|csv|txt)\b")
_URL = re.compile(r"https?://\S+")
_OWNER = re.compile(r"(?:владелец|владельца|owner)\W+(?:\w+\W+)?\d{4}-\d{2}-\d{2}", re.I)
_DERIVED = re.compile(r"^\s*derived:\s*([^;]+)")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        ast.Pow: operator.pow}


class SpecSourceError(ValueError):
    """A spec carries a number or choice without a checkable source."""


def _leaves(prefix: str, value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(f"{prefix}.{key}" if prefix else str(key), item)
    else:
        yield prefix, value


def spec_leaves(spec: StrategySpec) -> list[tuple[str, object]]:
    """(dotted path, value) for every leaf that needs a source; lists are one
    leaf, None needs none."""
    leaves = list(_leaves("", spec.params))
    leaves += [("costs.taker_fee_bps", spec.costs.taker_fee_bps),
               ("costs.slippage_bps", spec.costs.slippage_bps),
               ("min_leg_notional", spec.min_leg_notional)]
    return [(path, value) for path, value in leaves if value is not None]


def covering_key(path: str, sources: Mapping[str, str]) -> str | None:
    """The most specific source key covering `path` (itself or an ancestor)."""
    best = None
    for key in sources:
        if (path == key or path.startswith(key + ".")) and (best is None or len(key) > len(best)):
            best = key
    return best


def _resolve(token: str) -> Path | None:
    candidates = []
    if token.startswith(FRAB_PREFIX):
        candidates.append(FRAB_ROOT / token[len(FRAB_PREFIX):])
    else:
        candidates += [QLAB_ROOT / token, FRAB_ROOT / token]
    for path in candidates:
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=256)
def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_eval(expr: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "round" and len(node.args) == 1 and not node.keywords):
            return float(round(ev(node.args[0])))
        raise ValueError(f"not plain arithmetic: {expr!r}")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _numbers(value: object) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        return [n for item in value for n in _numbers(item)]
    return []


def _appears(number: float, text: str) -> bool:
    forms = {repr(number), f"{number:g}"}
    if number == int(number):
        forms.add(str(int(number)))
    if abs(number) < 1:
        forms.add(f"{number * 100:g}%")
    for form in forms:
        tail = r"0*(?![\d])" if "." in form else r"(?![\d])"
        if re.search(rf"(?<![\d.]){re.escape(form)}{tail}", text):
            return True
    return False


def source_problems(spec: StrategySpec) -> list[str]:
    """Everything wrong with `spec`'s sources; empty means the run may start."""
    problems: list[str] = []
    sources = spec.sources
    checked: dict[str, tuple[list[Path], bool]] = {}
    for key, citation in sources.items():
        if not citation or not citation.strip():
            problems.append(f"source for {key!r} is empty")
            continue
        derived = _DERIVED.match(citation)
        files, missing = [], []
        for token in _PATH_TOKEN.findall(citation):
            resolved = _resolve(token)
            (files if resolved else missing).append(resolved or token)
        problems += [f"source for {key!r} cites a file that does not exist: {m}" for m in missing]
        if not (derived or files or _URL.search(citation) or _OWNER.search(citation)):
            problems.append(
                f"source for {key!r} names nothing checkable (a file, a URL, an owner's dated "
                f"decision, or 'derived: <formula>'): {citation!r}"
            )
        checked[key] = (files, bool(derived))

    for path, value in spec_leaves(spec):
        key = covering_key(path, sources)
        if key is None:
            problems.append(f"no source for {path} = {value!r}")
            continue
        files, is_derived = checked.get(key, ([], False))
        numbers = _numbers(value)
        if is_derived:
            if key != path or len(numbers) != 1:
                problems.append(f"'derived:' must cover exactly one number; {key!r} covers {path}")
                continue
            try:
                result = _safe_eval(_DERIVED.match(sources[key]).group(1))
            except (ValueError, SyntaxError, ZeroDivisionError) as exc:
                problems.append(f"source for {key!r}: {exc}")
                continue
            if abs(result - numbers[0]) > 1e-9 * max(1.0, abs(numbers[0])):
                problems.append(f"source for {key!r} derives {result:g}, not {numbers[0]:g}")
            continue
        if files:
            text = "\n".join(_text(f) for f in files)
            absent = [n for n in numbers if not _appears(n, text)]
            if absent:
                names = ", ".join(str(f.relative_to(QLAB_ROOT.parent)) for f in files)
                problems.append(
                    f"{path} = {value!r}: {', '.join(f'{n:g}' for n in absent)} does not appear "
                    f"in the cited {names}"
                )
    return problems


def require_sources(spec: StrategySpec) -> None:
    problems = source_problems(spec)
    if problems:
        raise SpecSourceError(
            f"spec {spec.idea_id!r} cites no checkable source for:\n  " + "\n  ".join(problems)
        )


__all__ = ["SpecSourceError", "covering_key", "require_sources", "source_problems", "spec_leaves"]
