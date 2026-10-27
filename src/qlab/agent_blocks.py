"""The register of blocks the implementer agent has written (`night/agent_blocks.yaml`,
docs/IMPLEMENTER.md, «Из идеи в блок и в проверку за одну ночь»).

A block of the lab's composed strategies (`qlab.strategies.blocks`) is a function in a Python
dict: adding one is a code edit, which no agent may make. A block the implementer writes is a
whole strategy class in `src/qlab/strategies/agent/<module>.py` whose every grid parameter is a
key of the spec's `params`, so the same module serves every spec and every causal variant that
names it by `code_ref`. This register is the data side of that: what the block is, what each
parameter means, which card asked for it and which specs use it. It is written by CODE after the
implementer's guards pass (`register`), never by the agent, and read back into the implementer's
own brief (`block_lines`), so the next card with a similar idea reuses a block instead of
writing a second one.

Pure file work: no database, no agent, no tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

from qlab.pipeline.sources import QLAB_ROOT

REGISTRY = Path("night/agent_blocks.yaml")
HEADER = """# Blocks the implementer agent has written (qlab.agent_blocks, docs/IMPLEMENTER.md).
# Written by code after the guards pass, never by the agent. A block is a strategy class whose
# every grid parameter is a key of the spec's `params`: another spec names it by `code_ref`.
"""


@dataclass
class BlockEntry:
    name: str
    code_ref: str
    rule: str
    params: list[dict] = field(default_factory=list)
    """[{name, meaning}] -- what each parameter of the block means."""
    cards: list[str] = field(default_factory=list)
    specs: list[str] = field(default_factory=list)
    registered: str = ""


def _path(path: Path | None) -> Path:
    """The register's file; the module's `REGISTRY` is read at call time so a test can
    point it elsewhere."""
    path = REGISTRY if path is None else path
    return path if path.is_absolute() else QLAB_ROOT / path


def load(path: Path | None = None) -> list[BlockEntry]:
    """Every registered block; empty when the file is missing."""
    file = _path(path)
    if not file.is_file():
        return []
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return [BlockEntry(**{k: v for k, v in item.items() if k in BlockEntry.__dataclass_fields__})
            for item in raw.get("blocks") or []]


def _save(entries: list[BlockEntry], path: Path | None) -> None:
    file = _path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"blocks": [asdict(e) for e in entries]}, sort_keys=False,
                          allow_unicode=True, width=100)
    tmp = file.with_suffix(".tmp")
    tmp.write_text(HEADER + body, encoding="utf-8")
    tmp.replace(file)


def register(*, name: str, code_ref: str, rule: str, params: list[dict], card: str, spec: str,
             path: Path | None = None, today: date | None = None) -> BlockEntry:
    """Add the block, or add the card and the spec to a block that is already registered.
    Idempotent: the same call twice changes nothing the second time. The first registration
    keeps its description (a later card does not rewrite what the block is)."""
    entries = load(path)
    found = next((e for e in entries if e.name == name), None)
    if found is None:
        found = BlockEntry(name=name, code_ref=code_ref, rule=rule, params=params,
                           registered=(today or date.today()).isoformat())
        entries.append(found)
    if card not in found.cards:
        found.cards.append(card)
    if spec not in found.specs:
        found.specs.append(spec)
    _save(entries, path)
    return found


def block_lines(path: Path | None = None) -> list[str]:
    """The registered blocks as lines for an agent's brief: what each is and how to name it."""
    lines = []
    for e in load(path):
        params = "; ".join(f"{p['name']} = {p['meaning']}" for p in e.params)
        lines.append(f"- {e.name} (code_ref {e.code_ref}): {e.rule} Parameters: {params}")
    return lines


__all__ = ["BlockEntry", "REGISTRY", "block_lines", "load", "register"]
