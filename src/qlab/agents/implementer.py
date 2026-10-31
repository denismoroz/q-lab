"""The implementer agent: card -> strategy code and spec (docs/PLAN.md stage 5,
docs/IMPLEMENTER.md).

Owner, 2026-10-03: «делай все 3 одну за одной» -- the second of the three.
Until now every strategy was written by hand from its card. The implementer
writes two files and nothing else:

- `src/qlab/strategies/agent/<module>.py` -- a class implementing the
  Strategy protocol (`qlab.harness.strategy`);
- `specs/agent/<idea_id>.yaml` -- its spec, every number cited
  (`qlab.pipeline.sources`).

It may READ the repository (Read, Glob, Grep) and WRITE files (Write, Edit),
never run anything (no Bash). What it produces then goes through every guard
the framework has, run by code, not by the agent:

1. only the two allowed paths changed -- anything else is reverted and the
   attempt fails;
2. the spec loads and every number cites a checkable source;
3. the strategy module imports and declares the protocol;
4. the reviewer agent checks the code against the card
   (`qlab.agents.reviewer`); a verified gap blocks the run;
5. the stand evaluates it: the look-ahead guard, matched noise, the rules.

A card the data cannot express is not a failure: the agent declares what is
missing in `unexpressed_mechanisms` and the stand answers "not evaluable".

A card from the ideas inbox (`qlab.card_fields`, cards/SCHEMA.md, «Карточка из входящих идей»)
carries more: a `new_block`, a `grid`, a `baseline_spec` and a `data` block. For such a card the
agent writes a REUSABLE block (one module named after the block, every grid parameter a key of
the spec's `params`, nothing hidden in the class) and the spec carrying the grid's first
variant; code then checks that (guard 4), registers the block (`qlab.agent_blocks`) and runs the
spec, its causal version over the grid and the baseline on the same panel, all by code
(`qlab.implement_run`, docs/IMPLEMENTER.md). A card without those fields is handled as before.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from qlab import agent_blocks, tree_writes
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.card_fields import CardFieldError, CardFields, param_at, read_card_fields
from qlab.pipeline.sources import QLAB_ROOT

MODEL = "sonnet"  # docs/PLAN.md, "Модели для агентов системы": Implementer -> sonnet
CODE_DIR = Path("src/qlab/strategies/agent")
SPEC_DIR = Path("specs/agent")
TOOLS = "Read,Glob,Grep,Write,Edit"


def module_name(idea_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", idea_id.lower())


INSTRUCTIONS = """You implement ONE trading strategy for q-lab, a research lab that only
evaluates strategies (it never places orders). Work in the repository you are in. You may
read any file; you may write exactly two files and must not touch anything else:

  {code_path}   -- the strategy class
  {spec_path}   -- its spec

THE CARD (the source you implement; read it fully): {card_path}
{card_brief}
Read these first, they define what you write:
- src/qlab/harness/strategy.py   the Strategy protocol and the alignment rule: the weight
  at bar t may use data up to and including t, never after; the stand tests this
- src/qlab/harness/panel.py and src/qlab/data/panel.py   what a MarketPanel holds
  (prices, funding, tradeable, volume, high; NaN means unknown, never zero)
- src/qlab/pipeline/spec.py   every spec field and its rules
- docs/SOURCES.md   every number in the spec must cite where it comes from; a number
  cited to a file must appear in that file LITERALLY; derived numbers use
  "derived: <arithmetic>; <why>"
- an example pair: specs/trend-live-top25-cap.yaml with src/qlab/strategies/live/trend.py,
  and specs/xyz-equity-bab.yaml with src/qlab/strategies/bab.py

Data the stand can give you (spec `data.source`): "hyperliquid" (perps, and spot with
include_spot), "hyperliquid-xyz" (HIP-3 stocks, indices, commodities), "binance" (USDT
perps since 2019, spot by explicit instrument list), "aster" (USDT perps since 2021-09,
explicit instrument list only, funding settled every 1/2/4/8 hours depending on the symbol,
no spot); intervals 1d, 4h, 1h. Prices are closes, funding is per bar,
volume is base units, `high` is the bar's high. There is no order book, no liquidation feed,
no options, no outcome markets, no external stock prices.

Rules:
- Never invent a number. Parameters come from the card; cite "{card_path}" for them (the
  number must appear in that file literally) or a derived formula. Costs: the venue's row of
  night/costs.yaml (Hyperliquid perps: its `hyperliquid` row), cited as "night/costs.yaml, venue
  hyperliquid, taker_fee_bps" under costs.taker_fee_bps (and the same for slippage_bps); other
  numbers need a `costs.differs_from_table` entry with a checkable source, which you do not
  write. min_leg_notional and its `sources` entry: copy from
  specs/trend-live-top25-cap.yaml for Hyperliquid perps.
- If the card needs something the data cannot give, do NOT fake it with a proxy the card
  does not name: list it in `unexpressed_mechanisms` (one sentence each). An honest "not
  evaluable" is a valid outcome; a strategy that quietly does something else is not.
- The class: `name` attribute, `valid_intervals` tuple, `target_weights(self, panel,
  params) -> DataFrame` with the panel's index and columns, weights 0 where not tradeable,
  no look-ahead. Pure pandas/numpy. Module and class docstrings in English, saying what
  the card asks and what the code does.
- The spec: idea_id "{idea_id}", a Russian title, code_ref
  "qlab.strategies.agent.{module}:<ClassName>", params, data (start 2025-01-01 unless
  the card's DATA part below says otherwise, end {today}), costs, min_leg_notional,
  unexpressed_mechanisms, sources, params_fixed_at {today}, params_fixed_evidence
  "written by the implementer agent from {card_path} on {today}".
- Code, comments and the spec in English except the title.

When both files are written (the spec only, when the card says the block exists), stop.
Reply with one line: the class name, or
"NOT EXPRESSIBLE: <reason>" if even a declared-unexpressed spec makes no sense."""


def card_brief(fields: CardFields, code_path: Path, *, block_exists: bool,
               blocks: list[str] = ()) -> str:
    """The extra brief an inbox card adds to the instructions (empty for a hand-written card).
    What the agent must honour, in the order code will check it afterwards."""
    if not fields.inbox:
        return ""
    parts = [f"\nTHIS CARD COMES FROM THE IDEAS INBOX. The owner's idea, in words: "
             f"{fields.idea_text!r}. It carries more than a hand-written card; honour every part "
             "below, code checks each of them after you."]
    if fields.new_block is not None:
        block = fields.new_block
        listed = "\n".join(f"    - {p.name}: {p.meaning}" for p in block.params)
        parts.append(
            f"\nBLOCK. Write a REUSABLE block, not a one-off: the class is the block "
            f"\"{block.name}\" (CamelCase name), in the one module {code_path}. Its rule in "
            f"plain words, from the card: {block.rule}\nIts parameters and what each means:\n"
            f"{listed}\nEvery one of them, and every grid parameter below, is read by "
            "`target_weights` from `params` (a dotted path is a nested mapping); none is a "
            "constant, a default or a hidden number inside the class. The spec's `params` carry "
            "the card's FIRST variant. Another spec will name your module by code_ref with other "
            "numbers and the same class must serve it. The module docstring says the rule and "
            "lists each parameter with its meaning.")
        if block_exists:
            parts.append(
                f"The module {code_path} ALREADY EXISTS, written for an earlier card. Read it. "
                "Do NOT write or edit it (a change to it is reverted and the attempt fails): "
                "write only the spec, whose code_ref points at the existing class. If the "
                "card's rule needs behaviour that class does not have, reply \"NOT "
                f"EXPRESSIBLE: the block {block.name} does not express <what>\".")
    if fields.grid is not None:
        first = ", ".join(f"{p} = {v!r}" for p, v in fields.grid.first_variant().items())
        parts.append(
            f"\nGRID. Code runs the strategy over the card's variants ({fields.grid.count}), "
            "choosing between them every month from the past only. You do NOT write the grid and "
            "you do NOT write a Retune spec. Your spec's `params` carry the first variant: "
            f"{first}. Each of these paths ({', '.join(fields.grid.paths)}) must be a key of "
            "`params` (nested by dots) that the class reads. A grid parameter the class ignores "
            "makes the choice meaningless and the attempt fails.")
    if fields.readings:
        listed = "\n".join(f"    - {r}" for r in fields.readings)
        parts.append(
            "\nREADINGS. The idea's words allow more than one reading, and the card keeps each as "
            f"a variant of the grid:\n{listed}\nThe block must be able to run every one of them "
            "through its parameters: none may be left to a constant inside the class.")
    if fields.baseline_spec is not None:
        parts.append(
            f"\nBASELINE. The strategy this idea is compared with is {fields.baseline_spec}. "
            "Read it for the conventions, the costs, min_leg_notional and their `sources`. Do "
            "not edit it. Do not copy its params unless the card says the idea keeps them.")
    if fields.data is not None:
        d = fields.data
        where = (f"coins {', '.join(d.coins)}" if d.coins else f"coins by the rule: "
                 f"{d.universe_rule}")
        honest = (f"start the spec's data at {d.honest_from} (the first day the card's data "
                  "honestly exists)" if d.honest_from else "start 2025-01-01")
        interval = d.interval or "as the card says"
        parts.append(
            f"\nDATA. The card says: venue {d.venue}, interval {interval}, {where}. In the "
            f'spec: data.source "{d.venue}" (if the stand has no such source, say so in '
            "unexpressed_mechanisms, do not substitute one), the same interval, "
            f"data.instruments the card's coins when it lists them, and {honest}.")
    if blocks:
        parts.append("\nBLOCKS ALREADY WRITTEN (reuse one before you write another):\n"
                     + "\n".join(blocks))
    return "\n".join(parts) + "\n"


@dataclass
class ImplementOutcome:
    idea_id: str
    code_path: Path
    spec_path: Path
    agent_reply: str | None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)
    not_expressible: str | None = None
    fields: CardFields | None = None
    """The card's inbox fields (None for a card that has none)."""
    block_reused: bool = False
    """The card's block already existed and the agent wrote only the spec."""

    @property
    def ok(self) -> bool:
        """Code and spec written and passing the code-run guards."""
        return not self.problems and self.not_expressible is None


def _changed_paths() -> list[str]:
    done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                          cwd=QLAB_ROOT, capture_output=True, text=True, check=True)
    # The files the night's code steps write into the tree (a twin, an overlay variant) are not an
    # agent's: steps run together (docs/PIPELINE.md) and the agents' checks must not undo them.
    return tree_writes.without_code_writes(
        [line[3:].strip() for line in done.stdout.splitlines() if line.strip()])


def _revert(path: str) -> None:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=QLAB_ROOT,
                             capture_output=True).returncode == 0
    if tracked:
        subprocess.run(["git", "checkout", "--", path], cwd=QLAB_ROOT, check=True)
    else:
        (QLAB_ROOT / path).unlink(missing_ok=True)


def implement(card: Path, *, budget: CandidateBudget | ExplicitNoBudget, session,
              model: str = MODEL, max_turns: int = 60) -> ImplementOutcome:
    """Run the implementer on `card`, then the code-run guards 1-3 (the
    reviewer and the stand are the caller's next steps)."""
    import importlib

    import yaml

    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec

    idea_id = yaml.safe_load(card.read_text(encoding="utf-8"))["idea_id"]
    try:
        fields = read_card_fields(card, QLAB_ROOT)
    except CardFieldError as exc:
        # A broken card costs no tokens: the agent is not called.
        return ImplementOutcome(
            idea_id=idea_id, code_path=CODE_DIR / f"{module_name(idea_id)}.py",
            spec_path=SPEC_DIR / f"{idea_id}.yaml", agent_reply=None,
            problems=[f"the card's fields are wrong: {exc}"])
    module = fields.new_block.module if fields.new_block else module_name(idea_id)
    code_path = CODE_DIR / f"{module}.py"
    spec_path = SPEC_DIR / f"{idea_id}.yaml"
    # A block that already exists is reused as it is: the agent writes only the spec, and any
    # change to the module is undone (the module may be uncommitted, git would not see it).
    block_exists = fields.new_block is not None and (QLAB_ROOT / code_path).is_file()
    module_before = (QLAB_ROOT / code_path).read_text(encoding="utf-8") if block_exists else None
    allowed = {str(spec_path)} if block_exists else {str(code_path), str(spec_path)}
    before = set(_changed_paths())
    (QLAB_ROOT / CODE_DIR).mkdir(parents=True, exist_ok=True)
    (QLAB_ROOT / SPEC_DIR).mkdir(parents=True, exist_ok=True)
    init = QLAB_ROOT / CODE_DIR / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Strategies written by the implementer agent (docs/IMPLEMENTER.md)."""\n')

    from datetime import timedelta

    last_closed = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    brief = card_brief(fields, code_path, block_exists=block_exists,
                       blocks=agent_blocks.block_lines())
    prompt = INSTRUCTIONS.format(code_path=code_path, spec_path=spec_path, card_path=card,
                                 idea_id=idea_id, module=module, today=last_closed,
                                 card_brief=brief)
    result = run_agent(budget, prompt, session=session, agent="implementer", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", TOOLS, "--disallowedTools", "Bash"))
    outcome = ImplementOutcome(idea_id=idea_id, code_path=code_path, spec_path=spec_path,
                               agent_reply=(result.text or "").strip()[:500],
                               fields=fields if fields.inbox else None,
                               block_reused=block_exists)
    if result.exit_code != 0:
        outcome.problems.append(
            f"the agent stopped before finishing (exit {result.exit_code}; its spending cap or "
            "turn limit)")

    # Guard 1: only the two allowed paths (and the package marker) changed.
    for path in set(_changed_paths()) - before:
        if path not in allowed and path != str(CODE_DIR / "__init__.py"):
            _revert(path)
            outcome.reverted.append(path)
    if module_before is not None and (QLAB_ROOT / code_path).read_text(
            encoding="utf-8") != module_before:
        (QLAB_ROOT / code_path).write_text(module_before, encoding="utf-8")
        outcome.reverted.append(str(code_path))
    if outcome.reverted:
        outcome.problems.append(f"wrote outside its allowed files (reverted): {outcome.reverted}")
    reply = (outcome.agent_reply or "").lstrip("*_` ").strip()
    if reply.upper().startswith("NOT EXPRESSIBLE"):
        # An honest answer, not a broken attempt: recorded as such.
        outcome.not_expressible = reply
        return outcome
    if not (QLAB_ROOT / spec_path).is_file() or not (QLAB_ROOT / code_path).is_file():
        outcome.problems.append("the agent did not write both files")
        return outcome

    # Guard 2: the spec loads and cites a checkable source for every number.
    try:
        spec = load_spec(QLAB_ROOT / spec_path)
    except Exception as exc:  # noqa: BLE001
        outcome.problems.append(f"spec does not load: {exc}")
        return outcome
    outcome.problems += [f"source: {p}" for p in source_problems(spec)]

    # Guard 3: the module imports and the class looks like a Strategy.
    try:
        mod_name, _, cls = spec.code_ref.partition(":")
        strategy = getattr(importlib.import_module(mod_name), cls)()
        if not (hasattr(strategy, "name") and callable(getattr(strategy, "target_weights", None))):
            outcome.problems.append("the class does not implement the Strategy protocol")
    except Exception as exc:  # noqa: BLE001
        outcome.problems.append(f"strategy does not import: {type(exc).__name__}: {exc}")
    # Guard 4: what an inbox card asked for is in the spec and in the code.
    if fields.inbox:
        outcome.problems += card_problems(fields, spec, QLAB_ROOT)
    return outcome


def card_problems(fields: CardFields, spec, root: Path) -> list[str]:
    """The checks an inbox card adds (guard 4): the spec carries the grid's first variant, every
    grid parameter is a name the strategy's code mentions (a tripwire against a constant hidden
    in the class: not proof, a parameter the code never names cannot be read from `params`),
    and the spec's data is the card's. Pure: reads the spec and the module."""
    problems: list[str] = []
    if fields.grid is not None:
        source = ""
        module = spec.code_ref.split(":")[0]
        file = root / "src" / Path(*module.split(".")).with_suffix(".py")
        if file.is_file():
            source = file.read_text(encoding="utf-8")
        for path, value in fields.grid.first_variant().items():
            found, current = param_at(spec.params, path)
            if not found:
                problems.append(f"grid: {path} is not a parameter of the written spec")
            elif current != value:
                problems.append(f"grid: the spec carries {path} = {current!r}, the card's first "
                                f"variant is {value!r}")
            if source and path.split(".")[-1] not in source:
                problems.append(f"grid: the code never names {path.split('.')[-1]!r}: it cannot "
                                "read that parameter from params")
    if fields.data is not None:
        d = fields.data
        if spec.data.source != d.venue and getattr(spec, "execution_venue", None) != d.venue:
            problems.append(f"data: the card says venue {d.venue}, the spec reads "
                            f"{spec.data.source}")
        if d.interval and spec.data.interval != d.interval:
            problems.append(f"data: the card says interval {d.interval}, the spec has "
                            f"{spec.data.interval}")
    return problems


def register_block(outcome: ImplementOutcome, *, path: Path | None = None) -> bool:
    """Put the block the card asked for into the register (`qlab.agent_blocks`). Called by the
    stage after every guard and the reviewer have passed; False when the card has no block."""
    if outcome.fields is None or outcome.fields.new_block is None:
        return False
    from qlab.pipeline.spec import load_spec

    block = outcome.fields.new_block
    spec = load_spec(QLAB_ROOT / outcome.spec_path)
    agent_blocks.register(
        name=block.name, code_ref=spec.code_ref, rule=block.rule,
        params=[{"name": p.name, "meaning": p.meaning} for p in block.params],
        card=str(outcome.fields.card), spec=str(outcome.spec_path), path=path)
    return True


__all__ = ["ImplementOutcome", "MODEL", "card_brief", "card_problems", "implement",
           "module_name", "register_block"]
