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
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
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
perps since 2019, spot by explicit instrument list); intervals 1d, 4h, 1h. Prices are
closes, funding is per bar, volume is base units, `high` is the bar's high. There is no
order book, no liquidation feed, no options, no outcome markets, no external stock prices.

Rules:
- Never invent a number. Parameters come from the card; cite "{card_path}" for them (the
  number must appear in that file literally) or a derived formula. Costs: copy the costs,
  min_leg_notional and their `sources` entries from specs/trend-live-top25-cap.yaml for
  Hyperliquid perps.
- If the card needs something the data cannot give, do NOT fake it with a proxy the card
  does not name: list it in `unexpressed_mechanisms` (one sentence each). An honest "not
  evaluable" is a valid outcome; a strategy that quietly does something else is not.
- The class: `name` attribute, `valid_intervals` tuple, `target_weights(self, panel,
  params) -> DataFrame` with the panel's index and columns, weights 0 where not tradeable,
  no look-ahead. Pure pandas/numpy. Module and class docstrings in English, saying what
  the card asks and what the code does.
- The spec: idea_id "{idea_id}", a Russian title, code_ref
  "qlab.strategies.agent.{module}:<ClassName>", params, data (start 2025-01-01, end
  {today}), costs, min_leg_notional, unexpressed_mechanisms, sources, params_fixed_at
  {today}, params_fixed_evidence "written by the implementer agent from {card_path} on
  {today}".
- Code, comments and the spec in English except the title.

When both files are written, stop. Reply with one line: the class name, or
"NOT EXPRESSIBLE: <reason>" if even a declared-unexpressed spec makes no sense."""


@dataclass
class ImplementOutcome:
    idea_id: str
    code_path: Path
    spec_path: Path
    agent_reply: str | None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)
    not_expressible: str | None = None

    @property
    def ok(self) -> bool:
        """Code and spec written and passing the code-run guards."""
        return not self.problems and self.not_expressible is None


def _changed_paths() -> list[str]:
    done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                          cwd=QLAB_ROOT, capture_output=True, text=True, check=True)
    return [line[3:].strip() for line in done.stdout.splitlines() if line.strip()]


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
    module = module_name(idea_id)
    code_path = CODE_DIR / f"{module}.py"
    spec_path = SPEC_DIR / f"{idea_id}.yaml"
    allowed = {str(code_path), str(spec_path)}
    before = set(_changed_paths())
    (QLAB_ROOT / CODE_DIR).mkdir(parents=True, exist_ok=True)
    (QLAB_ROOT / SPEC_DIR).mkdir(parents=True, exist_ok=True)
    init = QLAB_ROOT / CODE_DIR / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Strategies written by the implementer agent (docs/IMPLEMENTER.md)."""\n')

    from datetime import timedelta

    last_closed = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    prompt = INSTRUCTIONS.format(code_path=code_path, spec_path=spec_path, card_path=card,
                                 idea_id=idea_id, module=module, today=last_closed)
    result = run_agent(budget, prompt, session=session, agent="implementer", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", TOOLS, "--disallowedTools", "Bash"))
    outcome = ImplementOutcome(idea_id=idea_id, code_path=code_path, spec_path=spec_path,
                               agent_reply=(result.text or "").strip()[:500])

    # Guard 1: only the two allowed paths (and the package marker) changed.
    for path in set(_changed_paths()) - before:
        if path not in allowed and path != str(CODE_DIR / "__init__.py"):
            _revert(path)
            outcome.reverted.append(path)
    if outcome.reverted:
        outcome.problems.append(f"wrote outside its two files (reverted): {outcome.reverted}")
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
    return outcome


__all__ = ["ImplementOutcome", "MODEL", "implement", "module_name"]
