"""The reviewer agent: does the strategy code express what its source
describes? (docs/PLAN.md stage 5, docs/REVIEWER.md)

The role the plan calls Adversary (opus: "поиск look-ahead и дыр — единственное
место, где нужен сильный судья"). The automatic guards catch a peek into the
future (`qlab.harness.lookahead`) and a number from nowhere
(`qlab.pipeline.sources`); what they cannot catch is a mechanism the source
describes and the code silently leaves out -- FRAB's two-phase breakeven, Bv2's
margin pool -- which once made the framework reject a working strategy.

**The model proposes, the code decides** (CLAUDE.md). The agent returns
findings, and every finding must carry evidence the code verifies:

- `source_quote` -- a verbatim quote from the source text it was given
  (whitespace-normalised substring match);
- `code_ref` -- `path:line` or `path:line-line` inside the code it was given.

A finding whose evidence does not check out is discarded and recorded as
rejected, with the reason. What the pipeline does with accepted findings is
decided by code (`qlab.pipeline.evaluate.not_evaluable_reasons`).

The call runs with its own system prompt and no tools (`--system-prompt`,
`--tools ""`): everything the agent needs is in the prompt, line-numbered, so
it cannot wander and does not pay for Claude Code's own ~40k-token prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.spec import StrategySpec

MODEL = "opus"  # docs/PLAN.md, "Модели для агентов системы": Adversary -> opus
KINDS = ("unexpressed_mechanism", "parameter_mismatch", "lookahead_risk", "other")

SYSTEM_PROMPT = """You review a trading strategy's implementation against its source.

You get: SOURCE texts (a strategy card, a paper's quotes, or the production code the
implementation must reproduce), the SPEC (parameters and data), and the CODE of the
implementation with line numbers. The implementation computes target portfolio
weights per bar; a weight at bar t may use data up to and including t, never after.

Find what is wrong or missing. Kinds:
- unexpressed_mechanism: something the source does that the code does not do at all
  (a rule, a state, an exit, a sizing or margin mechanism, a filter).
- parameter_mismatch: a number or choice in the spec/code that differs from the source.
- lookahead_risk: code that may use data after bar t to decide at bar t.
- other: anything else that makes the implementation not the strategy described.

Every finding MUST carry evidence:
- source_quote: copied VERBATIM from a SOURCE text (required for unexpressed_mechanism
  and parameter_mismatch); copy characters exactly, do not paraphrase.
- code_ref: "path:LINE" or "path:START-END" using the line numbers shown (required for
  parameter_mismatch and lookahead_risk; optional otherwise).
Findings without verifiable evidence are discarded automatically, so do not guess.
Do not report style, naming or performance. Do not judge whether the strategy is good.

Answer with ONE JSON object and nothing else:
{"findings": [{"kind": "...", "summary": "one sentence", "source_quote": "...",
"code_ref": "...", "spec_path": "..."}]}
An empty list is a valid answer."""


@dataclass(frozen=True)
class Finding:
    kind: str
    summary: str
    source_quote: str
    code_ref: str
    spec_path: str


@dataclass(frozen=True)
class ReviewOutcome:
    accepted: list[Finding]
    rejected: list[tuple[Finding | dict, str]]
    raw_text: str | None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def numbered(text: str) -> str:
    return "\n".join(f"{i:>4}| {line}" for i, line in enumerate(text.splitlines(), start=1))


BLOCKING_KINDS = ("unexpressed_mechanism", "parameter_mismatch", "lookahead_risk")


def review_key(spec: StrategySpec, code: Mapping[str, str]) -> str:
    """Changes whenever the implementation changes -- the spec's strategy
    fields or the code. Title, sources and the answers to a review do not
    change what was reviewed."""
    h = hashlib.sha256(spec.model_dump_json(
        include={"code_ref", "params", "data", "costs", "min_leg_notional"}).encode())
    for name in sorted(code):
        h.update(name.encode())
        h.update(code[name].encode())
    return h.hexdigest()


def unanswered_blocking(spec: StrategySpec, accepted: list[dict]) -> list[dict]:
    """Accepted findings that block a run: a missing mechanism, a parameter
    that differs from the source, or a possible peek -- unless the spec answers
    the finding by its summary in `review_answers`."""
    return [f for f in accepted
            if f.get("kind") in BLOCKING_KINDS and f.get("summary") not in spec.review_answers]


def build_prompt(spec_text: str, sources: Mapping[str, str], code: Mapping[str, str]) -> str:
    parts = ["## SOURCE"]
    parts += [f"### {name}\n{text}" for name, text in sources.items()]
    parts += ["## SPEC", spec_text, "## CODE (line-numbered)"]
    parts += [f"### {name}\n{numbered(text)}" for name, text in code.items()]
    return "\n\n".join(parts)


def parse_findings(text: str | None) -> list[dict] | None:
    """The findings list of the agent's JSON answer, or None if there is none."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    findings = payload.get("findings") if isinstance(payload, dict) else None
    return findings if isinstance(findings, list) else None


_REF = re.compile(r"^(?P<path>[^:]+):(?P<a>\d+)(?:-(?P<b>\d+))?$")


def _check_ref(ref: str, code: Mapping[str, str]) -> str | None:
    match = _REF.match(ref.strip())
    if not match or match["path"] not in code:
        return f"code_ref {ref!r} is not path:line in the code given"
    lines = len(code[match["path"]].splitlines())
    a, b = int(match["a"]), int(match["b"] or match["a"])
    if not (1 <= a <= b <= lines):
        return f"code_ref {ref!r} is outside {match['path']} (1-{lines})"
    return None


def verify(raw: list[dict], sources: Mapping[str, str], code: Mapping[str, str]) -> ReviewOutcome:
    """Keep the findings whose evidence checks out; reject the rest, saying why."""
    source_text = _norm("\n".join(sources.values()))
    accepted, rejected = [], []
    for item in raw:
        if not isinstance(item, dict):
            rejected.append((item, "not an object"))
            continue
        f = Finding(
            kind=str(item.get("kind", "")),
            summary=str(item.get("summary", "")).strip(),
            source_quote=str(item.get("source_quote") or "").strip(),
            code_ref=str(item.get("code_ref") or "").strip(),
            spec_path=str(item.get("spec_path") or "").strip(),
        )
        problems = []
        if f.kind not in KINDS:
            problems.append(f"unknown kind {f.kind!r}")
        if not f.summary:
            problems.append("no summary")
        needs_quote = f.kind in ("unexpressed_mechanism", "parameter_mismatch")
        needs_ref = f.kind in ("parameter_mismatch", "lookahead_risk")
        if f.source_quote:
            if _norm(f.source_quote) not in source_text:
                problems.append("source_quote is not verbatim in the sources")
        elif needs_quote:
            problems.append("no source_quote")
        if f.code_ref:
            ref_problem = _check_ref(f.code_ref, code)
            if ref_problem:
                problems.append(ref_problem)
        elif needs_ref:
            problems.append("no code_ref")
        if not f.source_quote and not f.code_ref:
            problems.append("no evidence at all")
        if problems:
            rejected.append((f, "; ".join(problems)))
        else:
            accepted.append(f)
    return ReviewOutcome(accepted=accepted, rejected=rejected, raw_text=None)


def code_files(spec: StrategySpec) -> dict[str, str]:
    """The implementation's own module, and the wrapped strategy's for a
    wrapper, keyed by repo-relative path."""
    from qlab.pipeline.sources import QLAB_ROOT

    refs = [spec.code_ref]
    inner = spec.params.get("inner_code_ref")
    if isinstance(inner, str):
        refs.append(inner)
    out = {}
    for ref in refs:
        module = ref.split(":")[0]
        rel = Path("src") / Path(*module.split(".")).with_suffix(".py")
        path = QLAB_ROOT / rel
        if path.is_file():
            out[str(rel)] = path.read_text(encoding="utf-8")
    return out


def run_review(
    *,
    spec: StrategySpec,
    spec_text: str,
    sources: Mapping[str, str],
    code: Mapping[str, str],
    budget: CandidateBudget | ExplicitNoBudget,
    session,
    model: str = MODEL,
) -> ReviewOutcome:
    """Run the reviewer, verify its evidence, and record the review."""
    from datetime import UTC, datetime

    from qlab.registry.models import Review

    prompt = build_prompt(spec_text, sources, code)
    result = run_agent(budget, prompt, session=session, agent="reviewer", model=model,
                       max_turns=1, extra_args=("--system-prompt", SYSTEM_PROMPT, "--tools", ""))
    raw = parse_findings(result.text)
    if raw is None:
        outcome = ReviewOutcome(accepted=[], rejected=[({}, "the answer held no findings JSON")],
                                raw_text=result.text)
    else:
        verified = verify(raw, sources, code)
        outcome = ReviewOutcome(verified.accepted, verified.rejected, result.text)
    if raw is not None:  # an unreadable answer is not a review of anything
        session.add(Review(
            idea_id=spec.idea_id,
            review_key=review_key(spec, code),
            at=datetime.now(UTC),
            model=model,
            sources=sorted(sources),
            accepted=[f.__dict__ for f in outcome.accepted],
            rejected=[{**(f.__dict__ if isinstance(f, Finding) else {"raw": f}), "why": why}
                      for f, why in outcome.rejected],
        ))
        session.flush()
    return outcome


__all__ = [
    "BLOCKING_KINDS",
    "KINDS",
    "MODEL",
    "SYSTEM_PROMPT",
    "Finding",
    "ReviewOutcome",
    "build_prompt",
    "code_files",
    "parse_findings",
    "review_key",
    "run_review",
    "unanswered_blocking",
    "verify",
]
