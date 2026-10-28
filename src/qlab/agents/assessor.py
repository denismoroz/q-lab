"""The assessor agent: a second opinion on a verdict the code has already
reached (docs/ASSESSOR.md).

Owner, 2026-10-24: «может не стоит полагаться на алгоритм оценки полностью -
может стоит иметь агента который будет это оценивать?»; on the proposal --
the code keeps the verdict, an agent reads the whole run and says where it
disagrees, the owner settles a dispute -- «спасибо. делай».

What it is for: a rule applied where it cannot tell anything (strategy-b-v2's
noise bar on 170 days rejected good and empty books alike) is plain to anyone
reading the whole run and invisible to a threshold. The agent reads the run
whole -- the verdict and its reason, every number, the market regimes, the
family's other runs, the production paper book -- and answers in plain words.

**The model proposes, the code decides** (CLAUDE.md). The verdict is never
changed. Every claim the agent makes must carry evidence the code checks:

- `fact` + `value` -- a key of the facts it was given and the number there
  (within `TOLERANCE`); or
- `quote` -- copied verbatim from the context texts it was given.

A claim whose evidence does not check out is discarded with the reason. A
view other than "agree" stands only with at least one verified claim
(`dispute`); otherwise it is recorded as "unsure". Like the reviewer, the call
has its own system prompt and no tools: everything is in the prompt.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.capital_text import CAP_NUMBERS
from qlab.year_view import years_line

MODEL = "opus"  # docs/PLAN.md, "Модели для агентов системы": the strong judge -> opus
VIEWS = ("agree", "too_harsh", "too_lenient", "unsure")
DISPUTES = ("too_harsh", "too_lenient")
TOLERANCE = 0.01
"""A claimed number matches a fact within 1% of it, or 0.005 absolute for
numbers near zero -- the agent rounds when it writes in words."""

SYSTEM_PROMPT = """You give a second opinion on a trading-strategy verdict.

A research lab screens strategies with fixed, versioned rules applied by code. You get
one run: the verdict (route) and its reason, the RULES as applied, numeric FACTS about
the run (and the latest runs of related variants of the same strategy), and CONTEXT
texts (the production paper book, earlier verdicts). The code's verdict stands whatever
you say; your opinion is shown to the owner beside it, and a dispute is his to settle.

Judge whether the verdict fits the evidence as a whole:
- agree: the verdict follows from the evidence.
- too_harsh: the evidence supports a better outcome than the verdict (e.g. a rule was
  applied where it cannot tell anything, or other variants/paper contradict it).
- too_lenient: the evidence is weaker than the verdict suggests (luck, one regime,
  too little data, a weakness the rules do not look at).
- unsure: the evidence does not settle it.
Look especially at: whether the period is long enough for the rule that decided; each
market regime separately (does the strategy do what it claims in bull/flat/bear?); what
related variants and the paper book show; what single fact would change your view.

Compare before you conclude:
- hold_ann_return / hold_ann_return_ex_best_1pct are holding the same coins over the same
  days. A return or a "without the best 1% of days" figure means little until set beside
  them: crypto earns in few days, so holding it also collapses without its best days.
- noise_* and regime_*_noise_percentile rank the strategy against random books that are
  half long and half short, so they never lose with the market: a book that holds spot
  looks worse than them in falls by construction. Do not read that as bad timing.
- hedge_timing_percentile, when present, is the direct test of timing: the strategy's own
  spot with its own hedges shifted in time (same amount of hedging, random moments);
  spot_only_ann_return and always_hedged_ann_return are the two plain alternatives.

Every claim MUST carry evidence:
- "fact": a key copied exactly from FACTS, and "value": the number there; or
- "quote": text copied VERBATIM from CONTEXT.
Claims without verifiable evidence are discarded automatically, so do not guess numbers.
Write "summary" and each claim's "text" in plain Russian for a non-specialist: no trading
jargon, no English trading words, explain any term the first time. 3-6 claims.
"checks": up to 3 concrete measurements that would settle the question (Russian).

Answer with ONE JSON object and nothing else:
{"view": "agree|too_harsh|too_lenient|unsure", "summary": "...",
 "claims": [{"text": "...", "fact": "...", "value": 0.0}, {"text": "...", "quote": "..."}],
 "checks": ["..."]}"""


@dataclass(frozen=True)
class Claim:
    text: str
    fact: str = ""
    value: float | None = None
    quote: str = ""


@dataclass
class AssessmentOutcome:
    view: str
    summary: str
    accepted: list[Claim] = field(default_factory=list)
    rejected: list[tuple[Claim | object, str]] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    raw_text: str | None = None

    @property
    def dispute(self) -> bool:
        return self.view in DISPUTES


@dataclass(frozen=True)
class Brief:
    """What the agent is given: numeric facts (claims are checked against
    them) and context texts (quotes are checked against them)."""

    idea_id: str
    trial_id: int
    facts: dict[str, float]
    context: dict[str, str]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _finite(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def matches(claimed: float, actual: float) -> bool:
    return abs(claimed - actual) <= max(TOLERANCE * abs(actual), 0.005)


FAMILY_KEYS = ("ann_return_net", "sharpe_net", "max_dd", "noise_return_percentile",
               "noise_power", "forward_days", "forward_days_needed", "selection_days",
               "fit_ann_return_net", "fit_sharpe_net", "pre_ann_return_net", "pre_sharpe_net",
               "pre_days", *CAP_NUMBERS)


def build_brief(session: Session, trial_id: int, *, paper: list[str] | None = None) -> Brief:
    """The facts and context of one trial, read from the registry: its own
    metrics, the latest run of each other idea in its family, its earlier
    verdicts, the rules it was judged by, and the production paper lines
    that name its strategy."""
    from sqlalchemy import select

    from qlab.registry.families import family_keys
    from qlab.registry.models import Idea, IdeaStatus, Spec, Trial, Verdict

    trial = session.get(Trial, trial_id)
    if trial is None:
        raise ValueError(f"no trial {trial_id}")
    spec = session.get(Spec, trial.spec_id)
    idea = session.get(Idea, spec.idea_id)
    facts = {f"trial.{k}": float(v) for k, v in (trial.metrics or {}).items() if _finite(v)}

    keys = family_keys(session)
    family = sorted(i for i, k in keys.items() if k == keys.get(idea.id) and i != idea.id)
    related = []
    for other in family:
        member = session.get(Idea, other)
        if member is None or member.status == IdeaStatus.RETIRED:
            continue  # withdrawn by the owner: not evidence about this one
        last = session.execute(
            select(Trial).join(Spec, Trial.spec_id == Spec.id)
            .where(Spec.idea_id == other, Trial.metrics.is_not(None))
            .order_by(Trial.id.desc()).limit(1)).scalars().first()
        if last is None:
            continue
        related.append(f"- {other} ({member.status.value}; trial {last.id}, "
                       f"{last.started_at:%Y-%m-%d}): "
                       f"{getattr(last.route, 'value', last.route)} -- {last.route_reason}")
        for k, v in last.metrics.items():
            if _finite(v) and (k in FAMILY_KEYS or (k.startswith("regime_")
                               and k.endswith(("_ann_return", "_noise_percentile", "_days")))):
                facts[f"{other}.{k}"] = float(v)

    earlier = session.execute(
        select(Trial).join(Spec, Trial.spec_id == Spec.id)
        .where(Spec.idea_id == idea.id, Trial.id < trial.id, Trial.route.is_not(None))
        .order_by(Trial.id.desc()).limit(5)).scalars().all()
    verdicts = session.execute(select(Verdict).where(Verdict.trial_id == trial.id)).scalars().all()
    rules = sorted({(v.note or "").split(";")[0] for v in verdicts})
    rule_lines = [
        f"- [{(v.note or '').split(';')[0]}] {v.rule_id}: {v.metric} {v.comparator} "
        f"{v.threshold} -> value {v.value}, passed {v.passed}" for v in verdicts]

    context = {
        "verdict": "\n".join([
            f"idea {idea.id}: {idea.title}; status {getattr(idea.status, 'value', idea.status)}",
            f"route: {getattr(trial.route, 'value', trial.route)}",
            f"reason: {trial.route_reason}",
            f"rules version: {verdicts[0].rules_version if verdicts else 'unknown'}",
            f"periods judged: {', '.join(r for r in rules if r)}",
        ]),
        "rules": "\n".join(rule_lines),
        "earlier verdicts": "\n".join(
            f"- trial {t.id} ({t.started_at:%Y-%m-%d}): {getattr(t.route, 'value', t.route)} "
            f"-- {t.route_reason}" for t in earlier),
        "related variants": "\n".join(related),
        # The result year by year (docs/YEARS.md): is the profit in one lucky period?
        "by calendar year": years_line(trial.metrics) or "",
    }
    if paper:
        word = (keys.get(idea.id) or "").lower()
        section, keep = [], False
        for line in paper:
            if line.startswith("**"):
                keep = bool(word) and word in line.lower()
            if keep:
                section.append(line)
        if section:
            context["production paper book vs the stand's replay"] = "\n".join(section)
    return Brief(idea_id=idea.id, trial_id=trial.id, facts=facts,
                 context={k: v for k, v in context.items() if v})


def build_prompt(brief: Brief) -> str:
    parts = ["## CONTEXT"]
    parts += [f"### {name}\n{text}" for name, text in brief.context.items()]
    parts.append("## FACTS (key = value)\n"
                 + "\n".join(f"{k} = {v:.6g}" for k, v in sorted(brief.facts.items())))
    return "\n\n".join(parts)


def parse_answer(text: str | None) -> dict | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def verify(answer: dict, brief: Brief) -> AssessmentOutcome:
    """Keep the claims whose evidence checks out; a dispute needs one."""
    context = _norm("\n".join(brief.context.values()))
    accepted, rejected = [], []
    for item in answer.get("claims") or []:
        if not isinstance(item, dict):
            rejected.append((item, "not an object"))
            continue
        value = item.get("value")
        claim = Claim(text=str(item.get("text", "")).strip(),
                      fact=str(item.get("fact") or "").strip(),
                      value=float(value) if _finite(value) else None,
                      quote=str(item.get("quote") or "").strip())
        problems = []
        if not claim.text:
            problems.append("no text")
        if claim.fact:
            if claim.fact not in brief.facts:
                problems.append(f"fact {claim.fact!r} is not among the facts given")
            elif claim.value is None:
                problems.append("fact without a number")
            elif not matches(claim.value, brief.facts[claim.fact]):
                problems.append(f"{claim.fact} is {brief.facts[claim.fact]:.6g}, "
                                f"not {claim.value:.6g}")
        if claim.quote and _norm(claim.quote) not in context:
            problems.append("quote is not verbatim in the context")
        if not claim.fact and not claim.quote:
            problems.append("no evidence at all")
        (rejected.append((claim, "; ".join(problems))) if problems else accepted.append(claim))
    view = str(answer.get("view", "")).strip()
    if view not in VIEWS:
        rejected.append(({"view": view}, f"unknown view {view!r}"))
        view = "unsure"
    if view in DISPUTES and not accepted:
        rejected.append(({"view": view}, "a dispute without one verified claim"))
        view = "unsure"
    checks = [str(c).strip() for c in answer.get("checks") or [] if str(c).strip()][:3]
    return AssessmentOutcome(view=view, summary=str(answer.get("summary", "")).strip(),
                             accepted=accepted, rejected=rejected, checks=checks)


def run_assessment(
    *,
    brief: Brief,
    budget: CandidateBudget | ExplicitNoBudget,
    session: Session,
    model: str = MODEL,
) -> AssessmentOutcome:
    """Run the assessor, verify its claims, and record the assessment."""
    from datetime import UTC, datetime

    from qlab.registry.models import Assessment

    result = run_agent(budget, build_prompt(brief), session=session, agent="assessor",
                       model=model, max_turns=1,
                       extra_args=("--system-prompt", SYSTEM_PROMPT, "--tools", ""))
    answer = parse_answer(result.text)
    if answer is None:
        return AssessmentOutcome(view="unsure", summary="", raw_text=result.text,
                                 rejected=[({}, "the answer held no JSON")])
    outcome = verify(answer, brief)
    outcome.raw_text = result.text
    session.add(Assessment(
        idea_id=brief.idea_id, trial_id=brief.trial_id, at=datetime.now(UTC), model=model,
        view=outcome.view, summary=outcome.summary,
        accepted=[c.__dict__ for c in outcome.accepted],
        rejected=[{**(c.__dict__ if isinstance(c, Claim) else {"raw": c}), "why": why}
                  for c, why in outcome.rejected],
        checks=outcome.checks,
    ))
    session.flush()
    return outcome


def latest_paper_lines(state_dir=None) -> list[str]:
    """The paper reconciliation of the latest night that has one."""
    from pathlib import Path

    from qlab.night import STATE_DIR

    for path in sorted(Path(state_dir or STATE_DIR).glob("*.json"), reverse=True):
        try:
            paper = json.loads(path.read_text(encoding="utf-8")).get("paper")
        except (OSError, json.JSONDecodeError):
            continue
        if paper:
            return list(paper)
    return []


def contested(view: str, accepted: list) -> bool:
    """A stored assessment disputes its verdict."""
    return view in DISPUTES and bool(accepted)


__all__ = [
    "DISPUTES",
    "MODEL",
    "SYSTEM_PROMPT",
    "VIEWS",
    "AssessmentOutcome",
    "Brief",
    "Claim",
    "build_brief",
    "build_prompt",
    "contested",
    "latest_paper_lines",
    "matches",
    "parse_answer",
    "run_assessment",
    "verify",
]
