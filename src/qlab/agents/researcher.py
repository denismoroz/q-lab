"""The researcher agent: studies one strategy across everything the framework
knows about it and says what else is needed or possible (docs/RESEARCHER.md).

Owner, 2026-10-03: «я бы прогнал trend через ресерчера, чтобы он оценил —
может ещё что-то нужно/можно сделать»; «b2 так же — помучал на стенде с
режимами».

Code gathers the evidence (`build_context`): every related idea's latest
routed trial -- route, reason, returns, Sharpe, drawdown, place among matched
noise, forward days, the per-regime breakdown and the claims checked -- and
the latest paper reconciliation. The agent (opus: judgement, not volume)
reads it with the strategy's documents and frab's own research and writes:

- a memo for the owner, in Russian (`docs/research/<name>-<date>.md`): what
  works, what does not and why, hypotheses ranked, what to do about
  production;
- at most `max_experiments` experiment specs (`specs/research/<name>/`), each
  declared before it runs, every number cited (`qlab.pipeline.sources`).

The agent proposes; code runs the experiments on the stand (each a recorded
trial, counted in deflation) and appends their results to the memo. It may
read anything and write only those two places; it runs nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qlab.agents.implementer import _changed_paths, _revert
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

MODEL = "opus"
MEMO_DIR = Path("docs/research")
SPEC_DIR = Path("specs/research")
CONTEXT_DIR = Path("data/research")


def build_context(name: str, idea_ids: list[str], session) -> str:
    """The evidence, written by code: no number in it comes from the agent."""
    from qlab.registry.models import Spec, Trial

    lines = [f"# Evidence for the research on {name} (built {datetime.now(UTC):%Y-%m-%d})", ""]
    for idea_id in idea_ids:
        trial = (session.query(Trial).join(Spec, Spec.id == Trial.spec_id)
                 .filter(Spec.idea_id == idea_id, Trial.route.is_not(None))
                 .order_by(Trial.id.desc()).first())
        if trial is None:
            lines += [f"## {idea_id}", "no routed trial", ""]
            continue
        m = trial.metrics or {}
        route = getattr(trial.route, "value", trial.route)
        lines += [f"## {idea_id} -- trial {trial.id} ({trial.started_at:%Y-%m-%d})",
                  f"route: {route} -- {trial.route_reason}"]
        keys = [k for k in sorted(m) if not k.startswith("fit_regime")]
        lines += [f"- {k}: {m[k]:.4g}" if isinstance(m[k], int | float) else f"- {k}: {m[k]}"
                  for k in keys]
        fit = {k: v for k, v in m.items() if k.startswith("fit_regime")}
        if fit:
            lines.append("selection-period regime breakdown: " + ", ".join(
                f"{k[4:]}={v:.4g}" for k, v in sorted(fit.items())))
        lines.append("")
    states = sorted(Path("data/night").glob("*.json"))
    if states:
        state = json.loads(states[-1].read_text())
        if state.get("paper"):
            lines += [f"## Paper reconciliation ({states[-1].stem})", *state["paper"], ""]
    return "\n".join(lines)


INSTRUCTIONS = """You are q-lab's researcher. q-lab is a research lab that finds and
filters trading strategies; it never places orders, and its verdicts are made by code, not
by you. Study the strategy "{name}" and say what else is needed or possible.

The evidence, built by code from the registry (every number you may quote): {context_path}
Documents to read: {docs}
Specs of the strategy and its variants: {specs}

Know the discipline before proposing anything (read docs/FIT_VS_FORWARD.md, docs/REGIMES.md,
docs/SOURCES.md, docs/RETUNE.md): results on the data a choice was made on prove nothing;
every experiment is a trial counted in deflation; a strategy's regime may be detected only
from the past; a number must come from a source (frab's research, a card, the owner) -- never
invent one; a grid of variants is declared before it runs and reported whole.

Write two kinds of files and nothing else:

1. {memo_path} -- a memo for the owner IN RUSSIAN, no jargon (explain a term on first use):
   what the strategy does, what the evidence says works and does not, and why; hypotheses for
   improvement ranked by expected value and cost; what you would do about production (capital,
   configuration) and why; what the framework cannot yet answer about it. Quote numbers only
   from the evidence file and the documents, naming where each comes from.

2. At most {max_experiments} experiment specs in {spec_dir}/ -- each a variant worth running
   on the stand that tests one hypothesis of the memo: copy the closest existing spec, change
   only what the hypothesis needs, give it a new idea_id "{name}-research-<short>", set
   params_fixed_at {today} with params_fixed_evidence "declared by the researcher on {today}:
   <hypothesis>", and cite a source for every changed number in `sources`. Prefer strategy
   code that already exists (wrappers such as qlab.strategies.regime_gate:RegimeGate and
   qlab.strategies.retune:Retune, or existing parameters); you cannot write strategy code.
   Write no spec if nothing is worth a trial.

Reply with one line: the memo path and the experiment spec paths."""


@dataclass
class ResearchOutcome:
    memo: Path | None
    specs: list[Path] = field(default_factory=list)
    reply: str | None = None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)


def research(name: str, *, idea_ids: list[str], specs: list[Path], docs: list[Path],
             budget: CandidateBudget | ExplicitNoBudget, session, max_experiments: int = 3,
             model: str = MODEL, max_turns: int = 60) -> ResearchOutcome:
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec

    today = datetime.now(UTC).date()
    (QLAB_ROOT / CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    context_path = CONTEXT_DIR / f"{name}-{today}.md"
    (QLAB_ROOT / context_path).write_text(build_context(name, idea_ids, session),
                                          encoding="utf-8")
    memo_path = MEMO_DIR / f"{name}-{today}.md"
    spec_dir = SPEC_DIR / name
    (QLAB_ROOT / MEMO_DIR).mkdir(parents=True, exist_ok=True)
    (QLAB_ROOT / spec_dir).mkdir(parents=True, exist_ok=True)
    before = set(_changed_paths())
    prompt = INSTRUCTIONS.format(
        name=name, context_path=context_path, docs=", ".join(map(str, docs)),
        specs=", ".join(map(str, specs)), memo_path=memo_path, spec_dir=spec_dir,
        max_experiments=max_experiments, today=(today - timedelta(days=0)).isoformat())
    result = run_agent(budget, prompt, session=session, agent="researcher", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", "Read,Glob,Grep,Write",
                                   "--disallowedTools", "Bash,Edit",
                                   "--add-dir", str(QLAB_ROOT.parent / "funding-rate-arbitrage")))
    outcome = ResearchOutcome(memo=None, reply=(result.text or "").strip()[:600])
    for path in sorted(set(_changed_paths()) - before):
        if path == str(memo_path):
            outcome.memo = memo_path
        elif path.startswith(f"{spec_dir}/") and path.endswith(".yaml"):
            outcome.specs.append(Path(path))
        else:
            _revert(path)
            outcome.reverted.append(path)
    if outcome.reverted:
        outcome.problems.append(f"wrote outside its places (reverted): {outcome.reverted}")
    if outcome.memo is None:
        outcome.problems.append("no memo written")
    if len(outcome.specs) > max_experiments:
        outcome.problems.append(f"{len(outcome.specs)} experiments, at most {max_experiments}")
    for spec_path in outcome.specs:
        try:
            problems = source_problems(load_spec(QLAB_ROOT / spec_path))
            outcome.problems += [f"{spec_path}: {p}" for p in problems]
        except Exception as exc:  # noqa: BLE001
            outcome.problems.append(f"{spec_path} does not load: {exc}")
    return outcome


__all__ = ["MODEL", "ResearchOutcome", "build_context", "research"]
