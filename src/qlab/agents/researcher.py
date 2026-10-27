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
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from qlab.agents.implementer import _changed_paths, _revert
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT
from qlab.year_view import YEAR_KEY, years_line

MODEL = "opus"
MEMO_DIR = Path("docs/research")
SPEC_DIR = Path("specs/research")
CONTEXT_DIR = Path("data/research")


CUT_SENTENCE = ("До даты отреза {cut}; данные после неё — для проверки исправлений, "
                "на них не смотреть.")


def drawdown_evidence(specs: list[Path], session) -> list[str]:
    """One section per strategy spec: where it lost money before the cut
    (`qlab.diagnose`). Built from cached panels only -- nothing is fetched; a spec
    whose panel is not cached says so. No number here comes from the agent."""
    from qlab.diagnose import diagnose_spec, render
    from qlab.night import last_closed_day, with_end
    from qlab.pipeline.spec import load_spec
    from qlab.regimes import load as load_regimes
    from qlab.rules.loader import load_latest

    lines: list[str] = []
    try:
        ruleset = load_latest()
        share = ruleset.holdout_share.share if ruleset.holdout_share is not None else None
        rules_version = ruleset.version
    except Exception:  # noqa: BLE001 - the section says what is missing
        share, rules_version = None, "unknown"
    regimes = load_regimes()
    for path in specs:
        full = path if path.is_absolute() else QLAB_ROOT / path
        try:
            spec = load_spec(full)
            lines += [f"## Drawdown evidence: {path} (idea {spec.idea_id}, data source "
                      f"{spec.data.source}, window {spec.data.start}..{spec.data.end})"]
            found = diagnose_spec(spec, session, share=share, regimes=regimes)
            if found is None:  # the nightly recheck keeps the panel of the extended window
                extended = with_end(spec, last_closed_day())
                if extended is not spec:
                    found = diagnose_spec(extended, session, share=share, regimes=regimes)
                    lines[-1] += f" -- panel of the window extended to {extended.data.end}"
        except Exception as exc:  # noqa: BLE001 - one broken spec must not stop the research
            lines += [f"evidence not built: {type(exc).__name__}: {exc}", ""]
            continue
        if found is None:
            lines += ["the panel for this spec's data window is not cached, so the drawdown "
                      "evidence is not built (nothing is fetched for it)", ""]
            continue
        lines += [CUT_SENTENCE.format(cut=found.cut.day), *render(found)]
        if "holdout_share" in found.cut.source:
            lines.append(f"(the held-out share is read from ruleset {rules_version})")
        lines.append("")
    return lines


def build_context(name: str, idea_ids: list[str], session, specs: list[Path] | None = None
                  ) -> str:
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
        keys = [k for k in sorted(m) if not k.startswith("fit_regime") and not YEAR_KEY.match(k)]
        lines += [f"- {k}: {m[k]:.4g}" if isinstance(m[k], int | float) else f"- {k}: {m[k]}"
                  for k in keys]
        if years := years_line(m):  # the per-year keys, as one line (docs/YEARS.md)
            lines.append(f"by calendar year (judged part): {years}")
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
    if specs:
        lines += ["# Where the strategy loses (drawdown evidence, before the cut only)", "",
                  *drawdown_evidence(specs, session)]
    return "\n".join(lines)


INSTRUCTIONS = """You are q-lab's researcher. q-lab is a research lab that finds and
filters trading strategies; it never places orders, and its verdicts are made by code, not
by you. Work on the strategy "{name}": make it better, and say what else is needed or possible.

The owner's decision, 2026-10-25, verbatim: «я предлагаю статус ждет проверку вперед - для
reserchera - игнорировать. а сформулировать так - 4 стратегии где он может работать. - если он
по этим стратегиям может улучшить - отлично - пускай улучшает. ... данные есть - отлично -
проверили - если данных мало - смотрим где стратегия плохо работает - основные просадки и их
пытаемся исправить». So "waiting for the forward test" is NOT a reason to stop: ignore that
status. Find where the strategy loses (the drawdown evidence in the evidence file: its deepest
drawdowns, its returns by market regime, its worst months) and propose fixes for those losses
as experiment specs.

The evidence, built by code from the registry (every number you may quote): {context_path}
Documents to read: {docs}
Specs of the strategy and its variants: {specs}

Know the discipline before proposing anything (read docs/FIT_VS_FORWARD.md, docs/REGIMES.md,
docs/SOURCES.md, docs/RETUNE.md): results on the data a choice was made on prove nothing;
every experiment is a trial counted in deflation; a strategy's regime may be detected only
from the past; a number must come from a source (frab's research, a card, the owner) -- never
invent one; a grid of variants is declared before it runs and reported whole.

A fix must be judged on data it did not see. The drawdown evidence is computed only on the
data BEFORE a cut date (stated in each section of the evidence file); the data after the cut
is held back for judging your fix. Design a fix from the evidence before the cut only: do not
look at the strategy's results after the cut anywhere (older memos and research documents that
quote whole-window numbers do not count as evidence for choosing a fix), and do not tune
anything to them.

Write two kinds of files and nothing else:

1. {memo_path} -- a memo for the owner IN RUSSIAN, no jargon (explain a term on first use):
   what the strategy does, what the evidence says works and does not, and why; where it
   loses (the drawdowns, in plain words) and what fix you propose for each; hypotheses for
   improvement ranked by expected value and cost; what you would do about production (capital,
   configuration) and why; what the framework cannot yet answer about it. Quote numbers only
   from the evidence file and the documents, naming where each comes from.
   If you propose to drop one of the strategy's active variants (the lab keeps only a few per
   strategy), put a line of exactly this form on a line of its own, one line per variant:
   DROP: <idea_id> — <why, one sentence>
   Code reads these lines and shows them to the owner. You cannot drop anything yourself:
   statuses are changed by code and by the owner, never on your word.

2. At most {max_experiments} experiment specs in {spec_dir}/ -- each a variant worth running
   on the stand that tests one fix of the memo: copy the closest existing spec, change
   only what the fix needs, give it a new idea_id "{name}-research-<short>", and cite a source
   for every changed number in `sources`. Prefer strategy code that already exists (wrappers
   such as qlab.strategies.regime_gate:RegimeGate and qlab.strategies.retune:Retune, or
   existing parameters); you cannot write strategy code. Write no spec if nothing is worth a
   trial.
   Where long history exists, prefer a variant on it: Binance (`data.source: binance`, history
   since 2019; see specs/bv2-binance.yaml, specs/trend-binance-cap.yaml). Set
   params_fixed_at to the CUT DATE stated in the evidence file, with params_fixed_evidence
   "declared by the researcher on {today}: fix designed on the data before <cut date>;
   <hypothesis>". The data after the cut is then the fix's forward test, and the stand judges
   it at once. Never set params_fixed_at to {today} for a fixed-parameter fix on history: it
   has no forward test until data after today accumulates.
   Where only a short history exists (Hyperliquid only, as for FRAB), the same applies with
   the cut of the evidence, but its held-out part is short: say in the memo that the result is
   preliminary. A preliminary result does not block anything; you keep working.
   To test a parameter idea on history at once, you may also offer it as a grid to
   qlab.strategies.retune:Retune with `selects_causally: true` (docs/RETUNE.md): the wrapper
   picks among the grid using only the past, so every day after its first choice is a forward
   test. For such a spec params_fixed_at is NOT today but the date of the wrapper's FIRST
   choice -- the start of the data where the whole book exists, plus the inner strategy's
   warm-up, plus one whole re-tuning period -- as in specs/trend-retune-monthly.yaml; its
   params_fixed_evidence says that arithmetic and that the grid was declared {today}. With
   today's date the whole history would be warm-up.

{focus}Reply with one line: the memo path and the experiment spec paths."""


@dataclass(frozen=True)
class DropProposal:
    """The researcher's suggestion to drop a variant. A suggestion only: code
    never changes a status on the agent's word (the owner decides)."""

    idea_id: str
    why: str


# `DROP: <idea_id> — <why>` on a line of its own (a list bullet or bold marker is
# tolerated); the dash is an em dash, an en dash, "--" or a spaced hyphen.
_DROP_LINE = re.compile(
    r"^[ \t>*_-]*DROP:[*_ \t]*`?(?P<idea>[A-Za-z0-9][A-Za-z0-9._-]*)`?[ \t]*"
    r"(?:[—–]|--|-(?=[ \t]))[ \t]*(?P<why>\S.*?)[ \t*_]*$", re.MULTILINE)


def parse_drop_proposals(memo_text: str) -> list[DropProposal]:
    """Every `DROP: <idea_id> — <why>` line of a memo, in order, one per idea
    (the first wins). Lines that do not fit the form are not proposals."""
    seen: dict[str, DropProposal] = {}
    for match in _DROP_LINE.finditer(memo_text):
        seen.setdefault(match["idea"], DropProposal(match["idea"], match["why"]))
    return list(seen.values())


@dataclass
class ResearchOutcome:
    memo: Path | None
    specs: list[Path] = field(default_factory=list)
    reply: str | None = None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)
    drops: list[DropProposal] = field(default_factory=list)


def research(name: str, *, idea_ids: list[str], specs: list[Path], docs: list[Path],
             budget: CandidateBudget | ExplicitNoBudget, session, max_experiments: int = 3,
             model: str = MODEL, max_turns: int = 60, focus: str | None = None
             ) -> ResearchOutcome:
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec

    today = datetime.now(UTC).date()
    (QLAB_ROOT / CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    context_path = CONTEXT_DIR / f"{name}-{today}.md"
    (QLAB_ROOT / context_path).write_text(build_context(name, idea_ids, session, specs),
                                          encoding="utf-8")
    memo_path = MEMO_DIR / f"{name}-{today}.md"
    spec_dir = SPEC_DIR / name
    (QLAB_ROOT / MEMO_DIR).mkdir(parents=True, exist_ok=True)
    (QLAB_ROOT / spec_dir).mkdir(parents=True, exist_ok=True)
    before = set(_changed_paths())
    prompt = INSTRUCTIONS.format(
        name=name, context_path=context_path, docs=", ".join(map(str, docs)),
        specs=", ".join(map(str, specs)), memo_path=memo_path, spec_dir=spec_dir,
        max_experiments=max_experiments, today=(today - timedelta(days=0)).isoformat(),
        focus=(f"The owner's question, verbatim -- answer it first: {focus}\n\n" if focus else ""))
    result = run_agent(budget, prompt, session=session, agent="researcher", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", "Read,Glob,Grep,Write",
                                   "--disallowedTools", "Bash,Edit",
                                   "--add-dir", str(QLAB_ROOT.parent / "funding-rate-arbitrage")))
    outcome = ResearchOutcome(memo=None, reply=(result.text or "").strip()[:600])
    if result.exit_code != 0:
        outcome.problems.append(
            f"the agent stopped before finishing (exit {result.exit_code}; its spending cap or "
            "turn limit): what it wrote may be incomplete")
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
    else:
        outcome.drops = parse_drop_proposals((QLAB_ROOT / outcome.memo).read_text(
            encoding="utf-8"))
    if len(outcome.specs) > max_experiments:
        # The limit is the code's (a strategy's free variant slots, night/limits.yaml): what is
        # over it never reaches the stand. A night rehearsal of 2026-10-27 ran one anyway.
        extra = outcome.specs[max_experiments:]
        for path in extra:
            (QLAB_ROOT / path).unlink(missing_ok=True)
        outcome.specs = outcome.specs[:max_experiments]
        outcome.problems.append(f"{len(outcome.specs) + len(extra)} experiments, at most "
                                f"{max_experiments}: not run and removed {[str(p) for p in extra]}")
    for spec_path in outcome.specs:
        try:
            problems = source_problems(load_spec(QLAB_ROOT / spec_path))
            outcome.problems += [f"{spec_path}: {p}" for p in problems]
        except Exception as exc:  # noqa: BLE001
            outcome.problems.append(f"{spec_path} does not load: {exc}")
    return outcome


def second_stage_lines(experiment, judged, opened: dict, result, *, noise_trials: int,
                       deployable_capital_usd: float) -> list[str]:
    """The second stage of an experiment that survived stage 1 on Binance's history, as nested
    report lines (docs/TWO_STAGE.md). The same call the generator makes after a stage 1
    survivor (`qlab.pipeline.stage2.run_stage2`, not a measurement): the same mechanism and
    numbers on Hyperliquid, the idea's status moved by both stages together. Code only, no
    tokens. `judged` and `opened` are `own_numbers.prepare`'s: an experiment with open numbers
    carries to the second stage the value chosen at its cut. Empty when the experiment does not
    go on (stage 1 did not survive, or it already runs on Hyperliquid); a failure is a line,
    never raised -- stage 1 is on record and stays so."""
    from dataclasses import asdict

    from qlab import own_numbers
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope
    from qlab.twin_stage import goes_to_stage2

    route = str(result.routing.route)
    try:
        config = stage2.load_config()
        if not config.enabled or not stage2.survives(route, config) \
                or not goes_to_stage2(experiment):
            return []
        spec = experiment
        if opened:
            with session_scope() as session:
                chosen = own_numbers.chosen_params(judged, session)
            if not chosen:
                return ["  - второй этап: на отрезе не выбрано ни одного значения, нести на "
                        "Hyperliquid нечего"]
            spec = own_numbers.fixed_at_choice(experiment, chosen,
                                               date.fromisoformat(opened["cut"]))
        outcome = stage2.run_stage2(
            spec, stage1_route=route, stage1_trial_id=result.trial_id, noise_trials=noise_trials,
            deployable_capital_usd=deployable_capital_usd, config=config)
        return [f"  - {line}" for line in stage2.report_lines({"route": route}, asdict(outcome))]
    except Exception as exc:  # noqa: BLE001 - one line in the memo, the other experiments run
        return [f"  - второй этап: сбой {type(exc).__name__}: {exc}"]


def run_experiments(memo: Path, spec_paths: list[Path], *, name: str, noise_trials: int = 200,
                    deployable_capital_usd: float = 3000.0) -> list[str]:
    """Run declared experiments on the stand (matched noise, every rule) and
    append their results to the memo -- numbers written by code, not by the
    researcher. An experiment that survives stage 1 on Binance's history goes on to the
    second stage on Hyperliquid at once (`second_stage_lines`); its lines follow the
    experiment's, indented."""
    from qlab import own_numbers
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType
    from qlab.rules.loader import load_latest

    lines = ["", "## Эксперименты на стенде (записано кодом, не исследователем)", ""]
    for spec_path in spec_paths:
        try:
            experiment = load_spec(spec_path)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- `{spec_path}`: спек не загружается: {exc}")
            continue
        if source_problems(experiment):
            lines.append(f"- `{spec_path}`: не запущен — числа без источника")
            continue
        try:
            with session_scope() as session:
                if session.get(Idea, experiment.idea_id) is None:
                    repo.upsert_idea(session, id=experiment.idea_id, title=experiment.title,
                                     source_type=SourceType.INTERNAL,
                                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                                     notes=f"Declared by the researcher ({name}).")
                # Open numbers (qlab.own_numbers, owner 2026-10-26): every value a trial,
                # then the spec that chooses one at the cut and is judged after it.
                judged, opened = own_numbers.prepare(
                    experiment, session, load_latest(),
                    deployable_capital_usd=deployable_capital_usd)
                result = evaluate_spec_with_shape_aware_bar(
                    judged, session=session, ruleset=load_latest(),
                    deployable_capital_usd=deployable_capital_usd, n_trials=noise_trials).candidate
            m = result.metrics or {}
            own = (f" [свои числа: {len(opened['candidate_trials'])} вариантов записаны "
                   f"испытаниями, выбор на данных до {opened['cut']}, вердикт — по дням с него]"
                   if opened else "")
            regimes = ", ".join(f"{r}: {m[f'regime_{r}_return']:+.1%}" for r in
                                ("bull", "flat", "bear") if f"regime_{r}_return" in m)
            lines.append(
                f"- `{experiment.idea_id}` (прогон {result.trial_id}){own}: "
                f"{result.routing.route} — "
                f"в год {m.get('ann_return_net', float('nan')):+.1%}, "
                f"Шарп {m.get('sharpe_net', float('nan')):.2f}, "
                f"просадка {m.get('max_dd', float('nan')):.1%}, "
                f"обгоняет шум {m.get('noise_return_percentile', float('nan')):.0%}, "
                f"нужно капитала ${m.get('min_capital_usd', float('nan')):,.0f}; {regimes}")
            if years := years_line(m):
                lines.append(f"  - {years}")
            lines += second_stage_lines(experiment, judged, opened, result,
                                        noise_trials=noise_trials,
                                        deployable_capital_usd=deployable_capital_usd)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- `{experiment.idea_id}`: прогон упал: {type(exc).__name__}: {exc}")
    with memo.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


__all__ = ["CUT_SENTENCE", "MODEL", "DropProposal", "ResearchOutcome", "build_context",
           "drawdown_evidence", "parse_drop_proposals", "research", "run_experiments",
           "second_stage_lines"]
