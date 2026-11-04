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

import yaml

from qlab import fixation, plain
from qlab.agents.implementer import _changed_paths, _revert
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.capital_text import basis_ru, headline, min_capital_ru, numbers_ru
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
        by_data = ruleset.splits_by_data
    except Exception:  # noqa: BLE001 - the section says what is missing
        share, rules_version, by_data = None, "unknown", False
    regimes = load_regimes()
    for path in specs:
        full = path if path.is_absolute() else QLAB_ROOT / path
        try:
            spec = load_spec(full)
            lines += [f"## Drawdown evidence: {path} (idea {spec.idea_id}, data source "
                      f"{spec.data.source}, window {spec.data.start}..{spec.data.end})"]
            found = diagnose_spec(spec, session, share=share, regimes=regimes, by_data=by_data)
            if found is None:  # the nightly recheck keeps the panel of the extended window
                extended = with_end(spec, last_closed_day())
                if extended is not spec:
                    found = diagnose_spec(extended, session, share=share, regimes=regimes,
                                          by_data=by_data)
                    lines[-1] += f" -- panel of the window extended to {extended.data.end}"
        except Exception as exc:  # noqa: BLE001 - one broken spec must not stop the research
            lines += [f"evidence not built: {type(exc).__name__}: {exc}", ""]
            continue
        if found is None:
            lines += ["the panel for this spec's data window is not cached, so the drawdown "
                      "evidence is not built (nothing is fetched for it)", ""]
            continue
        lines += [CUT_SENTENCE.format(cut=found.cut.day), *render(found)]
        if "holdout_share" in found.cut.source or "the stand's cut" in found.cut.source:
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
        # The leading numbers are on the strategy's own money (docs/CAPITAL_BASE.md): reason on
        # these; the keys without `cap_` below are the stand's share of the book's balance.
        lines += [f"numbers on the strategy's own money ({basis_ru(m)}): {numbers_ru(m)}",
                  "(keys cap_* and noise_cap_* are on the strategy's own money -- the rules "
                  "decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, "
                  "total_return_net, noise_return_percentile, regime_*, are the stand's share "
                  "of the book's balance and are not comparable across books)"]
        keys = [k for k in sorted(m) if not k.startswith(("fit_regime", "fit_cap_regime"))
                and not YEAR_KEY.match(k)]
        lines += [f"- {k}: {m[k]:.4g}" if isinstance(m[k], int | float) else f"- {k}: {m[k]}"
                  for k in keys]
        if years := years_line(m):  # the per-year keys, as one line (docs/YEARS.md)
            lines.append(f"by calendar year (judged part): {years}")
        fit = {k: v for k, v in m.items() if k.startswith(("fit_regime", "fit_cap_regime"))}
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
   If you see an idea that needs NEW code -- a rule or a block that no existing strategy or
   setting expresses, not a different number of an existing strategy -- end the memo with a
   fenced block of exactly this form: one idea per line, in plain words, at most three. Code
   puts them in the owner's ideas inbox, where an agent turns them into cards for the
   implementer. Write no block when you have no such idea.
```ideas
- "<the idea, in the words you would tell the owner>"
```

2. At most {max_experiments} experiment specs in {spec_dir}/ -- each a variant worth running
   on the stand that tests one fix of the memo: copy the closest existing spec, change
   only what the fix needs, give it a new idea_id "{name}-research-<short>", and cite a source
   for every changed number in `sources`. Prefer strategy code that already exists (wrappers
   such as qlab.strategies.regime_gate:RegimeGate, or existing parameters); you cannot write
   strategy code. Write no spec if nothing is worth a trial.
   Where long history exists, prefer a variant on it: Binance (`data.source: binance`, history
   since 2019; see specs/bv2-binance.yaml, specs/trend-binance-cap.yaml).

   SPOT PRICES (docs/SPOT_PRICES.md). The stand marks a spot leg at its perp's price by default.
   An experiment whose rule reads the coin's OWN SPOT CLOSES (the spot's return, a spot-perp gap,
   a leftover valued at the spot) must declare `data.spot_prices: own` (the spot market's own
   closes; Hyperliquid and Binance both have them) and must NOT list the perp mark in
   `unexpressed_mechanisms`: copy that line away from the spec you copy. Code refuses a spec that
   does, and a spec that is "not expressed" is not judged at all. A bar without a spot print is
   carried at the last own close and the run states how many were; never the perp's price.

   COINS (owner, 2026-11-04, docs/ONE_CHECK.md, «Почему опыт ждал 27 дней»). Prefer a RULE to a
   coin list: let the book take the venue's listing under the trailing volume floor
   (`data.min_daily_volume_usd`, `data.instruments` unset), so no list is written by hand and
   there is nothing to date. Write a list only when the idea is about NAMED coins (Bv2's BTC and
   ETH). Then you need no hedge about it: code checks, from the venue's own listing dates and
   volumes, whether each coin was among the top ten by trailing volume on every past day
   (night/knowable_coins.yaml); if so the list is judged from the start of the data, if not
   only from the day it was knowable. Never shorten the data window to "the day the list was
   written".

   ONE IDEA, ONE CHECK (owner, 2026-10-31, docs/ONE_CHECK.md; read it). An experiment does not
   get a "twin" any more and you do not write a causal wrapper: the idea declares up front the
   values of the numbers it changes, code chooses among them ONCE, on the first 2/3 of the
   venue's whole history, holds the choice on the last third, and the verdict comes at once
   (rules 2026-11-04.1: every spec's window is cut 2/3 selection, 1/3 forward test, by its days). So
   every experiment spec carries a
   `choice` block. `params` holds the FIRST value of every list; the block holds the lists:

     choice:
       axes:                       # a dotted path of params -> the WHOLE list of its values
         signal.window_days: [3, 7, 14, 30]
         rebalance_days: [1, 7]
       source: "night/own_numbers.yaml"

   `source` names the file the values come from: night/own_numbers.yaml (then each list is the
   WHOLE list that file declares for that parameter, never a subset you picked) or a file that
   holds sourced values (a document of the strategy, frab's research, the base spec): every value
   must literally stand in the cited file, and a number needs a source like any other. The
   product of the lists may not exceed max_candidates of night/own_numbers.yaml. When the fix is
   one named set of numbers against another (a base and a variant), give them whole instead:

     choice:
       candidates:                 # at least two; the first is what `params` holds
         fix: {{hedge_margin_headroom: 1.5}}
         base: {{hedge_margin_headroom: 1.0}}
       source: "specs/bv2-live.yaml, docs/research/<the document that gives 1.5>"

   Do not set params_fixed_at, params_fixed_evidence or selects_causally for such a spec: code
   makes its choice at the cut itself. Every value of every list is a trial of its own, so a long
   list raises the bar the best of them has to clear.

   A spec with no list is refused, unless it says why there is nothing to choose from:

     choice:
       nothing_to_choose: "<a sentence: why no number of this fix has an alternative the lab
                           has declared or a source gives>"

   Then it is judged on the last third of its window, as every spec is (rules 2026-11-04.1: the
   cut is always 2/3 and 1/3 of the data, never params_fixed_at). Still write params_fixed_at, the
   day the code names (it is a record, and the page shows when it falls inside the last third;
   owner 2026-10-29, docs/FIXATION.md: the start of the LAST THIRD of the spec's own window, with
   params_fixed_evidence "declared by the researcher on {today}: fix designed on the data before
   <that day>; <hypothesis>"; a LATER day only with a source cited under sources.params_fixed_at).
   Never set params_fixed_at to {today} for a fixed-parameter fix on history: it would record
   that the numbers saw the whole last third, and the page marks such a forward test as not
   clean. Use this only when it is true: the stand shows the
   sentence on the idea's page.
   Where only a short history exists (Hyperliquid only, as for FRAB), the same applies with
   the cut of the evidence, but its held-out part is short: say in the memo that the result is
   preliminary. A preliminary result does not block anything; you keep working.

   PLAIN WORDS (owner, 2026-11-03: «смотрю и не могу понять - а что она вообще делает и как»).
   The idea's page opens with what the strategy does in plain Russian. For a variant of a
   strategy whose code already has a text in strategies/plain/ (Bv2, Trend, FRAB, XSMOM, ...)
   you write nothing: code computes how the variant differs from its base from the two specs.
   When your experiment is a NEW mechanism or a NEW family -- code no text in strategies/plain/
   covers -- add `plain_summary` to the spec: 3 to 6 short sentences in Russian, no jargon
   (cards/SCHEMA.md, «Правила заполнения»: a term is explained at its first use, no English
   trading words in Russian letters): what it holds, what it buys and sells and when, why it
   should earn, what it risks, how much money it runs on. Optionally `plain_title`, at most 80
   characters. Code checks that it exists and passes the same jargon check the cards pass.

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
    from qlab import duplicates, one_check
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec
    from qlab.spot_declaration import spot_declaration_problems

    today = datetime.now(UTC).date()
    (QLAB_ROOT / CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    context_path = CONTEXT_DIR / f"{name}-{today}.md"
    evidence = build_context(name, idea_ids, session, specs)
    try:  # the open ideas in plain words and the rule that doubles are refused (docs/DUPLICATES.md)
        evidence += "\n" + duplicates.prompt_block(duplicates.open_index(session))
    except Exception:  # noqa: BLE001 - the evidence stands without it
        pass
    (QLAB_ROOT / context_path).write_text(evidence, encoding="utf-8")
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
            spec = load_spec(QLAB_ROOT / spec_path)
            problems = [*source_problems(spec), *one_check.declaration_problems(spec),
                        *spot_declaration_problems(spec)]
            # the page opens with plain words: a new mechanism or family carries its own
            spec_text = (QLAB_ROOT / spec_path).read_text(encoding="utf-8")
            problems += [f"plain words: {p}" for p in plain.spec_problems(
                yaml.safe_load(spec_text) or {}, spec_text)]
            outcome.problems += [f"{spec_path}: {p}" for p in problems]
        except Exception as exc:  # noqa: BLE001
            outcome.problems.append(f"{spec_path} does not load: {exc}")
    return outcome


def second_stage_lines(experiment, judged, opened: dict, result, *, noise_trials: int) -> list[str]:
    """The second stage of an experiment that survived stage 1 on Binance's history, as nested
    report lines (docs/TWO_STAGE.md). The same call the generator makes after a stage 1
    survivor (`qlab.pipeline.stage2.run_stage2`, not a measurement): the same mechanism and
    numbers on Hyperliquid, the idea's status moved by both stages together. Code only, no
    tokens. `experiment` is what the stand ran: the idea's one check (docs/ONE_CHECK.md), which
    goes to Hyperliquid whole, the same lists chosen month by month there. `judged` and `opened`
    are `own_numbers.prepare`'s, for an experiment of the older kind that leaves numbers open in
    its params: it carries to the second stage the value chosen at its cut. Empty when the
    experiment does not go on (stage 1 did not survive, or it already runs on Hyperliquid); a
    failure is a line, never raised -- stage 1 is on record and stays so."""
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
            config=config)
        return [f"  - {line}" for line in stage2.report_lines({"route": route}, asdict(outcome))]
    except Exception as exc:  # noqa: BLE001 - one line in the memo, the other experiments run
        return [f"  - второй этап: сбой {type(exc).__name__}: {exc}"]


def run_experiments(memo: Path, spec_paths: list[Path], *, name: str,
                    noise_trials: int = 200) -> list[str]:
    """Run declared experiments on the stand (matched noise, every rule) and
    append their results to the memo -- numbers written by code, not by the
    researcher. An experiment is ONE idea with ONE check (docs/ONE_CHECK.md): it declares the
    values it leaves open, every value is a trial, code chooses among them month by month from the
    past, and the verdict comes on Binance's whole history; one that declares nothing is refused
    unless it says why there is nothing to choose. An experiment that survives stage 1 goes on to
    the second stage on Hyperliquid at once (`second_stage_lines`); its lines follow the
    experiment's, indented."""
    from qlab import duplicates, one_check
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType
    from qlab.rules.loader import load_latest
    from qlab.spot_declaration import spot_declaration_problems

    lines = ["", "## Эксперименты на стенде (записано кодом, не исследователем)", ""]
    open_ideas = None  # the open ideas, read once, grown by every experiment that goes on
    for spec_path in spec_paths:
        try:
            # The day the numbers were fixed is named by code (owner, 2026-10-29, docs/FIXATION.md):
            # an earlier or missing day is corrected before anything runs, and the memo says so.
            if (named := fixation.normalise_file(spec_path)) is not None:
                lines.append(f"- день фиксации назвал код: {named}")
            experiment = load_spec(spec_path)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- `{spec_path}`: спек не загружается: {exc}")
            continue
        if source_problems(experiment):
            lines.append(f"- `{spec_path}`: не запущен — числа без источника")
            continue
        if wrong := spot_declaration_problems(experiment):
            lines.append(f"- `{spec_path}`: не запущен — цена спота объявлена неверно: "
                         f"{'; '.join(wrong)}")
            continue
        if declared := one_check.declaration_problems(experiment):
            lines.append(f"- `{spec_path}`: не запущен — у идеи один способ проверки, и он не "
                         f"объявлен: {'; '.join(declared)}")
            continue
        try:
            with session_scope() as session:
                if duplicates.is_new(session, experiment.idea_id):
                    # A double of an open idea is not run (owner, 2026-11-04, docs/DUPLICATES.md).
                    if open_ideas is None:
                        open_ideas = duplicates.open_index(session)
                    if double := duplicates.refusal(experiment, open_ideas):
                        duplicates.note_refusal(experiment, double, f"исследователь {name}")
                        lines.append(f"- `{spec_path}`: не запущен — {double.text()}")
                        continue
                    open_ideas.append(duplicates.as_open(experiment))
                if session.get(Idea, experiment.idea_id) is None:
                    repo.upsert_idea(session, id=experiment.idea_id, title=experiment.title,
                                     source_type=SourceType.INTERNAL,
                                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                                     notes=f"Declared by the researcher ({name}).")
                    session.flush()
                prepared = one_check.prepare(experiment, session, load_latest(),
                                             spec_path=spec_path)
                if prepared.heavy:
                    one_check.note_refusal(experiment.idea_id, prepared.heavy, str(spec_path))
                    lines.append(f"- `{experiment.idea_id}`: не запущен — данные проверки слишком "
                                 f"тяжёлые для одной ночи (слова кода: {prepared.heavy}); "
                                 f"вручную: `qlab twins run --only {experiment.idea_id} "
                                 "--allow-heavy`")
                    continue
                result = evaluate_spec_with_shape_aware_bar(
                    prepared.spec, session=session, ruleset=load_latest(),
                    n_trials=noise_trials).candidate
            m = result.metrics or {}
            own = _check_words(prepared)
            reg = "cap_regime_" if headline(m).on_capital else "regime_"
            regimes = ", ".join(f"{r}: {m[f'{reg}{r}_return']:+.1%}" for r in
                                ("bull", "flat", "bear") if f"{reg}{r}_return" in m)
            lines.append(
                f"- `{experiment.idea_id}` (прогон {result.trial_id}){own}: "
                f"{result.routing.route} — "
                f"{numbers_ru(m)}, "
                f"{min_capital_ru(m)}; {regimes}")
            if years := years_line(m):
                lines.append(f"  - {years}")
            lines += second_stage_lines(prepared.spec, prepared.spec, {}, result,
                                        noise_trials=noise_trials)
        except one_check.OneCheckRefused as exc:
            one_check.note_refusal(experiment.idea_id, str(exc), str(spec_path))
            lines.append(f"- `{experiment.idea_id}`: проверка не построена — {exc} "
                         "(слова кода)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- `{experiment.idea_id}`: прогон упал: {type(exc).__name__}: {exc}")
    with memo.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def _check_words(prepared) -> str:
    """The short note after an experiment's name: what its one check chose among."""
    if prepared.declared == "nothing":
        return f" [выбирать не из чего: {prepared.reason}; судится после дня фиксации]"
    return (f" [одна проверка: код выбирает раз в месяц по прошлому между "
            f"{len(prepared.candidates)} вариантами ({', '.join(prepared.candidates)}), каждый "
            f"записан испытанием; история Binance, судится с {prepared.first_choice}]")


__all__ = ["CUT_SENTENCE", "MODEL", "DropProposal", "ResearchOutcome", "build_context",
           "drawdown_evidence", "parse_drop_proposals", "research", "run_experiments",
           "second_stage_lines"]
