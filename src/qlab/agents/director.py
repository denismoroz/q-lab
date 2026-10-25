"""The research director: an expert who watches the DIRECTIONS of the search and
corrects what the idea generator and the researcher do (docs/DIRECTOR.md).

Owner, 2026-10-25: «я думаю в агентах не хватает агента эксперта - который бы
следил за направлениями поиска и корректировал действия генератора идей
исследователя»; «запуск руководителя раз в 2 дня - добавляй».

Code builds the evidence (`build_evidence`); the agent (opus: judgement, not
volume) reads it with the catalogue and the documents and writes a memo for the
owner plus a machine-readable block that code validates (`qlab.directions`) into
`night/directions.yaml`.

**The director sets directions, not parameter values.** The evidence file holds
no per-parameter or per-variant performance numbers -- no Sharpe, no return, no
drawdown of any single variant, no metric of any trial -- on purpose: a director
that saw how each variant did on forward data would steer the search towards
what already happened to work, and the forward test would stop being forward.
What it sees is aggregated: how many ideas each direction made, where the stand
routed them (reject, needs-forward ...), which RULES they failed (rule ids, not
values), what a direction cost in tokens, which mechanisms the catalogue can
and cannot build, the researcher's DROP proposals (numbers masked) and the
decay detector's lines for the books that are running (strategy-level, as the
owner's own detector prints them). The instructions forbid the agent to dig
for variant results elsewhere; code cannot check that it did not (the same
limit as the researcher's cut date), so the memo says what it relied on.

Verdicts stay with the code. The director cannot change a status, close a
family or touch a spec: `propose_close_family` is shown to the owner and nothing
acts on it. Its spending goes through the budget guard (`qlab.budget.run_agent`):
without a budget the call fails instead of spending silently.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from qlab import directions as dirs
from qlab.agents.generator import CATALOGUE, Mechanism, catalogue, mechanism_of
from qlab.agents.implementer import _changed_paths
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.funnel import (
    AGENT_WORK_START,
    NOISE_PREFIX,
    SURVIVOR_STATUSES,
    PeriodFunnel,
    idea_source,
    latest_routed_trials,
    period_funnel,
)
from qlab.pipeline.sources import QLAB_ROOT

MODEL = "opus"
MEMO_DIR = Path("docs/research")
CONTEXT_DIR = Path("data/research")
NIGHT_DIR = Path("data/night")
GENERATED_DIR = Path("specs/generated")
TOOLS = "Read,Glob,Grep,Write"
RECENT_DAYS = 14

APPLIED_HEADING = "## Что принял код (записано кодом, не руководителем)"
GAP_HEADING = "Каких параметров не хватает"

_DIGITS = re.compile(r"\d[\d.,]*%?")
_RESEARCHER_MARK = re.compile(r"^Declared by the researcher \((?P<name>[^)]+)\)")
_GENERATOR_MARK = re.compile(r"^Generated from \S+ \((?P<mechanism>[^)]+)\)")
_GAP_ROW = re.compile(r"^\|\s*`(?P<id>[^`]+)`\s*\|(?P<gap>[^|]*)\|(?P<usable>[^|]*)\|\s*$")


# --------------------------------------------------------------------------
# Which direction an idea belongs to


@dataclass
class DirectionStats:
    """What one direction of the search has produced, aggregated. Holds counts
    and rule ids only -- never a metric of any trial."""

    direction: str
    kind: str  # generator | researcher | implementer
    ideas: int = 0
    created_recently: int = 0
    never_evaluated: int = 0
    statuses: Counter = field(default_factory=Counter)
    routes: Counter = field(default_factory=Counter)
    failed_rules: Counter = field(default_factory=Counter)
    #: Rules of kind `infrastructure` (venue_supported ...): a fact about OUR setup, not
    #: about the idea, so kept apart from `failed_rules` (rules 2026-10-01.1).
    failed_infrastructure: Counter = field(default_factory=Counter)
    survivors: list[str] = field(default_factory=list)


def generator_mechanism(idea_id: str, notes: str | None, root: Path) -> str | None:
    """The mechanism a generated idea tests: the `# mechanism:` header of its
    spec, else the id in the idea's note."""
    spec = root / GENERATED_DIR / f"{idea_id}.yaml"
    if spec.is_file():
        found = mechanism_of(spec.read_text(encoding="utf-8"))
        if found:
            return found
    match = _GENERATOR_MARK.match(notes or "")
    return match["mechanism"].strip() if match else None


def researcher_strategy(session: Session, idea_id: str, notes: str | None, plan: dict) -> str:
    """The researcher strategy an experiment belongs to: named in the idea's
    note, else the root of its parent chain, else the id's prefix."""
    from qlab.registry.strategies import strategy_root

    strategies = plan.get("strategies") or {}
    match = _RESEARCHER_MARK.match(notes or "")
    if match and match["name"] in strategies:
        return match["name"]
    root = strategy_root(session, idea_id)
    for name, entry in strategies.items():
        if entry.get("ideas") and strategy_root(session, entry["ideas"][0]) == root:
            return name
    prefix = idea_id.split("-research-")[0]
    return prefix if prefix in strategies else "unassigned"


def agent_ideas(session: Session) -> dict:
    """Idea id -> (idea, which agent made it), for every idea an agent created."""
    from qlab.registry.models import Idea

    ideas = {}
    for idea in session.query(Idea).filter(~Idea.id.startswith(NOISE_PREFIX)).all():
        source = idea_source(idea.id, idea.notes)
        if source is not None:
            ideas[idea.id] = (idea, source)
    return ideas


def _infrastructure_rule_ids(versions: set[str]) -> set[tuple[str, str]]:
    """`(rules_version, rule_id)` of every rule of kind `infrastructure` in the given
    versions. A version whose file cannot be loaded contributes nothing: its failed
    rules then count as strategy rules, the stricter reading."""
    from qlab.rules import RuleKind, load

    found: set[tuple[str, str]] = set()
    for version in versions:
        try:
            ruleset = load(version)
        except (FileNotFoundError, ValueError):
            continue
        found.update((version, r.id) for r in ruleset.rules if r.kind == RuleKind.INFRASTRUCTURE)
    return found


def direction_stats(session: Session, *, root: Path, plan: dict, config: dirs.DirectorConfig,
                    now: datetime) -> list[DirectionStats]:
    """Every agent-created idea, aggregated by direction: generator ideas by
    mechanism family, researcher experiments by strategy, implementer ideas
    together. The latest routed trial of each idea decides its route; the rules
    it failed are the `verdict` rows of that trial."""
    from qlab.registry.models import Verdict

    ideas = agent_ideas(session)
    latest = latest_routed_trials(session, list(ideas))
    failed: dict[int, list[str]] = {}
    failed_infra: dict[int, list[str]] = {}
    if latest:
        rows = (session.query(Verdict.trial_id, Verdict.rule_id, Verdict.rules_version)
                .filter(Verdict.trial_id.in_([t for t, _ in latest.values()]),
                        Verdict.passed.is_(False)).all())
        infrastructure = _infrastructure_rule_ids({version for _, _, version in rows})
        for trial_id, rule_id, version in rows:
            target = failed_infra if (version, rule_id) in infrastructure else failed
            target.setdefault(trial_id, []).append(rule_id)

    stats: dict[str, DirectionStats] = {}
    recent = now - timedelta(days=RECENT_DAYS)
    for idea_id in sorted(ideas):
        idea, source = ideas[idea_id]
        if source == "generator":
            mechanism = generator_mechanism(idea_id, idea.notes, root)
            direction = f"generator / {config.family_of(mechanism) if mechanism else 'unknown'}"
        elif source == "researcher":
            direction = f"researcher / {researcher_strategy(session, idea_id, idea.notes, plan)}"
        else:
            direction = "implementer"
        row = stats.setdefault(direction, DirectionStats(direction, source))
        row.ideas += 1
        created = idea.created_at
        created = created.replace(tzinfo=UTC) if created.tzinfo is None else created
        row.created_recently += created >= recent
        row.statuses[getattr(idea.status, "value", idea.status)] += 1
        if idea_id not in latest:
            row.never_evaluated += 1
            continue
        trial_id, route = latest[idea_id]
        row.routes[route] += 1
        row.failed_rules.update(sorted(set(failed.get(trial_id, []))))
        row.failed_infrastructure.update(sorted(set(failed_infra.get(trial_id, []))))
        if idea.status in SURVIVOR_STATUSES or route == "paper":
            row.survivors.append(idea_id)
    return sorted(stats.values(), key=lambda s: s.direction)


# --------------------------------------------------------------------------
# The catalogue's coverage


@dataclass
class MechanismCoverage:
    id: str
    family: str
    buildable_now: bool | None  # the catalogue's own table ("можно ли использовать")
    lacks: str  # parameters without values, from the same table
    not_expressed: str  # the catalogue's field «Чего блоки не выражают», shortened
    usable: str
    parameter_rows: int
    ideas_made: int


def parse_gap_table(catalogue_text: str) -> dict[str, tuple[str, str]]:
    """Mechanism id -> (what parameters it lacks, can the generator use it), the rows of
    the catalogue's last table («Каких параметров не хватает»)."""
    at = catalogue_text.find(GAP_HEADING)
    out: dict[str, tuple[str, str]] = {}
    if at < 0:
        return out
    for line in catalogue_text[at:].splitlines():
        match = _GAP_ROW.match(line.strip())
        if match:
            out[match["id"]] = (match["gap"].strip(), match["usable"].strip())
    return out


def catalogue_coverage(mechanisms: list[Mechanism], gaps: dict[str, tuple[str, str]],
                       config: dirs.DirectorConfig, made: Counter) -> list[MechanismCoverage]:
    rows = []
    for m in mechanisms:
        gap, usable = gaps.get(m.id, ("", ""))
        rows.append(MechanismCoverage(
            id=m.id, family=config.family_of(m.id),
            buildable_now=usable.lower().startswith("да") if usable else None,
            lacks=_shorten(gap, 300), not_expressed=_shorten(m.unexpressed, 300),
            usable=_shorten(usable, 300),
            parameter_rows=len(m.params), ideas_made=made.get(m.id, 0)))
    return sorted(rows, key=lambda r: (r.family, r.id))


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# Tokens, DROP proposals, the decay detector


def nights_in(night_dir: Path, since: datetime, until: datetime):
    """(day, saved state) of every night inside the period, oldest first."""
    for path in sorted(night_dir.glob("*.json")):
        try:
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
            state = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if since - timedelta(days=1) < day <= until:
            yield path.stem, state


def researcher_tokens_by_strategy(night_dir: Path, since: datetime, until: datetime
                                  ) -> dict[str, int]:
    """Tokens the nightly researcher spent per strategy, from the nights' saved
    state (the registry's ledger does not say which strategy a call was for)."""
    total: Counter = Counter()
    for _, state in nights_in(night_dir, since, until):
        for entry in state.get("research") or []:
            if entry.get("tokens") and not entry.get("stopped"):
                total[entry["strategy"]] += int(entry["tokens"])
    return dict(total)


def latest_decay_lines(night_dir: Path) -> tuple[str, list[str]] | None:
    """The newest night's decay-detector lines (day, lines), if any night has them."""
    for path in sorted(night_dir.glob("*.json"), reverse=True):
        try:
            decay = json.loads(path.read_text(encoding="utf-8")).get("decay")
        except (ValueError, OSError):
            continue
        if decay and decay.get("lines"):
            return path.stem, list(decay["lines"])
    return None


def mask_numbers(text: str) -> str:
    """Numbers in quoted text are replaced by `#`: a DROP reason may quote how a
    variant did, and that must not reach the director."""
    return _DIGITS.sub("#", text)


def drop_proposals(plan: dict, memo_dir: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Per strategy: (memo name, idea, masked reason) of the DROP lines in its
    newest memo."""
    from qlab.agents.researcher import parse_drop_proposals
    from qlab.night import latest_memo

    out: dict[str, list[tuple[str, str, str]]] = {}
    for name in plan.get("strategies") or {}:
        memo = latest_memo(name, memo_dir)
        if memo is None:
            continue
        found = parse_drop_proposals(memo.read_text(encoding="utf-8"))
        if found:
            out[name] = [(memo.name, d.idea_id, mask_numbers(d.why)) for d in found]
    return out


# --------------------------------------------------------------------------
# The evidence file


def _counts(counter: Counter) -> str:
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{key} {n:,}" for key, n in ranked) or "none"


def _funnel_lines(period: PeriodFunnel, researcher_tokens: dict[str, int]) -> list[str]:
    by = {row.source: row for row in period.by_source}
    lines = [f"### {period.label} ({period.since:%Y-%m-%d} .. {period.until:%Y-%m-%d})",
             f"- ideas made by agents: {period.created} (generator {by['generator'].created}, "
             f"researcher {by['researcher'].created}, implementer {by['implementer'].created}); "
             f"the scout wrote {period.scout_drafts} card drafts in {period.scout_calls} calls",
             f"- evaluated on the stand: {period.evaluated}; where the latest trial sent them: "
             f"{_counts(Counter(period.routes))}",
             f"- survivors (passed every rule, waiting for the owner): {period.survivors}"
             + (f" ({', '.join(period.survivor_ids)})" if period.survivor_ids else ""),
             f"- tokens: {period.tokens_total:,} in all; by agent: "
             f"{_counts(Counter(period.tokens_by_agent))}; by stage: "
             f"{_counts(Counter(period.tokens_by_stage))}",
             "- price of one survivor: "
             + (f"{period.tokens_per_survivor:,} tokens" if period.tokens_per_survivor
                else "none (no survivors)")]
    if researcher_tokens:
        lines.append("- researcher tokens by strategy (from the nights' saved state): "
                     f"{_counts(Counter(researcher_tokens))}")
    return lines


def build_evidence(session: Session, *, root: Path, config: dirs.DirectorConfig, plan: dict,
                   now: datetime | None = None, night_dir: Path | None = None,
                   catalogue_path: Path = CATALOGUE) -> str:
    """The evidence file, written by code. No per-parameter performance of any
    individual variant is in it (module docstring); every number is a count, a
    token total, or something the decay detector printed."""
    now = now or datetime.now(UTC)
    night_dir = night_dir if night_dir is not None else root / NIGHT_DIR
    catalogue_file = catalogue_path if catalogue_path.is_absolute() else root / catalogue_path
    mechanisms = catalogue(catalogue_file)
    gaps = parse_gap_table(catalogue_file.read_text(encoding="utf-8"))
    stats = direction_stats(session, root=root, plan=plan, config=config, now=now)
    made = Counter(mechanism for idea_id, (idea, source) in agent_ideas(session).items()
                   if source == "generator"
                   and (mechanism := generator_mechanism(idea_id, idea.notes, root)))
    coverage = catalogue_coverage(mechanisms, gaps, config, made)
    strategy_names = sorted(plan.get("strategies") or {})
    family_names = sorted({c.family for c in coverage})

    lines = [f"# Evidence for the research director (built {now:%Y-%m-%d}, by code)", "",
             "Every number below is a count, a token total, or a line the decay detector "
             "printed for a running book. There is NO per-parameter or per-variant "
             "performance (no return, Sharpe or drawdown of any variant, no metric of any "
             "trial) in this file, on purpose: you set directions, not parameter values, and "
             "must not look for variant results elsewhere (docs/research memos, night reports, "
             "specs of single variants). Rejection reasons are rule ids; values are not shown.",
             "", "## Names you may use in the directions block", "",
             f"- mechanism ids (the catalogue {catalogue_path}): "
             f"{', '.join(m.id for m in mechanisms)}",
             f"- mechanism families (night/director.yaml): {', '.join(family_names)}",
             f"- researcher strategies (night/research.yaml): {', '.join(strategy_names)}",
             "- priorities: " + ", ".join(dirs.PRIORITIES), ""]

    lines += ["## Funnel", ""]
    for label, since in (("last-14-days", now - timedelta(days=RECENT_DAYS)),
                         ("since-agents-started", AGENT_WORK_START)):
        period = period_funnel(session, since, now, label=label, night_dir=night_dir)
        lines += [*_funnel_lines(period, researcher_tokens_by_strategy(night_dir, since, now)),
                  ""]

    lines += ["## Verdicts by direction", "",
              "Every idea an agent created, by the direction it belongs to: generator ideas by "
              "the mechanism family of their spec's `# mechanism:` header, researcher "
              "experiments by their strategy. Route = where the idea's LATEST routed trial sent "
              "it; failed rules = strategy rule ids that trial failed (an idea counts once per "
              "rule); infrastructure facts = failed rules of kind `infrastructure` "
              "(venue_supported: no order adapter of ours on the data venue), counted apart; "
              "status = the idea's status today.", ""]
    if not stats:
        lines += ["no agent-created idea in the registry", ""]
    for row in stats:
        lines += [f"### {row.direction}",
                  f"- ideas: {row.ideas} (made in the last {RECENT_DAYS} days: "
                  f"{row.created_recently}); never evaluated: {row.never_evaluated}",
                  f"- status today: {_counts(row.statuses)}",
                  f"- route of the latest trial: {_counts(row.routes)}",
                  f"- rules failed by the latest trial: {_counts(row.failed_rules)}",
                  "- infrastructure facts failed (about our setup, not about the idea; "
                  f"never a reason to close a direction): {_counts(row.failed_infrastructure)}",
                  f"- survivors: {', '.join(row.survivors) or 'none'}", ""]

    lines += ["## Catalogue coverage", "",
              f"{catalogue_path}: {len(mechanisms)} mechanisms. `buildable now` = the catalogue's "
              "own table says the generator can use it; `lacks` = the parameters the table has "
              "no values for (or new blocks it needs); `not expressed` = what the blocks cannot "
              "express of the source's version (recorded in differs_from_source, or fatal when "
              "the first item says the mechanism cannot be assembled). `ideas made` counts "
              "generator ideas that tested the mechanism.", ""]
    families: dict[str, list[MechanismCoverage]] = {}
    for row in coverage:
        families.setdefault(row.family, []).append(row)
    for family, rows in families.items():
        buildable = sum(1 for r in rows if r.buildable_now)
        lines += [f"### family {family}: {len(rows)} mechanism(s), {buildable} buildable now"]
        for r in rows:
            state = {True: "yes", False: "NO", None: "unknown"}[r.buildable_now]
            lines += [f"- {r.id}: buildable now {state}; parameter rows {r.parameter_rows}; "
                      f"ideas made {r.ideas_made}",
                      f"    lacks: {r.lacks or 'nothing recorded'}",
                      f"    not expressed: {r.not_expressed or 'nothing recorded'}"]
        lines.append("")
    unlisted = [r.id for r in coverage if r.family == r.id]
    if unlisted:
        lines += [f"(mechanisms with no family in night/director.yaml, each counted as its own: "
                  f"{', '.join(unlisted)})", ""]

    lines += ["## The researcher: DROP proposals and the owner's ideas", ""]
    drops = drop_proposals(plan, root / MEMO_DIR)
    for name, entry in (plan.get("strategies") or {}).items():
        lines.append(f"### strategy {name}")
        for memo_name, idea, why in drops.get(name, []):
            lines.append(f"- DROP proposal in {memo_name}: {idea} -- {why} (numbers masked)")
        if name not in drops:
            lines.append("- no DROP proposal in its newest memo")
        for idea in entry.get("owner_ideas") or []:
            lines.append(f"- the owner's idea, verbatim: «{idea}»")
        lines.append("")

    lines += ["## Decay detector (running books against the stand's own history)", ""]
    decay = latest_decay_lines(night_dir)
    if decay is None:
        lines += ["no night has decay-detector lines yet", ""]
    else:
        lines += [f"night {decay[0]}:", *decay[1], ""]

    lines += ["## Your previous directions", ""]
    found = dirs.read_directions(root)
    if found.status == "missing":
        lines += ["none: this is the first run; its directions will wait for the owner's "
                  "review before any code acts on them", ""]
    else:
        lines += [f"status: {found.status}"
                  + (" (the generator and the researcher ignore the file until the owner "
                     "clears `pending_owner_review`)" if found.status == "pending" else ""),
                  "```yaml",
                  (root / dirs.DIRECTIONS).read_text(encoding="utf-8").rstrip(), "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The agent


INSTRUCTIONS = """You are q-lab's research director. q-lab is a research lab that finds and
filters trading strategies; it never places orders, and its verdicts are made by code, not by
you. Two agents search for strategies: the idea generator (writes new specs from the catalogue
of published mechanisms, {catalogue}) and the researcher (improves four named strategies,
night/research.yaml). Nobody has been watching the DIRECTIONS of the search as a whole. You do.

The owner's words, 2026-10-25, verbatim: «я думаю в агентах не хватает агента эксперта - который
бы следил за направлениями поиска и корректировал действия генератора идей исследователя».
And: «запуск руководителя раз в 2 дня - добавляй». So you run every two days.

The evidence, built by code from the registry (read it first; every number you may quote is in
it): {context_path}. You may also read the catalogue {catalogue}, docs/NIGHT.md,
docs/FIT_VS_FORWARD.md and docs/BLOCKS.md.

Decide, for each direction the evidence shows: keep going, give it more priority, pause it,
or propose to close it -- and say what the search lacks: mechanisms the catalogue should gain,
data the lab should obtain. Typical questions: which families of mechanisms keep ending in the
same rejection, and is that the family's fault or a gap of the blocks; which families have
mechanisms nobody has tried; where the tokens go against what comes out; which of the
researcher's strategies deserves the night's budget first when it is short.

What you must NOT do:
- You set directions, not parameter values. Never choose, suggest or hint at a parameter of any
  strategy or variant (a window, a threshold, a coin list, a timeframe, a number of coins). The
  evidence file has no per-variant performance on purpose: otherwise the choice of direction
  would be tuned to forward data. Do not look for it: do not open the researcher's memos in
  docs/research, docs/STATE.md, the cards in cards/, the night reports, data/night, or the
  specs of single variants -- they quote how variants did. Say in the memo what you relied on.
- You pass no verdicts and change no status. A pause or a priority is advice that code applies
  after checking it; `propose_close_family` is only shown to the owner, who decides.
- Never invent a number. A direction with few trials is "too early to tell", not "dead": every
  pause or closing proposal must name the evidence in the file that supports it (which rules
  keep failing, which part of the catalogue is missing), and an empty `pause_mechanisms` is a
  fine answer.
- Write exactly one file, {memo_path}, and nothing else; run nothing.

The memo is for the owner, IN RUSSIAN, in plain words (explain a term at its first use; do not
write English trading words in Russian letters). Sections: what the funnel says about the
search as a whole; direction by direction, what the evidence shows and what you conclude; the
decisions you make (priorities, preferred and paused mechanisms) and why; what you propose to
close, if anything (the owner decides); what the catalogue and the data lack; what you relied
on and what you could not see.

End the memo with ONE fenced block, exactly in this shape (YAML; keys you do not need may be
omitted). Code parses it, drops what does not check out, and tells the owner what it dropped:

```directions
generator:
  prefer_mechanisms: [<mechanism ids from the catalogue>]
  pause_mechanisms:
    - {{id: <mechanism id>, why: "<one sentence, plain words>"}}
researcher:
  <strategy name from night/research.yaml>: {{priority: high|normal|low, note: "<plain text>"}}
catalogue_requests:
  - {{family: <family>, look_for: "<what a future catalogue agent should look for>", why: "<why>"}}
data_requests:
  - {{what: "<data the lab lacks>", why: "<why>"}}
propose_close_family:
  - {{family: <a mechanism family or a researcher strategy>, why: "<why>"}}
```

Rules for the block: mechanism ids, families and strategy names must be exactly those listed in
the evidence file (an unknown one is rejected). A researcher `note` is plain text WITHOUT any
digit (a digit is a parameter value; the note is dropped if it has one). The researcher reads
the note as advice for where to look, next to the owner's own question; the generator reads the
preferred and paused lists in its instructions, and code rejects a spec of a paused mechanism.

Reply with one line: the memo path."""


@dataclass
class DirectorOutcome:
    memo: Path | None
    reply: str | None = None
    problems: list[str] = field(default_factory=list)
    stray: list[str] = field(default_factory=list)


def direct(*, budget: CandidateBudget | ExplicitNoBudget, session: Session,
           config: dirs.DirectorConfig, plan: dict, model: str = MODEL,
           max_turns: int | None = None, catalogue_path: Path = CATALOGUE) -> DirectorOutcome:
    """One agent call: build the evidence, let the director write its memo.
    The call goes through the budget guard. Applying the memo's block is a
    separate, token-free step (`apply_memo`), so a restart does not pay twice."""
    today = datetime.now(UTC).date()
    root = QLAB_ROOT
    (root / CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    (root / MEMO_DIR).mkdir(parents=True, exist_ok=True)
    context_path = CONTEXT_DIR / f"director-{today}.md"
    (root / context_path).write_text(
        build_evidence(session, root=root, config=config, plan=plan,
                       catalogue_path=catalogue_path), encoding="utf-8")
    memo_path = MEMO_DIR / f"director-{today}.md"

    directions_file = root / dirs.DIRECTIONS
    kept = directions_file.read_bytes() if directions_file.is_file() else None
    before = set(_changed_paths())
    started = datetime.now().timestamp()
    prompt = INSTRUCTIONS.format(catalogue=catalogue_path, context_path=context_path,
                                 memo_path=memo_path)
    result = run_agent(budget, prompt, session=session, agent="director", model=model,
                       max_turns=max_turns or config.max_turns, cwd=root,
                       extra_args=("--allowedTools", TOOLS, "--disallowedTools", "Bash,Edit"))
    outcome = DirectorOutcome(memo=None, reply=(result.text or "").strip()[:600])
    if result.exit_code != 0:
        outcome.problems.append(
            f"the agent stopped before finishing (exit {result.exit_code}; its spending cap or "
            "turn limit): what it wrote may be incomplete")

    # The directions file belongs to code: whatever the agent did to it is undone, and only
    # the validated block of its memo can change it. Other strays are reported, not reverted
    # (2026-10-25: a guard that reverted "outside" files destroyed another agent's work).
    if (directions_file.read_bytes() if directions_file.is_file() else None) != kept:
        if kept is None:
            directions_file.unlink()
        else:
            directions_file.write_bytes(kept)
        outcome.problems.append(f"the agent changed {dirs.DIRECTIONS}: restored")
    for path in sorted(set(_changed_paths()) - before):
        if path not in (str(memo_path), str(dirs.DIRECTIONS)):
            outcome.stray.append(path)
    if outcome.stray:
        outcome.problems.append(
            f"files outside its place changed during the run (not reverted -- maybe someone "
            f"else's work; check them): {outcome.stray}")
    memo = root / memo_path
    if memo.is_file() and memo.stat().st_mtime >= started - 1:
        outcome.memo = memo_path
    else:
        outcome.problems.append("no memo written")
    return outcome


# --------------------------------------------------------------------------
# From the memo to night/directions.yaml


@dataclass
class ApplyResult:
    directions: dirs.Directions | None
    problems: list[str]
    written: Path | None = None


def _code_section(result: ApplyResult) -> list[str]:
    d = result.directions
    lines = [APPLIED_HEADING, ""]
    if d is None:
        lines += ["Блок направлений не принят, файл `night/directions.yaml` не менялся:", ""]
        lines += [f"- {p}" for p in result.problems]
        return lines
    lines += ["Принято кодом и записано в `night/directions.yaml`"
              + (" — **с пометкой «ждёт проверки владельца»: генератор и исследователь файл "
                 "пока не читают**" if d.pending_owner_review else "") + ":", "",
              f"- предпочитать механизмы: {', '.join(d.prefer_mechanisms) or '—'}",
              f"- приостановлены механизмы: "
              f"{', '.join(f'{p.id} ({p.why})' for p in d.pause_mechanisms) or '—'}"]
    for name, s in d.researcher.items():
        lines.append(f"- исследователь, {name}: важность {s.priority}"
                     + (f"; пометка: {s.note}" if s.note else ""))
    lines += [f"- просьб к каталогу: {len(d.catalogue_requests)}, просьб о данных: "
              f"{len(d.data_requests)}",
              f"- предложено закрыть (решает владелец, код этого не делает): "
              f"{', '.join(c.family for c in d.propose_close_family) or '—'}"]
    if result.problems:
        lines += ["", "Отброшено кодом:", *[f"- {p}" for p in result.problems]]
    return lines


def apply_memo(memo: Path, *, root: Path, mechanism_ids: set[str],
               strategy_names: set[str], family_names: set[str], day: date) -> ApplyResult:
    """Take the directions block of the memo, validate it, write the file, and add
    a section to the memo saying what code accepted. No tokens. Idempotent: a repeat
    on the same memo replaces the section it added."""
    memo_file = memo if memo.is_absolute() else root / memo
    text = memo_file.read_text(encoding="utf-8")
    if APPLIED_HEADING in text:
        text = text[:text.index(APPLIED_HEADING)].rstrip() + "\n"
    block = dirs.extract_block(text)
    if block is None:
        result = ApplyResult(None, ["the memo has no ```directions block"])
    else:
        # The flag is decided from the file this run replaces, before it is replaced.
        pending = dirs.pending_for_new(root)
        parsed = dirs.parse_block(block, mechanism_ids=mechanism_ids,
                                  strategy_names=strategy_names, family_names=family_names,
                                  day=day, memo=str(memo), pending_owner_review=pending)
        result = ApplyResult(parsed.directions, parsed.problems)
        if parsed.directions is not None:
            result.written = dirs.write_directions(parsed.directions, root)
    memo_file.write_text(text.rstrip() + "\n\n" + "\n".join(_code_section(result)) + "\n",
                         encoding="utf-8")
    return result


def known_names(plan: dict, config: dirs.DirectorConfig, mechanisms: list[Mechanism]
                ) -> tuple[set[str], set[str], set[str]]:
    """(mechanism ids, strategy names, family names) a block may use."""
    return ({m.id for m in mechanisms}, set(plan.get("strategies") or {}),
            {config.family_of(m.id) for m in mechanisms})


def summary(directions: dirs.Directions) -> dict:
    """What the night keeps of the directions for its report."""
    return yaml.safe_load(yaml.safe_dump(directions.to_dict()))


__all__ = ["APPLIED_HEADING", "INSTRUCTIONS", "MODEL", "ApplyResult", "DirectionStats",
           "DirectorOutcome", "MechanismCoverage", "agent_ideas", "apply_memo", "build_evidence",
           "catalogue_coverage", "direct", "direction_stats", "known_names", "mask_numbers",
           "parse_gap_table", "summary"]
