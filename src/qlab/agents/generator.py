"""The idea generator: published mechanisms -> new strategy specs, no new code
(docs/BLOCKS.md, docs/MECHANISMS.md).

The left half of the search loop (find an idea -> write code) used to need an
agent to write Python. A strategy composed of ready blocks
(`qlab.strategies.blocks:Composed`) is only a YAML spec, and the catalogue
`docs/MECHANISMS.md` lists published mechanisms with the numbers their sources
report. This agent reads the catalogue and writes up to `max_specs` new specs
under `specs/generated/`. It may read anything and write only there; it runs
nothing. Code then guards what it wrote and runs what passes:

1. anything written outside `specs/generated/` is reverted, and an existing
   generated spec that was edited is restored;
2. a spec must load, carry the code_ref `qlab.strategies.blocks:Composed`, a
   `gen-` idea id, a `# mechanism: <id>` header naming a catalogue mechanism,
   Binance daily data from 2019-09-08 on up to the end the agent was told, and
   the costs of the reference Binance spec;
3. every number in `params` is one of the values the mechanism's own table gives
   for exactly that parameter (or the framework's `sizing.gross` 1), cites the
   catalogue and appears in it literally; a `derived:` formula may combine only
   values from the mechanism's table; a parameter the table has no row for
   cannot be used;
4. `qlab.pipeline.sources.source_problems` passes, the idea is not already in
   the registry or in `specs/generated/`, and a mechanism the blocks do not
   fully express is declared in `unexpressed_mechanisms`;
5. the stand evaluates each valid spec exactly as it evaluates the
   researcher's experiments (matched noise, every rule); a spec that fails a
   guard is reported and not run. Every run writes its trial.

The verdict is the code's, never the agent's. The agent's spending goes
through the budget guard (`qlab.budget.run_agent`): without a budget the call
fails instead of spending silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from qlab.agents.implementer import _changed_paths, _revert
from qlab.budget import CandidateBudget, ExplicitNoBudget, run_agent
from qlab.pipeline.sources import QLAB_ROOT

MODEL = "sonnet"  # docs/PLAN.md: the agents that write specs run on sonnet
CATALOGUE = Path("docs/MECHANISMS.md")
SPEC_DIR = Path("specs/generated")
CONTEXT_DIR = Path("data/research")
REFERENCE_SPEC = Path("specs/trend-binance-cap.yaml")
CODE_REF = "qlab.strategies.blocks:Composed"
ID_PREFIX = "gen-"
DATA_SOURCE = "binance"
DATA_INTERVAL = "1d"
DATA_START = date(2019, 9, 8)
"""The first day Binance USDT perpetuals exist in the archive; the reference
Binance specs start here."""
TOOLS = "Read,Glob,Grep,Write"
MECHANISM_HEADER = re.compile(r"^#\s*mechanism:\s*(\S+)\s*$", re.MULTILINE)
RESULT_METRICS = ("ann_return_net", "sharpe_net", "max_dd", "noise_return_percentile",
                  "min_capital_usd", "forward_days", "forward_days_needed")


# --------------------------------------------------------------------------
# The catalogue


@dataclass(frozen=True)
class ParamRow:
    """One row of a mechanism's parameter table."""

    param: str  # the dotted path in the spec's params, e.g. "signal.lookback_days"
    values: tuple[float, ...]  # every number in the "values from the source" cell
    values_text: str
    source: str


@dataclass(frozen=True)
class Mechanism:
    id: str
    summary: str
    blocks: str
    checked: str
    params: tuple[ParamRow, ...]
    unexpressed: str
    sources: str

    @property
    def fully_expressed(self) -> bool:
        """The blocks express the whole mechanism: the catalogue's "not
        expressed" field is empty or says there is nothing."""
        text = re.sub(r"[\s*_`.]+", " ", self.unexpressed).strip().lower()
        return text in ("", "-", "—", "–") or text.startswith(("ничего", "нет", "всё выражается"))


_LABELS = {
    "суть": "summary",
    "блоки": "blocks",
    "где проверено": "checked",
    "параметры": "params",
    "чего блоки не выражают": "unexpressed",
    "источники": "sources",
}
_LABEL_LINE = re.compile(r"^\*\*([^*]+?)\s*:?\*\*\s*:?\s*(.*)$")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


_PARENS = re.compile(r"\([^()]*\)")
_LEADING_VALUES = re.compile(r"\s*-?\d+(?:\.\d+)?%?(?:\s*,\s*-?\d+(?:\.\d+)?%?)*")


def cell_numbers(text: str) -> tuple[float, ...]:
    """The values a table cell gives: the comma-separated numbers it opens
    with; `5%` also yields 0.05. Notes in parentheses and anything after the
    first word are commentary, not values -- "28 (4 недели)" is 28, not 4, and
    "365 (12 месяцев); другие сроки таблицы 2" is 365 only."""
    plain = text
    while (stripped := _PARENS.sub("", plain)) != plain:
        plain = stripped
    lead = _LEADING_VALUES.match(plain)
    if lead is None:
        return ()
    out: list[float] = []
    for match in _NUMBER.finditer(lead.group()):
        value = float(match.group())
        out.append(value)
        if lead.group()[match.end():match.end() + 1] == "%":
            out.append(value / 100)
    return tuple(dict.fromkeys(out))


def _param_rows(lines: list[str]) -> tuple[ParamRow, ...]:
    rows: list[ParamRow] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: ") or cells[0].lower().startswith("параметр"):
            continue
        rows.append(ParamRow(param=cells[0].strip("`"), values=cell_numbers(cells[1]),
                             values_text=cells[1], source=cells[2]))
    return tuple(rows)


def parse_catalogue(text: str) -> list[Mechanism]:
    """Mechanisms of a catalogue text in `docs/MECHANISMS.md`'s format."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("### "):
            sections.append((line[4:].strip().strip("`"), []))
        elif re.match(r"#{1,2} ", line):
            sections.append(("", []))  # a heading above the level: ends the section
        elif sections:
            sections[-1][1].append(line)
    out = []
    for mech_id, body in sections:
        if not mech_id:
            continue
        fields: dict[str, list[str]] = {}
        current: str | None = None
        for line in body:
            m = _LABEL_LINE.match(line.strip())
            if m and m.group(1).strip().lower() in _LABELS:
                current = _LABELS[m.group(1).strip().lower()]
                fields[current] = [m.group(2)] if m.group(2) else []
            elif current is not None:
                fields[current].append(line)

        def text_of(name: str, fields=fields) -> str:
            return "\n".join(fields.get(name, [])).strip()

        out.append(Mechanism(
            id=mech_id, summary=text_of("summary"), blocks=text_of("blocks"),
            checked=text_of("checked"), params=_param_rows(fields.get("params", [])),
            unexpressed=text_of("unexpressed"), sources=text_of("sources")))
    return out


def catalogue(path: Path = CATALOGUE) -> list[Mechanism]:
    """Parse the catalogue file (relative paths are read from the project root)."""
    path = path if path.is_absolute() else QLAB_ROOT / path
    return parse_catalogue(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The guards


@dataclass
class GeneratedSpec:
    path: Path  # relative to the project root
    spec: object  # qlab.pipeline.spec.StrategySpec
    mechanism: str


@dataclass
class Rejected:
    path: str
    problems: list[str]


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else QLAB_ROOT / path


def _param_leaves(prefix: str, value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _param_leaves(f"{prefix}.{key}" if prefix else str(key), item)
    else:
        yield prefix, value


def mechanism_of(spec_text: str) -> str | None:
    """The mechanism id a generated spec declares in its `# mechanism:` header."""
    m = MECHANISM_HEADER.search(spec_text)
    return m.group(1) if m else None


FRAMEWORK_VALUES: dict[str, tuple[float, ...]] = {"sizing.gross": (1.0,)}
"""Numbers a spec may use without a catalogue row. `sizing.gross` 1 means the
whole capital, no leverage: the stand accounts every portfolio that way, so it
is not a finding of any paper and no mechanism needs to give it. Nothing else
belongs here: any other number must come from the mechanism's own table."""


DERIVABLE = frozenset({"min_history_days"})
"""Parameters a `derived:` formula may set. The history a coin needs before it
is traded is bookkeeping -- enough bars to compute the longest window the
blocks read (e.g. "derived: 28; the signal's lookback") -- not a finding of a
paper. A formula elsewhere would let catalogue numbers be combined into values
no paper used (a rebalance of 7 * 2 = 14 where the paper gave only 7)."""


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _in(number: float, values) -> bool:
    return any(_close(number, v) for v in values)


def _catalogue_citation_problems(spec, mechanism: Mechanism | None, catalogue_path: Path,
                                 catalogue_text: str) -> list[str]:
    """Every numeric leaf of `params` is a value the spec's mechanism gives in the
    table row for exactly that parameter path, or a `FRAMEWORK_VALUES` entry. A
    parameter the table has no row for cannot be used, even if its number
    appears somewhere else in the catalogue. A `derived:` leaf is exempt from
    the row check (the formula is evaluated by `source_problems`), but each
    operand of the formula must be a value from some row of the same mechanism.
    (`source_problems` then checks that a number cited to a file appears in it
    literally.)"""
    from qlab.pipeline.sources import _DERIVED, _numbers, covering_key

    problems: list[str] = []
    rows = {r.param: r for r in mechanism.params} if mechanism else {}
    mechanism_values = tuple(v for r in rows.values() for v in r.values)
    name = mechanism.id if mechanism else "?"
    for path, value in _param_leaves("", spec.params):
        numbers = _numbers(value)
        if not numbers:
            continue
        key = covering_key(path, spec.sources)
        citation = spec.sources.get(key, "") if key else ""
        derived = _DERIVED.match(citation)
        if derived and path not in DERIVABLE:
            problems.append(f"{path}: a formula may set only {sorted(DERIVABLE)}; {path} must be "
                            f"a value from {name}'s own row")
            continue
        if derived:
            operands = [float(n) for n in _NUMBER.findall(derived.group(1))]
            if mechanism is not None:
                absent = [n for n in operands if not _in(n, mechanism_values)]
                if absent:
                    problems.append(f"{path}: derived from numbers that are not values in any "
                                    f"row of {name} in {catalogue_path}: {absent}")
            continue
        if str(catalogue_path) not in citation:
            problems.append(f"{path} = {value!r} does not cite {catalogue_path}: the numbers come "
                            "from the catalogue only")
        if mechanism is None:
            continue  # reported once, at the missing mechanism
        row = rows.get(path)
        allowed = (*(row.values if row else ()), *FRAMEWORK_VALUES.get(path, ()))
        off = [n for n in numbers if not _in(n, allowed)]
        if off and not allowed:
            problems.append(f"{path} = {value!r}: the catalogue gives no value for {path} in "
                            f"{name}")
        elif off:
            given = row.values_text if row and row.values else ""
            if path in FRAMEWORK_VALUES:
                framework = [f"{v:g}" for v in FRAMEWORK_VALUES[path]]
                given = ", ".join(filter(None, [given, *framework]))
            problems.append(f"{path} = {value!r}: {off} is not among the catalogue's values "
                            f"for it in {name} ({given})")
    return problems


def check_generated(path: Path, *, mechanisms: list[Mechanism], catalogue_path: Path = CATALOGUE,
                    data_end: date | None = None, registry_idea_ids: frozenset[str] = frozenset(),
                    other_params: dict[tuple[str, str], str] | None = None,
                    reference: Path | None = REFERENCE_SPEC
                    ) -> tuple[GeneratedSpec | None, list[str]]:
    """Every guard on one spec file the agent wrote. Returns the spec (if it
    loads) and the problems; a spec with problems is reported, not run."""
    from qlab.pipeline.sources import source_problems
    from qlab.pipeline.spec import load_spec

    try:
        spec = load_spec(_abs(path))
    except Exception as exc:  # noqa: BLE001 - the report says why
        return None, [f"does not load: {exc}"]
    problems: list[str] = []
    if path.stem != spec.idea_id:
        problems.append(f"file name {path.name!r} is not idea_id {spec.idea_id!r}")
    if not spec.idea_id.startswith(ID_PREFIX):
        problems.append(f"idea_id {spec.idea_id!r} does not start with {ID_PREFIX!r}")
    if spec.idea_id in registry_idea_ids:
        problems.append(f"idea_id {spec.idea_id!r} is already in the registry")
    if spec.code_ref != CODE_REF:
        problems.append(f"code_ref is {spec.code_ref!r}, must be exactly {CODE_REF!r}")
    mechanism_id = mechanism_of(_abs(path).read_text(encoding="utf-8"))
    mechanism = next((m for m in mechanisms if m.id == mechanism_id), None)
    if mechanism is None:
        problems.append("no '# mechanism: <id>' header naming a mechanism of the catalogue"
                        if mechanism_id is None
                        else f"mechanism {mechanism_id!r} is not in the catalogue")

    data = spec.data
    if (data.source, data.interval) != (DATA_SOURCE, DATA_INTERVAL):
        problems.append(f"data must be {DATA_SOURCE} {DATA_INTERVAL}, got "
                        f"{data.source} {data.interval}")
    if data.start < DATA_START:
        problems.append(f"data.start {data.start} is before {DATA_START}, where Binance's "
                        "history begins")
    if data_end is not None and data.end != data_end:
        problems.append(f"data.end is {data.end}, must be the latest snapshot end {data_end}")
    if data.instruments is not None or spec.required_instruments is not None:
        problems.append("no hand-picked instrument list: the universe block selects coins")
    if data.min_daily_volume_usd is not None:
        problems.append("no liquidity floor outside the universe block")
    for name in ("exploratory", "params_fit_from"):
        if getattr(spec, name) is not None:
            problems.append(f"{name} is not the agent's to set")
    if spec.review_answers or spec.selects_causally:
        problems.append("review_answers and selects_causally are not the agent's to set")

    if reference is not None and _abs(reference).is_file():
        ref = load_spec(_abs(reference))
        if (spec.costs != ref.costs or spec.min_leg_notional != ref.min_leg_notional):
            problems.append(f"costs and min_leg_notional must be copied from {reference}")
    if mechanism is not None and not mechanism.fully_expressed and not spec.unexpressed_mechanisms:
        problems.append("the catalogue says the blocks do not express the whole mechanism "
                        f"({mechanism.unexpressed[:120]!r}) but unexpressed_mechanisms is empty")

    key = (spec.code_ref, json.dumps(spec.params, sort_keys=True, default=str))
    if (other := (other_params or {}).get(key)) is not None and other != spec.idea_id:
        problems.append(f"same params as the already existing {other}")

    catalogue_text = _abs(catalogue_path).read_text(encoding="utf-8")
    problems += _catalogue_citation_problems(spec, mechanism, catalogue_path, catalogue_text)
    problems += [f"source: {p}" for p in source_problems(spec)]
    return GeneratedSpec(path=path, spec=spec, mechanism=mechanism_id or ""), problems


# --------------------------------------------------------------------------
# The agent


INSTRUCTIONS = """You are q-lab's idea generator. q-lab is a research lab that finds and
filters trading strategies; it never places orders, and its verdicts are made by code, not by
you. Your task: turn published mechanisms into NEW strategy specs. You write no code: a
strategy is a spec composed of ready blocks (code_ref {code_ref}).

Read these first, they define what you write:
- {catalogue} -- the ONLY source of mechanisms and numbers you may use. Each mechanism lists
  the blocks that express it, a table of parameter values from its sources, and what the
  blocks cannot express.
- docs/BLOCKS.md and src/qlab/strategies/blocks/ -- the blocks and exactly which keys each
  one takes (no defaults; a missing or extra key is an error).
- docs/SOURCES.md and src/qlab/pipeline/sources.py -- every number must cite where it comes
  from; a number cited to a file must appear in that file LITERALLY.
- docs/FIT_VS_FORWARD.md and src/qlab/pipeline/spec.py -- every spec field and what
  params_fixed_at means.
- {reference} -- a Binance spec whose costs, min_leg_notional and the `sources` entries for
  them you copy unchanged.
- {context_path} -- written by code: the idea ids already in the registry, the specs
  already generated, and the data end you must use.

Write at most {max_specs} specs, each ONE file {spec_dir}/<idea_id>.yaml, and nothing else:
you may not write anywhere else, and a stray file is reverted. Do not edit existing files.
Each spec tests ONE mechanism of the catalogue that is not yet in the registry or in
{spec_dir}/ (a different mechanism, or the same one with a different set of the catalogue's
values, but never a copy). Write fewer than {max_specs} -- or none -- if nothing new is worth
a trial, and say why in your reply. Every run is a trial counted in deflation.

Shape of each spec (comments allowed; the first line is mandatory):

  # mechanism: <mechanism id exactly as in the catalogue>
  idea_id: gen-<short-name>
  title: <a Russian title in plain words>
  code_ref: {code_ref}
  params:
    universe, signal, portfolio, sizing, rebalance_days, min_history_days  (as in BLOCKS.md)
  data:
    source: {data_source}
    interval: {data_interval}
    start: <{data_start} or later>
    end: {data_end}
  costs, min_leg_notional, simultaneous_legs   (copied from {reference})
  noise_universe: panel
  unexpressed_mechanisms: []
  params_fixed_at / params_fixed_evidence   (see below)
  sources: ...

Rules:
- Use only mechanisms and numbers from {catalogue}. Never invent a number. EVERY number in
  params must be one of the values in the chosen mechanism's OWN table, in the row for that
  very parameter (the row's name is the dotted path in params: signal.lookback_days,
  portfolio.quantiles, universe.n, universe.window_days, rebalance_days, min_history_days,
  sizing.window_days, ...). A value from another mechanism, from another row, or from a note
  or the prose of the catalogue is not allowed, and code rejects the spec. A parameter the
  mechanism's table has no row for (or an empty row) cannot be used at all: choose blocks
  that need only the parameters the table gives (for example universe rule `all`, which takes
  none, and sizing `equal`), and if the mechanism leaves you no usable values, skip it and
  say so in your reply.
- Portfolios: the papers sort coins into quantile groups (long the top fifth, short the
  bottom fifth), so use `long_short_quantile` or `long_only_quantile` with the mechanism's
  portfolio.quantiles value. `top_n` (a count of coins) is allowed only if the mechanism's
  table has a portfolio.top_n row with a value.
- Other blocks the catalogue's "blocks" field may name (their parameter paths are checked the
  same way, against the mechanism's own row):
  * portfolio `long_short_fraction` / `long_only_fraction` take portfolio.fraction, the SHARE
    of the eligible coins per side ("the top and bottom 30%" is 0.3; at most 0.5 for
    long_short_fraction, at most 1 for long_only_fraction);
  * portfolio `spot_perp_carry` (a coin held against its perpetual to earn the funding) takes
    portfolio.top_n, and needs `include_spot: true` in the spec's data block; pair it with
    the `funding` signal;
  * sizing `cap_weight` (coins of each side weighted by market capitalisation) takes only
    sizing.gross, so it needs no catalogue number besides the framework's 1.0.
- The one framework value: sizing.gross 1.0 (the whole capital, no leverage; this is how the
  stand accounts every portfolio, so no paper has to give it). Cite it as "{catalogue},
  framework value: gross 1 = the whole capital, no leverage". Any other sizing.gross is
  rejected.
- For EACH number in params, `sources` must have an entry (at the number's own dotted path
  such as signal.lookback_days, or at an ancestor such as signal) whose text names the
  catalogue file {catalogue}, the mechanism id and the paper, for example:
  "{catalogue}, mechanism <id>, row signal.lookback_days: 7, 14, 28 (Author Year)". Name the
  paper as author and year, with no web address. The number must appear in {catalogue}
  literally. Only min_history_days may be a formula (when the table has no row for it): the
  history a coin needs before it is traded, e.g. "derived: 28; the signal's lookback" -- every
  operand must be a value from a row of the mechanism you chose, and the formula must evaluate
  to exactly the number. Every other parameter is a value from its own row, never a formula.
  If the catalogue gives no sizing, use the `equal` block. Strings and block names need a
  citation too (an ancestor entry such as `signal` covers them).
- Costs, min_leg_notional and their sources: copy from {reference}. Do not change data.source,
  interval, or add an instrument list, a liquidity floor or a regime_claim.
- data.end must be exactly {data_end} (the latest Binance snapshot end).
- If the catalogue says the blocks cannot express part of the mechanism, either pick another
  mechanism or list that part in unexpressed_mechanisms (one sentence each); a non-empty list
  makes the stand answer "not evaluable". Never fake it with a block the catalogue does not
  name for the mechanism.
- noise_universe: "panel" (the strategy chooses its coins by its own universe rule, so its
  random books may choose among the whole panel); "held" only for a fixed instrument list,
  which is not allowed here.
- params_fixed_at is the day after the last data the numbers could have been fitted to
  (docs/FIT_VS_FORWARD.md): data BEFORE it is a selection period where a pass proves nothing,
  data FROM it is a forward test. Decide it from the catalogue's dates. If the catalogue gives
  the sample period of the sources whose numbers you use, it is the day after the LATEST sample
  end among them; if it gives only the publication date, it is the publication date (the end
  of the year when only a year is known) of the LATEST of those sources. If that day is before
  the spec's data.start, use data.start (the sources could not have seen our history). If no
  date can be established from the catalogue, OMIT params_fixed_at and
  params_fixed_evidence entirely: the whole window is then the selection period, which is the
  honest reading of an unknown date. Never set it to {today}: today's date would turn the
  whole history into selection data. params_fixed_evidence says which sources and which dates
  gave the day.

Reply with one line per spec: idea_id, the mechanism id, and the params_fixed_at you chose
with the reason in a few words (or "no spec: <reason>")."""


@dataclass
class GenerateOutcome:
    accepted: list[GeneratedSpec] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    reply: str | None = None
    problems: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)


def latest_binance_end(session) -> date | None:
    """The end of the newest Binance snapshot in the registry."""
    from sqlalchemy import func

    from qlab.registry.models import DataSnapshot

    return (session.query(func.max(DataSnapshot.range_end))
            .filter(DataSnapshot.source == DATA_SOURCE).scalar())


def existing_generated(spec_dir: Path = SPEC_DIR) -> dict[str, tuple[str, str]]:
    """idea_id -> (code_ref, params as JSON) of every spec already in `spec_dir`."""
    from qlab.pipeline.spec import load_spec

    out = {}
    for path in sorted(_abs(spec_dir).glob("*.yaml")):
        try:
            spec = load_spec(path)
        except Exception:  # noqa: BLE001 - an unloadable file still reserves its stem
            out[path.stem] = ("", "")
            continue
        out[spec.idea_id] = (spec.code_ref, json.dumps(spec.params, sort_keys=True, default=str))
    return out


def generate(*, budget: CandidateBudget | ExplicitNoBudget, session, data_end: date,
             catalogue_path: Path = CATALOGUE, max_specs: int = 3,
             registry_idea_ids: list[str] | tuple[str, ...] = (), model: str = MODEL,
             max_turns: int = 40) -> GenerateOutcome:
    """One agent call that writes up to `max_specs` new specs, then every
    code-side guard. The agent call goes through the budget guard."""
    today = datetime.now(UTC).date()
    outcome = GenerateOutcome()
    mechanisms = catalogue(catalogue_path)
    if not mechanisms:
        outcome.problems.append(f"no mechanisms in {catalogue_path}: the agent was not called")
        return outcome

    spec_dir = SPEC_DIR
    (QLAB_ROOT / spec_dir).mkdir(parents=True, exist_ok=True)
    (QLAB_ROOT / CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
    existing = existing_generated(spec_dir)
    context_path = CONTEXT_DIR / f"generate-{today}.md"
    (QLAB_ROOT / context_path).write_text("\n".join([
        f"# Context for the idea generator (built {today}, by code)", "",
        f"Data end to use (latest Binance snapshot end): {data_end}", "",
        f"## Specs already in {spec_dir}/ (do not duplicate)", "",
        *[f"- {i}" for i in sorted(existing)], "",
        "## Idea ids already in the registry (do not duplicate)", "",
        *[f"- {i}" for i in sorted(registry_idea_ids)], ""]), encoding="utf-8")

    originals = {p: p.read_bytes() for p in sorted(_abs(spec_dir).glob("*.yaml"))}
    before = set(_changed_paths())
    prompt = INSTRUCTIONS.format(
        code_ref=CODE_REF, catalogue=catalogue_path, reference=REFERENCE_SPEC,
        context_path=context_path, max_specs=max_specs, spec_dir=spec_dir,
        data_source=DATA_SOURCE, data_interval=DATA_INTERVAL, data_start=DATA_START,
        data_end=data_end, today=today)
    result = run_agent(budget, prompt, session=session, agent="generator", model=model,
                       max_turns=max_turns, cwd=QLAB_ROOT,
                       extra_args=("--allowedTools", TOOLS, "--disallowedTools", "Bash,Edit"))
    outcome.reply = (result.text or "").strip()[:800]
    if result.exit_code != 0:
        outcome.problems.append(
            f"the agent stopped before finishing (exit {result.exit_code}; its spending cap or "
            "turn limit): what it wrote may be incomplete")

    # Guard 1: only new files directly under specs/generated/.
    candidates: list[Path] = []
    edited = {str(p.relative_to(QLAB_ROOT)) for p in originals}  # restored below, never run
    for path in sorted(set(_changed_paths()) - before - edited):
        rest = path[len(f"{spec_dir}/"):] if path.startswith(f"{spec_dir}/") else ""
        if rest and "/" not in rest and rest.endswith(".yaml"):
            candidates.append(Path(path))
        else:
            _revert(path)
            outcome.reverted.append(path)
    for path, content in originals.items():
        if not path.is_file() or path.read_bytes() != content:
            path.write_bytes(content)
            outcome.reverted.append(f"{path.relative_to(QLAB_ROOT)} (an existing spec; restored)")
    if outcome.reverted:
        outcome.problems.append(f"wrote outside its places (reverted): {outcome.reverted}")

    # Guards 2-4, per spec.
    seen = {(ref, params): idea for idea, (ref, params) in existing.items() if ref}
    registry = frozenset(registry_idea_ids) | frozenset(existing)
    for path in candidates:
        if len(outcome.accepted) >= max_specs:
            outcome.rejected.append(Rejected(str(path), [f"more than {max_specs} specs"]))
            continue
        generated, problems = check_generated(
            path, mechanisms=mechanisms, catalogue_path=catalogue_path, data_end=data_end,
            registry_idea_ids=registry, other_params=seen)
        if problems or generated is None:
            outcome.rejected.append(Rejected(str(path), problems))
            continue
        seen[(generated.spec.code_ref, json.dumps(generated.spec.params, sort_keys=True,
                                                  default=str))] = generated.spec.idea_id
        outcome.accepted.append(generated)
    if not candidates:
        outcome.problems.append("no spec written")
    return outcome


# --------------------------------------------------------------------------
# The stand


@dataclass
class GeneratedResult:
    idea_id: str
    mechanism: str
    spec_path: str
    trial_id: int | None = None
    route: str | None = None
    reason: str | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None


def run_generated_one(item: GeneratedSpec, *, noise_trials: int = 200,
                      deployable_capital_usd: float = 3000.0) -> GeneratedResult:
    """One generated spec on the stand in its own transaction (matched noise,
    every rule, the trial written like every stand run's)."""
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType
    from qlab.rules.loader import load_latest

    spec = item.spec
    result = GeneratedResult(idea_id=spec.idea_id, mechanism=item.mechanism,
                             spec_path=str(item.path))
    try:
        with session_scope() as session:
            if session.get(Idea, spec.idea_id) is None:
                repo.upsert_idea(
                    session, id=spec.idea_id, title=spec.title, source_type=SourceType.INTERNAL,
                    asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                    notes=f"Generated from {CATALOGUE} ({item.mechanism})")
            out = evaluate_spec_with_shape_aware_bar(
                spec, session=session, ruleset=load_latest(),
                deployable_capital_usd=deployable_capital_usd, n_trials=noise_trials).candidate
            result.trial_id = out.trial_id
            result.route = str(out.routing.route)
            result.reason = out.routing.reason
            metrics = out.metrics or {}
            result.metrics = {k: metrics[k] for k in RESULT_METRICS if k in metrics}
    except Exception as exc:  # noqa: BLE001 - recorded in the report, the others still run
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def run_generated(items: list[GeneratedSpec], *, noise_trials: int = 200,
                  deployable_capital_usd: float = 3000.0) -> list[GeneratedResult]:
    """Evaluate each valid generated spec on the stand; one failure does not
    stop the others."""
    return [run_generated_one(item, noise_trials=noise_trials,
                              deployable_capital_usd=deployable_capital_usd) for item in items]


def result_line(r: GeneratedResult) -> str:
    """One result for the owner, in plain Russian."""
    if r.error:
        return f"`{r.idea_id}`: прогон упал — {r.error}"
    m = r.metrics

    def pct(key: str) -> str:
        v = m.get(key)
        return f"{v:+.1%}" if isinstance(v, int | float) and v == v else "—"

    sharpe = m.get("sharpe_net")
    return (f"`{r.idea_id}` (механизм {r.mechanism}, прогон {r.trial_id}): {r.route} — "
            f"в год {pct('ann_return_net')}, "
            f"Шарп {f'{sharpe:.2f}' if isinstance(sharpe, int | float) else '—'}, "
            f"просадка {pct('max_dd')}, "
            f"обгоняет шум {pct('noise_return_percentile').lstrip('+')}")


__all__ = ["GenerateOutcome", "GeneratedResult", "GeneratedSpec", "MODEL", "Mechanism",
           "ParamRow", "Rejected", "catalogue", "check_generated", "generate",
           "latest_binance_end", "parse_catalogue", "result_line", "run_generated",
           "run_generated_one"]
