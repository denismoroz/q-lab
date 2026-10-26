"""The research director's directions: the machine-readable half of its memo,
validated by code, and how the generator, the researcher and the night read it.

Owner, 2026-10-25: «я думаю в агентах не хватает агента эксперта - который бы
следил за направлениями поиска и корректировал действия генератора идей
исследователя»; «запуск руководителя раз в 2 дня - добавляй».

The director agent (`qlab.agents.director`) writes a memo for the owner and, at
its end, a fenced block (```directions ... ```) in YAML. This module is the
code side of that block: it parses it, rejects what does not check out, and
keeps the result in `night/directions.yaml` (history: `data/directions/<date>.yaml`).

What a direction can say -- and nothing else:

* generator: `prefer_mechanisms` (ids that exist in the catalogue), `pause_mechanisms`
  (ids that exist, each with a reason). A spec of a paused mechanism is rejected by the
  generator's guard; the preferred ones are only named in its instructions;
* researcher: per strategy of `night/research.yaml`, `priority` (high / normal / low)
  and a plain-text `note`. A note with a digit in it is dropped: the director sets
  directions, not parameter values (a number in a note is a value to tune to);
* `catalogue_requests`, `data_requests`: tasks for later; nothing acts on them yet;
* `code_requests`: code the lab lacks to test a direction (owner, 2026-10-26: «будет дописывать
  код если руководитель посчитает что направление - следует проверить»): which repository
  (`frab` or `qlab`), what to make, why. The programmer agent (`qlab.agents.programmer`) takes
  them one a night, on a branch; nothing reaches a main branch without the owner. Like a note,
  a request has no digit in it: it asks for a capability, never a value;
* `propose_close_family`: shown to the owner. **Code never acts on it**; closing a family
  of search is the owner's decision.

**Owner review before code acts.** The first directions ever written are marked
`pending_owner_review: true`, and while the flag is true the generator and the researcher
behave as if the file did not exist. A later run keeps the flag while the file it replaces
still carries it (nobody has reviewed anything yet); once the owner has cleared it
(`qlab director --approve`, or by hand), later directions apply at once. No file at all is
today's behaviour: nothing is steered.

Nothing here calls an agent or touches the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import yaml

DIRECTIONS = Path("night/directions.yaml")
HISTORY_DIR = Path("data/directions")
CONFIG = Path("night/director.yaml")

PRIORITIES = ("high", "normal", "low")
"""High first: the order the night walks the researcher's strategies in when the budget is short."""

BLOCK_KEYS = ("generator", "researcher", "catalogue_requests", "data_requests",
              "code_requests", "propose_close_family")
CODE_REPOS = ("frab", "qlab")
"""Where the programmer agent may be asked to write (qlab.agents.programmer)."""

_FENCE = re.compile(r"^```directions[ \t]*\r?\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
_DIGIT = re.compile(r"\d")

FILE_HEADER = """\
# The research director's directions (qlab.directions, docs/DIRECTOR.md).
# Written by CODE from the director agent's memo after validation -- do not edit the
# lists by hand (the next director run replaces them). The one hand edit that belongs
# to the owner: `pending_owner_review: false` once the first directions are reviewed
# (or `qlab director --approve`). While it is true the generator and the researcher
# ignore this file.
"""


# --------------------------------------------------------------------------
# The shapes


@dataclass(frozen=True)
class Pause:
    id: str
    why: str


@dataclass(frozen=True)
class StrategyDirection:
    priority: str = "normal"
    note: str | None = None


@dataclass(frozen=True)
class CatalogueRequest:
    family: str
    look_for: str
    why: str


@dataclass(frozen=True)
class DataRequest:
    what: str
    why: str


@dataclass(frozen=True)
class CodeRequest:
    repo: str
    what: str
    why: str
    strategy: str | None = None


@dataclass(frozen=True)
class CloseProposal:
    family: str
    why: str


@dataclass(frozen=True)
class Directions:
    date: date
    memo: str | None = None
    pending_owner_review: bool = False
    prefer_mechanisms: tuple[str, ...] = ()
    pause_mechanisms: tuple[Pause, ...] = ()
    researcher: dict[str, StrategyDirection] = field(default_factory=dict)
    catalogue_requests: tuple[CatalogueRequest, ...] = ()
    data_requests: tuple[DataRequest, ...] = ()
    propose_close_family: tuple[CloseProposal, ...] = ()
    problems: tuple[str, ...] = ()
    code_requests: tuple[CodeRequest, ...] = ()
    """What code rejected or dropped from the block (unknown ids, numbers in notes ...)."""

    def paused(self) -> dict[str, str]:
        return {p.id: p.why for p in self.pause_mechanisms}

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "memo": self.memo,
            "pending_owner_review": self.pending_owner_review,
            "generator": {
                "prefer_mechanisms": list(self.prefer_mechanisms),
                "pause_mechanisms": [{"id": p.id, "why": p.why} for p in self.pause_mechanisms],
            },
            "researcher": {name: {"priority": d.priority, "note": d.note}
                           for name, d in self.researcher.items()},
            "catalogue_requests": [{"family": r.family, "look_for": r.look_for, "why": r.why}
                                   for r in self.catalogue_requests],
            "data_requests": [{"what": r.what, "why": r.why} for r in self.data_requests],
            "code_requests": [{"repo": r.repo, "what": r.what, "why": r.why,
                               "strategy": r.strategy} for r in self.code_requests],
            "propose_close_family": [{"family": c.family, "why": c.why}
                                     for c in self.propose_close_family],
            "problems": list(self.problems),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Directions:
        gen = raw.get("generator") or {}
        day = raw["date"]
        return cls(
            date=day if isinstance(day, date) else date.fromisoformat(str(day)),
            memo=raw.get("memo"),
            pending_owner_review=bool(raw.get("pending_owner_review", False)),
            prefer_mechanisms=tuple(gen.get("prefer_mechanisms") or ()),
            pause_mechanisms=tuple(Pause(p["id"], p["why"])
                                   for p in gen.get("pause_mechanisms") or ()),
            researcher={name: StrategyDirection(d.get("priority", "normal"), d.get("note"))
                        for name, d in (raw.get("researcher") or {}).items()},
            catalogue_requests=tuple(CatalogueRequest(r["family"], r["look_for"], r["why"])
                                     for r in raw.get("catalogue_requests") or ()),
            data_requests=tuple(DataRequest(r["what"], r["why"])
                                for r in raw.get("data_requests") or ()),
            propose_close_family=tuple(CloseProposal(c["family"], c["why"])
                                       for c in raw.get("propose_close_family") or ()),
            problems=tuple(raw.get("problems") or ()),
            code_requests=tuple(CodeRequest(r["repo"], r["what"], r["why"], r.get("strategy"))
                                for r in raw.get("code_requests") or ()))


# --------------------------------------------------------------------------
# Parsing and validating the memo's block


def extract_block(memo_text: str) -> str | None:
    """The body of the memo's first ```directions fence; None when there is none."""
    match = _FENCE.search(memo_text)
    return match["body"] if match else None


@dataclass
class ParseResult:
    directions: Directions | None
    problems: list[str] = field(default_factory=list)


def _text(value: object) -> str | None:
    """A non-empty string, or None."""
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    return None


def _items(value: object, where: str, problems: list[str]) -> list[dict]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        problems.append(f"{where}: a list was expected, got {type(value).__name__}: ignored")
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        else:
            problems.append(f"{where}: an item must be a mapping, got {item!r}: ignored")
    return out


def parse_block(text: str, *, mechanism_ids: set[str] | frozenset[str],
                strategy_names: set[str] | frozenset[str], family_names: set[str] | frozenset[str],
                day: date, memo: str | None = None, pending_owner_review: bool = False
                ) -> ParseResult:
    """Validate the YAML text of a directions block. A bad item is dropped and
    named in `problems`; a block that is not a mapping (or not YAML) gives no
    directions at all, so the previous file stays."""
    problems: list[str] = []
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return ParseResult(None, [f"the directions block is not valid YAML: "
                                  f"{str(exc).splitlines()[0] if str(exc) else exc}"])
    if not isinstance(raw, dict):
        return ParseResult(None, ["the directions block must be a mapping"])
    for key in raw:
        if key not in BLOCK_KEYS:
            problems.append(f"unknown key {key!r} in the block: ignored")

    # generator
    gen = raw.get("generator") or {}
    if not isinstance(gen, dict):
        problems.append("generator: a mapping was expected: ignored")
        gen = {}
    for key in gen:
        if key not in ("prefer_mechanisms", "pause_mechanisms"):
            problems.append(f"generator.{key}: unknown key: ignored")
    pauses: list[Pause] = []
    for item in _items(gen.get("pause_mechanisms"), "generator.pause_mechanisms", problems):
        mech, why = _text(item.get("id")), _text(item.get("why"))
        if mech is None or mech not in mechanism_ids:
            problems.append(f"pause_mechanisms: {item.get('id')!r} is not a mechanism of the "
                            "catalogue: ignored")
        elif why is None:
            problems.append(f"pause_mechanisms: {mech} has no reason (`why`): ignored")
        elif mech not in {p.id for p in pauses}:
            pauses.append(Pause(mech, why))
    paused = {p.id for p in pauses}
    prefer: list[str] = []
    raw_prefer = gen.get("prefer_mechanisms") or []
    if not isinstance(raw_prefer, list):
        problems.append("generator.prefer_mechanisms: a list was expected: ignored")
        raw_prefer = []
    for mech in raw_prefer:
        if not isinstance(mech, str) or mech not in mechanism_ids:
            problems.append(f"prefer_mechanisms: {mech!r} is not a mechanism of the catalogue: "
                            "ignored")
        elif mech in paused:
            problems.append(f"prefer_mechanisms: {mech} is also paused: the pause stands")
        elif mech not in prefer:
            prefer.append(mech)

    # researcher
    researcher: dict[str, StrategyDirection] = {}
    raw_researcher = raw.get("researcher") or {}
    if not isinstance(raw_researcher, dict):
        problems.append("researcher: a mapping was expected: ignored")
        raw_researcher = {}
    for name, entry in raw_researcher.items():
        if name not in strategy_names:
            problems.append(f"researcher: {name!r} is not a strategy of night/research.yaml: "
                            "ignored")
            continue
        entry = entry if isinstance(entry, dict) else {}
        priority = entry.get("priority", "normal")
        if priority not in PRIORITIES:
            problems.append(f"researcher.{name}: priority {priority!r} is not one of "
                            f"{'/'.join(PRIORITIES)}: normal taken")
            priority = "normal"
        note = _text(entry.get("note"))
        if note is not None and _DIGIT.search(note):
            problems.append(f"researcher.{name}: the note has a digit in it (a direction is not "
                            "a parameter value): the note is dropped")
            note = None
        researcher[name] = StrategyDirection(priority, note)

    # requests and proposals
    catalogue_requests: list[CatalogueRequest] = []
    for item in _items(raw.get("catalogue_requests"), "catalogue_requests", problems):
        family, look_for, why = (_text(item.get("family")), _text(item.get("look_for")),
                                 _text(item.get("why")))
        if family and look_for and why:
            catalogue_requests.append(CatalogueRequest(family, look_for, why))
        else:
            problems.append(f"catalogue_requests: {item!r} needs family, look_for and why: ignored")
    data_requests: list[DataRequest] = []
    for item in _items(raw.get("data_requests"), "data_requests", problems):
        what, why = _text(item.get("what")), _text(item.get("why"))
        if what and why:
            data_requests.append(DataRequest(what, why))
        else:
            problems.append(f"data_requests: {item!r} needs what and why: ignored")
    code_requests: list[CodeRequest] = []
    for item in _items(raw.get("code_requests"), "code_requests", problems):
        repo, what, why = _text(item.get("repo")), _text(item.get("what")), _text(item.get("why"))
        strategy = _text(item.get("strategy"))
        if repo not in CODE_REPOS or not what or not why:
            problems.append(f"code_requests: {item!r} needs repo ({'/'.join(CODE_REPOS)}), what "
                            "and why: ignored")
        elif _DIGIT.search(what) or _DIGIT.search(why):
            problems.append(f"code_requests: {what!r} has a digit in it (a request asks for a "
                            "capability, not a value): ignored")
        elif strategy is not None and strategy not in strategy_names:
            problems.append(f"code_requests: {strategy!r} is not a strategy of "
                            "night/research.yaml: ignored")
        else:
            code_requests.append(CodeRequest(repo, what, why, strategy))
    closes: list[CloseProposal] = []
    known = set(family_names) | set(strategy_names)
    for item in _items(raw.get("propose_close_family"), "propose_close_family", problems):
        family, why = _text(item.get("family")), _text(item.get("why"))
        if family is None or family not in known:
            problems.append(f"propose_close_family: {item.get('family')!r} is neither a mechanism "
                            "family nor a strategy of the researcher: ignored")
        elif why is None:
            problems.append(f"propose_close_family: {family} has no reason (`why`): ignored")
        elif family not in {c.family for c in closes}:
            closes.append(CloseProposal(family, why))

    return ParseResult(Directions(
        date=day, memo=memo, pending_owner_review=pending_owner_review,
        prefer_mechanisms=tuple(prefer), pause_mechanisms=tuple(pauses), researcher=researcher,
        catalogue_requests=tuple(catalogue_requests), data_requests=tuple(data_requests),
        propose_close_family=tuple(closes), problems=tuple(problems),
        code_requests=tuple(code_requests)), problems)


# --------------------------------------------------------------------------
# The file


def _abs(root: Path | None, path: Path) -> Path:
    return path if root is None or path.is_absolute() else root / path


@dataclass(frozen=True)
class DirectionsFile:
    status: str  # missing | pending | active | corrupt
    directions: Directions | None = None
    error: str | None = None


def read_directions(root: Path | None = None, path: Path = DIRECTIONS) -> DirectionsFile:
    """What `night/directions.yaml` is today. A file that cannot be read is
    `corrupt` and acts as no file (today's behaviour), never as a crash."""
    file = _abs(root, path)
    if not file.is_file():
        return DirectionsFile("missing")
    try:
        directions = Directions.from_dict(yaml.safe_load(file.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001 - a hand-damaged file must not stop the night
        return DirectionsFile("corrupt", error=f"{type(exc).__name__}: {exc}")
    return DirectionsFile("pending" if directions.pending_owner_review else "active", directions)


def active_directions(root: Path | None = None, path: Path = DIRECTIONS) -> Directions | None:
    """The directions the generator and the researcher obey: None while the file
    is missing, corrupt or waiting for the owner's review."""
    found = read_directions(root, path)
    return found.directions if found.status == "active" else None


def has_previous(root: Path | None = None) -> bool:
    """Whether any directions were ever written (the file, or a history copy)."""
    if _abs(root, DIRECTIONS).is_file():
        return True
    history = _abs(root, HISTORY_DIR)
    return history.is_dir() and any(history.glob("*.yaml"))


def pending_for_new(root: Path | None = None) -> bool:
    """Whether directions written now wait for the owner: the first ones ever do,
    and so do those replacing a file nobody has reviewed yet."""
    if not has_previous(root):
        return True
    return read_directions(root).status == "pending"


def render(directions: Directions) -> str:
    return FILE_HEADER + yaml.safe_dump(directions.to_dict(), allow_unicode=True,
                                        sort_keys=False, width=100)


def write_directions(directions: Directions, root: Path | None = None) -> Path:
    """Write `night/directions.yaml` and its history copy `data/directions/<date>.yaml`."""
    text = render(directions)
    target = _abs(root, DIRECTIONS)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    history = _abs(root, HISTORY_DIR)
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{directions.date.isoformat()}.yaml").write_text(text, encoding="utf-8")
    return target


def approve(root: Path | None = None) -> Directions | None:
    """Clear `pending_owner_review` (the owner's review is done). None when
    there is no readable file. The history copy keeps what the director wrote."""
    found = read_directions(root)
    if found.directions is None:
        return None
    approved = replace(found.directions, pending_owner_review=False)
    _abs(root, DIRECTIONS).write_text(render(approved), encoding="utf-8")
    return approved


# --------------------------------------------------------------------------
# How the others read them


_RANK = {"high": 0, "normal": 1, "low": 2}


def priority_rank(directions: Directions | None, strategy: str) -> int:
    """0 for high, 1 for normal (also: no directions, or none for this strategy), 2 for low."""
    if directions is None or strategy not in directions.researcher:
        return _RANK["normal"]
    return _RANK[directions.researcher[strategy].priority]


def order_strategies(names: list[str], directions: Directions | None,
                     unfinished: list[str] | None = None) -> list[str]:
    """The night's order of the researcher's strategies: high priority first, then
    normal, then low; inside one priority the strategies a budget stop left
    unfinished go first, then by name. Without directions this is today's order
    (unfinished first, then by name). A low-priority strategy is the one cut when
    the budget is short -- that is what the priority is for."""
    first = set(unfinished or ())
    return sorted(names, key=lambda n: (priority_rank(directions, n), n not in first, n))


def researcher_focus(name: str, directions: Directions | None) -> str | None:
    """The part of the researcher's focus that comes from the director."""
    if directions is None or name not in directions.researcher:
        return None
    d = directions.researcher[name]
    if d.priority == "normal" and d.note is None:
        return None
    lines = [f"The research director's direction for this strategy (dated "
             f"{directions.date.isoformat()}): priority {d.priority}."]
    if d.note:
        lines.append("Its note -- a direction for where to look, not a value to tune to; "
                     f"treat it as advice, the evidence decides: {d.note}")
    return "\n".join(lines)


def generator_text(directions: Directions | None) -> str:
    """The part of the generator's instructions that comes from the director."""
    if directions is None or not (directions.prefer_mechanisms or directions.pause_mechanisms):
        return ""
    lines = [f"The research director's directions (dated {directions.date.isoformat()}):"]
    if directions.prefer_mechanisms:
        lines.append("- prefer these mechanisms when you choose what to test (advice, not an "
                     f"order): {', '.join(directions.prefer_mechanisms)}")
    if directions.pause_mechanisms:
        lines.append("- do NOT write a spec for these mechanisms; they are paused and code "
                     "rejects such a spec:")
        lines += [f"    {p.id} -- {p.why}" for p in directions.pause_mechanisms]
    return "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------
# night/director.yaml


@dataclass(frozen=True)
class DirectorConfig:
    enabled: bool
    cadence_days: int
    max_turns: int
    families: dict[str, tuple[str, ...]]

    def family_of(self, mechanism_id: str) -> str:
        """The family a catalogue mechanism belongs to; a mechanism the config does not
        list is a family of its own (the evidence says so)."""
        for family, ids in self.families.items():
            if mechanism_id in ids:
                return family
        return mechanism_id


def load_config(path: Path = CONFIG) -> DirectorConfig | None:
    """night/director.yaml; None when the file is missing."""
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cadence = raw.get("cadence_days", 2)
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
        raise ValueError(f"{path}: cadence_days must be a whole number of days, got {cadence!r}")
    families = {}
    for family, ids in (raw.get("families") or {}).items():
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError(f"{path}: families.{family} must be a list of mechanism ids")
        families[str(family)] = tuple(ids)
    return DirectorConfig(enabled=bool(raw.get("enabled", False)), cadence_days=cadence,
                          max_turns=int(raw.get("max_turns", 40)), families=families)


__all__ = ["BLOCK_KEYS", "CONFIG", "DIRECTIONS", "FILE_HEADER", "HISTORY_DIR", "PRIORITIES",
           "CatalogueRequest", "CloseProposal", "DataRequest", "Directions", "DirectionsFile",
           "DirectorConfig", "ParseResult", "Pause", "StrategyDirection", "active_directions",
           "approve", "extract_block", "generator_text", "has_previous", "load_config",
           "order_strategies", "parse_block", "pending_for_new", "priority_rank",
           "read_directions", "render", "researcher_focus", "write_directions"]
