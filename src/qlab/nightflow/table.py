"""The night's steps as one declared table (`night/pipeline.yaml`, docs/PIPELINE.md).

Owner, 2026-10-31: «разбиваем весь процесс на шаги ... не зависимые шаги должны выполняться по
любому. если какой-то шаг - падает - это не должно останавливать весь процесс.» The table is the
single place that says what the steps are and what each depends on. Everything else (the runner,
the journal, the API, the page) reads it; nothing here runs a step.

Edges have a kind (`EDGE_KINDS`). Only `data` edges block: a step whose `data` parent failed (or was
switched off and not done earlier the same day) cannot run and is marked blocked. The other kinds
only order the steps (`fresh`, `exclusive`, `budget`), so a failed parent never stops them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PIPELINE = Path(__file__).resolve().parents[3] / "night" / "pipeline.yaml"

#: data -- blocks a step whose parent failed; fresh -- ordering for a fresh reading; exclusive --
#: the two must not overlap; budget -- only the order in which tokens are spent.
EDGE_KINDS = ("data", "fresh", "exclusive", "budget")
STEP_KINDS = ("data", "code", "agent")

#: The lanes of the night (docs/PIPELINE.md, «Дорожки»): steps of one lane run one at a time, in
#: the table's order among those that are ready; lanes run together. `final` is the ladder and
#: the report, which wait for every other lane. A step with no lane (a test's table) is alone in a
#: lane of its own, so the limit on the number of steps is then the only one.
LANES = ("calc", "checks", "agents", "final")
LANE_TITLES = {
    "calc": ("Расчёты", "Всё, что гонит идеи через стенд и пишет испытания: по одному, чтобы одну "
             "идею не посчитать дважды."),
    "checks": ("Проверки", "Чтение боевого хоста и кэша панелей стенда; испытаний не пишут."),
    "agents": ("Агенты", "Шаги, которые тратят токены: по одному, гард бюджета не рассчитан на "
               "два сразу."),
    "final": ("Итог", "Лестница и отчёт: после всех остальных дорожек."),
}


class TableError(ValueError):
    """The table is malformed: a duplicate id, an unknown parent, a parent below its child."""


@dataclass(frozen=True)
class Edge:
    step: str
    kind: str
    why: str


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    description: str
    kind: str
    tokens: bool
    items: str = ""
    depends_on: tuple[Edge, ...] = ()
    done_prefixes: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    refresh: bool = False
    reads_all: str = ""
    lane: str = ""

    @property
    def lane_key(self) -> str:
        """The lane the scheduler keeps one step at a time in: the declared, else the step's own."""
        return self.lane or self.id

    @property
    def hard_parents(self) -> tuple[str, ...]:
        return tuple(e.step for e in self.depends_on if e.kind == "data")


@dataclass
class Table:
    steps: list[Step]
    by_id: dict[str, Step] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {s.id: s for s in self.steps}

    # ------------------------------------------------------------------ shape

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def parents(self, step_id: str, kinds: tuple[str, ...] | None = None) -> list[str]:
        return [e.step for e in self.by_id[step_id].depends_on if kinds is None or e.kind in kinds]

    def children(self, step_id: str, kinds: tuple[str, ...] | None = None) -> list[str]:
        return [s.id for s in self.steps
                if any(e.step == step_id and (kinds is None or e.kind in kinds)
                       for e in s.depends_on)]

    def descendants(self, step_id: str, kinds: tuple[str, ...] | None = None) -> list[str]:
        """Every step reachable below `step_id` along edges of `kinds` (all kinds when None), in
        the table's order. `kinds=("data",)` is what a failure blocks."""
        seen: set[str] = set()
        todo = [step_id]
        while todo:
            for child in self.children(todo.pop(), kinds):
                if child not in seen:
                    seen.add(child)
                    todo.append(child)
        return [s for s in self.ids if s in seen]

    def ancestors(self, step_id: str, kinds: tuple[str, ...] | None = None) -> list[str]:
        seen: set[str] = set()
        todo = [step_id]
        while todo:
            for parent in self.parents(todo.pop(), kinds):
                if parent not in seen:
                    seen.add(parent)
                    todo.append(parent)
        return [s for s in self.ids if s in seen]

    def layers(self, kinds: tuple[str, ...] | None = None) -> dict[str, int]:
        """The wave of each step: 0 for a step with no parent, else one more than the deepest
        parent, along edges of `kinds`. Steps of one wave have no path between them, so they could
        run together (the table's order is a valid order, so one pass is enough)."""
        wave: dict[str, int] = {}
        for step in self.steps:
            above = [wave[e.step] for e in step.depends_on if kinds is None or e.kind in kinds]
            wave[step.id] = 1 + max(above) if above else 0
        return wave

    def can_run_with(self, step_id: str, kinds: tuple[str, ...] | None = None) -> list[str]:
        """Steps with no path to or from `step_id` along edges of `kinds`: nothing in the graph
        forces either to wait for the other."""
        related = set(self.ancestors(step_id, kinds)) | set(self.descendants(step_id, kinds))
        return [s for s in self.ids if s != step_id and s not in related]

    def lanes(self) -> dict[str, list[str]]:
        """The steps of each declared lane, in the table's order; the lanes in `LANES` order."""
        out: dict[str, list[str]] = {name: [] for name in LANES}
        for step in self.steps:
            if step.lane:
                out.setdefault(step.lane, []).append(step.id)
        return {name: ids for name, ids in out.items() if ids}

    def ordered(self, a: str, b: str) -> bool:
        """Whether one of the two steps is an ancestor of the other, along edges of any kind:
        they can never run at the same time."""
        return a in self.ancestors(b) or b in self.ancestors(a)

    def restart_set(self, step_id: str) -> list[str]:
        """What a restart of `step_id` runs: the step, everything that depends on it by `data`
        (its result is the input of those), and the steps that only read (`refresh`: the ladder
        and the report), in the table's order."""
        chosen = {step_id, *self.descendants(step_id, ("data",))}
        chosen |= {s.id for s in self.steps if s.refresh}
        return [s for s in self.ids if s in chosen]


def _edges(raw: dict, ids_above: list[str]) -> tuple[Edge, ...]:
    edges = [Edge(step=str(e["step"]), kind=str(e["kind"]), why=str(e["why"]))
             for e in raw.get("depends_on") or []]
    if raw.get("reads_all"):
        seen = {e.step for e in edges}
        edges += [Edge(step=s, kind="fresh", why=str(raw["reads_all"]))
                  for s in ids_above if s not in seen]
    return tuple(edges)


def parse(doc: dict) -> Table:
    steps: list[Step] = []
    for raw in doc.get("steps") or []:
        above = [s.id for s in steps]
        step = Step(
            id=str(raw["id"]), title=str(raw["title"]), description=str(raw["description"]),
            kind=str(raw["kind"]), tokens=bool(raw["tokens"]), items=str(raw.get("items") or ""),
            depends_on=_edges(raw, above), done_prefixes=tuple(raw.get("done_prefixes") or ()),
            state_keys=tuple(raw.get("state_keys") or ()), refresh=bool(raw.get("refresh", False)),
            reads_all=str(raw.get("reads_all") or ""), lane=str(raw.get("lane") or ""))
        steps.append(step)
    table = Table(steps)
    validate(table)
    return table


def validate(table: Table) -> None:
    seen: dict[str, int] = {}
    for at, step in enumerate(table.steps):
        if step.id in seen:
            raise TableError(f"step {step.id!r} is declared twice")
        seen[step.id] = at
        if step.kind not in STEP_KINDS:
            raise TableError(f"step {step.id!r} has kind {step.kind!r}; one of {STEP_KINDS}")
        if step.kind == "agent" and not step.tokens:
            raise TableError(f"agent step {step.id!r} must say it spends tokens")
        for edge in step.depends_on:
            if edge.kind not in EDGE_KINDS:
                raise TableError(f"step {step.id!r}: edge to {edge.step!r} has kind {edge.kind!r}; "
                                 f"one of {EDGE_KINDS}")
            if edge.step not in seen:
                raise TableError(f"step {step.id!r} depends on {edge.step!r}, which is not "
                                 "declared above it (the table's order must be a valid order)")
            if not edge.why.strip():
                raise TableError(f"step {step.id!r}: the edge to {edge.step!r} has no reason")
        if len({e.step for e in step.depends_on}) != len(step.depends_on):
            raise TableError(f"step {step.id!r} names a parent twice")
        if step.lane and step.lane not in LANES:
            raise TableError(f"step {step.id!r} is in lane {step.lane!r}; one of {LANES}")
    spenders = {s.lane for s in table.steps if s.tokens and s.lane}
    if len(spenders) > 1:
        raise TableError("steps that spend tokens must share one lane (the token guard is not "
                         f"safe for two at once); they are in {sorted(spenders)}")
    _validate_ownership(table)


def _validate_ownership(table: Table) -> None:
    """No two steps write one key of the night's state or one done-marker prefix at the same time
    (`qlab.nightflow.statehub`): a key may have several owners only if every two of them are ordered
    by the table (one is the other's ancestor), and a step may not name a key the state keeps for
    the run itself."""
    from qlab.nightflow.statehub import HUB_KEYS

    keys: dict[str, list[str]] = {}
    prefixes: dict[str, list[str]] = {}
    for step in table.steps:
        for key in step.state_keys:
            if key in HUB_KEYS:
                raise TableError(f"step {step.id!r} owns {key!r}, which the run itself keeps")
            keys.setdefault(key, []).append(step.id)
        for prefix in step.done_prefixes:
            prefixes.setdefault(prefix, []).append(step.id)
    for what, owners_of in (("state key", keys), ("done-marker prefix", prefixes)):
        for name, owners in owners_of.items():
            for i, a in enumerate(owners):
                for b in owners[i + 1:]:
                    if not table.ordered(a, b):
                        raise TableError(f"steps {a!r} and {b!r} both own the {what} {name!r} "
                                         "but nothing orders them: they could write it together")


def load_table(path: Path | None = None) -> Table:
    target = path or PIPELINE
    return parse(yaml.safe_load(target.read_text(encoding="utf-8")) or {})
