"""What has already been tried (`night/tried.yaml`, docs/IDEAS.md): the prior-art registry the
carder agent reads before it writes a card, so nobody repeats a finished experiment.

One entry per tested variant or family: where it is written down and what came of it, with the
numbers as that source writes them. The graveyard (rejected and retired ideas) is NOT copied
here: code reads it from the registry at run time (`qlab.agents.carder.graveyard_lines`).

An entry is a fact about a past experiment, not a verdict on a new idea: a new idea that
resembles an entry is allowed when it says what is different (the card's `vs_tried`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from qlab.pipeline.sources import QLAB_ROOT

TRIED = Path("night/tried.yaml")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


@dataclass(frozen=True)
class Tried:
    id: str
    family: str
    what: str  # Russian, one line
    file: str  # where it is written down: a path from q-lab's root, or funding-rate-arbitrage/...
    section: str
    result: str  # one line, the numbers as the source writes them

    @property
    def line(self) -> str:
        return f"{self.id} [{self.family}]: {self.what} -- {self.result}"

    @property
    def where(self) -> str:
        return f"{self.file}, {self.section}"


@dataclass(frozen=True)
class TriedRegistry:
    version: str
    families: dict[str, str]
    entries: tuple[Tried, ...]

    def ids(self) -> set[str]:
        return {e.id for e in self.entries}

    def get(self, tried_id: str) -> Tried | None:
        return next((e for e in self.entries if e.id == tried_id), None)

    def unknown(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self.ids()]


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else QLAB_ROOT / path


FRAB = "funding-rate-arbitrage"


def frab_root() -> Path | None:
    """FRAB's repository: next to q-lab, or where `night/programmer.yaml` says it is (a worktree
    of q-lab is not next to it). None when neither exists."""
    candidates = [QLAB_ROOT.parent / FRAB]
    try:
        config = yaml.safe_load((QLAB_ROOT / "night/programmer.yaml").read_text(encoding="utf-8"))
        candidates.append(Path(config["repos"]["frab"]["path"]))
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        pass
    return next((c for c in candidates if c.is_dir()), None)


def resolve(file: str) -> Path:
    """The file an entry cites: q-lab's own, or `funding-rate-arbitrage/...` in FRAB's
    repository."""
    if file.startswith(f"{FRAB}/"):
        return (frab_root() or QLAB_ROOT.parent / FRAB) / file.removeprefix(f"{FRAB}/")
    return QLAB_ROOT / file


def load_tried(path: Path = TRIED) -> TriedRegistry:
    """The registry, validated: unique lower-case-with-dashes ids, a declared family, a Russian
    `what`, a cited file and section, a result. Raises ValueError listing every problem."""
    raw = yaml.safe_load(_abs(path).read_text(encoding="utf-8")) or {}
    families = raw.get("families") or {}
    problems: list[str] = []
    entries: list[Tried] = []
    seen: set[str] = set()
    for i, row in enumerate(raw.get("entries") or [], 1):
        where = row.get("where") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(where, dict):
            problems.append(f"entry {i}: not a mapping with `where: {{file, section}}`")
            continue
        entry_id = row.get("id")
        label = f"entry {i} ({entry_id})"
        if not isinstance(entry_id, str) or not _ID.match(entry_id):
            problems.append(f"{label}: id must be lower-case-with-dashes")
            continue
        if entry_id in seen:
            problems.append(f"{label}: the id repeats")
        seen.add(entry_id)
        if row.get("family") not in families:
            problems.append(f"{label}: family {row.get('family')!r} is not declared in `families`")
        for key in ("what", "result"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                problems.append(f"{label}: `{key}` is empty")
        if not _CYRILLIC.search(str(row.get("what"))):
            problems.append(f"{label}: `what` is not in Russian")
        if "\n" in str(row.get("what", "")).strip() or "\n" in str(row.get("result", "")).strip():
            problems.append(f"{label}: `what` and `result` are one line each")
        if not all(isinstance(where.get(k), str) and where[k].strip() for k in ("file", "section")):
            problems.append(f"{label}: `where` needs a `file` and a `section`")
            continue
        entries.append(Tried(id=entry_id, family=str(row.get("family")),
                             what=str(row.get("what")).strip(), file=where["file"].strip(),
                             section=where["section"].strip(),
                             result=str(row.get("result")).strip()))
    for family, text in families.items():
        if not isinstance(text, str) or not _CYRILLIC.search(text):
            problems.append(f"family {family}: the description must be Russian text")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return TriedRegistry(version=str(raw.get("version", "")), families=dict(families),
                         entries=tuple(entries))


def missing_files(registry: TriedRegistry) -> list[str]:
    """Cited files that are not there. A repository next to q-lab that is absent (a worktree,
    another machine) is not reported: only a file that should be here and is not."""
    out = []
    for e in registry.entries:
        if e.file.startswith(f"{FRAB}/") and frab_root() is None:
            continue
        if not resolve(e.file).is_file():
            out.append(f"{e.id}: {e.file}")
    return out


def render(registry: TriedRegistry) -> str:
    """The registry as the carder's brief shows it: families, then one line per entry."""
    lines = [f"# Already tried (night/tried.yaml, version {registry.version})", ""]
    for family, text in registry.families.items():
        lines += [f"## {family} -- {text}"]
        lines += [f"- {e.line}  [{e.where}]" for e in registry.entries if e.family == family]
        lines.append("")
    return "\n".join(lines)


_WORD = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9-]{2,}", re.IGNORECASE)
_STOP = frozenset("""что как это для при или его все она они так там тут уже ещё еще бы
если чтобы когда только можно нужно надо есть был была были будет быть the and for with that this
from not are was into than then also can use used
""".split())


def stems(text: str) -> set[str]:
    """Words of a text reduced to their first five letters: enough to match a Russian word in
    another ending, and an English one in another form."""
    return {w[:5] for w in (m.lower() for m in _WORD.findall(text)) if w not in _STOP}


def rarity(documents: list[set[str]]) -> dict[str, float]:
    """How telling each stem is across the documents: ln(N / the documents holding it). A word
    every entry uses (the registry is full of «страховка») tells nothing; a rare one does."""
    import math

    counts: dict[str, int] = {}
    for doc in documents:
        for stem in doc:
            counts[stem] = counts.get(stem, 0) + 1
    return {stem: math.log(len(documents) / n) for stem, n in counts.items()}


def nearest(registry: TriedRegistry, text: str, *, limit: int = 3, least: float = 3.0
            ) -> list[Tried]:
    """The entries that share the most telling words with an idea (their weights summed, at
    least `least`), best first. A heuristic that points; the carder judges whether they are the
    same idea."""
    wanted = stems(text)
    docs = [stems(f"{e.id.replace('-', ' ')} {e.what} {e.result}") for e in registry.entries]
    weight = rarity(docs)
    scored = []
    for e, doc in zip(registry.entries, docs, strict=True):
        score = sum(weight[s] for s in wanted & doc)
        if score >= least:
            scored.append((score, e))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [e for _, e in scored[:limit]]


__all__ = ["FRAB", "TRIED", "Tried", "TriedRegistry", "frab_root", "load_tried",
           "missing_files", "nearest", "rarity", "render", "resolve", "stems"]
