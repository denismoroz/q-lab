"""Which files of the working tree the night's code steps wrote.

The agent steps (the implementer, the researcher, the carder, the director, the scout, the
generator, the cataloguer) check, after the agent's call, which files of
the tree changed (`git status`), and undo every change that is not one of the files the agent may
write. That was safe while nothing else wrote the tree during the call. With steps running together
(docs/PIPELINE.md, «Дорожки») the calculations lane writes spec files at the same time (a causal
twin next to its spec, an overlay variant, a second-stage spec), and the check would take them for
the agent's and delete them. The functions that write these files tell this module (`note`); the
checks (`implementer._changed_paths`, the cataloguer's) ask it and leave those files alone. The
list is of this process (one night); a file an agent changes that a code step also wrote is not
noticed, which no agent has a reason to do."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_LOCK = threading.Lock()
_WRITES: list[str] = []


def note(path: Path | str) -> None:
    """Record that a code step has just written `path`. A path outside the repository is not the
    tree's business."""
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return
    with _LOCK:
        if rel not in _WRITES:
            _WRITES.append(rel)


def written() -> list[str]:
    with _LOCK:
        return list(_WRITES)


def is_written_by_code(change: str, written_paths: Iterable[str]) -> bool:
    """Whether the changed path `change` of `git status` is, or (a new directory is listed as one
    entry) contains, a file that a code step wrote."""
    change = change.strip().strip('"')
    return any(w == change or (change.endswith("/") and w.startswith(change))
               for w in written_paths)


def without_code_writes(changes: Iterable[str]) -> list[str]:
    """The changed paths of `git status` minus the ones code steps wrote in this process."""
    mine = written()
    return [c for c in changes if not is_written_by_code(c, mine)]
