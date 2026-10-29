"""The day a spec's numbers are fixed is named by code, not by an agent (docs/FIXATION.md).

Owner, 2026-10-29, on docs/EVALUATION_PATHS.md (question В9, «выбор до отреза одним способом»): for
the ideas of the generator and the experiments of the researcher the day `params_fixed_at` is the
start of the LAST THIRD of the spec's data window, the share the rules hold out
(`holdout_share`, 1/3 since rules 2026-10-24.1). Data before it is the selection period, data
from it is the forward test (docs/FIT_VS_FORWARD.md). Before, every agent chose its own day: the
generator read it from the dates of a paper, the researcher from the cut of an evidence file built
on another window, so two ideas on one stand were judged on forward tests of unrelated lengths.

An agent may put the day LATER than the named one, and only with a source: a paper whose numbers
were fitted up to a later day (`sources["params_fixed_at"]`, a citation that names something
checkable like any other). Earlier is never allowed: the days between were not held out. A spec
that selects causally (`selects_causally`) names its first choice instead and is left alone.

Pure functions plus one file rewrite (`normalise_file`): code corrects an agent's date to the named
one rather than throwing the spec away, and says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

NAMED_BY_CODE = ("named by code: the start of the last third of the data window "
                 "({start}..{end}, the rules' holdout_share {share:.4g}; docs/FIXATION.md)")
SOURCE_KEY = "params_fixed_at"
"""The `sources` key under which an agent cites a later day."""


def named_date(start: date, end: date, share: float) -> date:
    """The first day of the held-out part of the window `[start, end]`: the window is cut after
    `1 - share` of its days, rounded to a whole day."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"share must be between 0 and 1, got {share}")
    if end <= start:
        raise ValueError(f"empty window {start}..{end}")
    days = (end - start).days
    return start + timedelta(days=int(round(days * (1.0 - share))))


def holdout_share() -> float:
    """The share of the window the latest rules hold out for the forward test."""
    from qlab.rules.loader import load_latest

    held = load_latest().holdout_share
    if held is None:
        raise ValueError("the latest rules have no holdout_share: no day can be named")
    return held.share


@dataclass(frozen=True)
class Fixation:
    named: date
    given: date | None
    ok: bool
    problem: str | None = None


def check(spec, share: float | None = None) -> Fixation:
    """Whether `spec`'s `params_fixed_at` is the named day, or a later one with a source."""
    share = holdout_share() if share is None else share
    named = named_date(spec.data.start, spec.data.end, share)
    given = spec.params_fixed_at
    if spec.selects_causally:  # names the wrapper's first choice, not a cut of the window
        return Fixation(named, given, True)
    if given is None:
        return Fixation(named, given, False,
                        f"params_fixed_at is missing: the code names {named} (the start of the "
                        "last third of the window)")
    if given < named:
        return Fixation(named, given, False,
                        f"params_fixed_at {given} is earlier than the day the code names, "
                        f"{named} (the start of the last third of the window): days between "
                        "were not held out")
    if given > named and not (spec.sources.get(SOURCE_KEY) or "").strip():
        return Fixation(named, given, False,
                        f"params_fixed_at {given} is later than the day the code names, {named}: "
                        f"a later day needs sources[{SOURCE_KEY!r}], the source whose numbers "
                        "were fitted up to it")
    return Fixation(named, given, True)


def problems(spec, share: float | None = None) -> list[str]:
    found = check(spec, share)
    return [] if found.ok else [found.problem or "params_fixed_at is not the named day"]


_TOP_KEY = re.compile(r"^[A-Za-z_][\w]*\s*:")


def _drop_key(lines: list[str], key: str) -> list[str]:
    """`lines` without the top-level `key:` entry (its line and the indented / blank lines of a
    block scalar that follow it)."""
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith(f"{key}:"):
            skipping = True
            continue
        if skipping and (not line.strip() or line[0] in " \t-"):
            continue
        skipping = False
        out.append(line)
    return out


def normalise_file(path: Path, share: float | None = None) -> str | None:
    """Correct the file of an agent's spec so its `params_fixed_at` is the named day, unless the
    agent put a later one with a source. Returns a sentence saying what was changed, or None when
    nothing was. A file that does not load, or no longer loads after the change, is left as it was
    (it fails loudly in its own check)."""
    from qlab.pipeline.spec import load_spec

    try:
        spec = load_spec(path)
    except Exception:  # noqa: BLE001 - the caller's check says why
        return None
    found = check(spec, share)
    if found.ok:
        return None
    share = holdout_share() if share is None else share
    original = path.read_text(encoding="utf-8")
    lines = _drop_key(_drop_key(original.split("\n"), "params_fixed_at"), "params_fixed_evidence")
    while lines and not lines[-1].strip():
        lines.pop()
    evidence = NAMED_BY_CODE.format(start=spec.data.start, end=spec.data.end, share=share)
    if found.given is not None and spec.params_fixed_evidence:
        said = " ".join(spec.params_fixed_evidence.split())
        evidence += f"; the agent had written {found.given} ({said})"
    lines += [f"params_fixed_at: {found.named.isoformat()}",
              f"params_fixed_evidence: {_quote(evidence)}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        load_spec(path)
    except Exception:  # noqa: BLE001
        path.write_text(original, encoding="utf-8")
        return None
    before = "no day" if found.given is None else str(found.given)
    return f"{path.name}: params_fixed_at {before} -> {found.named} (named by code)"


def _quote(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False)


__all__ = ["Fixation", "NAMED_BY_CODE", "SOURCE_KEY", "check", "holdout_share", "named_date",
           "normalise_file", "problems"]
