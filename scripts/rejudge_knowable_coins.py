"""List the open ideas whose judged window was cut by a coin list, and what the knowable-coins
check (`qlab.knowable_coins`, docs/ONE_CHECK.md) makes of it: judged_from before -> after.

Dry run by default: it reads the specs, the registry (read-only) and the raw candles, and writes
nothing. `--apply` rewrites `judged_from`, `instruments_chosen_at` and `coins_knowable` in the
spec FILES that carry the dates as literals (second-stage specs, older twins); an idea that
declares a `choice` is rebuilt by the night's check from its own spec and needs nothing written.
Neither mode touches the registry: the stand version bump (2026-11-04.1) makes `qlab restamp`
measure the ideas again.

    uv run python scripts/rejudge_knowable_coins.py --root /path/to/q-lab
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent
CLOSED = ("rejected", "retired", "decayed")
"""Idea statuses that are not open: the idea has been judged and put away."""


@dataclass(frozen=True)
class Row:
    idea_id: str
    status: str
    path: str
    kind: str
    """`literal` (the dates stand in the spec file) or `check` (the night's one check builds them)."""
    venue: str
    coins: tuple[str, ...]
    data_start: date
    before: date | None
    after: date | None
    plain: str
    known_from: date | None

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _coins(raw: dict) -> list[str]:
    names = (raw.get("data") or {}).get("instruments") or raw.get("required_instruments") or []
    return [n for n in dict.fromkeys(str(n).removesuffix("-SPOT") for n in names)]


def open_status(registry: Path) -> dict[str, str]:
    """Idea id -> status, read through a read-only connection."""
    uri = f"file:{registry}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return dict(conn.execute("select id, status from idea"))


def scan(root: Path, registry: Path, config=None, only: str | None = None) -> list[Row]:
    from qlab import knowable_coins as kc
    from qlab import one_check
    from qlab.pipeline.spec import load_spec

    config = config or kc.load_config(HERE / kc.CONFIG, root=root)
    statuses = open_status(registry)

    def volume_of(venue: str):
        return kc.raw_volume(venue, config)

    rows: dict[str, Row] = {}
    for path in sorted((root / "specs").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict) or "idea_id" not in raw or "code_ref" not in raw:
            continue
        idea = str(raw["idea_id"])
        status = statuses.get(idea)
        if status is None or status in CLOSED or idea in rows or (only and only != idea):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            spec = load_spec(path)
        except Exception:  # noqa: BLE001 - a spec that does not load is not judged by it
            continue
        row = _literal(spec, status, rel, config, volume_of) if spec.instruments_chosen_at else None
        if row is None and one_check.declares_lists(spec):
            row = _check(spec, status, rel, path, config, volume_of, root)
        if row is not None:
            rows[idea] = row
    return list(rows.values())


def _literal(spec, status, rel, config, volume_of) -> Row | None:
    from qlab import knowable_coins as kc

    coins = _coins({"data": {"instruments": spec.data.instruments},
                    "required_instruments": spec.required_instruments})
    volume = volume_of(spec.data.source)
    if not coins or volume is None:
        return None
    known = kc.check(volume, coins, venue=spec.data.source, start=spec.data.start,
                     end=spec.data.end, config=config)
    return Row(spec.idea_id, status, rel, "literal", spec.data.source, tuple(coins),
               spec.data.start, spec.instruments_chosen_at, kc.list_day(known, spec.instruments_chosen_at),
               known.plain, known.known_from)


def _check(spec, status, rel, path, config, volume_of, root) -> Row | None:
    from qlab import one_check

    def listing():
        return one_check.binance_listing(store=root / "data" / "one_check", fetch=_no_network)

    kwargs = {"venue_names": listing(), "spec_path": path, "knowable_config": config}
    old = one_check.check_spec(spec, volume_of=lambda venue: None, **kwargs)
    new = one_check.check_spec(spec, volume_of=volume_of, **kwargs)
    if old.spec is None or new.spec is None or new.spec.coins_knowable is None:
        return None
    known = new.spec.coins_knowable
    return Row(spec.idea_id, status, rel, "check", new.spec.data.source, tuple(known.coins),
               new.spec.data.start, old.spec.judged_from, new.spec.judged_from, known.plain,
               known.known_from)


def _no_network():
    raise OSError("dry run: the listing kept in data/one_check/ is used, nothing is fetched")


def render(rows: list[Row]) -> str:
    lines = []
    changed = [r for r in rows if r.changed]
    lines.append(f"open ideas with a coin list the check looked at: {len(rows)}; "
                 f"judged window changed: {len(changed)}")
    for row in sorted(changed, key=lambda r: (r.venue, r.idea_id)):
        before = row.before.isoformat() if row.before else "data start"
        after = row.after.isoformat() if row.after else "data start"
        lines.append(f"{row.idea_id} [{row.status}, {row.venue}, {row.kind}] "
                     f"coins {','.join(row.coins)}: judged_from {before} -> {after} "
                     f"(data from {row.data_start}) | {row.path}")
        lines.append(f"    {row.plain}")
    same = [r for r in rows if not r.changed]
    for row in sorted(same, key=lambda r: r.idea_id):
        lines.append(f"unchanged: {row.idea_id} [{row.status}, {row.venue}] "
                     f"judged_from {row.before or 'data start'} | {row.plain}")
    return "\n".join(lines)


def apply(rows: list[Row], root: Path, config) -> list[str]:
    """Rewrite the literal dates in the spec files of the changed `literal` rows."""
    from qlab import knowable_coins as kc

    done = []
    for row in rows:
        if row.kind != "literal" or not row.changed:
            continue
        path = root / row.path
        text = path.read_text(encoding="utf-8")
        known = kc.check(kc.raw_volume(row.venue, config), row.coins, venue=row.venue,
                         start=row.data_start, end=_end(text), config=config)
        text = _set_line(text, "judged_from", row.after)
        text = _set_line(text, "instruments_chosen_at", row.after)
        block = yaml.safe_dump({"coins_knowable": known.model_dump(mode="json")},
                               allow_unicode=True, sort_keys=False)
        text = re.sub(r"(?m)^coins_knowable:.*(?:\n[ \t].*)*\n?", "", text).rstrip("\n") + "\n" + block
        path.write_text(text, encoding="utf-8")
        done.append(row.path)
    return done


def _end(text: str) -> date:
    return date.fromisoformat(str(yaml.safe_load(text)["data"]["end"]))


def _set_line(text: str, key: str, value: date | None) -> str:
    pattern = re.compile(rf"(?m)^{key}:.*$")
    if value is None:
        return pattern.sub("", text).replace("\n\n\n", "\n\n")
    line = f"{key}: '{value.isoformat()}'"
    return pattern.sub(line, text) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="the q-lab checkout to read")
    parser.add_argument("--registry", type=Path, default=None, help="default: <root>/data/qlab.db")
    parser.add_argument("--only", default=None, help="one idea id")
    parser.add_argument("--apply", action="store_true", help="write the literal spec files")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    os.chdir(root)  # the lab reads specs/, night/ and data/ relative to its root
    from qlab import knowable_coins as kc

    config = kc.load_config(HERE / kc.CONFIG, root=root)
    rows = scan(root, args.registry or root / "data" / "qlab.db", config, args.only)
    print(render(rows))
    if args.apply:
        for path in apply(rows, root, config):
            print(f"rewrote {path}")
    else:
        print("\ndry run: nothing written (use --apply for the literal spec files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
