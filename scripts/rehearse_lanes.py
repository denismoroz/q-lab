"""Rehearse the night's lanes on a COPY of the project and of the registry (docs/PIPELINE.md).

Never run this in the real checkout: it starts a night. It is the proof that steps running in lanes
leave every trial exactly as the same steps one after another do.

    # run `qlab night run ...` with the agent steps replaced by sleeping stubs (no token is spent);
    # `--stub step=seconds[:spec,spec]` (the specs are run on the stand from that step's thread)
    python scripts/rehearse_lanes.py run --stub assess=30 --stub research=60:specs/x.yaml -- \\
        --off director --concurrency 3

    # the trials, verdicts and spec rows added to a registry after a baseline: identical or not
    python scripts/rehearse_lanes.py compare BASE.db RUN_A.db RUN_B.db

    # when the steps of the latest run started and ended, and which of them overlapped
    python scripts/rehearse_lanes.py timeline RUN.db

The copy is made by hand: the project without `data/`, then `data/` cloned (`cp -Rc`) with the
registry from sqlite's online backup. `QLAB_DB` must name the copy's registry and the working
directory must be the copy."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rehearse_lanes.py run")
    parser.add_argument("--stub", action="append", default=[],
                        help="step=seconds[:spec,spec]: replace the agent step with a stub")
    own, rest = parser.parse_known_args(argv)
    rest = [a for a in rest if a != "--"]
    plan: dict[str, tuple[float, list[str]]] = {}
    for item in own.stub:
        step, _, body = item.partition("=")
        seconds, _, specs = body.partition(":")
        plan[step] = (float(seconds), [s for s in specs.split(",") if s])

    import qlab.night as night
    import qlab.nightflow.table as table_module
    from qlab.cli import app
    from qlab.nightflow import rehearsal

    real_steps, real_load = night.night_steps, table_module.load_table
    night.night_steps = lambda: rehearsal.stubbed_steps(real_steps(), plan)
    table_module.load_table = lambda path=None: rehearsal.table_with_stub_keys(
        real_load(path), list(plan))
    app(["night", "run", *rest], prog_name="qlab")
    return 0


def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _canon(text: str | None) -> str:
    if text is None:
        return "null"

    def round_floats(value):
        if isinstance(value, float):
            return float(f"{value:.10g}")
        if isinstance(value, dict):
            return {k: round_floats(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [round_floats(v) for v in value]
        return value
    return json.dumps(round_floats(json.loads(text)), sort_keys=True, default=str)


def _trials(db: sqlite3.Connection, first_id: int) -> Counter:
    rows = db.execute(
        "select s.idea_id, s.code_ref, s.params, t.snapshot_id, t.config_hash, t.status, t.route,"
        " t.route_reason, t.metrics from trial t join spec s on s.id = t.spec_id"
        " where t.id > ?", (first_id,))
    return Counter((r[0], r[1], _canon(r[2]), r[3], r[4], r[5], r[6], r[7], _canon(r[8]))
                   for r in rows)


def _specs(db: sqlite3.Connection, first_id: int) -> Counter:
    return Counter((r[0], r[1], r[2], _canon(r[3])) for r in db.execute(
        "select idea_id, version, code_ref, params from spec where id > ?", (first_id,)))


def _verdicts(db: sqlite3.Connection, first_id: int) -> Counter:
    rows = db.execute(
        "select idea_id, stage, rule_id, rules_version, metric, value, comparator, threshold,"
        " passed, data_range_start, data_range_end from verdict where id > ?", (first_id,))
    return Counter(tuple(round(v, 10) if isinstance(v, float) else v for v in r) for r in rows)


def _ideas(db: sqlite3.Connection) -> Counter:
    return Counter(db.execute("select id, status, parent_id from idea"))


def _compare(argv: list[str]) -> int:
    base_path, a_path, b_path = argv
    base, a, b = _connect(base_path), _connect(a_path), _connect(b_path)
    first_trial = base.execute("select coalesce(max(id), 0) from trial").fetchone()[0]
    first_spec = base.execute("select coalesce(max(id), 0) from spec").fetchone()[0]
    first_verdict = base.execute("select coalesce(max(id), 0) from verdict").fetchone()[0]
    same = True
    for name, fn, first in (("trials", _trials, first_trial), ("spec rows", _specs, first_spec),
                            ("verdicts", _verdicts, first_verdict)):
        ca, cb = fn(a, first), fn(b, first)
        only_a, only_b = ca - cb, cb - ca
        print(f"{name}: A {sum(ca.values())} rows ({len(ca)} distinct), "
              f"B {sum(cb.values())} rows ({len(cb)} distinct), "
              f"only in A {sum(only_a.values())}, only in B {sum(only_b.values())}")
        if only_a or only_b:
            same = False
            for label, extra in (("A", only_a), ("B", only_b)):
                for row, count in list(extra.items())[:5]:
                    print(f"   only in {label} x{count}: {str(row)[:300]}")
    ia, ib = _ideas(a), _ideas(b)
    print(f"idea statuses: {'identical' if ia == ib else 'DIFFERENT'} ({sum(ia.values())} ideas)")
    same = same and ia == ib
    print("IDENTICAL" if same else "DIFFERENT")
    return 0 if same else 1


def _aware(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _timeline(argv: list[str]) -> int:
    db = _connect(argv[0])
    run = db.execute("select id, night_date, started_at, finished_at, state, concurrency from "
                     "pipeline_run order by id desc limit 1").fetchone()
    print(f"run {run[0]} night {run[1]} state {run[4]} concurrency {run[5]}")
    rows = db.execute("select step_id, state, started_at, finished_at, duration_s, items_done, "
                      "items_failed, error from pipeline_step_run where run_id = ? order by id",
                      (run[0],)).fetchall()
    t0 = min(_aware(r[2]) for r in rows)
    total = 0.0
    spans = []
    for step, state, start, end, seconds, done, failed, error in rows:
        s = (_aware(start) - t0).total_seconds()
        e = (_aware(end) - t0).total_seconds() if end else float("nan")
        spans.append((step, s, e, state))
        total += seconds or 0.0
        print(f"  {step:<24} {state:<11} start {s:8.1f}s end {e:8.1f}s took {seconds or 0:8.1f}s "
              f"items {done}/{failed}" + (f"  ERROR {error[:90]}" if error else ""))
    wall = max(e for _, _, e, _ in spans if e == e)
    ran = [x for x in spans if x[3] not in ("off", "skipped", "blocked", "cut")]
    overlaps = sorted((a[0], b[0]) for i, a in enumerate(ran) for b in ran[i + 1:]
                      if a[1] < b[2] and b[1] < a[2])
    print(f"wall {wall:.1f}s, sum of the steps {total:.1f}s, {len(overlaps)} overlapping pairs")
    for a, b in overlaps:
        print(f"   {a} || {b}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "compare", "timeline"):
        print(__doc__)
        return 2
    return {"run": _run, "compare": _compare, "timeline": _timeline}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
