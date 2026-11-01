"""Rehearse "one idea, one check" on a COPY of the project and of the registry (docs/ONE_CHECK.md).

Never run `run`, `retire` or `waiting` in the real checkout: they write to the registry named by
`QLAB_DB`, and the script refuses to start unless it names a file outside the real `data/`.

    # 1. the copy: the project without data/, data/ cloned (cp -Rc, copy on write: it costs next to
    #    nothing), the registry from sqlite's online backup, the sibling funding-rate-arbitrage link
    python scripts/rehearse_one_check.py prepare /path/to/scratch/one-check-rehearsal

    # 2. inside the copy, with QLAB_DB naming the copy's registry:
    #    a stub researcher (no agent, no token) declares a new idea; `night run --only research ...`
    #    runs it through its ONE check to a verdict on both venues; the report and the page follow
    python scripts/rehearse_one_check.py run

    # 3. the owner's retirements on the copy, with a before/after table of status and route
    python scripts/rehearse_one_check.py retire [--apply]

    # 4. what the registry holds for the new idea: one idea, one ladder, no twin in the journal
    python scripts/rehearse_one_check.py show IDEA

    # 5. throw the copy away
    python scripts/rehearse_one_check.py cleanup /path/to/scratch/one-check-rehearsal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

IDEA = "xsmom-research-rehearsal"
STRATEGY = "xsmom"
PAIRS = ("xsmom-research-nobull", "xsmom-research-liq1m")
OWNER_WORDS = ("владелец 2026-10-31: «не уверен что нам нужна вот такая дубликация. поидее просто "
               "для стратегии достаточна проверка. так мы гоняем все 2х хотя все равно судим по "
               "двойнику»; «да, делай после пайплайна»; 2026-11-01: «да, делай по порядку»")


def _guard() -> Path:
    """The copy's registry; refuses anything but a rehearsal copy (made by `prepare`)."""
    db = os.environ.get("QLAB_DB")
    if not db:
        sys.exit("QLAB_DB must name the copy's registry")
    here = Path.cwd().resolve()
    if not (here / ".rehearsal").is_file() or not Path(db).resolve().is_relative_to(here):
        sys.exit("refusing: this is not a rehearsal copy (no .rehearsal marker in the working "
                 "directory, or QLAB_DB lies outside it); run `prepare` and work inside the copy")
    return Path(db)


def _prepare(target: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if target.exists():
        sys.exit(f"{target} exists")
    copy = target / "q-lab"
    copy.mkdir(parents=True)
    subprocess.run(["rsync", "-a", "--exclude", "data", "--exclude", ".git", "--exclude",
                    "web/node_modules", "--exclude", ".venv", f"{root}/", f"{copy}/"], check=True)
    subprocess.run(["cp", "-Rc", str(root / "data"), str(copy / "data")], check=True)
    for stale in (copy / "data").glob("qlab.db*"):
        stale.unlink()
    src = sqlite3.connect(f"file:{root / 'data' / 'qlab.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(copy / "data" / "qlab.db")
    src.backup(dst)
    dst.close()
    src.close()
    (target / "funding-rate-arbitrage").symlink_to(root.parent / "funding-rate-arbitrage")
    (copy / ".venv").symlink_to(root / ".venv")
    (copy / ".rehearsal").write_text("a rehearsal copy\n", encoding="utf-8")
    print(f"copy: {copy}\nQLAB_DB={copy / 'data' / 'qlab.db'}")


# --------------------------------------------------------------------------------------------
# The stub researcher


def _stub_spec() -> str:
    """The experiment a stub researcher writes: the 20-leg XSMOM book with its rebalance open."""
    import yaml

    base = yaml.safe_load(Path("specs/xsmom-honest-20legs.yaml").read_text(encoding="utf-8"))
    base["idea_id"] = IDEA
    base["title"] = "XSMOM, честный универс, 20 ног: репетиция одной проверки"
    base["params"]["rebalance_days"] = 1
    for key in ("params_fixed_at", "params_fixed_evidence"):
        base.pop(key, None)
    base["choice"] = {"axes": {"rebalance_days": [1, 7]}, "source": "night/own_numbers.yaml"}
    sources = dict(base.get("sources") or {})
    sources["rebalance_days"] = ("night/own_numbers.yaml, the declared list for rebalance_days; "
                                 "owner 2026-10-26")
    base["sources"] = sources
    return yaml.safe_dump(base, sort_keys=False, allow_unicode=True)


def _install_stubs() -> None:
    """The researcher is replaced by a stub that declares one idea, and the budget by one that
    never calls anything: no agent runs and no token is spent."""
    import types

    import qlab.agents.researcher as researcher
    import qlab.budget as budget

    def stub_research(name, *, idea_ids, specs, docs, budget, session, max_experiments=2, **kw):
        memo = researcher.MEMO_DIR / f"{name}-rehearsal.md"
        memo.parent.mkdir(parents=True, exist_ok=True)
        memo.write_text("# Репетиция одной проверки\n\nЗаписку пишет заглушка, не агент.\n",
                        encoding="utf-8")
        out = researcher.ResearchOutcome(memo=memo, reply="stub")
        if name == STRATEGY:
            path = researcher.SPEC_DIR / name / f"{IDEA}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_stub_spec(), encoding="utf-8")
            out.specs.append(path)
        return out

    class _Cand:
        spent_tokens = 0

    class _Stage:
        def candidate(self, _name):
            return _Cand()

    class _Night:
        @staticmethod
        def open(_session, _reading):
            return _Night()

        def stage(self, _stage, candidates):
            return _Stage()

    researcher.research = stub_research
    budget.NightBudget = _Night
    budget.probe = lambda _session: types.SimpleNamespace(
        report=types.SimpleNamespace(reading=None))


def _only_xsmom_researched() -> None:
    import yaml

    path = Path("night/research.yaml")
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan["strategies"] = {STRATEGY: plan["strategies"][STRATEGY]}
    plan.pop("rotation", None)
    plan["max_experiments"] = 1
    path.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run(argv: list[str]) -> int:
    _guard()
    ap = argparse.ArgumentParser(prog="rehearse_one_check.py run")
    ap.add_argument("--date", default=date.today().isoformat())
    own = ap.parse_args(argv)
    _only_xsmom_researched()
    _install_stubs()
    from qlab.cli import app

    app(["night", "run", "--date", own.date, "--no-graveyard", "--only", "research",
         "--only", "twins", "--only", "ladder", "--only", "report"], standalone_mode=False)
    return _show([IDEA])


# --------------------------------------------------------------------------------------------
# What the registry holds


def _show(argv: list[str]) -> int:
    db = _guard()
    idea = argv[0] if argv else IDEA
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    print(f"== ideas named like {idea}")
    for r in con.execute("select id, status, parent_id from idea where id like ? order by id",
                         (idea + "%",)):
        print(f"  {r['id']}  {r['status']}  parent={r['parent_id']}")
    print("== trials (first the idea's own, then its second stage)")
    for r in con.execute("""select s.idea_id, t.id, t.route, json_extract(s.data_requirements,
        '$.source') as venue, json_extract(s.params,'$.refit_every') as refit
        from trial t join spec s on s.id = t.spec_id where s.idea_id like ? order by t.id""",
                         (idea + "%",)):
        print(f"  {r['idea_id']:42} trial {r['id']}  {r['venue']:12} refit={r['refit']}  "
              f"route={r['route']}")
    print("== the journal: items of the night's steps that name the idea")
    try:
        for r in con.execute("""select sr.step_id, i.name from pipeline_item i
            join pipeline_step_run sr on sr.id = i.step_run_id where i.name like ?
            order by sr.id""", (idea + "%",)):
            print(f"  {r['step_id']:12} {r['name']}")
    except sqlite3.Error as exc:
        print(f"  (journal not readable: {exc})")
    twin_items = con.execute(
        "select count(*) from pipeline_item i join pipeline_step_run sr on sr.id = i.step_run_id "
        "where i.name like ? and sr.step_id like 'twins%'", (idea + "%",)).fetchone()[0]
    print(f"twin items for the idea in the journal: {twin_items}")
    from qlab.ladder import build_ladder, load_ledgers
    from qlab.registry.db import session_scope

    with session_scope() as session:
        ladder = build_ladder(session, idea, load_ledgers())
        if ladder is None:
            print("no ladder (the idea does not exist)")
            return 1
        print("== the one ladder")
        for step in ladder.steps:
            print(f"  {step.number}. {step.outcome.value:10} {step.detail[:110]}")
        print(f"  by venue: {ladder.summary_ru}")
    return 0


# --------------------------------------------------------------------------------------------
# The owner's retirements, on the copy


def _snapshot(con: sqlite3.Connection) -> dict[str, tuple]:
    out = {}
    for (idea_id, status) in con.execute("select id, status from idea"):
        route = con.execute(
            "select t.route from trial t join spec s on s.id=t.spec_id where s.idea_id=? "
            "and t.route is not null order by t.id desc limit 1", (idea_id,)).fetchone()
        out[idea_id] = (status, route[0] if route else None)
    return out


def _retire(argv: list[str]) -> int:
    db = _guard()
    ap = argparse.ArgumentParser(prog="rehearse_one_check.py retire")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ideas", nargs="*", default=None)
    own = ap.parse_args(argv)
    con = sqlite3.connect(db)
    before = _snapshot(con)
    pairs = [(o, o + "-causal") for (o,) in con.execute(
        "select o.id from idea o join idea c on c.id = o.id || '-causal' order by o.id")]
    print("pair | original before | twin before")
    for o, c in pairs:
        print(f"{o} | {before[o]} | {before[c]}")
    if not own.apply:
        return 0
    ideas = own.ideas or []
    if not ideas:
        sys.exit("--ideas IDEA... names the originals to retire")
    subprocess.run([sys.executable, "-c", "from qlab.cli import app; app()", "retire", *ideas,
                    "--reason", OWNER_WORDS], check=True)
    after = _snapshot(sqlite3.connect(db))
    changed = {k: (before.get(k), after[k]) for k in after if before.get(k) != after[k]}
    print(json.dumps(changed, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "show", "retire", "cleanup"))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    own = parser.parse_args()
    if own.command == "prepare":
        _prepare(Path(own.rest[0]))
        return 0
    if own.command == "cleanup":
        shutil.rmtree(own.rest[0])
        return 0
    return {"run": _run, "show": _show, "retire": _retire}[own.command](own.rest)


if __name__ == "__main__":
    raise SystemExit(main())
