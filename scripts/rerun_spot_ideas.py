"""Judge three ideas again now that the stand has the spot market's own prices
(docs/SPOT_PRICES.md).

They were refused with "implementation does not express: ... the stand marks a spot column at its
perp's price" because their rule reads the coin's own spot closes:

- `bv3-research-fast-release` (the researcher's first Bv3 experiment, Binance and the second stage
  on Hyperliquid),
- `bv2-fast-release-by-own-trend` (the implementer's narrow card of the same idea),
- `frab-lot-leftover` (the coin left over after a lot-rounded sale, valued at the spot).

What it does, in order:

1. Reads each idea's spec file and shows the change (always). The change: `data.spot_prices: own`
   is declared and the line of `unexpressed_mechanisms` that said the spot is priced at the perp is
   removed. Nothing else in the file changes, and the script refuses to write if the edit would.
2. `--apply` writes the changed spec files. `--run` applies and then judges the ideas the way the
   night does: the researcher's experiment through `qlab.agents.researcher.run_experiments`
   (the idea's one check on Binance, the second stage on Hyperliquid), the implementer's cards
   through `qlab.implement_run.run_card` (check, baseline, second stage), every run a recorded
   trial. Code only: no agent, no tokens.

A dry run (no flag) touches nothing: no file, no registry row, no ledger.

    uv run python scripts/rerun_spot_ideas.py                  # dry run: shows the changes
    uv run python scripts/rerun_spot_ideas.py --apply          # writes the three spec files
    uv run python scripts/rerun_spot_ideas.py --run            # applies and judges, 200 noise books
    uv run python scripts/rerun_spot_ideas.py --run --only frab-lot-leftover --noise-trials 20

Run it from the repository root: the spec, ledger and data paths of the stand are relative to it.
`QLAB_DB` picks the registry (a copy for a rehearsal); `--ledger` the implementer's ledger
(default `data/implement/ledger.json`; a rehearsal gives its own); `--memo` where the lines of the
researcher's experiment are written (default `data/research/spot-rerun-<day>.md`).
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Target:
    idea_id: str
    spec: Path
    kind: str  # "research": a researcher's experiment; "card": an implementer's card
    card: str | None = None
    family: str | None = None  # the researcher's strategy name


TARGETS = (
    Target("bv3-research-fast-release", Path("specs/research/bv3/bv3-research-fast-release.yaml"),
           "research", family="bv3"),
    Target("bv2-fast-release-by-own-trend", Path("specs/agent/bv2-fast-release-by-own-trend.yaml"),
           "card", card="cards/bv2-fast-release-by-own-trend.yaml"),
    Target("frab-lot-leftover", Path("specs/agent/frab-lot-leftover.yaml"),
           "card", card="cards/frab-lot-leftover.yaml"),
)


def changes(target: Target) -> tuple[str, str, list[str]] | str:
    """(old text, new text, removed lines) for a target, or the reason it cannot be changed."""
    from qlab.spot_declaration import declare_own_in_spec_text

    if not target.spec.is_file():
        return f"the spec file {target.spec} does not exist here"
    old = target.spec.read_text(encoding="utf-8")
    try:
        new, removed = declare_own_in_spec_text(old)
    except ValueError as exc:
        return str(exc)
    return old, new, removed


def show(target: Target) -> tuple[bool, str | None]:
    """Print the change; (has a change to make, the reason it cannot)."""
    found = changes(target)
    print(f"\n== {target.idea_id}  ({target.spec})")
    if isinstance(found, str):
        print(f"   CANNOT: {found}")
        return False, found
    old, new, removed = found
    if old == new:
        print("   already declares data.spot_prices: own and has no perp-mark line: no change")
        return False, None
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after", n=0,
                                     lineterm=""):
        print("   " + line)
    return True, None


def apply(target: Target) -> bool:
    found = changes(target)
    if isinstance(found, str):
        return False
    old, new, _ = found
    if old != new:
        target.spec.write_text(new, encoding="utf-8")
    return True


def judge_experiment(target: Target, memo: Path, noise_trials: int) -> list[str]:
    """The researcher's experiment, as the night judges it (one check on Binance, then the
    second stage on Hyperliquid); the lines are appended to `memo`."""
    from qlab.agents.researcher import run_experiments

    return run_experiments(memo, [target.spec], name=target.family or target.idea_id,
                           noise_trials=noise_trials)


def judge_card(target: Target, ledger_path: Path, noise_trials: int) -> list[str]:
    """The implementer's card: check, baseline and second stage (`qlab.implement_run`). The
    card's old record (blocked by the reviewer, or finished as not evaluable) is kept in the new
    one as `previous`; the runs start from nothing, on the changed spec."""
    from qlab import implement_run, night, twin_stage

    class Context(implement_run.RunContext):
        def save(self) -> None:  # the ledger only: the night's own state is not this script's
            twin_stage.save_ledger(self.ledger, self.ledger_path)

    today = datetime.now(UTC).date()
    ledger = twin_stage.load_ledger(ledger_path)
    old = ledger.get(target.card)
    record = implement_run.new_record(target.idea_id, today)
    record.update(status="pending-run", attempts=(old or {}).get("attempts", 1),
                  rerun="data.spot_prices: own (docs/SPOT_PRICES.md)")
    if old:
        record["previous"] = {k: old.get(k) for k in ("status", "blocked", "day") if old.get(k)}
        if old.get("entry"):
            record["entry"] = old["entry"]
    ledger[target.card] = record
    ctx = Context(day=today, end=night.last_closed_day(), noise_trials=noise_trials, state={},
                  ledger=ledger, ledger_path=ledger_path,
                  stage2_config=implement_run.load_stage2_config())
    implement_run.run_card(target.card, record, ctx)
    ctx.save()
    entry = {**(record.get("entry") or {}), "runs": record["runs"]}
    return [f"- `{target.idea_id}` (карточка {target.card}): {implement_run.summary(entry)}",
            *implement_run.entry_lines(entry)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changed spec files")
    parser.add_argument("--run", action="store_true",
                        help="apply, then judge the ideas (records trials in the registry)")
    parser.add_argument("--only", action="append", default=[], metavar="IDEA",
                        help="restrict to this idea (repeatable)")
    parser.add_argument("--noise-trials", type=int, default=200)
    parser.add_argument("--ledger", type=Path, default=None,
                        help="implementer's ledger (default data/implement/ledger.json)")
    parser.add_argument("--memo", type=Path, default=None)
    args = parser.parse_args(argv)

    from qlab import implement_run

    targets = [t for t in TARGETS if not args.only or t.idea_id in args.only]
    unknown = set(args.only) - {t.idea_id for t in TARGETS}
    if unknown:
        print(f"unknown idea(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    mode = "run" if args.run else "apply" if args.apply else "dry run (nothing is written)"
    print(f"rerun_spot_ideas: {mode}; {len(targets)} idea(s)")
    ready = []
    for target in targets:
        has_change, cannot = show(target)
        if cannot is None:
            ready.append(target)
    if not (args.apply or args.run):
        print("\nDry run: nothing written. --apply writes the spec files, --run judges the ideas.")
        return 0
    for target in ready:
        apply(target)
        print(f"\nwritten: {target.spec}")
    if not args.run:
        return 0
    today = datetime.now(UTC).date()
    memo = args.memo or Path(f"data/research/spot-rerun-{today}.md")
    memo.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = args.ledger or implement_run.LEDGER
    print(f"\nmemo of the experiment: {memo}; ledger: {ledger_path}")
    for target in ready:
        print(f"\n== judging {target.idea_id}", flush=True)
        try:
            lines = (judge_experiment(target, memo, args.noise_trials)
                     if target.kind == "research"
                     else judge_card(target, ledger_path, args.noise_trials))
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            lines = [f"- `{target.idea_id}`: the run failed: {type(exc).__name__}: {exc}"]
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
