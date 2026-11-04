"""Find doubles among the OPEN ideas of the registry (docs/DUPLICATES.md).

Both checks of `qlab.duplicates`, across all open ideas, printed as groups:

1. the same fingerprint of the latest spec (the same strategy code, numbers after defaults, lists
   to choose among, data, costs and money);
2. the same results (daily returns on the strategy's money correlated at least 0.999 and the yearly
   return within 0.1 pp, same venue and window), from the stored forward series.

Read-only by default: it opens the registry, reads the spec files and the stored series, prints.
Nothing is retired, marked or written. `--mark` writes the «дубль X по результатам» readings to
`data/duplicates/ledger.json` (a stored reading, it changes no status).

    uv run python scripts/find_duplicates.py
    QLAB_DB=/path/qlab.db QLAB_FORWARD_SERIES=/path/forward_series \
        uv run python scripts/find_duplicates.py

Run it from the repository root: the spec paths are relative to it.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mark", action="store_true",
                        help="write the results readings to the duplicates ledger")
    args = parser.parse_args(argv)

    from qlab import duplicates
    from qlab.registry.db import session_scope

    with session_scope() as session:
        index = duplicates.open_index(session)
        print(f"open ideas: {len(index)}")
        by_print = duplicates.groups_by_fingerprint(index)
        by_results = duplicates.groups_by_results(session, index)
        titles = {i.idea_id: i for i in index}
        for name, groups in (("SAME FINGERPRINT", by_print), ("SAME RESULTS", by_results)):
            print(f"\n== {name}: {len(groups)} group(s)")
            for n, group in enumerate(groups, 1):
                print(f"\n[{n}] {group.detail}")
                for idea_id in group.ideas:
                    item = titles[idea_id]
                    print(f"    {idea_id}  [{item.status}]  {item.plain_title[:90]}")
        if args.mark:
            ledger = duplicates.load_ledger()
            for group in by_results:
                # the newest idea of a group is the double of the oldest
                from qlab.registry.models import Idea

                ordered = sorted(group.ideas, key=lambda i: (session.get(Idea, i).created_at, i))
                latest = duplicates.latest_series_trials(session)
                for later in ordered[1:]:
                    record = duplicates.find_same_results(
                        session, later, latest[later], latest=latest, candidates=ordered[:1])
                    if record is not None:
                        ledger["results"][later] = record
            duplicates.save_ledger(ledger)
            print("\nledger written:", duplicates.ledger_path())
    return 0


if __name__ == "__main__":
    sys.exit(main())
