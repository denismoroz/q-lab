"""By hand: `uv run python -m qlab.publications fetch [--days 90]` reads the paper
APIs and applies the topic rule (no tokens); `list` shows what is stored.

The night does the same on its own (`qlab.publications.stage`); this is for a
backfill and for looking at what the rule marks. The registry must be migrated
(`qlab init-db`)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from qlab.publications.filter import load_rules
from qlab.publications.stage import CONFIG, fetch_all, load_config
from qlab.registry.db import session_scope
from qlab.registry.models import Publication


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qlab.publications")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch", help="read the APIs, store, apply the topic rule")
    fetch.add_argument("--days", type=int, default=None,
                       help="read this far back instead of from the newest stored paper")
    fetch.add_argument("--config", type=Path, default=CONFIG)
    listing = sub.add_parser("list", help="stored papers")
    listing.add_argument("--candidates", action="store_true", help="only the marked ones")
    args = parser.parse_args(argv)

    with session_scope() as session:
        if args.cmd == "fetch":
            cfg = load_config(args.config)
            rules = load_rules(Path(cfg["filter"]))
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                summary = fetch_all(session, cfg, rules, client=client,
                                    now=datetime.now(UTC), days=args.days)
            print(json.dumps(summary, ensure_ascii=False, indent=1))
        else:
            query = session.query(Publication).order_by(Publication.published.desc())
            if args.candidates:
                query = query.filter(Publication.candidate.is_(True))
            for row in query:
                mark = "*" if row.candidate else " "
                print(f"{mark} {row.published:%Y-%m-%d} {row.source:6} {row.title} <{row.link}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
