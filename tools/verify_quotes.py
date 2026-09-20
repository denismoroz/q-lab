"""Verify that every extracted number is traceable to its source document.

Extraction is the one step in the pipeline where a model reads prose and writes
numbers, so it is the one step where a number can be invented. Every verdict in
`seed/graveyard.yaml` carries a `source_quote`; this script checks each quote
appears verbatim in the file it cites, under light normalisation (whitespace,
dash and quote-mark variants). A quote that cannot be found is not proof of
fabrication, but it means the number can no longer be audited — which is the
same thing as far as capital allocation is concerned.

Usage:  uv run python tools/verify_quotes.py [--seed seed/graveyard.yaml] [--sources <repo>]
Exit code 1 if any quote fails to verify.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import yaml

DEFAULT_SOURCES = Path("/Users/d/prj/funding-rate-arbitrage")


def normalise(text: str) -> str:
    """Fold away formatting that carries no meaning for a quote's identity."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("—", "-").replace("–", "-")
    for ch in "«»“”":
        text = text.replace(ch, '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=Path("seed/graveyard.yaml"))
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    args = parser.parse_args()

    if not args.sources.is_dir():
        print(f"source repository not available at {args.sources} — nothing to verify against")
        return 0

    data = yaml.safe_load(args.seed.read_text())
    cache: dict[str, str | None] = {}
    failures: list[tuple[str, str, str]] = []
    checked = 0

    for idea in data.get("ideas", []):
        for verdict in idea.get("verdicts", []):
            checked += 1
            quote, source_file = verdict.get("source_quote"), verdict.get("source_file")
            if not quote or not source_file:
                failures.append((idea["id"], verdict.get("metric", "?"), "no quote or source file"))
                continue
            if source_file not in cache:
                path = args.sources / source_file
                cache[source_file] = normalise(path.read_text()) if path.exists() else None
            haystack = cache[source_file]
            if haystack is None:
                failures.append((idea["id"], verdict.get("metric", "?"), f"missing: {source_file}"))
            elif normalise(quote) not in haystack:
                failures.append(
                    (idea["id"], verdict.get("metric", "?"), "quote not found in source")
                )

    print(f"checked {checked} verdicts, {checked - len(failures)} verified, {len(failures)} failed")
    for idea_id, metric, problem in failures:
        print(f"  {idea_id} / {metric}: {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
