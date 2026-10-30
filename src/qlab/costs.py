"""One cost table per venue (night/costs.yaml, docs/COSTS.md).

Owner, 2026-10-29, on docs/EVALUATION_PATHS.md (question В7): the same venue was priced by different
specs in different ways, and nothing said which number was the lab's. The table is the answer: a row
per venue, each number with its source. A spec's `costs` are the row of the venue its orders go to
(`execution_venue`, else `data.source`), or the spec says why they are not
(`sources["costs.differs_from_table"]`, checked like any other citation by
`qlab.pipeline.sources`). Specs and second stages made from now on take the row
(`qlab.pipeline.stage2`, the idea generator).

Pure reading: nothing here computes a metric or decides a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

TABLE = Path(__file__).resolve().parents[2] / "night" / "costs.yaml"
DEVIATION_KEY = "costs.differs_from_table"
"""The `sources` key of a spec whose costs are not its venue's row."""
TOLERANCE = 1e-9
ENFORCE = True
"""Whether `problems` asks anything. Only the test suite turns it off (conftest.py): the specs of
tests have made-up costs."""


@dataclass(frozen=True)
class VenueCosts:
    venue: str
    taker_fee_bps: float
    slippage_bps: float
    sources: dict[str, str]
    """`taker_fee_bps` / `slippage_bps` -> the citation of the number."""
    two_leg_taker_fee_bps: float | None = None
    """The taker fee of a book that holds a spot leg against a perpetual on this venue (the mean of
    the perpetual's and the spot's), when the row states it (Binance); cited in
    `sources["two_leg_taker_fee_bps"]`."""


@lru_cache(maxsize=4)
def _load(path: Path) -> dict[str, VenueCosts]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, VenueCosts] = {}
    try:
        for venue, row in raw["venues"].items():
            sources = {str(k): " ".join(str(v).split()) for k, v in row["sources"].items()}
            for key in ("taker_fee_bps", "slippage_bps"):
                if not sources.get(key):
                    raise ValueError(f"{path}: venue {venue!r} has no source for {key}")
            two = row.get("two_leg_taker_fee_bps")
            if two is not None and not sources.get("two_leg_taker_fee_bps"):
                raise ValueError(f"{path}: venue {venue!r} has no source for two_leg_taker_fee_bps")
            out[str(venue)] = VenueCosts(str(venue), float(row["taker_fee_bps"]),
                                         float(row["slippage_bps"]), sources,
                                         None if two is None else float(two))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    return out


def table(path: Path = TABLE) -> dict[str, VenueCosts]:
    """Venue -> its row of `night/costs.yaml`. A missing key or an uncited number is an error:
    nothing in the table has a default."""
    return _load(path)


def venue_costs(venue: str, path: Path = TABLE) -> VenueCosts | None:
    return table(path).get(venue)


def order_venue(spec) -> str:
    """The venue where the spec's orders would go: `execution_venue`, else the data's venue."""
    return spec.execution_venue or spec.data.source


def two_legged(spec) -> bool:
    """Whether the spec's book holds a spot leg against a perpetual (two legs at once)."""
    return int(getattr(spec, "simultaneous_legs", 1) or 1) >= 2


def expected_taker_fee_bps(spec, row: VenueCosts) -> float:
    """The taker fee the venue's row asks of this spec: the perpetual's, or for a two-legged book
    on a venue whose row states it, the mean of the two legs'."""
    if two_legged(spec) and row.two_leg_taker_fee_bps is not None:
        return row.two_leg_taker_fee_bps
    return row.taker_fee_bps


def matches(spec, row: VenueCosts) -> bool:
    return (abs(spec.costs.taker_fee_bps - expected_taker_fee_bps(spec, row)) <= TOLERANCE
            and abs(spec.costs.slippage_bps - row.slippage_bps) <= TOLERANCE)


def problems(spec, path: Path = TABLE) -> list[str]:
    """What is wrong with the costs of `spec` against the table; empty means they are fine."""
    if not ENFORCE:
        return []
    venue = order_venue(spec)
    row = venue_costs(venue, path)
    deviates = bool((spec.sources.get(DEVIATION_KEY) or "").strip())
    if row is None:
        return [f"night/costs.yaml has no row for venue {venue!r}: costs of this spec have "
                f"nothing to be compared with"] if not deviates else []
    if matches(spec, row):
        if deviates:
            return [f"{DEVIATION_KEY} is given but the costs are the table's row of {venue}: "
                    "drop the sentence"]
        return []
    if deviates:
        return []
    return [f"costs {spec.costs.taker_fee_bps:g} + {spec.costs.slippage_bps:g} bp are not the "
            f"table's row of {venue} ({expected_taker_fee_bps(spec, row):g} + "
            f"{row.slippage_bps:g} bp, "
            f"night/costs.yaml): say why in sources[{DEVIATION_KEY!r}]"]


def table_citation_problems(path: Path = TABLE) -> list[str]:
    """Problems with the table's own citations, checked as a spec's are
    (`qlab.pipeline.sources.citation_problems`): a cited file exists and holds the number, a
    `derived:` formula gives it."""
    from qlab.pipeline.sources import citation_problems

    out: list[str] = []
    for venue, row in table(path).items():
        for key in ("taker_fee_bps", "slippage_bps"):
            out += citation_problems(f"{venue}.{key}", row.sources[key], getattr(row, key))
        if row.two_leg_taker_fee_bps is not None:
            out += citation_problems(f"{venue}.two_leg_taker_fee_bps",
                                     row.sources["two_leg_taker_fee_bps"],
                                     row.two_leg_taker_fee_bps)
    return out


__all__ = ["DEVIATION_KEY", "TABLE", "VenueCosts", "expected_taker_fee_bps", "matches",
           "order_venue", "problems", "two_legged",
           "table", "table_citation_problems", "venue_costs"]
