"""Record of regime-detector variants tried (docs/REGIME_DETECT.md; owner,
2026-10-04: «да, делай оба» -- the detectors' own trials on record).

A research script calls `record` for every variant it measures, on the
development period or on the holdout; `summary` counts, per family, how many
variants were tried and how many times the holdout was looked at -- the
number a "best of many" must be read against, as `trial` is for strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

DEVELOPMENT, HOLDOUT = "development", "holdout"


def _code_sha() -> str:
    from qlab.pipeline.evaluate import _code_sha as sha

    return sha()


def record(family: str, variant: str, period: str, window: tuple, metrics: dict, script: str,
           *, chosen: bool = False, notes: str | None = None, session=None) -> int:
    """Write one detector trial; returns its id. `window` is (start, end)
    of the scored days, end exclusive."""
    from qlab.registry.db import session_scope
    from qlab.registry.models import DetectorTrial

    if period not in (DEVELOPMENT, HOLDOUT):
        raise ValueError(f"period must be {DEVELOPMENT!r} or {HOLDOUT!r}, not {period!r}")
    start, end = (pd.Timestamp(x).date() if not isinstance(x, date) else x for x in window)
    row = DetectorTrial(
        family=family, variant=variant, period=period, data_start=start, data_end=end,
        metrics={k: float(v) for k, v in metrics.items()}, script=script, code_sha=_code_sha(),
        chosen=chosen, recorded_at=datetime.now(UTC), notes=notes)
    if session is not None:
        session.add(row)
        session.flush()
        return row.id
    with session_scope() as own:
        own.add(row)
        own.flush()
        return row.id


@dataclass(frozen=True)
class FamilySummary:
    family: str
    development_variants: int
    holdout_looks: int
    chosen: list[str]


def summary(session) -> list[FamilySummary]:
    from qlab.registry.models import DetectorTrial

    rows = session.query(DetectorTrial).all()
    out = []
    for family in sorted({r.family for r in rows}):
        mine = [r for r in rows if r.family == family]
        out.append(FamilySummary(
            family=family,
            development_variants=len({r.variant for r in mine if r.period == DEVELOPMENT}),
            holdout_looks=sum(1 for r in mine if r.period == HOLDOUT),
            chosen=sorted({r.variant for r in mine if r.chosen})))
    return out


__all__ = ["DEVELOPMENT", "HOLDOUT", "FamilySummary", "record", "summary"]
