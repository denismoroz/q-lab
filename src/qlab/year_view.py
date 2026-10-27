"""The year-by-year view of a run, read back from its stored metrics (docs/YEARS.md).

`qlab.harness.metrics.year_breakdown` writes flat keys (`year_2021_return`, ...) into
a trial's metrics. This module turns them back into rows, one plain-Russian line for
the reports and the agents' briefs, and a structured form for the console's API.

Nothing here computes a metric from returns: it only reads, orders and words what the
stand already measured. Nothing here decides anything either -- the readings of the
ruleset (`qlab.rules.engine.read_informational`) are shown, never enforced.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from qlab.harness.metrics import COUNTED_YEAR_DAYS

YEAR_KEY = re.compile(r"^year_(\d{4})_(return|days|max_dd|sharpe)$")
SUMMARY_KEYS = ("years_counted", "years_positive", "years_positive_share",
                "ann_return_net_ex_best_year", "best_year_share_of_profit")
FULL_YEAR_DAYS = 360
"""A year with fewer days than this is marked with its day count in the line: a first
or last year of the judged part, or a year with gaps in the data."""
MINUS = "−"


@dataclass(frozen=True)
class YearRow:
    year: int
    ret: float
    days: int
    max_dd: float | None
    sharpe: float | None

    @property
    def counted(self) -> bool:
        return self.days >= COUNTED_YEAR_DAYS

    @property
    def partial(self) -> bool:
        return self.days < FULL_YEAR_DAYS


def is_year_key(key: str) -> bool:
    """A per-year key or a summary of them: what a part-specific copy of the metrics
    (`fit_`, `pre_`) leaves out and what a stage's result copies whole."""
    return bool(YEAR_KEY.match(key)) or key in SUMMARY_KEYS


def year_keys(metrics: Mapping[str, object]) -> list[str]:
    """Every year-view key present in `metrics`, in a stable order."""
    return sorted(k for k in metrics if is_year_key(k))


def year_rows(metrics: Mapping[str, object] | None) -> list[YearRow]:
    """The per-year rows stored in `metrics`, oldest first. Empty when none were stored
    (a run from before the view existed)."""
    parts: dict[int, dict[str, float]] = {}
    for key, value in (metrics or {}).items():
        found = YEAR_KEY.match(key)
        if found and isinstance(value, int | float) and value == value:
            parts.setdefault(int(found.group(1)), {})[found.group(2)] = float(value)
    return [YearRow(year=year, ret=p["return"], days=int(p["days"]),
                    max_dd=p.get("max_dd"), sharpe=p.get("sharpe"))
            for year, p in sorted(parts.items()) if "return" in p and "days" in p]


def best_year(rows: list[YearRow]) -> int | None:
    """The counted year with the highest return -- the one `ann_return_net_ex_best_year`
    removes."""
    counted = [r for r in rows if r.counted]
    return max(counted, key=lambda r: r.ret).year if len(counted) >= 2 else None


def _pct(value: float) -> str:
    text = f"{abs(value):.0%}" if abs(value) >= 0.01 else f"{abs(value):.1%}"
    return f"{'+' if value >= 0 else MINUS}{text}"


def years_line(metrics: Mapping[str, object] | None) -> str | None:
    """One compact line, e.g. «по годам: 2020 +41%, 2021 +88%, 2026 (282 дн.) +1%; без
    лучшего года (2021): +4% в год; прибыльных лет 5 из 7». None when the run stored no
    year view, or only a stretch too short to be called a year."""
    rows = year_rows(metrics)
    if not rows or not (len(rows) >= 2 or rows[0].counted):
        return None
    m = metrics or {}
    parts = [f"{r.year} ({r.days} дн.) {_pct(r.ret)}" if r.partial else f"{r.year} {_pct(r.ret)}"
             for r in rows]
    line = "по годам: " + ", ".join(parts)
    ex_best = m.get("ann_return_net_ex_best_year")
    if isinstance(ex_best, int | float) and ex_best == ex_best:
        line += f"; без лучшего года ({best_year(rows)}): {_pct(ex_best)} в год"
    counted, positive = m.get("years_counted"), m.get("years_positive")
    if isinstance(counted, int | float) and isinstance(positive, int | float):
        line += f"; прибыльных лет {positive:.0f} из {counted:.0f}"
    return line


def year_view(metrics: Mapping[str, object] | None, ruleset=None) -> dict | None:
    """The console's form of the view: the rows, the summaries and, with a ruleset, its
    informational readings. None when the run stored no year view."""
    from qlab.rules.engine import read_informational

    rows = year_rows(metrics)
    if not rows:
        return None
    m = metrics or {}
    summary = {k: m[k] for k in SUMMARY_KEYS
               if isinstance(m.get(k), int | float) and m[k] == m[k]}
    readings = [] if ruleset is None else [
        {"id": r.rule_id, "metric": r.metric, "value": r.value, "comparator": r.comparator,
         "threshold": r.threshold, "holds": r.holds}
        for r in read_informational(m, ruleset)]
    return {
        "years": [{"year": r.year, "return": r.ret, "days": r.days, "max_dd": r.max_dd,
                   "sharpe": r.sharpe, "counted": r.counted, "partial": r.partial}
                  for r in rows],
        "best_year": best_year(rows),
        "summary": summary,
        "readings": readings,
        "line": years_line(m),
    }


__all__ = ["FULL_YEAR_DAYS", "SUMMARY_KEYS", "YEAR_KEY", "YearRow", "best_year", "is_year_key",
           "year_keys", "year_rows", "year_view", "years_line"]
