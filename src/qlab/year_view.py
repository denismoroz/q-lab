"""The year-by-year view of a run, read back from its stored metrics (docs/YEARS.md).

`qlab.harness.metrics.year_breakdown` writes flat keys (`year_2021_return`, ...) into
a trial's metrics. This module turns them back into rows, one plain-Russian line for
the reports and the agents' briefs, and a structured form for the console's API.

Every year key has a twin on the strategy's own money (`cap_year_2025_return`,
docs/CAPITAL_BASE.md): the same year, read as a share of the capital the strategy started with
and summed rather than compounded when the strategy does not reinvest its profit. The view
shows both; `capital_view` gives the second under the stand's key names, so the same line, the
same rows and the same four signs are asked of it.

Nothing here computes a metric from returns: it only reads, orders and words what the
stand already measured. Nothing here decides anything either -- the readings of the
ruleset (`qlab.rules.engine.read_informational`) and the four signs built on them
(`qlab.year_signs`, rules 2026-10-28.2) are shown, never enforced.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from qlab.harness.metrics import CAP_REINVESTED, COUNTED_YEAR_DAYS, as_stand_keys, cap_key

YEAR_KEY = re.compile(r"^(?:cap_)?year_(\d{4})_(return|days|max_dd|sharpe)$")
SUMMARY_KEYS = ("years_counted", "years_positive", "years_positive_share",
                "ann_return_net_ex_best_year", "best_year_share_of_profit")
CAP_SUMMARY_KEYS = tuple(cap_key(k) for k in SUMMARY_KEYS)
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
    return bool(YEAR_KEY.match(key)) or key in SUMMARY_KEYS or key in CAP_SUMMARY_KEYS


def year_keys(metrics: Mapping[str, object]) -> list[str]:
    """Every year-view key present in `metrics`, in a stable order."""
    return sorted(k for k in metrics if is_year_key(k))


def capital_view(metrics: Mapping[str, object] | None) -> dict[str, object]:
    """The year view on the strategy's own money, under the stand's key names (empty for a
    run from before the capital summaries): feed it to `year_rows`, `years_line`,
    `year_view` or `qlab.year_signs.read_signs` to read the capital twin of what they show."""
    return as_stand_keys(metrics)


def reinvests(metrics: Mapping[str, object] | None) -> bool | None:
    """Whether the run's capital summaries compound (True) or sum (False); None when the run
    has none."""
    flag = (metrics or {}).get(CAP_REINVESTED)
    return None if not isinstance(flag, int | float) else bool(flag)


def year_rows(metrics: Mapping[str, object] | None) -> list[YearRow]:
    """The per-year rows stored in `metrics`, oldest first. Empty when none were stored
    (a run from before the view existed). Reads the stand's keys; ask `capital_view(metrics)`
    for the capital twin."""
    parts: dict[int, dict[str, float]] = {}
    for key, value in (metrics or {}).items():
        found = YEAR_KEY.match(key) if not key.startswith("cap_") else None
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


def years_line(metrics: Mapping[str, object] | None, *, signs: bool = False,
               capital: bool = True) -> str | None:
    """One compact line, e.g. «по годам: 2020 +41%, 2021 +88%, 2026 (282 дн.) +1%; без
    лучшего года (2021): +4% в год; прибыльных лет 5 из 7». None when the run stored no
    year view, or only a stretch too short to be called a year.

    With `signs`, the line ends with the four signs of the newest ruleset (А Б В Г, each
    «держится / не держится / не определено» with the reason): the form the reports use.
    The agents' briefs stay without them.

    With `capital` (the default) and a run that has the year view on the strategy's own money
    (docs/CAPITAL_BASE.md), the same line on that money follows when it differs from the
    stand's: «На деньгах стратегии (…): 2025 +15%, …»."""
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
    if signs:
        from qlab.year_signs import latest_ruleset, signs_sentence

        ruleset = latest_ruleset()
        if ruleset is not None and (tail := signs_sentence(m, ruleset)):
            line += "; " + tail
    if capital and (cap := capital_view(m)):
        cap_line = years_line(cap, signs=signs, capital=False)
        if cap_line and cap_line != line:
            how = ("прибыль не вкладывается обратно, годы складываются" if reinvests(m) is False
                   else "прибыль вкладывается, годы склеиваются")
            line += f". На деньгах стратегии ({how}): " + cap_line.removeprefix("по годам: ")
    return line


def _view_of(m: Mapping[str, object], ruleset) -> dict | None:
    """The view of one set of year keys (the stand's, or the capital twin under the stand's
    names)."""
    from qlab.rules.engine import read_informational
    from qlab.year_signs import read_signs

    rows = year_rows(m)
    if not rows:
        return None
    summary = {k: m[k] for k in SUMMARY_KEYS
               if isinstance(m.get(k), int | float) and m[k] == m[k]}
    readings = [] if ruleset is None else [
        {"id": r.rule_id, "metric": r.metric, "value": r.value, "comparator": r.comparator,
         "threshold": r.threshold, "holds": r.holds}
        for r in read_informational(m, ruleset)]
    signs = [] if ruleset is None else [r.to_json() for r in read_signs(m, ruleset)]
    return {
        "years": [{"year": r.year, "return": r.ret, "days": r.days, "max_dd": r.max_dd,
                   "sharpe": r.sharpe, "counted": r.counted, "partial": r.partial}
                  for r in rows],
        "best_year": best_year(rows),
        "summary": summary,
        "readings": readings,
        "signs": signs,
        "line": years_line(m),
    }


def year_view(metrics: Mapping[str, object] | None, ruleset=None) -> dict | None:
    """The console's form of the view: the rows, the summaries and, with a ruleset, its
    informational readings. None when the run stored no year view.

    `capital` is the same view on the strategy's own money (docs/CAPITAL_BASE.md), with
    `reinvested` saying whether its years compound or sum; None for a run from before the
    capital summaries."""
    m = metrics or {}
    view = _view_of(m, ruleset)
    if view is None:
        return None
    capital = _view_of(capital_view(m), ruleset)
    if capital is not None:
        capital["reinvested"] = reinvests(m)
    view["capital"] = capital
    return view


__all__ = ["CAP_SUMMARY_KEYS", "FULL_YEAR_DAYS", "SUMMARY_KEYS", "YEAR_KEY", "YearRow",
           "best_year", "capital_view", "is_year_key", "reinvests", "year_keys", "year_rows",
           "year_view", "years_line"]
