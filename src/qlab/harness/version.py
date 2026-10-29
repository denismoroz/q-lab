"""The stand's version: which way of MEASURING a run a trial was made under.

A result is a number together with the way it was measured. When the way changes (a new metric
base, new panels, new costs, a new liquidity floor), the numbers of runs made earlier keep what
they were and are no longer comparable with new ones. `STAND_VERSION` names the measurement; every
new trial records it (`trial.stand_version`, set by `qlab.registry.repo.add_trial`), and the night's
restamp stage (`qlab.restamp`, docs/RESTAMP.md) re-runs the ideas whose latest run was made under an
older one. A trial that predates the column has no version: it is older than the first one.

**Bump it** (a new string, `YYYY-MM-DD.N`, always larger than the last) whenever how a run is
measured changes:

* a metric is added, removed or computed differently (the stand's, the capital-based `cap_*`, the
  noise bar, the year summaries and signs);
* the base a percent is a percent of changes;
* the panel a strategy sees changes (what is masked, which instruments, which fill of gaps), or the
  data a spec asks for (the liquidity floor of stage 2, the data window);
* costs, fees, slippage or the funding accrual change;
* a wrapper or a strategy's book changes in a way that moves its results.

**Do not bump it** for a change that moves no number: wording, display, the report, a new column of
the console, a new file format of a stored series, a faster way to the same numbers.

The decisive rules have their own version (`rules/*.yaml`, stored on every verdict); a new rules
version also makes a run stale, without a bump here.

**Deliberately not bumped** (2026-10-29): the noise-book count (a book that gave no number is
replaced, `noise_n_*` metrics; `qlab.restamp` selects the runs that lacked it by its own reason)
and the protection readings of the Bv2 family (`prot_*` metrics, rules 2026-10-29.1; older runs are
measured by `qlab protection --compute`). A bump would measure ~35 ideas again, including a
two-hour candle refetch, for numbers that moved on a handful of runs.

History:

* `2026-10-28.1` -- the first version. Under it: year signs recorded beside the rules (rules
  2026-10-28.2), the percent of the strategy's own money (`cap_*` metrics), stage 2's $1M-a-day
  liquidity floor for strategies that pick from the whole listing, the retune cache key with the
  tradeable mask.
"""

from __future__ import annotations

import re

STAND_VERSION = "2026-10-28.1"

_FORMAT = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d+)$")


def version_key(version: str | None) -> tuple[str, int] | None:
    """`(date, serial)` of a version string such as `2026-10-28.1`, or None when it has not that
    form (or is None). Versions compare as these pairs."""
    match = _FORMAT.match(version or "")
    return None if match is None else (match.group(1), int(match.group(2)))


def is_older(version: str | None, current: str) -> bool:
    """Whether `version` is older than `current`. A missing version (a trial from before the
    version was recorded) is older than every one; a version that does not parse is treated the
    same way, so a typo cannot hide a stale run."""
    now = version_key(current)
    if now is None:
        raise ValueError(f"{current!r} is not a version of the form YYYY-MM-DD.N")
    seen = version_key(version)
    return seen is None or seen < now


__all__ = ["STAND_VERSION", "is_older", "version_key"]
