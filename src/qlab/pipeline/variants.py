"""Timeframe variants of a spec (docs/TASKS.md T25).

Owner, 2026-10-01: "я хочу чтобы были варианты проверки стратегии на разных
таймфреймах"; "разные тайфреймы для входа и разные для выхода"; "быть
гибкими". This module turns one spec into a declared SET of variants:

- `timeframe_variants`: the same strategy and params on other bar intervals;
- `split_cadence_variant`: the same strategy on fast bars, with entries and
  exits acted on at different cadences (`qlab.strategies.cadence`).

The discipline that keeps this from becoming a search for the best-looking
timeframe is the caller's and is stated in every generated header: the set
is written before any of it is run, every variant is a trial of the same
idea (so deflation counts them all), and the result is reported for the
whole set -- "holds on N of M" -- never for its best member.

A variant is refused for an interval the strategy does not declare in
`valid_intervals`: converting days to bars is not enough to make a strategy
mean the same thing on other bars (trend's daily volatility target was the
case that showed it).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from qlab.data.sources.base import INTERVAL_TO_TIMEDELTA
from qlab.pipeline.evaluate import resolve_strategy
from qlab.pipeline.spec import StrategySpec

SPLIT_CADENCE_CODE_REF = "qlab.strategies.cadence:SplitCadence"
_SPLIT_REQUEST = (
    "владелец 2026-10-01: «возможно нужно использовать разные тайфреймы для входа и разные "
    "для выхода»"
)


def _check_interval(code_ref: str, interval: str) -> None:
    if interval not in INTERVAL_TO_TIMEDELTA:
        raise ValueError(f"unknown interval {interval!r}")
    valid = getattr(resolve_strategy(code_ref), "valid_intervals", None)
    if valid is None:
        raise ValueError(
            f"{code_ref} declares no valid_intervals: it was never checked on any bar "
            "size but its own, so it cannot be varied across timeframes"
        )
    if interval not in valid:
        raise ValueError(f"{code_ref} is not valid on {interval!r} (declares {tuple(valid)})")


def _header(base: Path, what: str, members: list[str]) -> str:
    return (
        f"# GENERATED {date.today().isoformat()} by qlab.pipeline.variants from {base}.\n"
        f"# {what}\n"
        f"# The whole set, declared before any member is run: {', '.join(members)}.\n"
        "# Every member is a trial of the same idea (deflation counts them all) and the\n"
        "# result is reported for the whole set -- never for its best member\n"
        "# (docs/TASKS.md T25). Comments explaining the params live in the base spec.\n\n"
    )


def _raw(base: Path) -> dict:
    return yaml.safe_load(base.read_text(encoding="utf-8"))


def _dump(path: Path, header: str, raw: dict) -> Path:
    StrategySpec.model_validate(raw)  # a variant that does not load is never written
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def timeframe_variants(base: Path, intervals: list[str], out_dir: Path) -> list[Path]:
    """One spec per interval, everything else unchanged."""
    raw = _raw(base)
    for interval in intervals:
        _check_interval(raw["code_ref"], interval)
    written = []
    members = [f"{base.stem}-tf-{i}" for i in intervals]
    for interval in intervals:
        variant = {**raw, "data": {**raw["data"], "interval": interval}}
        variant["title"] = f"{raw['title']} [{interval} bars]"
        header = _header(base, "Timeframe variant: same strategy and params, other bars.", members)
        written.append(_dump(out_dir / f"{base.stem}-tf-{interval}.yaml", header, variant))
    return written


def split_cadence_variant(
    base: Path, *, fast: str, entry_every: str, exit_every: str, out_dir: Path
) -> Path:
    """The base strategy on `fast` bars, growing the book only at `entry_every`
    period closes and shrinking it at `exit_every` period closes."""
    raw = _raw(base)
    _check_interval(raw["code_ref"], fast)
    name = f"{base.stem}-split-{entry_every}-{exit_every}"
    variant = {
        **raw,
        "title": f"{raw['title']} [entry every {entry_every}, exit every {exit_every}, "
        f"{fast} bars]",
        "code_ref": SPLIT_CADENCE_CODE_REF,
        "params": {
            "inner_code_ref": raw["code_ref"],
            "inner_params": raw.get("params", {}),
            "entry_every": entry_every,
            "exit_every": exit_every,
        },
        "data": {**raw["data"], "interval": fast},
        # The inner strategy's citations move under `inner_params`; the
        # wrapper's own choices cite the owner's request (docs/SOURCES.md).
        "sources": {
            **{
                (k if k == "min_leg_notional" or k == "costs" or k.startswith("costs.")
                 else f"inner_params.{k}"): v
                for k, v in (raw.get("sources") or {}).items()
            },
            "inner_code_ref": "src/" + raw["code_ref"].split(":")[0].replace(".", "/") + ".py",
            "entry_every": _SPLIT_REQUEST,
            "exit_every": _SPLIT_REQUEST,
        },
    }
    header = _header(
        base,
        f"Split cadence: decide entries every {entry_every}, exits every {exit_every}.",
        [name],
    )
    return _dump(out_dir / f"{name}.yaml", header, variant)


__all__ = ["SPLIT_CADENCE_CODE_REF", "split_cadence_variant", "timeframe_variants"]
