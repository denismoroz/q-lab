"""`night/anomalies.yaml`: the anomaly detector's lines and the investigator's limits.

Every number the detector compares with lives in the file, each with its rationale, and
nothing here has a default: a missing key is an error (the same rule as `night/stage2.yaml`).
The file is versioned (`version`); a scan is stored with the version it was made under, so a
changed line is a new version and the old scans stay what they were.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qlab.pipeline.sources import QLAB_ROOT

CONFIG = Path("night/anomalies.yaml")

FLAGS = ("implausible_day", "concentrated_instruments", "concentrated_coin_days",
         "tail_move_coin_days", "untradeable_coin_days", "unlisted_on_execution_venue",
         "frozen_price", "venue_divergence", "stand_vs_paper", "too_good_for_family")
"""The detector's flags, in the order they are shown."""

REQUIRED: dict[str, tuple[str, ...]] = {
    "implausible_day": ("single_move", "market_move", "sigmas"),
    "concentrated_instruments": ("min_instruments", "top1_share", "top3_share"),
    "concentrated_coin_days": ("top_fraction", "min_top", "min_total", "share"),
    "tail_move_coin_days": ("min_move", "share"),
    "untradeable_coin_days": ("min_volume_usd", "delist_days", "share"),
    "unlisted_on_execution_venue": ("share",),
    "frozen_price": ("min_days", "share"),
    "venue_divergence": ("min_common_days", "gap_per_year"),
    "stand_vs_paper": ("gap",),
    "too_good_for_family": ("min_members", "fence_iqr"),
}
"""The numbers each flag reads from its block of the file."""

INVESTIGATOR_KEYS = ("model", "max_turns", "max_tokens_per_investigation", "max_per_night",
                     "max_attempts", "max_programmer_requests", "calls_dir")


@dataclass(frozen=True)
class FlagLine:
    """One flag's declared line: the numbers, why they are what they are, and whether they are
    still provisional."""

    name: str
    params: Mapping[str, Any]
    rationale: str
    provisional: bool

    def need(self, key: str) -> float:
        return float(self.params[key])


@dataclass(frozen=True)
class Scan:
    routes: tuple[str, ...]
    latest_only: bool
    max_scans_per_night: int


@dataclass(frozen=True)
class Investigator:
    model: str
    max_turns: int
    max_tokens_per_investigation: int
    max_per_night: int
    max_attempts: int
    max_programmer_requests: int
    calls_dir: Path
    """Where each investigation's log of toolkit calls is kept (data/anomalies/calls)."""
    passed_routes: tuple[str, ...] = ("paper",)
    investigate_routes: tuple[str, ...] | None = None
    """Routes whose flagged runs the agent investigates; None: every scanned route. Since
    2026-10-29 the code scans more routes than the agent reads (night/anomalies.yaml): a scan is
    seconds of code, an investigation is tokens."""


@dataclass(frozen=True)
class Config:
    version: str
    enabled: bool
    scan: Scan
    flags: Mapping[str, FlagLine]
    investigator: Investigator

    def flag(self, name: str) -> FlagLine:
        return self.flags[name]


def load_config(path: Path = CONFIG) -> Config | None:
    """The parsed file, None when it is missing (the stage is then off: a working directory
    without the project's night folder has nothing to configure). A present but incomplete file
    raises `ValueError` naming the missing key."""
    target = path if path.is_absolute() or path.is_file() else QLAB_ROOT / path
    if not target.is_file():
        return None
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return parse_config(raw, source=str(path))


def parse_config(raw: Mapping[str, Any], *, source: str = "night/anomalies.yaml") -> Config:
    def need(block: Mapping[str, Any], key: str, where: str) -> Any:
        if key not in block:
            raise ValueError(f"{source}: {where} needs `{key}`")
        return block[key]

    flags: dict[str, FlagLine] = {}
    block = need(raw, "flags", "the file")
    for name in FLAGS:
        entry = need(block, name, "flags")
        for key in REQUIRED[name]:
            need(entry, key, f"flag {name}")
        flags[name] = FlagLine(
            name=name, params={k: entry[k] for k in REQUIRED[name]},
            rationale=" ".join(str(need(entry, "rationale", f"flag {name}")).split()),
            provisional=bool(entry.get("provisional", False)))
    scan = need(raw, "scan", "the file")
    inv = need(raw, "investigator", "the file")
    for key in INVESTIGATOR_KEYS:
        need(inv, key, "investigator")
    return Config(
        version=str(need(raw, "version", "the file")),
        enabled=bool(need(raw, "enabled", "the file")),
        scan=Scan(routes=tuple(str(r) for r in need(scan, "routes", "scan")),
                  latest_only=bool(need(scan, "latest_only", "scan")),
                  max_scans_per_night=int(need(scan, "max_scans_per_night", "scan"))),
        flags=flags,
        investigator=Investigator(
            model=str(inv["model"]), max_turns=int(inv["max_turns"]),
            max_tokens_per_investigation=int(inv["max_tokens_per_investigation"]),
            max_per_night=int(inv["max_per_night"]), max_attempts=int(inv["max_attempts"]),
            max_programmer_requests=int(inv["max_programmer_requests"]),
            calls_dir=Path(str(inv["calls_dir"])),
            passed_routes=tuple(str(r) for r in inv.get("passed_routes", ("paper",))),
            investigate_routes=(None if inv.get("investigate_routes") is None else
                                tuple(str(r) for r in inv["investigate_routes"]))))


__all__ = ["CONFIG", "FLAGS", "REQUIRED", "Config", "FlagLine", "Investigator", "Scan",
           "load_config", "parse_config"]
