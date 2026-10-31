"""What it costs to fetch Hyperliquid's hourly candles, and whether a night can afford it.

Night of 2026-10-31: three ideas of the queue of closed ideas each needed the hourly candles of
every Hyperliquid contract (about an hour), and a restamp needed them once more for a window with
another end; the stages that took them ran until the stop time and the agent stages never started.
This module says, before a stage starts such an idea, whether its data is already cached (a
snapshot answers the request: quick) and, if not, about how long fetching it would take
(`night/hyperliquid_fetch.yaml`, a measurement). A stage that would not finish it within its own
time limit does not start it, and the report says «данные слишком тяжелы за ночь»; a hand run with
`--allow-heavy` fetches it. Reads the registry; fetches nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG = Path("night/hyperliquid_fetch.yaml")
HEAVY_RU = "данные слишком тяжелы за ночь"


@dataclass(frozen=True)
class FetchCost:
    all_contracts_minutes: float
    seconds_per_instrument: float


def load_config(path: Path = CONFIG) -> FetchCost:
    """`night/hyperliquid_fetch.yaml`; a missing key is an error (no silent default makes a heavy
    fetch look cheap)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return FetchCost(float(raw["all_contracts_minutes"]), float(raw["seconds_per_instrument"]))
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None


@dataclass(frozen=True)
class Estimate:
    """`minutes` to fetch the spec's data (0 when cached or quick); `cached` says a snapshot
    answers the request."""

    minutes: float
    cached: bool
    why: str


def estimate(spec, session, cost: FetchCost) -> Estimate:
    """About how long the data request of `spec` (a Hyperliquid spec) takes to fetch."""
    from qlab.pipeline.evaluate import resolve_cached_panel

    data = spec.data
    if data.source != "hyperliquid" or data.interval == "1d":
        return Estimate(0.0, True, "daily or not Hyperliquid: quick")
    if resolve_cached_panel(session, spec) is not None:
        return Estimate(0.0, True, "a snapshot answers the request")
    if not data.instruments:
        return Estimate(cost.all_contracts_minutes, False,
                        "часовые свечи всех контрактов площадки, около "
                        f"{cost.all_contracts_minutes:g} минут")
    minutes = len(data.instruments) * cost.seconds_per_instrument / 60
    return Estimate(minutes, False, f"часовые свечи {len(data.instruments)} инструментов, около "
                    f"{minutes:.0f} минут")


def heavy_reason(est: Estimate, limit_minutes: float) -> str | None:
    """Why the fetch is too heavy for a stage with `limit_minutes` for the whole night, or None."""
    if est.cached or est.minutes <= limit_minutes:
        return None
    return (f"{HEAVY_RU}: готового снимка нет, нужны {est.why}, а у этапа на всю ночь "
            f"{limit_minutes:g} минут; вручную: --allow-heavy")


__all__ = ["CONFIG", "Estimate", "FetchCost", "HEAVY_RU", "estimate", "heavy_reason",
           "load_config"]
