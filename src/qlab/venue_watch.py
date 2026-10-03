"""The venue watcher: what changed on the venues since yesterday
(docs/PLAN.md, M7: «Первым — наблюдатель за структурой рынка: запуски площадок,
новые инструменты, смена механик funding и комиссий. На ежедневно меняющемся
рынке это источник, который не иссякает»).

Owner, 2026-10-03: «делай все 3 одну за одной» -- the third, sources of new
candidates, venue watcher first.

It compares each venue's listing as recorded today with the previous one and
reports events -- facts, no judgement:

- Hyperliquid (main and HIP-3 `xyz`), from the recorder's daily snapshots
  (`data/recorded/<source>/meta/<date>.json`): instruments listed, delisted or
  relisted; maximum leverage, margin table or size precision changed; and,
  as context, the day's extremes -- highest and lowest funding, largest jumps
  in daily volume and open interest (ranked, never thresholded);
- Binance USDT perpetuals, from `exchangeInfo`, snapshotted here daily
  (`data/recorded/binance/meta/<date>.json`): contracts listed, delisted,
  status changed.

Events feed the morning report and the search stage (`qlab.night`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

RECORDED = Path("data/recorded")
EVENTS_DIR = Path("data/events")
TOP = 5  # how many extremes the report lists per measure (a display choice, not a threshold)


@dataclass(frozen=True)
class Event:
    source: str
    kind: str
    instrument: str
    detail: str


def _snapshots(source: str) -> list[Path]:
    return sorted((RECORDED / source / "meta").glob("*.json"))


def _hl_index(path: Path) -> dict[str, dict]:
    return {e["name"]: e for e in json.loads(path.read_text())["universe"]}


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def diff_hyperliquid(source: str, prev: dict[str, dict], cur: dict[str, dict]) -> list[Event]:
    events: list[Event] = []
    for name, e in cur.items():
        was = prev.get(name)
        dead, was_dead = bool(e.get("isDelisted")), bool(was and was.get("isDelisted"))
        if was is None:
            events.append(Event(source, "listed", name,
                                f"max leverage {e.get('maxLeverage')}"
                                + (" (already delisted)" if dead else "")))
            continue
        if dead and not was_dead:
            events.append(Event(source, "delisted", name, ""))
        elif was_dead and not dead:
            events.append(Event(source, "relisted", name, ""))
        for key, label in (("maxLeverage", "max leverage"), ("marginTableId", "margin table"),
                           ("szDecimals", "size precision")):
            if was.get(key) != e.get(key):
                events.append(Event(source, "changed", name,
                                    f"{label} {was.get(key)} -> {e.get(key)}"))
    for name in prev.keys() - cur.keys():
        events.append(Event(source, "removed", name, "no longer in the venue's listing"))

    live = {n: e for n, e in cur.items() if not e.get("isDelisted") and e.get("ctx")}
    funding = sorted(((_f(e["ctx"].get("funding")), n) for n, e in live.items()),
                     key=lambda x: x[0])
    for rate, name in [*funding[-TOP:][::-1], *funding[:TOP]]:
        if rate == rate:
            events.append(Event(source, "funding-extreme", name,
                                f"{rate * 24 * 365:+.0%} a year at today's hourly rate"))
    for key, label in (("dayNtlVlm", "daily volume"), ("openInterest", "open interest")):
        jumps = []
        for n, e in live.items():
            before = prev.get(n, {}).get("ctx") or {}
            a, b = _f(before.get(key)), _f(e["ctx"].get(key))
            if a > 0 and b == b:
                jumps.append((b / a, n, a, b))
        for ratio, name, a, b in sorted(jumps, reverse=True)[:TOP]:
            events.append(Event(source, f"{key}-jump", name,
                                f"{label} x{ratio:.1f} ({a:,.0f} -> {b:,.0f})"))
    return events


def snapshot_binance(day: date) -> Path:
    """Today's Binance USDT-perpetual listing (status, onboard date)."""
    import httpx

    from qlab.data.sources.binance import fetch_exchange_info

    with httpx.Client() as client:
        info = fetch_exchange_info(client)
    folder = RECORDED / "binance" / "meta"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day.isoformat()}.json"
    path.write_text(json.dumps({s: {"status": v["status"],
                                    "onboard_date": str(v["onboard_date"])}
                                for s, v in info.items()}))
    return path


def diff_binance(prev: dict[str, dict], cur: dict[str, dict]) -> list[Event]:
    events = []
    for sym, v in cur.items():
        was = prev.get(sym)
        if was is None:
            events.append(Event("binance", "listed", sym, f"status {v['status']}"))
        elif was["status"] != v["status"]:
            kind = "delisted" if v["status"] == "SETTLING" else "status"
            events.append(Event("binance", kind, sym, f"{was['status']} -> {v['status']}"))
    for sym in prev.keys() - cur.keys():
        events.append(Event("binance", "removed", sym, "no longer in exchangeInfo"))
    return events


def watch(day: date, *, binance: bool = True) -> list[Event]:
    """Every venue's events between its two latest snapshots up to `day`;
    written to `data/events/<day>.json`."""
    events: list[Event] = []
    for source in ("hyperliquid", "hyperliquid-xyz"):
        snaps = [p for p in _snapshots(source) if p.stem <= day.isoformat()]
        if len(snaps) >= 2:
            events += diff_hyperliquid(source, _hl_index(snaps[-2]), _hl_index(snaps[-1]))
    if binance:
        snapshot_binance(day)
        snaps = [p for p in _snapshots("binance") if p.stem <= day.isoformat()]
        if len(snaps) >= 2:
            events += diff_binance(json.loads(snaps[-2].read_text()),
                                   json.loads(snaps[-1].read_text()))
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVENTS_DIR / f"{day.isoformat()}.json").write_text(
        json.dumps([asdict(e) for e in events], indent=1))
    return events


__all__ = ["Event", "diff_binance", "diff_hyperliquid", "snapshot_binance", "watch"]
