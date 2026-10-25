"""Point-in-time asset classes of the instruments of a HIP-3 deployment (docs/ASSET_CLASSES.md).

Why. The betting-against-beta paper ranks "securities in an asset class (or within a country for
global equities)". The `xyz` deployment lists single stocks, funds, a preferred share, indices,
commodities and currencies side by side, and a rank across all of them is a different strategy.
Hyperliquid does carry the deployer's own description of every instrument it lists
(`perpAnnotation`: a category and a text naming the underlying security and its exchange); this
module turns it into a class by the versioned rule in `classification/<venue>.rule.yaml`.

What it does not do. It never looks at prices, and it never guesses: an instrument the venue has
not described is `unclassified` and belongs to no class.

Point in time. An instrument has a class from its listing date (the first daily candle; with no
candle, the first day it shows in a daily listing snapshot, which is a bound). A class changes
only when two daily snapshots of the annotation differ, and the new value is visible from the
first bar AFTER the day it was seen -- a later value is never applied backwards. A delisted
instrument keeps its class history; `ClassTable.class_at` returns nothing on or after the day it
was delisted, so a rank never counts it as alive.

Layout. The annotation snapshots are recorded data (`data/recorded/<source>/annotations/<day>.json`,
backed up with the rest of the record). The class table (`classification/<source>.yaml`) is
derived from them, from the rule and from the listing facts, is deterministic and is committed.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_RULE_PATH = Path("classification/hyperliquid-xyz.rule.yaml")
DEFAULT_TABLE_PATH = Path("classification/hyperliquid-xyz.yaml")
DEFAULT_RECORD_DIR = Path("data/recorded/hyperliquid-xyz")
DEFAULT_RAW_DIR = Path("data/raw/hyperliquid-xyz/1d")

UNCLASSIFIED = "unclassified"
TABLE_FORMAT_VERSION = 1
_EXCHANGE_TAG = "({exchange}: "


# --------------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassRule:
    """The mapping in `classification/<venue>.rule.yaml`, validated."""

    version: str
    venue: str
    placeholders: tuple[str, ...]
    category_to_asset_class: Mapping[str, str]
    stock_refinements: tuple[tuple[tuple[str, ...], str], ...]
    exchange_to_country: tuple[tuple[str, str], ...]
    countries_apply_to: frozenset[str]
    with_country: frozenset[str]
    sources: Mapping[str, object] = field(default_factory=dict)


def load_rule(path: Path = DEFAULT_RULE_PATH) -> ClassRule:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in (
        "version", "venue", "unclassified_when", "category_to_asset_class",
        "stock_refinements", "exchange_to_country", "countries_apply_to", "class_key",
    ):
        if key not in raw:
            raise ValueError(f"asset-class rule {path}: missing {key!r}")
    refinements = tuple(
        (tuple(str(p).lower() for p in row["match_any"]), str(row["asset_class"]))
        for row in raw["stock_refinements"]
    )
    # Longest exchange name first, so "NYSE Arca" is tried before "NYSE".
    exchanges = sorted(
        ((str(k), str(v)) for k, v in raw["exchange_to_country"].items()),
        key=lambda kv: -len(kv[0]),
    )
    return ClassRule(
        version=str(raw["version"]),
        venue=str(raw["venue"]),
        placeholders=tuple(
            str(x).lower() for x in raw["unclassified_when"].get("description_equals", [])
        ),
        category_to_asset_class={
            str(k).lower(): str(v) for k, v in raw["category_to_asset_class"].items()
        },
        stock_refinements=refinements,
        exchange_to_country=tuple(exchanges),
        countries_apply_to=frozenset(str(x) for x in raw["countries_apply_to"]),
        with_country=frozenset(str(x) for x in raw["class_key"].get("with_country", [])),
        sources=raw.get("sources", {}),
    )


# --------------------------------------------------------------------------------------------
# classifying one annotation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """The class of one instrument at one time. `class_key` is the group instruments are ranked
    inside; `asset_class` is the coarser kind ("stock", "fund", "index", ...)."""

    asset_class: str
    class_key: str
    country: str | None = None
    venue_category: str | None = None
    underlying: str | None = None
    note: str | None = None

    def matches(self, wanted: Collection[str]) -> bool:
        """True if `wanted` names this class either by kind ("stock") or by key ("stock:US")."""
        return self.asset_class in wanted or self.class_key in wanted

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _first_exchange(description: str, rule: ClassRule) -> str | None:
    """The country of the earliest "(<exchange>: " tag in `description`, or None."""
    best: tuple[int, str] | None = None
    for exchange, country in rule.exchange_to_country:
        pos = description.find(_EXCHANGE_TAG.format(exchange=exchange))
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, country)
    return None if best is None else best[1]


def _underlying(description: str) -> str:
    """The start of the venue's description, which names the security, its share class and its
    exchange: what the table keeps of the venue's text."""
    text = " ".join(description.split())
    return text if len(text) <= 240 else text[:237].rsplit(" ", 1)[0] + "..."


def classify(annotation: Mapping[str, object] | None, rule: ClassRule) -> Classification:
    """Apply the rule to one `perpAnnotation` answer (None when the venue has none)."""
    if annotation is None:
        return Classification(UNCLASSIFIED, UNCLASSIFIED, note="the venue serves no annotation")
    category = str(annotation.get("category") or "").strip()
    description = str(annotation.get("description") or "").strip()
    if description.lower() in rule.placeholders or not category:
        return Classification(
            UNCLASSIFIED, UNCLASSIFIED, venue_category=category or None,
            note="the annotation is a placeholder",
        )
    kind = rule.category_to_asset_class.get(category.lower())
    if kind is None:
        return Classification(
            UNCLASSIFIED, UNCLASSIFIED, venue_category=category,
            underlying=_underlying(description),
            note=f"category {category!r} is not in the rule",
        )
    if kind == "stock":
        lowered = f" {description.lower()}"
        for patterns, refined in rule.stock_refinements:
            if any(p in lowered for p in patterns):
                kind = refined
                break
    country = _first_exchange(description, rule) if kind in rule.countries_apply_to else None
    key = kind
    note = None
    if kind in rule.with_country:
        key = f"{kind}:{country or 'unknown'}"
        if country is None:
            note = "no known exchange named in the description"
    return Classification(
        asset_class=kind, class_key=key, country=country, venue_category=category,
        underlying=_underlying(description), note=note,
    )


# --------------------------------------------------------------------------------------------
# reading the venue and the record
# --------------------------------------------------------------------------------------------


def fetch_annotations(client, names: Collection[str]) -> dict[str, dict | None]:
    """`perpAnnotation` for every name (None where the venue has none)."""
    from qlab.data.sources import hyperliquid as hl

    return {
        name: hl._post(client, {"type": "perpAnnotation", "coin": name})  # type: ignore[misc]
        for name in sorted(names)
    }


def save_annotation_snapshot(
    record_dir: Path, day: date, annotations: Mapping[str, dict | None]
) -> Path:
    """Write `<record_dir>/annotations/<day>.json`; a day already recorded is overwritten (the
    venue's last word of that day)."""
    path = record_dir / "annotations" / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(annotations), indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def load_annotation_snapshots(record_dir: Path) -> dict[date, dict[str, dict | None]]:
    out: dict[date, dict[str, dict | None]] = {}
    for path in sorted((record_dir / "annotations").glob("*.json")):
        out[date.fromisoformat(path.stem)] = json.loads(path.read_text(encoding="utf-8"))
    return out


@dataclass(frozen=True)
class ListingFacts:
    """What the venue's own data says about when an instrument lived."""

    listed_from: date
    basis: str  # "first_candle" | "first_listing_snapshot"
    delisted_since: date | None = None
    delisted_basis: str | None = None  # "seen_delisted" | "on_or_before_first_snapshot"


def _listing_days(record_dir: Path) -> dict[date, dict[str, bool]]:
    """{snapshot day: {name: is_delisted}} from the recorder's daily listing files."""
    days: dict[date, dict[str, bool]] = {}
    for path in sorted((record_dir / "meta").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        days[date.fromisoformat(path.stem)] = {
            e["name"]: bool(e.get("isDelisted")) for e in payload["universe"]
        }
    return days


def _first_candle_days(record_dir: Path, raw_dir: Path | None) -> dict[str, date]:
    """The earliest daily candle we hold per instrument, from the recorded store and the fetch
    cache of the stand (the cache reaches back before the recorder began)."""
    first: dict[str, date] = {}

    def note(name: str, index: pd.Index) -> None:
        if len(index):
            day = pd.Timestamp(index.min()).date()
            first[name] = min(first.get(name, day), day)

    for path in (record_dir / "candles" / "1d").glob("*.parquet"):
        note(path.stem, pd.read_parquet(path, columns=[]).index)
    if raw_dir is not None and raw_dir.is_dir():
        for path in raw_dir.glob("*.parquet"):
            note(path.name.split("__", 1)[0], pd.read_parquet(path, columns=[]).index)
    return first


def listing_facts(
    record_dir: Path, raw_dir: Path | None = DEFAULT_RAW_DIR
) -> dict[str, ListingFacts]:
    """Listing and delisting dates per instrument, from HL's own data only."""
    days = _listing_days(record_dir)
    candles = _first_candle_days(record_dir, raw_dir)
    ordered = sorted(days)
    facts: dict[str, ListingFacts] = {}
    names = set(candles)
    for listing in days.values():
        names |= set(listing)
    for name in sorted(names):
        seen = [d for d in ordered if name in days[d]]
        if name in candles:
            listed, basis = candles[name], "first_candle"
        else:
            listed, basis = seen[0], "first_listing_snapshot"
        delisted_since: date | None = None
        delisted_basis: str | None = None
        if seen and days[seen[-1]][name]:
            run_start = seen[-1]
            for d in reversed(seen):
                if not days[d][name]:
                    break
                run_start = d
            delisted_since = run_start
            delisted_basis = (
                "on_or_before_first_snapshot" if run_start == ordered[0] else "seen_delisted"
            )
        facts[name] = ListingFacts(listed, basis, delisted_since, delisted_basis)
    return facts


# --------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------


def build_table(
    rule: ClassRule,
    snapshots: Mapping[date, Mapping[str, dict | None]],
    facts: Mapping[str, ListingFacts],
) -> dict:
    """The class table document. Deterministic: the same inputs always give the same bytes.

    For every instrument: its listing facts and a history of classes. The first snapshot that
    names the instrument gives the class from the listing date; a later snapshot that differs
    adds an entry visible from the first bar after the day it was seen.
    """
    if not snapshots:
        raise ValueError("no annotation snapshots: run `qlab data asset-classes` first")
    days = sorted(snapshots)
    names = sorted(set(facts) | {n for d in days for n in snapshots[d]})
    instruments: dict[str, dict] = {}
    for name in names:
        history: list[dict] = []
        previous: Classification | None = None
        for day in days:
            if name not in snapshots[day]:
                continue
            now = classify(snapshots[day][name], rule)
            if previous is not None and (
                now.asset_class, now.class_key, now.country
            ) == (previous.asset_class, previous.class_key, previous.country):
                continue
            fact = facts.get(name)
            if previous is None:
                valid_from = fact.listed_from if fact else day + timedelta(days=1)
            else:
                valid_from = day + timedelta(days=1)
            history.append({
                "valid_from": valid_from.isoformat(),
                "observed_on": day.isoformat(),
                **now.as_dict(),
            })
            previous = now
        entry: dict = {}
        fact = facts.get(name)
        if fact is not None:
            entry["listed_from"] = fact.listed_from.isoformat()
            entry["listed_from_basis"] = fact.basis
            if fact.delisted_since is not None:
                entry["delisted_since"] = fact.delisted_since.isoformat()
                entry["delisted_basis"] = fact.delisted_basis
        if history:
            entry["history"] = history
        else:
            never = Classification(
                UNCLASSIFIED, UNCLASSIFIED, note="never in an annotation snapshot"
            )
            entry["history"] = [{
                "valid_from": (fact.listed_from if fact else days[0]).isoformat(),
                "observed_on": days[0].isoformat(),
                **never.as_dict(),
            }]
        instruments[name] = entry
    return {
        "version": TABLE_FORMAT_VERSION,
        "venue": rule.venue,
        "rule_version": rule.version,
        "annotation_snapshots": [d.isoformat() for d in days],
        "sources": dict(rule.sources),
        "instruments": instruments,
    }


def write_table(document: Mapping, path: Path = DEFAULT_TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Derived by `qlab data asset-classes` from the recorded annotation snapshots, the\n"
        f"# rule classification/{document['venue']}.rule.yaml and the listing facts. Do not\n"
        "# edit by hand: a change belongs in the rule (a new version) or in the sources.\n"
    )
    path.write_text(
        header + yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _Entry:
    valid_from: date
    classification: Classification


@dataclass(frozen=True)
class _Instrument:
    listed_from: date
    delisted_since: date | None
    history: tuple[_Entry, ...]


class ClassTable:
    """Read side of the table: classes as known on a day, and as a table over panel bars."""

    def __init__(self, document: Mapping):
        if document.get("version") != TABLE_FORMAT_VERSION:
            raise ValueError(f"unknown class table version {document.get('version')!r}")
        self.rule_version = str(document["rule_version"])
        self.venue = str(document["venue"])
        self._instruments: dict[str, _Instrument] = {}
        for name, raw in document["instruments"].items():
            history = tuple(
                _Entry(
                    date.fromisoformat(str(h["valid_from"])),
                    Classification(**{
                        k: h.get(k)
                        for k in (
                            "asset_class", "class_key", "country", "venue_category",
                            "underlying", "note",
                        )
                    }),
                )
                for h in raw["history"]
            )
            listed = raw.get("listed_from")
            self._instruments[name] = _Instrument(
                listed_from=date.fromisoformat(str(listed)) if listed else history[0].valid_from,
                delisted_since=(
                    date.fromisoformat(str(raw["delisted_since"]))
                    if raw.get("delisted_since") else None
                ),
                history=history,
            )

    @property
    def names(self) -> list[str]:
        return sorted(self._instruments)

    def class_history(self, name: str) -> list[tuple[date, Classification]]:
        """Every class the instrument has had, delisted or not; empty for an unknown name."""
        inst = self._instruments.get(name)
        return [] if inst is None else [(e.valid_from, e.classification) for e in inst.history]

    def class_at(self, name: str, day: date) -> Classification | None:
        """The class known for `name` on `day`, or None before its listing, on or after its
        delisting, or for a name the table does not know."""
        inst = self._instruments.get(name)
        if inst is None or day < inst.listed_from:
            return None
        if inst.delisted_since is not None and day >= inst.delisted_since:
            return None
        current = None
        for entry in inst.history:
            if entry.valid_from <= day:
                current = entry.classification
        return current

    def labels_frame(
        self, index: pd.DatetimeIndex, columns: Collection[str], field_name: str = "class_key"
    ) -> pd.DataFrame:
        """`field_name` ("class_key", "asset_class" or "country") of every panel cell, as known
        at that bar; None where the instrument is unknown, unclassified, not yet listed or
        already delisted.
        Same shape rule as `coinmarketcap.as_of_frame`: columns the table does not know are
        all None."""
        if field_name not in ("class_key", "asset_class", "country"):
            raise ValueError(f"unknown label field {field_name!r}")
        days = [pd.Timestamp(t).date() for t in index]
        out = pd.DataFrame(None, index=index, columns=list(columns), dtype=object)
        for name in out.columns:
            inst = self._instruments.get(name)
            if inst is None:
                continue
            labels = []
            for day in days:
                cls = self.class_at(name, day)
                unlabelled = cls is None or cls.asset_class == UNCLASSIFIED
                labels.append(None if unlabelled else getattr(cls, field_name))
            out[name] = pd.Series(labels, index=index, dtype=object)
        return out

    def member_mask(
        self, index: pd.DatetimeIndex, columns: Collection[str], classes: Collection[str]
    ) -> pd.DataFrame:
        """True where the instrument's class at that bar is one of `classes` (kinds such as
        "stock" or keys such as "stock:US")."""
        wanted = set(classes)
        kinds = self.labels_frame(index, columns, "asset_class")
        keys = self.labels_frame(index, columns, "class_key")
        return kinds.isin(wanted) | keys.isin(wanted)


def load_table(path: Path = DEFAULT_TABLE_PATH) -> ClassTable:
    if not path.is_file():
        raise FileNotFoundError(
            f"no class table at {path}: run `qlab data asset-classes` first"
        )
    return ClassTable(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = [
    "DEFAULT_RAW_DIR",
    "DEFAULT_RECORD_DIR",
    "DEFAULT_RULE_PATH",
    "DEFAULT_TABLE_PATH",
    "UNCLASSIFIED",
    "ClassRule",
    "ClassTable",
    "Classification",
    "ListingFacts",
    "build_table",
    "classify",
    "fetch_annotations",
    "listing_facts",
    "load_annotation_snapshots",
    "load_rule",
    "load_table",
    "save_annotation_snapshot",
    "write_table",
]
