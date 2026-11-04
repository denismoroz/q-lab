"""From which day a hand-picked coin list may be judged (night/knowable_coins.yaml).

A coin list written on one day (Bv3's BTC and ETH, dated 2026-10-03) used to be judged only from
that day: a list picked by hand may know which coins did well later (survivorship), so the years
before it were warm-up (`qlab.twin.chosen_list`). That guard is right for a lucky pick and wrong
for a list anybody could have written on every past day. This module tells them apart by code,
from the venue's own data and point-in-time only.

A coin is KNOWABLE on a day when, looking only at the days before it, (1) it had been listed for at
least `window_days` days, and (2) its mean daily traded value (close * volume) over those days was
among the `top_k` largest of the venue's perpetuals. A list is knowable from the first day on which
every coin of it is. A coin that later falls out of the top does not matter: survivorship is
choosing the coins that rose into view, not the ones that faded out of it.

No agent is called, nothing is written: the module reads the raw daily candles and returns a
`CoinsKnowable` that the spec records.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from qlab.pipeline.spec import CoinsKnowable

CONFIG = Path("night/knowable_coins.yaml")
SPOT_SUFFIX = "-SPOT"
_STEM = re.compile(r"^(?P<coin>.+)__(?P<start>\d{4}-\d{2}-\d{2})__(?P<end>\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class KnowableConfig:
    top_k: int
    window_days: int
    raw_dir: Path
    interval: str
    source: str


def load_config(path: Path = CONFIG, root: Path | None = None) -> KnowableConfig:
    full = path if root is None or path.is_absolute() else root / path
    raw = yaml.safe_load(full.read_text(encoding="utf-8"))
    top_k, window = int(raw["top_k"]), int(raw["window_days"])
    if top_k < 1 or window < 1:
        raise ValueError(f"{full}: top_k and window_days must be positive")
    raw_dir = Path(raw["raw_dir"])
    return KnowableConfig(
        top_k=top_k, window_days=window,
        raw_dir=raw_dir if root is None or raw_dir.is_absolute() else root / raw_dir,
        interval=str(raw.get("interval", "1d")), source=CONFIG.as_posix())


def coin_of(column: str) -> str:
    """`BTC-SPOT` and `BTC` are the same coin."""
    return column.removesuffix(SPOT_SUFFIX).upper()


# --------------------------------------------------------------------------
# The data: daily traded value of every perpetual of a venue


def _newest_files(folder: Path) -> dict[str, Path]:
    """The parquet file with the longest span for each coin (a coin may hold several cached
    ranges): the latest end wins, then the earlier start."""
    best: dict[str, tuple[tuple[date, int], Path]] = {}
    for path in folder.glob("*.parquet"):
        match = _STEM.match(path.stem)
        if match is None or match["coin"].endswith(SPOT_SUFFIX) or ":" in match["coin"]:
            continue
        rank = (date.fromisoformat(match["end"]), -date.fromisoformat(match["start"]).toordinal())
        if match["coin"] not in best or rank > best[match["coin"]][0]:
            best[match["coin"]] = (rank, path)
    return {coin: path for coin, (_, path) in best.items()}


def read_daily_value(folder: Path) -> pd.DataFrame | None:
    """Daily traded value in USD (close * volume), one column per perpetual, index = UTC dates.
    None when the folder holds no candles. `volume` is in base units (qlab.data.sources.base), so
    the value is recomputed here from the same row's close, the way `capacity` does."""
    files = _newest_files(folder) if folder.is_dir() else {}
    columns: dict[str, pd.Series] = {}
    for coin, path in sorted(files.items()):
        try:
            frame = pd.read_parquet(path, columns=["price", "volume"])
        except Exception:  # noqa: BLE001 - a half-written file is a coin without history
            continue
        frame = frame.dropna(subset=["price", "volume"])
        if frame.empty:
            continue
        value = frame["price"] * frame["volume"]
        index = pd.DatetimeIndex(value.index)
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        value.index = index.tz_localize(None).normalize()
        columns[coin.upper()] = value[~value.index.duplicated(keep="last")]
    if not columns:
        return None
    return pd.DataFrame(columns).sort_index()


@functools.cache
def _cached_value(folder: str) -> pd.DataFrame | None:
    return read_daily_value(Path(folder))


def raw_volume(venue: str, config: KnowableConfig | None = None) -> pd.DataFrame | None:
    """The venue's daily traded value from the raw cache, or None when there is no cache (a
    checkout without data/): the list is then dated by hand, as before. Read once per process."""
    cfg = config or load_config()
    return _cached_value(str(cfg.raw_dir / venue / cfg.interval))


# --------------------------------------------------------------------------
# The check


def _plain(coins: Sequence[str], standing: dict[str, date | None], listed: dict[str, date | None],
           known_from: date | None, from_start: bool, first_day: date, config: KnowableConfig,
           venue: str) -> str:
    names = ", ".join(coins)
    where = {"binance": "на Binance", "hyperliquid": "на Hyperliquid"}.get(venue, f"на {venue}")
    word = _top_word(config.top_k)
    if known_from is None:
        never = [c for c in coins if standing[c] is None]
        return (f"Монета {', '.join(never)} за всю историю данных не была {word} по обороту "
                f"{where}: список считается выбранным вручную и оценивается со дня, когда его "
                "записали.")
    if from_start:
        return (f"Монеты {names} были {word} по обороту {where} (за предыдущие "
                f"{config.window_days} дней) на каждый день с {first_day} — история честная, "
                "оценка идёт с начала данных.")
    late = max(coins, key=lambda c: (standing[c] or date.min, c))
    came = listed.get(late)
    if came is not None and (known_from - came).days <= config.window_days + 1:
        return (f"Монета {late} появилась {where} только {came}; с {known_from} все монеты "
                f"({names}) были {word} по обороту (за предыдущие {config.window_days} дней) — "
                "история честная с этого дня, раньше — разгон.")
    return (f"Монета {late} вошла {word} по обороту {where} только {known_from} (до этого её "
            "нельзя было выбрать, не зная будущего): оценка идёт с этого дня, раньше — разгон.")


def _top_word(top_k: int) -> str:
    return "в первой десятке" if top_k == 10 else f"в первых {top_k}"


def check(volume_usd: pd.DataFrame, coins: Collection[str], *, venue: str, start: date,
          end: date, config: KnowableConfig) -> CoinsKnowable:
    """Whether the list `coins` was knowable on the days from `start` to `end` of `venue`.

    `volume_usd` is the venue's daily traded value (`read_daily_value`). For each day the rank is
    taken among the coins that have a full `window_days` window of earlier days; the window ends
    on the previous day, so a day never sees its own volume. A day with no candle inside a coin's
    window leaves the coin unranked that day (unknown is not zero)."""
    names = list(dict.fromkeys(coin_of(c) for c in coins))
    value = volume_usd.copy()
    value.columns = [coin_of(c) for c in value.columns]
    value = value.loc[:, ~value.columns.duplicated()]
    full = pd.date_range(value.index.min(), max(value.index.max(), pd.Timestamp(end)), freq="D")
    value = value.reindex(full)
    trailing = value.rolling(config.window_days, min_periods=config.window_days).mean().shift(1)
    rank = trailing.rank(axis=1, ascending=False, method="min")
    inside = rank <= config.top_k
    stamp = trailing.index
    days = stamp[(stamp >= pd.Timestamp(start)) & (stamp <= pd.Timestamp(end))]
    ranked = days[trailing.loc[days].notna().any(axis=1).to_numpy()]
    first_day = ranked[0].date() if len(ranked) else start
    standing: dict[str, date | None] = {}
    listed: dict[str, date | None] = {}
    for coin in names:
        listed[coin] = None if coin not in value.columns else (
            None if value[coin].first_valid_index() is None
            else value[coin].first_valid_index().date())
        if coin not in inside.columns:
            standing[coin] = None
            continue
        hits = inside.loc[days, coin]
        standing[coin] = hits[hits].index[0].date() if hits.any() else None
    known_from = None if any(d is None for d in standing.values()) or not standing \
        else max(d for d in standing.values() if d is not None)
    from_start = known_from is not None and known_from <= first_day
    return CoinsKnowable(
        venue=venue, top_k=config.top_k, window_days=config.window_days,
        checked_from=first_day, known_from=known_from, from_start=from_start, coins=standing,
        listed=listed,
        plain=_plain(names, standing, listed, known_from, from_start, first_day, config, venue),
        source=(f"{config.source}: top {config.top_k} of the venue's perpetuals by the mean daily "
                f"traded value of the previous {config.window_days} days, "
                f"{venue} candles in {config.raw_dir.as_posix()}"))


def list_day(known: CoinsKnowable | None, hand_picked: date | None) -> date | None:
    """The first day the list may be judged from: None = from the start of the data.

    A list knowable on every past day is judged from the start; knowable only from a day, from the
    earlier of that day and the hand-picked date (from the hand-picked date the list was surely
    known); not knowable, from the hand-picked date as before."""
    if known is None or known.known_from is None:
        return hand_picked
    if known.from_start:
        return None
    return known.known_from if hand_picked is None else min(known.known_from, hand_picked)


STORE = Path("data/one_check/coins_knowable")
"""The last check of each idea, kept for its page: a one-check idea's spec file does not carry the
check (the night builds it), so the page reads what the night found (`remember`, `recall`)."""


def remember(idea_id: str, known: CoinsKnowable, store: Path | None = None) -> None:
    path = (store or STORE) / f"{idea_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(known.model_dump_json(), encoding="utf-8")


def recall(idea_id: str, store: Path | None = None) -> CoinsKnowable | None:
    try:
        return CoinsKnowable.model_validate_json(
            ((store or STORE) / f"{idea_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = ["CONFIG", "KnowableConfig", "STORE", "check", "coin_of", "list_day", "load_config",
           "raw_volume", "read_daily_value", "recall", "remember"]
