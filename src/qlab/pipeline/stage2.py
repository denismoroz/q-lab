"""The second stage of the idea generator's evaluation (docs/TWO_STAGE.md, T45).

Stage 1 judges a generated spec as written, on Binance's long history (perpetuals
since 2019, delisted ones included: an honest universe). An idea that survives it
gets its forward check where q-lab's paper and live execution is: the SAME
mechanism -- the same blocks and the same parameter values, nothing refitted --
is built on Hyperliquid and judged there, with Hyperliquid's instruments, fees
and funding, on data the stage 1 fit never saw (everything after the
`params_fixed_at` stage 1 inherited). The idea needs BOTH stages to survive.

Everything here is code. No agent is called and no token is spent, so the token
budget guard has nothing to guard (a test pins that).

* `translate` -- the stage 1 spec -> the stage 2 spec (pure);
* `book_window` -- from which day the market holds enough coins for the portfolio
  rule, and whether it ever does (pure, on a panel);
* `combine` -- the idea's route from the two stages' routes (pure);
* confirmation by stage 1's history (`qlab.pipeline.confirm`, rules 2026-10-27.1): where
  the comparison with random books cannot decide on Hyperliquid's short window, stage 2
  is a consistency check with stage 1's long history, not a second proof;
* `run_stage2` -- the whole step: child idea, translation, window, run, verdict,
  the parent's status, all of it written to the registry;
* `report_lines` -- the stages side by side, in plain Russian.

The other way round (owner, 2026-11-05, on an idea born on Hyperliquid: «почему вот эта стратегия
не тестировалась на бинансе?»; «да, делай и прогоняй все»): an idea whose own spec is on
Hyperliquid's data gets a Binance stage under `<idea>-bn`, by the same functions with the Binance
config (`night/stage2_binance.yaml`, `config_for`): the same mechanism and numbers on Binance's
perpetuals since 2019, Binance's costs and minimum leg, coin names mapped back (kPEPE -> 1000PEPE,
a bridge token UBTC -> BTC-SPOT), coins Binance lacks dropped and counted. Binance has no order
adapter of ours, so a pass there is `needs-infrastructure`, as for every Binance pass.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from qlab import knowable_coins, tree_writes
from qlab.capital_text import CAP_NUMBERS, numbers_ru
from qlab.costs import two_legged, venue_costs
from qlab.pipeline import confirm
from qlab.pipeline.spec import SpecCosts, SpecData, StrategySpec
from qlab.registry.stages import BINANCE, HYPERLIQUID, NOTES_PREFIX, child_id
from qlab.venues.config import load_venue
from qlab.year_view import year_keys, years_line

CONFIG = Path("night/stage2.yaml")
BINANCE_CONFIG = Path("night/stage2_binance.yaml")
"""The second stage of an idea born on Hyperliquid: the same step, on Binance."""
VENUE_TITLE = {HYPERLIQUID: "Hyperliquid", BINANCE: "Binance"}
SPEC_DIR = Path("specs/generated/stage2")
FAILURES = Path("data/stage2/failures.json")
"""The second stages that crashed and have not been decided since: parent id -> attempts so far,
the error, whether the idea was given up on (`record_failure`)."""
MAX_FAILED_ATTEMPTS = 2
"""Consecutive crashes of an idea's second stage after which the Hyperliquid side is recorded as
not run (a failure) and the idea takes Binance's verdict; one crash is not yet an answer."""
PERMANENT_ERRORS = frozenset({
    "ValueError", "KeyError", "LookupError", "TypeError", "AttributeError", "AssertionError",
    "ImportError", "ModuleNotFoundError", "NotImplementedError", "ZeroDivisionError"})
"""Exceptions that a second attempt on the same data and code would raise again: the reason is
permanent, so the first crash already settles the Hyperliquid side as not run. A locked database,
a dropped connection or a timeout is not in the list: it can pass."""
COMPOSED_CODE_REF = "qlab.strategies.blocks:Composed"
RESULT_METRICS = ("ann_return_net", "total_return_net", "sharpe_net", "max_dd",
                  "noise_return_percentile", "noise_power", "min_capital_usd", "forward_days",
                  "forward_days_needed", "selection_days", "ann_return_net_ex_best_1pct",
                  *CAP_NUMBERS)
UNDECIDED = ("not-evaluable", "needs-more-data", "error")
"""Stage 2 routes that decided nothing: the idea waits."""


# --------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class SpotLeg:
    """A coin's spot leg on the stage 2 venue: the venue's token and where that is cited."""
    token: str
    source: str


@dataclass(frozen=True)
class Stage2Config:
    enabled: bool
    venue: str
    data_start: date
    taker_fee_bps: float
    slippage_bps: float
    cost_sources: Mapping[str, str]
    stage1_survives: tuple[str, ...]
    spot_legs: Mapping[str, SpotLeg] = field(default_factory=dict)
    """Coin -> the venue's spot token the lab has verified, with its source."""
    no_spot: Mapping[str, str] = field(default_factory=dict)
    """Coin -> why the venue has no usable spot token for it (a cited reason)."""
    min_daily_volume_usd: float | None = None
    """The execution venue's liquidity floor: a strategy that picks its coins from the venue's
    whole listing trades a coin on a day in stage 2 only if the live engine would (owner,
    2026-10-28, «да, делай», on docs/research/xsmom-anomaly-2026-10-28.md: 88% of an XSMOM
    second stage's money came from coins under the live engine's $1M a day).
    Stage 2 takes the stricter of this and stage 1's own floor; a named instrument list keeps
    stage 1's. Cited in `cost_sources`."""
    two_leg_taker_fee_bps: float | None = None
    """The venue's fee for a book whose two legs (spot against a perpetual) move together
    (`qlab.costs.two_legged`): Binance's row has one, Hyperliquid's does not."""


def load_config(path: Path = CONFIG) -> Stage2Config:
    """`night/stage2.yaml`. A missing key is an error: nothing here has a default
    value, for the same reason a spec's costs have none."""
    return _load_config(path)


def _load_config(path: Path) -> Stage2Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        costs = raw["costs"]
        sources = {str(k): " ".join(str(v).split())
                   for k, v in (costs["sources"] or {}).items()}
        two_leg = None
        if "taker_fee_bps" in costs:  # numbers of its own: a config written before the table
            taker, slippage = float(costs["taker_fee_bps"]), float(costs["slippage_bps"])
        else:  # the venue's row of the cost table (qlab.costs, night/costs.yaml): one source
            row = venue_costs(str(raw["venue"]))
            if row is None:
                raise ValueError(f"{path}: night/costs.yaml has no row for {raw['venue']!r}")
            taker, slippage = row.taker_fee_bps, row.slippage_bps
            sources.update({"costs.taker_fee_bps": row.sources["taker_fee_bps"],
                            "costs.slippage_bps": row.sources["slippage_bps"]})
            two_leg = row.two_leg_taker_fee_bps
            if two_leg is not None:
                sources["costs.two_leg_taker_fee_bps"] = row.sources["two_leg_taker_fee_bps"]
        config = Stage2Config(
            enabled=bool(raw["enabled"]), venue=str(raw["venue"]),
            data_start=date.fromisoformat(str(raw["data_start"])),
            taker_fee_bps=taker, slippage_bps=slippage, cost_sources=sources,
            stage1_survives=tuple(str(r) for r in raw["stage1_survives"]),
            spot_legs={str(coin): SpotLeg(str(leg["token"]), " ".join(str(leg["source"]).split()))
                       for coin, leg in (raw["spot_legs"] or {}).items()},
            no_spot={str(coin): " ".join(str(leg["source"]).split())
                     for coin, leg in (raw["no_spot"] or {}).items()},
            min_daily_volume_usd=(None if raw.get("min_daily_volume_usd") is None
                                  else float(raw["min_daily_volume_usd"])),
            two_leg_taker_fee_bps=two_leg)
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    cited = {"costs.taker_fee_bps", "costs.slippage_bps"}
    if config.min_daily_volume_usd is not None:
        cited.add("min_daily_volume_usd")
    absent = cited - set(config.cost_sources)
    if absent:
        raise ValueError(f"{path}: no source for {sorted(absent)}")
    return config


def config_for(venue: str) -> Stage2Config:
    """The second stage's config for a second venue: Hyperliquid's (`night/stage2.yaml`) or
    Binance's (`night/stage2_binance.yaml`, for an idea born on Hyperliquid)."""
    return _load_config(BINANCE_CONFIG) if venue == BINANCE else load_config()


def second_venue_of(spec: StrategySpec) -> str:
    """The venue an idea's second stage is judged on: Binance for a spec on Hyperliquid's own
    data, Hyperliquid for anything else (a Binance spec; an idea that declares lists of values
    runs its check on Binance's history whatever venue its file names, docs/ONE_CHECK.md)."""
    from qlab import one_check

    # A mixed book (Hyperliquid perps against another venue's spot, `data.spot_source`) already
    # holds Binance's side; it has no separate «Binance version», and the stand refuses to build
    # one (2026-11-06: seven FRAB-on-Binance-spot ideas failed in under a second each).
    if (spec.data.source == HYPERLIQUID and not one_check.declares_lists(spec)
            and spec.data.spot_source is None):
        return BINANCE
    return HYPERLIQUID


def failure_key(parent_id: str, venue: str = HYPERLIQUID) -> str:
    """The key of an idea's second stage in the ledger of crashes (`FAILURES`): the parent's id for
    the Hyperliquid stage (as it always was), the child's id for the Binance one, so the two
    stages of an idea born on Hyperliquid (its stricter look there and its Binance stage) never
    count each other's crashes."""
    return parent_id if venue == HYPERLIQUID else child_id(parent_id, venue)


def stricter_floor(stage1: float | None, venue: float | None) -> float | None:
    """The higher of two liquidity floors; `None` means no floor."""
    floors = [f for f in (stage1, venue) if f is not None]
    return max(floors) if floors else None


def independent_venues() -> bool:
    """Whether the newest rules judge the two venues independently (`VenueSplit.independent`,
    from 2026-10-30.2). A rules file that cannot be read means the older, stricter reading."""
    from qlab.rules.loader import load_latest

    try:
        split = load_latest().venue_split
    except Exception:  # noqa: BLE001 - the rules are read again by the run itself
        return False
    return split is not None and split.independent


def survives(route: str | None, config: Stage2Config, *, independent: bool | None = None) -> bool:
    """Whether a stage 1 route goes on to stage 2.

    Before rules 2026-10-30.2 only a route that passed or still waits did, and a reject was final.
    From them (`independent`, default: what the newest rules say) the second venue is always
    judged, whatever Binance said; a run with no route has nothing to anchor a second stage on."""
    if route is None:
        return False
    if independent is None:
        independent = independent_venues()
    return True if independent else route in config.stage1_survives


# --------------------------------------------------------------------------
# Instruments: stage 1's names on the other venue

_UNIT_PREFIXES = ("1000000", "1000", "1M", "k")
SPOT_SUFFIX = "-SPOT"


def venue_name(name: str, venue_names: Collection[str]) -> str | None:
    """The column the other venue calls `name`, or None when it lists no such coin.

    Venues list a low-priced coin per many units under different names (Binance
    1000PEPE, 1000000MOG, 1MBABYDOGE; Hyperliquid kPEPE); the prefix is dropped
    and the coin looked up again (`coinmarketcap.venue_symbol`, the mapping the
    lab already uses for market-cap weights). Prices scale, returns do not."""
    from qlab.data.sources.coinmarketcap import venue_symbol

    suffix = SPOT_SUFFIX if name.endswith(SPOT_SUFFIX) else ""
    base = name[: -len(SPOT_SUFFIX)] if suffix else name
    # Spot columns are matched among spot columns, perpetuals among perpetuals.
    names = ({n[: -len(SPOT_SUFFIX)] for n in venue_names if n.endswith(SPOT_SUFFIX)} if suffix
             else {n for n in venue_names if not n.endswith(SPOT_SUFFIX)})
    candidates = [base]
    for prefix in _UNIT_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix):
            candidates.append(base[len(prefix):])
    for candidate in candidates:
        found = venue_symbol(candidate, names)
        if found is not None:
            return found + suffix
    return None


def venue_listing(venue: str = HYPERLIQUID) -> list[str]:
    """What Hyperliquid lists: its perpetual contracts and, as `<coin>-SPOT`, the coins that
    also have a USDC-quoted spot token (`qlab.data.sources.hyperliquid.discover_spot_universe`,
    the lab's own resolution of spot tokens: the token named like the coin, else "U" + coin).
    Stage 1's spot columns are matched against the second list. For `binance`: Binance's
    perpetuals and their spot columns, the listing kept a month
    (`qlab.one_check.binance_listing`)."""
    if venue == BINANCE:
        from qlab import one_check

        return one_check.binance_listing()
    from qlab.data.sources.hyperliquid import discover_spot_universe, discover_universe

    return [*discover_universe(), *discover_spot_universe()]


def hyperliquid_token_coins(path: Path = CONFIG) -> dict[str, str]:
    """Hyperliquid's bridge tokens of the lab's verified spot legs, back to their coins
    (`night/stage2.yaml` spot_legs, UBTC -> BTC): how a Hyperliquid spec that names the token is
    read on Binance. Empty when the config cannot be read."""
    try:
        legs = _load_config(path).spot_legs
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return {leg.token: coin for coin, leg in legs.items()}


def canonical_name(name: str, token_coins: Mapping[str, str]) -> str:
    """A Hyperliquid column named by its bridge token (`UBTC-SPOT`) under the coin's name
    (`BTC-SPOT`); any other name as it is."""
    coin = spot_coin(name)
    if coin is not None and coin in token_coins:
        return f"{token_coins[coin]}{SPOT_SUFFIX}"
    return name


def spot_coin(name: str) -> str | None:
    """The coin of a spot column name (`BTC-SPOT` -> `BTC`), None for a perpetual."""
    return name[: -len(SPOT_SUFFIX)] if name.endswith(SPOT_SUFFIX) else None


def absence_reason(name: str, config: Stage2Config) -> str:
    """Why stage 1's column `name` has no counterpart on the stage 2 venue, in words that
    cite their source. A spot leg gets the reason of its coin."""
    coin = spot_coin(name)
    if coin is None:
        return f"{name} is not listed on {config.venue}"
    if config.venue != HYPERLIQUID:
        return (f"{name} has no {config.venue} spot leg: the lab's listing of {config.venue} "
                f"(qlab.one_check.binance_listing) has no spot column {name}")
    base = coin
    for prefix in _UNIT_PREFIXES:
        if coin.startswith(prefix) and len(coin) > len(prefix):
            base = coin[len(prefix):]
            break
    for candidate in (coin, base):
        if candidate in config.no_spot:
            return (f"{name} has no {config.venue} spot leg: {config.no_spot[candidate]}")
    return (f"{name} has no {config.venue} spot leg: spotMeta lists no USDC-quoted token named "
            f"{base} or U{base} (the lab's rule, qlab.data.sources.hyperliquid.fetch_spot_meta)")


def map_instruments(instruments: list[str], venue_names: Collection[str]
                    ) -> tuple[list[str], list[str]]:
    """(mapped names without repeats, in order; the stage 1 names the venue lacks)."""
    mapped: list[str] = []
    dropped: list[str] = []
    for name in instruments:
        found = venue_name(name, venue_names)
        if found is None:
            dropped.append(name)
        elif found not in mapped:
            mapped.append(found)
    return mapped, dropped


def _spot_legs(instruments: list[str], config: Stage2Config
               ) -> tuple[list[str], dict[str, str]]:
    """Notes and cited sources for the spot columns of the stage 2 instrument list: which
    Hyperliquid token each is, and where that comes from. Only the lab's verified tokens
    (`night/stage2.yaml`, `spot_legs`) carry their own citation; any other coin is
    resolved by the lab's rule from the live listing and cites that rule."""
    notes: list[str] = []
    sources: dict[str, str] = {}
    for name in instruments:
        coin = spot_coin(name)
        if coin is None:
            continue
        if config.venue != HYPERLIQUID:
            citation = ("src/qlab/data/sources/binance.py: Binance's own spot market of the coin, "
                        f"{coin}USDT, the lab's column {name}")
            notes.append(f"{name} is {config.venue}'s own {coin}USDT spot market ({citation}).")
            sources[f"data.spot_legs.{coin}"] = citation
            continue
        leg = config.spot_legs.get(coin)
        if leg is not None:
            notes.append(f"{name} is {config.venue}'s {leg.token}/USDC spot token ({leg.source}).")
            sources[f"data.spot_legs.{coin}"] = leg.source
        else:
            citation = ("src/qlab/data/sources/hyperliquid.py fetch_spot_meta: the spot token "
                        f"named {coin} or U{coin} in the live spotMeta; kept only on bars "
                        "where its price is within frab's 3% of the perp "
                        "(src/qlab/data/snapshot.py spot_is_another_asset)")
            notes.append(f"{name} is the {config.venue} spot token resolved from the live "
                         f"listing, not one of the lab's verified tokens ({citation}).")
            sources[f"data.spot_legs.{coin}"] = citation
    return notes, sources


# --------------------------------------------------------------------------
# The translation


@dataclass(frozen=True)
class Translation:
    spec: StrategySpec | None
    problems: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    """Why the book cannot be held on the venue: a REQUIRED column (a spot leg, usually) it
    does not list. The run is "not evaluable" with these reasons; nothing is dropped
    silently, because a book without one of its legs is another mechanism."""
    dropped_candidates: list[str] = field(default_factory=list)
    """Candidates of a coin-list rule (`params.universe.candidates`, Bv2's top_liquid) that the
    venue cannot hold: no perpetual, or no spot leg under the same coin, in the stage 2 list.
    Stage 1's names; the rule chooses among the rest."""


#: The coin-list rule whose candidates name venue columns (`_bv2_mechanisms.UniverseRule`).
_COIN_LIST_RULE = "top_liquid"


def _universe_candidates(params: object, held: Callable[[str], str | None], path: str = "params"
                         ) -> tuple[object, list[str], list[str]]:
    """`params` with the candidates of every coin-list rule in it (`universe: {rule: top_liquid,
    candidates: [...]}`, at any depth: a wrapper such as `Retune` keeps it under `base_params`)
    renamed to the venue's coins and cut to the ones the venue can hold; (new params, stage 1
    names dropped, problems). `held(coin)` is the venue's name of a coin whose perpetual and spot
    leg are both in the stage 2 list, None when one is missing.

    It is the same translation the instrument list gets ("монет первого этапа, которых нет на
    Hyperliquid, не торгуем"): a candidate is a name of a column, not a number, and a candidate
    with no column was a crash of the run (bv3-research-topk-coins-hl, 2026-11-04: «Bv2 needs
    column '1000BONK'; this panel has none»). A rule left with no candidate is a problem."""
    if isinstance(params, Mapping):
        out: dict = {}
        dropped: list[str] = []
        problems: list[str] = []
        for key, value in params.items():
            if (key == "universe" and isinstance(value, Mapping)
                    and value.get("rule") == _COIN_LIST_RULE
                    and isinstance(value.get("candidates"), list)):
                kept: list[str] = []
                for coin in value["candidates"]:
                    found = held(str(coin))
                    if found is None:
                        dropped.append(str(coin))
                    elif found not in kept:
                        kept.append(found)
                if not kept:
                    problems.append(f"none of the {len(value['candidates'])} candidates of "
                                    f"{path}.universe has both a perpetual and a spot leg here")
                out[key] = {**value, "candidates": kept}
                continue
            out[key], more, issues = _universe_candidates(value, held, f"{path}.{key}")
            dropped += more
            problems += issues
        return out, dropped, problems
    return params, [], []


def strategy_sources(code_ref: str) -> tuple[str, ...] | None:
    """The data sources the strategy declares (`valid_sources`) it can run on, or None when it
    declares none. A code reference that cannot be resolved declares nothing here: the run
    reports it on its own."""
    from qlab.pipeline.evaluate import resolve_strategy

    try:
        declared = getattr(resolve_strategy(code_ref), "valid_sources", None)
    except Exception:  # noqa: BLE001 - not this function's error to report
        return None
    return None if declared is None else tuple(declared)


def coins_without_data(spec: StrategySpec, config: Stage2Config,
                       venue_names: Collection[str]) -> dict[str, str]:
    """The columns of the spec's instrument list that the venue lists but that have no candle in
    the second stage's window ({column: reason}): a perpetual contract delisted before the window
    began, a spot market with no candle in it.
    They are a coin the venue does not list for this run (`translate(no_data=...)`), not a failure
    of the run (night of 2026-10-31: MATIC, delisted on Hyperliquid in 2024, stopped
    bv2-research-top10-pool's second stage with «no candle data»)."""
    import pandas as pd

    from qlab.data.sources.hyperliquid import delisted_without_candles, spot_without_candles

    if config.venue != "hyperliquid" or not spec.data.instruments:
        return {}
    mapped = [found for name in spec.data.instruments
              if (found := venue_name(name, venue_names)) is not None]
    perps = list(dict.fromkeys(n for n in mapped if not n.endswith(SPOT_SUFFIX)))
    spots = list(dict.fromkeys(spot_coin(n) for n in mapped if n.endswith(SPOT_SUFFIX)))
    if not perps and not spots:
        return {}
    start = pd.Timestamp(max(spec.data.start, config.data_start), tz="UTC")
    end = pd.Timestamp(spec.data.end, tz="UTC")
    found = dict(delisted_without_candles(perps, start, end)) if perps else {}
    found.update(spot_without_candles(spots, start, end) if spots else {})
    return found


def translate(spec: StrategySpec, config: Stage2Config, *,
              venue_names: Collection[str] | None = None,
              no_data: Mapping[str, str] | None = None,
              volume_of: Callable[[str], pd.DataFrame | None] | None = None) -> Translation:
    """The stage 2 spec of a stage 1 spec.

    Kept exactly: the code, every parameter value (a deep copy, never an edit),
    the interval, the end of the data, `params_fixed_at` and its evidence, the
    fit window, the noise universe, the sources of the parameters. Switched: the
    venue (`data.source`), the start of the data (the later of stage 1's and the
    config's), the costs (Hyperliquid's, cited), the idea id and title. A stage 1
    `instruments_chosen_at` later than the start becomes the child's `judged_from`: the days
    before the list was known are warm-up here as there. A list that has a date (or a knowable-
    coins check, `qlab.knowable_coins`) is checked AGAIN on the venue's own listing dates and
    volumes (`volume_of`, default the raw cache): its coins may have been obvious picks on Binance
    and not on Hyperliquid, and the child's `judged_from` follows the venue it is judged on.

    With `instruments` or `required_instruments` named, `venue_names` (what the
    venue lists) maps them; a coin it does not list is dropped from the list --
    not traded -- and named in `dropped`. `problems` non-empty means no spec could
    be made (the whole list is missing).

    The candidates of a coin-list rule (`params.universe.candidates`) are names of columns too:
    each is renamed to the venue's coin and kept only when the stage 2 list has both its
    perpetual and its spot leg, the rest named in `dropped_candidates` -- the one place a
    parameter is touched, and only its names."""
    problems: list[str] = []
    dropped: list[str] = []
    data = spec.data
    reverse = config.venue != HYPERLIQUID
    # Back to Binance (an idea born on Hyperliquid): a column Hyperliquid names by its bridge token
    # is read under its coin first (UBTC-SPOT -> BTC-SPOT); kPEPE -> 1000PEPE is `venue_name`'s.
    tokens = hyperliquid_token_coins() if reverse else {}
    instruments = (None if data.instruments is None
                   else [canonical_name(n, tokens) for n in data.instruments])
    no_data = dict(no_data or {})
    if no_data and venue_names is not None:
        # A contract with no candle in the window is a coin the venue does not list for this run,
        # and so is its spot leg: the existing path drops it and the note says why.
        gone = {*no_data, *(f"{name}{SPOT_SUFFIX}" for name in no_data)}
        venue_names = [n for n in venue_names if n not in gone]
    if instruments is not None:
        if venue_names is None:
            raise ValueError("the spec names instruments: venue_names is needed to map them")
        instruments, dropped = map_instruments(instruments, venue_names)
        if not instruments:
            problems.append(f"none of the {len(data.instruments)} instruments of stage 1 is "
                            f"listed on {config.venue}")
    required = (None if spec.required_instruments is None
                else [canonical_name(n, tokens) for n in spec.required_instruments])
    unavailable: list[str] = []
    if required is not None and venue_names is not None:
        mapped = [venue_name(n, venue_names) or n for n in required]
        unavailable = [absence_reason(n, config) for n in dict.fromkeys(required)
                       if venue_name(n, venue_names) is None]
        required = list(dict.fromkeys(mapped))  # an unmapped name stays: the run says it is absent
    params = copy.deepcopy(spec.params)
    dropped_candidates: list[str] = []
    if venue_names is not None:
        listed = set(instruments) if instruments is not None else set(venue_names)

        def held(coin: str) -> str | None:
            coin = spot_coin(canonical_name(f"{coin}{SPOT_SUFFIX}", tokens)) or coin
            perp = venue_name(coin, venue_names)
            if perp is None or perp not in listed or f"{perp}{SPOT_SUFFIX}" not in listed:
                return None
            spot = venue_name(f"{coin}{SPOT_SUFFIX}", venue_names)
            return perp if spot == f"{perp}{SPOT_SUFFIX}" else None

        params, dropped_candidates, issues = _universe_candidates(params, held)
        problems += issues
    only_on = strategy_sources(spec.code_ref)
    if only_on is not None and config.venue not in only_on:
        problems.append(
            f"the strategy {spec.code_ref} runs only on a panel of {', '.join(only_on)} "
            f"(valid_sources), so it has no version on {config.venue}")
    if problems:
        return Translation(None, problems, dropped, unavailable, dropped_candidates)

    child_data = SpecData(
        source=config.venue, interval=data.interval, include_spot=data.include_spot,
        spot_prices=data.spot_prices, spot_source=None,
        # Back to Binance the window is Binance's whole history: the Hyperliquid start was where
        # Hyperliquid's data begins, not a choice of the idea (a coin listed later starts when its
        # candles do).
        start=config.data_start if reverse else max(data.start, config.data_start),
        end=data.end, instruments=instruments,
        # The venue's floor is how its live engine picks coins from the listing, so it applies
        # to a strategy that picks from the venue's whole listing; a named list (Bv2's coins
        # and their spot tokens, FRAB's) is traded live whatever its volume, and the floor
        # would drop spot tokens the live book holds.
        min_daily_volume_usd=(data.min_daily_volume_usd if instruments is not None
                              else stricter_floor(data.min_daily_volume_usd,
                                                  config.min_daily_volume_usd)))
    # A list chosen later than the venue's data begins (a causal twin's hand-picked coins,
    # `instruments_chosen_at`) is judged from that day here as on the twin's own history:
    # earlier days know how the list ended up.
    known_from = (spec.instruments_chosen_at
                  if spec.instruments_chosen_at is not None and instruments is not None
                  and spec.instruments_chosen_at > child_data.start else None)
    known = None
    list_day = spec.instruments_chosen_at
    if instruments is not None and (list_day is not None or spec.coins_knowable is not None):
        volume = (volume_of or knowable_coins.raw_volume)(config.venue)
        if volume is not None:
            known = knowable_coins.check(
                volume, instruments, venue=config.venue, start=child_data.start,
                end=child_data.end, config=knowable_coins.load_config())
            list_day = knowable_coins.list_day(known, spec.instruments_chosen_at)
            known_from = list_day if list_day is not None and list_day > child_data.start else None
    notes = [
        f"Stage 2 of the evaluation (docs/TWO_STAGE.md): the same mechanism and the same "
        f"parameter values as stage 1 ({spec.idea_id}) on {config.venue}'s own perpetual "
        f"contracts listed at each bar; nothing was refitted.",
        f"The costs are {config.venue}'s (published base-tier taker fee, frab's slippage "
        f"estimate), not stage 1's.",
    ]
    if reverse:
        notes.append(
            f"The idea was born on Hyperliquid ({spec.idea_id}'s own data is Hyperliquid's); this "
            f"is its {config.venue} verdict, judged independently (docs/VENUE_SPLIT.md; owner, "
            "2026-11-05: «почему вот эта стратегия не тестировалась на бинансе?»). The data starts "
            f"with {config.venue}'s archive on {config.data_start}, not on Hyperliquid's first "
            "day; numbers inside the parameters (a wrapper's scoring costs, its grid) are stage "
            "1's, unchanged, and a wrapper that chooses by code chooses again on this data. The "
            "facts about execution are this venue's: the lab has no order adapter here, so a pass "
            "is needs-infrastructure.")
    if child_data.min_daily_volume_usd != data.min_daily_volume_usd:
        before = ("no floor" if data.min_daily_volume_usd is None
                  else f"${data.min_daily_volume_usd:,.0f}")
        notes.append(f"A coin is traded on a day only with at least "
                     f"${child_data.min_daily_volume_usd:,.0f} of daily volume, the floor "
                     f"{config.venue}'s live engine uses; stage 1 had {before}.")
    if dropped:
        notes.append(f"{len(dropped)} instrument(s) of stage 1 are not listed on {config.venue} "
                     f"and are not traded: {', '.join(dropped)}.")
    if dropped_candidates:
        notes.append(f"{len(dropped_candidates)} candidate(s) of the coin-list rule "
                     f"(universe.candidates) have no perpetual or no spot leg on {config.venue} "
                     "and are not candidates here; the rule chooses among the rest: "
                     f"{', '.join(dropped_candidates)}.")
    if no_data:
        notes.append("Of those, listed by the venue but with no candle in the window, so treated "
                     "as not listed: " + "; ".join(f"{name} ({why})" for name, why in
                                                   sorted(no_data.items())) + ".")
    if known is not None:
        notes.append(f"Knowable-coins check on {config.venue}'s own listing and volumes: "
                     f"{known.plain}")
    if known_from is not None:
        notes.append(f"Judged from {known_from}, the day the instrument list was known "
                     "(instruments_chosen_at); the days before it are warm-up.")
    sources = {k: v for k, v in spec.sources.items() if k != "costs" and not k.startswith("costs.")
               and not (reverse and k.startswith("data.spot_legs."))}
    sources.update({k: v for k, v in config.cost_sources.items()
                    if k != "costs.two_leg_taker_fee_bps"})
    taker = config.taker_fee_bps
    if config.two_leg_taker_fee_bps is not None and two_legged(spec):
        # Spot against a perpetual on this venue: both legs' fees, the table's two-leg number.
        taker = config.two_leg_taker_fee_bps
        sources["costs.taker_fee_bps"] = config.cost_sources["costs.two_leg_taker_fee_bps"]
        notes.append(f"The book has {spec.simultaneous_legs} legs that move together: the taker "
                     f"fee is {config.venue}'s two-leg number of night/costs.yaml, {taker:g} bp.")
    min_leg = spec.min_leg_notional
    if (spec.execution_venue or spec.data.source) != config.venue:
        # Stage 1 was judged for another venue (Binance, docs/VENUE_SPLIT.md), with that venue's
        # minimum leg: the second stage is judged for ITS venue, with the venue's own.
        venue = load_venue(config.venue)
        if venue is not None and venue.min_leg_notional_usd is not None:
            min_leg = venue.min_leg_notional_usd
            sources["min_leg_notional"] = (
                f"venues/{config.venue}.yaml min_leg_notional_usd: {min_leg:g}, the venue's own "
                + ("minimum order (funding-rate-arbitrage/research/SCREENING.md section 2: "
                   "«HL ~$10»)" if not reverse else
                   "MIN_NOTIONAL of the ordinary contract (exchangeInfo read 2026-10-30, 516 of "
                   "522 contracts); the contracts that differ are read per contract "
                   "(min_leg_notional_by_instrument, "
                   "seed/venue_facts/2026-10-30-binance_um_min_notional.csv)"))
            notes.append(f"The minimum leg is {config.venue}'s own, ${min_leg:g} "
                         f"(venues/{config.venue}.yaml), not stage 1's ${spec.min_leg_notional:g}.")
    spot_notes, spot_sources = _spot_legs(instruments or [], config)
    notes += spot_notes
    sources.update(spot_sources)
    evidence = spec.params_fixed_evidence
    own_list: dict[str, object] = {}
    if known is not None:
        # the venue's own check replaces the date the stage 1 spec carried
        own_list = {"instruments_chosen_at": list_day, "coins_knowable": known}
        sources["coins_knowable"] = known.source
        if list_day is not None:
            sources["instruments_chosen_at"] = (
                f"{known.source}: the first day every coin of the list was knowable on "
                f"{config.venue}" if list_day == known.known_from else
                sources.get("instruments_chosen_at")
                or f"the list's hand-picked date {list_day} (stage 1 spec {spec.idea_id})")
        else:
            sources.pop("instruments_chosen_at", None)
    reverse_only: dict[str, object] = {}
    if reverse:
        # The facts about execution are the data's venue's (docs/VENUE_SPLIT.md): no
        # `execution_venue` naming Hyperliquid survives the move to Binance.
        reverse_only = {"execution_venue": None}
    child = spec.model_copy(update={**own_list, **reverse_only,
        "idea_id": child_id(spec.idea_id, config.venue),
        "title": f"{spec.title} (второй этап: {config.venue})",
        "params": params,
        "data": child_data,
        "costs": SpecCosts(taker_fee_bps=taker, slippage_bps=config.slippage_bps),
        "required_instruments": required,
        "min_leg_notional": min_leg,
        "exploratory": None,
        "judged_from": known_from,
        "differs_from_source": [*spec.differs_from_source, *notes],
        "params_fixed_evidence": (None if evidence is None
                                  else f"Inherited from stage 1 ({spec.idea_id}): {evidence}"),
        "sources": sources,
    })
    return Translation(child, [], dropped, unavailable, dropped_candidates)


# --------------------------------------------------------------------------
# From which day the market holds the book


@dataclass(frozen=True)
class BookWindow:
    needed: int
    block: str
    most: int
    judged_from: date | None
    judged_bars: int
    full_share: float | None
    reason: str | None
    """Why the mechanism cannot be tested on this market; None when it can."""


def book_window(panel, params: Mapping[str, object]) -> BookWindow:
    """From which bar the market has, at once, as many coins as the portfolio rule
    needs (`PORTFOLIO_MIN_COINS`), and whether it ever does.

    The panel may start later than the mechanism can trade (a coin needs
    `min_history_days` of prices before it is admitted): the first bars carry no
    book. Judging them as flat days would dilute the return and the Sharpe, so the
    judged window starts at the first bar with a full book (`StrategySpec.
    judged_from`); a market that never has one cannot test the mechanism, and
    that is "not evaluable", never a fail."""
    from qlab.strategies.blocks.composed import eligible_coins
    from qlab.strategies.blocks.portfolio import min_coins

    portfolio = params["portfolio"]
    block = str(portfolio["block"])  # type: ignore[index]
    needed = min_coins(portfolio)  # type: ignore[arg-type]
    count = eligible_coins(panel, params).sum(axis=1)
    most = int(count.max()) if len(count) else 0
    full = (count >= needed).to_numpy()
    if not full.any():
        return BookWindow(needed, block, most, None, 0, None, (
            f"the market never had {needed} admissible coins at once, which the portfolio "
            f"rule `{block}` needs to hold its book (most on any day: {most}), so the "
            "mechanism cannot be held there"))
    position = int(full.argmax())
    bars = len(full) - position
    share = float(full[position:].mean())
    if bars < 2:
        return BookWindow(needed, block, most, None, bars, share, (
            f"the market had {needed} admissible coins at once only on the last day of the "
            "data: too short to measure a return"))
    return BookWindow(needed, block, most, count.index[position].date(), bars, share, None)


# --------------------------------------------------------------------------
# Both stages together


@dataclass(frozen=True)
class Combined:
    route: str
    """The route that moves the idea's status (a `TrialRoute` value)."""
    reason: str


PASSED = ("paper", "shelf", "needs-infrastructure")
"""Routes of a run that passed every rule of its venue."""


VENUE_PENDING = ("needs-forward", "needs-more-data", "error")
"""Routes of a venue that has not spoken yet: more data, or a crashed run tried again."""


def venue_state(route: str | None) -> str:
    """`pass`, `reject`, `pending` (will decide with data), or `none` (the venue cannot judge the
    idea: not evaluable, or no run)."""
    if route in PASSED:
        return "pass"
    if route == "reject":
        return "reject"
    if route in VENUE_PENDING:
        return "pending"
    return "none"


def combine_venues(binance: str | None, hyperliquid: str | None, *, why: str = "") -> Combined:
    """The idea's route when the two venues are judged independently (rules from 2026-10-30.2,
    docs/VENUE_SPLIT.md; owner, 2026-10-30: «проходить если где-то»). `binance` is the route of the
    Binance run, `hyperliquid` that of the Hyperliquid run; either may be None (no run).

    * a venue passed: the idea passes, with the better route (a Hyperliquid pass is `paper`, a
      Binance pass alone `needs-infrastructure`: the lab has no Binance order adapter);
    * both rejected: reject; a reject on one venue with the other unable to judge at all
      (not evaluable, no run) is a reject too, nothing is pending;
    * a reject on one venue with the other still undecided: the idea stays open, undecided
      (`needs-forward`), because that venue has not spoken yet;
    * nothing passed or rejected: the undecided venue's route stands, Binance's first."""
    b, h = venue_state(binance), venue_state(hyperliquid)
    seen = f"Binance {binance or 'no run'}, Hyperliquid {hyperliquid or 'no run'}"
    tail = f": {why}" if why else ""
    if h == "pass":
        return Combined(str(hyperliquid), f"passed on Hyperliquid ({hyperliquid}); {seen}{tail}")
    if b == "pass":
        return Combined("needs-infrastructure", f"passed on Binance ({binance}); {seen}{tail}. "
                        "The idea stands on Binance alone, where the lab has no order adapter")
    if "reject" in (b, h):
        other = h if b == "reject" else b
        if other in ("reject", "none"):
            return Combined("reject", f"no venue passed and none is pending; {seen}{tail}")
        return Combined("needs-forward", f"one venue rejected and the other has not decided yet; "
                        f"{seen}{tail}")
    if b == "pending":
        return Combined(str(binance), f"{seen}{tail}")
    if h == "pending":
        return Combined(str(hyperliquid), f"{seen}{tail}")
    return Combined("not-evaluable", f"neither venue could judge the idea; {seen}{tail}")


def combine_for(config: Stage2Config, stage1_route: str, stage2_route: str | None, *,
                why: str = "", best_of_venues: bool = False,
                independent: bool = False) -> Combined:
    """`combine` with the venues in their places: for an idea born on Hyperliquid (the Binance
    config) stage 1 is the Hyperliquid verdict and stage 2 the Binance one. Only independent venues
    (rules from 2026-10-30.2) ever judged such an idea, so that direction is `combine_venues`."""
    if config.venue == BINANCE:
        return combine_venues(stage2_route, stage1_route or None, why=why)
    return combine(stage1_route, stage2_route, config=config, stage2_reason=why,
                   best_of_venues=best_of_venues, independent=independent)


def parent_route(session, idea_id: str, route: str, reason: str
                 ) -> tuple[str, str, tuple[str, ...] | None]:
    """(route, reason, venue ideas) a Binance run moves its idea's status by, under independent
    venues: the Binance route together with the second stage's latest run on record. The second
    stage is not run here; only what it already said is read. An idea with no second-stage run
    keeps the Binance route (the second stage, if it follows, settles the idea itself)."""
    from qlab.registry.stages import latest_routed, stage2_child

    child = stage2_child(session, idea_id)
    found = None if child is None else latest_routed(session, child.id)
    if child is None or found is None:
        return route, reason, None
    hl = getattr(found[0].route, "value", found[0].route)
    combined = combine_venues(route, hl, why=f"Binance: {reason}")
    return combined.route, combined.reason, (child.id,)


def combine(stage1_route: str, stage2_route: str | None, *, config: Stage2Config,
            stage2_reason: str = "", best_of_venues: bool = False,
            independent: bool = False) -> Combined:
    """The idea's route from the two stages (the two venues).

    Rules before 2026-10-30.1 (`best_of_venues=False`): it needs both to survive:

    * a reject at either stage rejects (stage 2's with the data range of stage 2:
      the child's verdicts carry it, and `apply_route(also_ideas=...)` reads it);
    * stage 1 still waiting for forward data keeps the idea waiting, whatever
      stage 2 says (it passed nothing yet);
    * otherwise stage 2's route stands: it holds everything stage 1 checked, on the
      venue where the orders go (a stage 1 `needs-infrastructure` is Binance's
      missing adapter, which stage 2 settles);
    * a stage 2 that decided nothing (not evaluable, no data, a crash) never
      passes the idea: it waits, with the reason.

    Rules from 2026-10-30.1 (`best_of_venues=True`, `RuleSet.venue_split`, docs/VENUE_SPLIT.md):
    stage 1 is the verdict for Binance, stage 2 the verdict for Hyperliquid, and the idea's route
    is the better of the two:

    * Binance has not decided (it waits for forward data) or was not passed: the idea takes
      Binance's route, whatever Hyperliquid says (Hyperliquid's consistency check needs Binance's
      history to have decided; a Binance reject is final, as before);
    * Binance passed and Hyperliquid passed: Hyperliquid's route (`paper`: the lab can trade
      there);
    * Binance passed and Hyperliquid rejected or decided nothing: the idea stays passed on Binance
      alone, `needs-infrastructure` (the lab has no Binance adapter), with the Hyperliquid result
      named. A Hyperliquid reject is the child's own verdict, with its own range of data; it no
      longer kills the idea.

    Rules from 2026-10-30.2 (`independent=True`, `RuleSet.venue_split.independent`): the venues are
    judged independently, see `combine_venues`.

    A stage 2 route that came from the consistency check with stage 1
    (`qlab.pipeline.confirm`) is an ordinary stage 2 route here. It is only ever
    given for a stage 1 that decided (`paper`, `shelf`, `needs-infrastructure`); and
    were it ever given for a stage 1 that still waits, the `needs-forward` line below
    keeps the idea waiting: "too early" on stage 1 never turns into a pass."""
    if independent:
        return combine_venues(stage1_route, stage2_route, why=stage2_reason)
    if stage1_route not in config.stage1_survives:
        return Combined(stage1_route, f"stage 1 {stage1_route}: stage 2 not run")
    if best_of_venues:
        return _best_of_venues(stage1_route, stage2_route, stage2_reason)
    if stage2_route == "reject":
        return Combined("reject", f"stage 2 ({config.venue}) rejected: {stage2_reason}")
    if stage2_route is None or stage2_route in UNDECIDED:
        why = stage2_reason or "stage 2 did not finish"
        return Combined("needs-forward", f"stage 1 {stage1_route}; stage 2 decided nothing "
                        f"({stage2_route or 'no run'}): {why}")
    if stage1_route == "needs-forward":
        return Combined("needs-forward", f"stage 1 still waits for forward data; stage 2 "
                        f"{stage2_route}: {stage2_reason}")
    return Combined(stage2_route, f"stage 1 {stage1_route}; stage 2 {stage2_route}: "
                    f"{stage2_reason}")


def _best_of_venues(binance: str, hyperliquid: str | None, why: str) -> Combined:
    """`combine` under `RuleSet.venue_split`: the better of the Binance and Hyperliquid routes."""
    if binance not in PASSED:
        # Binance waits for forward data: nothing passed yet. Hyperliquid's result is named.
        return Combined(binance, f"Binance {binance}; Hyperliquid {hyperliquid or 'no run'}: {why}")
    if hyperliquid in ("paper", "shelf"):
        return Combined(hyperliquid, f"passed on Binance ({binance}) and on Hyperliquid "
                        f"({hyperliquid}): {why}")
    seen = ("rejected" if hyperliquid == "reject"
            else f"decided nothing ({hyperliquid or 'no run'})")
    return Combined("needs-infrastructure", f"passed on Binance ({binance}); Hyperliquid {seen}: "
                    f"{why}. The idea stands on Binance alone, where the lab has no order adapter")


# --------------------------------------------------------------------------
# The step


@dataclass
class Stage2Result:
    parent_id: str
    idea_id: str
    stage1_route: str | None = None
    stage1_trial_id: int | None = None
    status: str = "skipped"
    """skipped | evaluated | not-evaluable | error"""
    measurement: bool = False
    """Run by hand on an idea that did not survive stage 1: a measurement of the
    mechanism on the second venue. The trial and the child's own verdicts are written
    (every run is), the parent's status is not touched."""
    skipped: str | None = None
    trial_id: int | None = None
    route: str | None = None
    reason: str | None = None
    metrics: dict = field(default_factory=dict)
    venue: str | None = None
    data_start: str | None = None
    data_end: str | None = None
    judged_from: str | None = None
    needed_coins: int | None = None
    most_coins: int | None = None
    full_share: float | None = None
    dropped: list = field(default_factory=list)
    no_data: dict = field(default_factory=dict)
    """Contracts the venue lists that have no candle in the window (delisted before it): a part of
    `dropped`, with the reason each."""
    dropped_candidates: list = field(default_factory=list)
    """Candidates of a coin-list rule the venue cannot hold (`Translation.dropped_candidates`)."""
    unavailable: list = field(default_factory=list)
    """Required columns the venue does not list, with the cited reason each (the run is
    not evaluable for them)."""
    spec_path: str | None = None
    combined_route: str | None = None
    combined_reason: str | None = None
    independent: bool = False
    """The venues were judged independently (rules from 2026-10-30.2): the first venue's verdict
    did not decide whether this one ran."""
    parent_status: str | None = None
    confirmation: dict | None = None
    """The consistency check with stage 1's history (`qlab.pipeline.confirm.Confirmation`
    as a dict), when the noise comparison could not decide on this window: the
    percentile among stage 1's same-length segments, how many there were, the stage 1
    trial compared to, and the rules version that decided. None when the comparison
    with random books decided as before."""
    error: str | None = None
    failure: dict | None = None
    """The idea's crashes in a row when this run crashed (`record_failure`): attempts, whether it
    is given up on."""
    venue_routes: dict | None = None
    """The route standing for each venue after the run, as the shared status function read them
    from the registry ({"binance": ..., "hyperliquid": ...}; independent venues only): what the
    report names, so an idea born on Hyperliquid whose stricter look there rejected is not called
    passed there because its first look was."""


def write_spec(spec: StrategySpec, parent_id: str, spec_dir: Path) -> Path:
    """The stage 2 spec as a file, for whoever wants to read or re-run it."""
    path = spec_dir / f"{spec.idea_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False,
                          allow_unicode=True, width=100)
    path.write_text(
        f"# Stage 2 of {parent_id} (qlab.pipeline.stage2, docs/TWO_STAGE.md): made by code from\n"
        f"# the stage 1 spec. The parameter values are stage 1's, untouched; do not edit by hand.\n"
        + body, encoding="utf-8")
    tree_writes.note(path)
    return path


def _ensure_child(session, spec: StrategySpec, parent_id: str, *, measurement: bool) -> None:
    from qlab.registry import repo
    from qlab.registry.models import AssetClass, Idea

    parent = session.get(Idea, parent_id)
    if parent is None:
        raise ValueError(f"stage 2 of {parent_id!r}: the idea is not in the registry")
    child = session.get(Idea, spec.idea_id)
    if child is None:
        venue = VENUE_TITLE.get(spec.data.source, spec.data.source)
        child = repo.upsert_idea(
            session, id=spec.idea_id, title=spec.title, source_type=parent.source_type,
            asset_class=AssetClass.CRYPTO_PERP, profile=parent.profile,
            notes=f"{NOTES_PREFIX}{parent_id} (docs/TWO_STAGE.md): the same mechanism and "
                  f"parameters on {venue}"
                  + ("; the idea was born on Hyperliquid, this is its Binance verdict "
                     "(docs/VENUE_SPLIT.md)" if spec.data.source == BINANCE else "")
                  + ("; a MEASUREMENT run by hand on an idea that did not survive stage 1, "
                     "its parent's status is not touched" if measurement else ""))
    child.parent_id = parent_id
    session.flush()


def _settle(session, result: Stage2Result, combined: Combined, *, rules_version: str | None,
            end: date | None, parent_id: str, child_id: str, independent: bool = False,
            note: str = "") -> None:
    """Move the parent's status by the two stages together.

    Independent venues (rules from 2026-10-30.2) read both venues' verdicts from the registry,
    through the same function every first-stage run uses (`qlab.pipeline.venue_status.settle`);
    `combined` is then only what the caller expected. The older rules take `combined` as given."""
    from qlab.registry.lifecycle import apply_route
    from qlab.registry.models import Idea, TrialRoute

    if independent:
        from qlab.pipeline.venue_status import settle

        settled = settle(session, parent_id, trial_id=result.trial_id,
                         rules_version=rules_version, run_data_end=end, note=note)
        result.venue_routes = {"binance": settled.view.binance,
                               "hyperliquid": settled.view.hyperliquid}
        result.combined_route = settled.view.combined.route
        result.combined_reason = settled.reason
        result.parent_status = session.get(Idea, parent_id).status.value
        return
    apply_route(
        session, idea_id=parent_id, route=TrialRoute(combined.route),
        route_reason=combined.reason, trial_id=result.trial_id, rules_version=rules_version,
        run_data_end=end, also_ideas=(child_id,))
    result.combined_route = combined.route
    result.combined_reason = combined.reason
    result.parent_status = session.get(Idea, parent_id).status.value


def run_stage2(spec: StrategySpec, *, stage1_route: str | None, stage1_trial_id: int | None,
               noise_trials: int,
               config: Stage2Config | None = None, spec_dir: Path = SPEC_DIR,
               venue_names: Collection[str] | None = None,
               measure: bool = False, settle_note: str = "",
               only_a_pass_moves: bool = False) -> Stage2Result:
    """Stage 2 of one idea whose stage 1 (`spec`, run as trial `stage1_trial_id`)
    routed `stage1_route`. Never raises: a failure is the result's `error`, and the
    idea waits (stage 2 is tried again on the next run).

    `measure=True` runs the stage on an idea that did not survive stage 1 (by hand,
    `qlab stage2 --measure`): the mechanism is measured on the second venue and the
    parent's status is left alone.

    `settle_note` is put first in the reason the idea's status transition carries (the night's
    queue of closed ideas says there why a closed idea was looked at again);
    `only_a_pass_moves` leaves the idea's status alone unless the venues together pass it (a
    closed idea is revived by a pass, never reopened by a result that cannot decide yet)."""
    result = Stage2Result(parent_id=spec.idea_id, idea_id=child_id(spec.idea_id),
                          stage1_route=stage1_route, stage1_trial_id=stage1_trial_id,
                          measurement=measure)
    try:
        config = config or load_config()
        result.idea_id = child_id(spec.idea_id, config.venue)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # Stage 1 is on record and must stay so: a broken config is this result's error.
        result.status = "error"
        result.error = f"night/stage2.yaml: {type(exc).__name__}: {exc}"
        return result
    result.venue = config.venue
    if not config.enabled:
        result.skipped = ("stage 2 is switched off (night/stage2.yaml)" if config.venue != BINANCE
                          else "stage 2 is switched off (night/stage2_binance.yaml)")
        return result
    if not measure and not survives(stage1_route, config):
        result.skipped = (f"stage 1 routed {stage1_route}: only survivors go on" if stage1_route
                          else "the idea has no stage 1 run to anchor a second stage on")
        return result
    try:
        if _id_taken(spec.idea_id, config.venue):
            result.skipped = (f"the second stage's idea id {result.idea_id!r} belongs to another "
                              f"idea of the registry, not to a second stage of {spec.idea_id!r}")
            return result
        _run(spec, result, config, noise_trials=noise_trials, spec_dir=spec_dir,
             venue_names=venue_names, measure=measure, settle_note=settle_note,
             only_a_pass_moves=only_a_pass_moves)
    except Exception as exc:  # noqa: BLE001 - recorded in the report, the idea waits
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        if not measure:
            _settle_after_error(result, config, closed_before=only_a_pass_moves)
    return result


def load_failures(path: Path | None = None) -> dict:
    try:
        data = json.loads((path or FAILURES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_failures(data: dict, path: Path | None = None) -> None:
    target = path or FAILURES
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True),
                      encoding="utf-8")


def is_permanent(error: str) -> bool:
    """Whether a crash (`"TypeName: message"`) would happen again on the same data and code."""
    return error.split(":", 1)[0].strip() in PERMANENT_ERRORS


def record_failure(parent_id: str, error: str, *, today: date | None = None,
                   path: Path | None = None) -> dict:
    """Count one more crash of the idea's second stage and say whether it is given up on:
    `MAX_FAILED_ATTEMPTS` crashes in a row, or a permanent reason. A decided run clears the count
    (`clear_failure`), so the crashes are consecutive by construction."""
    data = load_failures(path)
    attempts = int((data.get(parent_id) or {}).get("attempts", 0)) + 1
    permanent = is_permanent(error)
    entry = {"attempts": attempts, "error": error[:400], "permanent": permanent,
             "given_up": permanent or attempts >= MAX_FAILED_ATTEMPTS,
             "day": (today or date.today()).isoformat()}
    data[parent_id] = entry
    _save_failures(data, path)
    return entry


def clear_failure(parent_id: str, *, path: Path | None = None) -> None:
    data = load_failures(path)
    if parent_id in data:
        del data[parent_id]
        _save_failures(data, path)


def stays_closed(session, parent_id: str, stage1_trial_id: int | None, *,
                 closed_before: bool = False) -> bool:
    """Whether the idea is closed (rejected, retired, decayed) and that is not the very first
    stage's doing a moment ago. A closed idea is revived by a pass only (owner: «воскрешение с
    кладбища только на данных, которых не было»); a crash or a result that cannot decide yet never
    reopens it. An idea that this flow's first stage has just rejected is another matter: the
    second venue has not spoken yet, and it is judged by `combine_venues` as before.
    `closed_before` says the caller knows the idea was closed before the flow began (the queue of
    closed ideas)."""
    from qlab.registry.lifecycle import CLOSED_STATUSES
    from qlab.registry.models import Idea, StageTransition

    parent = session.get(Idea, parent_id)
    if parent is None or parent.status not in CLOSED_STATUSES:
        return False
    if closed_before:
        return True
    move = (session.query(StageTransition)
            .filter(StageTransition.idea_id == parent_id,
                    StageTransition.to_status == parent.status)
            .order_by(StageTransition.id.desc()).first())
    just_now = (move is not None and stage1_trial_id is not None
                and move.trial_id == stage1_trial_id)
    return not just_now


def _id_taken(parent_id: str, venue: str = HYPERLIQUID) -> bool:
    from qlab.registry.db import session_scope
    from qlab.registry.stages import child_id_taken

    with session_scope() as session:
        return child_id_taken(session, parent_id, venue)


def _settle_after_error(result: Stage2Result, config: Stage2Config, *,
                        closed_before: bool = False) -> None:
    """A stage 2 that crashed does not pass the idea, and does not reopen a closed one.

    * An idea that was closed before this flow stays closed: only a pass revives it.
    * An open idea (or one this flow's first stage has just rejected) waits, with the reason:
      the venue has not spoken. After `MAX_FAILED_ATTEMPTS` crashes in a row, or at once when
      the reason is permanent, the Hyperliquid side is recorded as not run (a failure) and the
      idea takes Binance's verdict: a crash is no reason to wait for ever."""
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    split = load_latest().venue_split
    independent = split is not None and split.independent
    result.independent = independent
    failure = record_failure(failure_key(result.parent_id, config.venue), result.error or "")
    result.failure = failure
    gave_up = bool(independent and failure["given_up"])
    why = result.error or ""
    if gave_up:
        how = ("постоянная причина" if failure["permanent"]
               else f"{failure['attempts']} попытки подряд")
        why = f"{VENUE_TITLE.get(config.venue, config.venue)} не проводился: сбой ({how}): {why}"
    # A crashed second stage has not spoken: for independent venues that is "error" (pending, tried
    # again) until it is given up on, then "none" (the venue did not judge, the first venue's
    # verdict stands); never a pass.
    combined = combine_for(config, result.stage1_route or "", ("none" if gave_up else "error")
                           if independent else None, why=why,
                           best_of_venues=split is not None, independent=independent)
    try:
        with session_scope() as session:
            if independent:
                # The crash is in the ledger now (`record_failure` above): the shared function
                # reads it, so a venue still being retried is pending and a given-up one is
                # «не проводился: сбой»; a closed idea stays closed unless a venue passed.
                _settle(session, result, combined, rules_version=load_latest().version,
                        end=None, parent_id=result.parent_id, child_id=result.idea_id,
                        independent=True)
                return
            if stays_closed(session, result.parent_id, result.stage1_trial_id,
                            closed_before=closed_before):
                from qlab.registry.models import Idea

                result.combined_route = combined.route
                result.combined_reason = (f"{combined.reason}; the idea stays closed: only a pass "
                                          "revives a closed idea, a crash is not one")
                result.parent_status = session.get(Idea, result.parent_id).status.value
                return
            _settle(session, result, combined, rules_version=load_latest().version,
                    end=None, parent_id=result.parent_id, child_id=result.idea_id,
                    independent=independent)
    except Exception as exc:  # noqa: BLE001 - the error above is the one worth reporting
        result.error = f"{result.error}; and the idea's status was not updated: {exc}"


def _give_up_second_venue(spec: StrategySpec, result: Stage2Result, note: str,
                          venue: str = HYPERLIQUID) -> None:
    """The second stage cannot be made (`translate` refused): settle the idea with the second
    venue as one that cannot judge it."""
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    record_failure(failure_key(spec.idea_id, venue), f"ValueError: {result.reason}")
    try:
        with session_scope() as session:
            _settle(session, result, Combined("reject", ""), rules_version=load_latest().version,
                    end=None, parent_id=spec.idea_id, child_id=result.idea_id, independent=True,
                    note=note)
    except Exception as exc:  # noqa: BLE001 - the stage's own reason is the one worth reporting
        result.error = f"the idea's status was not updated: {exc}"


def _run(spec: StrategySpec, result: Stage2Result, config: Stage2Config, *,
         noise_trials: int, spec_dir: Path,
         venue_names: Collection[str] | None, measure: bool, settle_note: str = "",
         only_a_pass_moves: bool = False) -> None:
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.pipeline.evaluate import evaluate_spec, resolve_panel
    from qlab.pipeline.sources import source_problems
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    if spec.data.instruments is not None or spec.required_instruments is not None:
        if venue_names is None:
            venue_names = venue_listing(config.venue)
    no_data = (coins_without_data(spec, config, venue_names)
               if spec.data.instruments is not None else {})
    translation = translate(spec, config, venue_names=venue_names, no_data=no_data)
    result.dropped = translation.dropped
    result.dropped_candidates = translation.dropped_candidates
    result.no_data = no_data
    result.unavailable = translation.unavailable
    if translation.spec is None:
        result.status = "not-evaluable"
        result.reason = "; ".join(translation.problems)
        if not measure and independent_venues():
            # The venue cannot express the idea at all: no trial to anchor, but the venue's
            # answer is «cannot judge», not «owed». Recorded as a permanent give-up so the shared
            # status function stops waiting for it, and the idea is settled now.
            _give_up_second_venue(spec, result, settle_note, config.venue)
        return  # nothing to anchor a trial on: no data was ever requested
    child = translation.spec
    ruleset = load_latest()

    with session_scope() as session:
        # The data first, the child's registry row after: fetching Hyperliquid's hourly candles of
        # every contract takes about an hour, and a write transaction opened before it (the new
        # child idea) would hold SQLite's one write lock all that time.
        panel = resolve_panel(session, child)
        _ensure_child(session, child, spec.idea_id, measurement=measure)
        window = None
        if spec.code_ref == COMPOSED_CODE_REF:
            window = book_window(panel, child.params)
            result.needed_coins, result.most_coins = window.needed, window.most
            result.full_share = window.full_share
            if window.judged_from is not None:
                if child.judged_from is not None:
                    window = dataclasses.replace(
                        window, judged_from=max(window.judged_from, child.judged_from))
                child = child.model_copy(update={
                    "judged_from": window.judged_from,
                    "differs_from_source": [*child.differs_from_source, (
                        f"Judged from {window.judged_from}: the first day {config.venue} has "
                        f"the {window.needed} coins the portfolio rule needs at once "
                        f"(snapshot {panel.snapshot_id}); the days before are warm-up with no "
                        "book.")]})
                result.judged_from = window.judged_from.isoformat()
        result.data_start = child.data.start.isoformat()
        result.data_end = child.data.end.isoformat()
        result.spec_path = str(write_spec(child, spec.idea_id, spec_dir))
        problems = source_problems(child)
        if problems:
            raise ValueError("the stage 2 spec cites no checkable source for: "
                             + "; ".join(problems))

        reviewer = None
        if (ruleset.stage2_confirmation is not None and config.venue == HYPERLIQUID
                and result.stage1_route in confirm.STAGE1_CONFIRMS):
            # (Binance's window is the long one: its noise bar decides on its own, and stage 1's
            # Hyperliquid history is too short to be compared with.)
            # Only a stage 1 that decided can be confirmed; a stage 1 that waits passes nothing.
            # A measurement shows the check too; it still moves no idea's status.
            shared = (ruleset.venue_split is not None
                      and ruleset.venue_split.shared_coins_history)
            reviewer = confirm.Stage2Reviewer(
                stage1_route=result.stage1_route,
                load_history=lambda: confirm.load_stage1_history(
                    session, spec, result.stage1_trial_id,
                    venue_names=(None if not shared else
                                 (venue_names if venue_names is not None else venue_listing))))
        extra = [*translation.unavailable, *([window.reason] if window is not None
                                             and window.reason is not None else [])]
        if extra:
            evaluation = evaluate_spec(
                child, session=session, ruleset=ruleset,
                not_evaluable_extra=extra)
        else:
            review = {} if reviewer is None else {"review_route": reviewer}
            evaluation = evaluate_spec_with_shape_aware_bar(
                child, session=session, ruleset=ruleset, n_trials=noise_trials,
                **review).candidate
        if reviewer is not None and reviewer.outcome is not None:
            result.confirmation = dataclasses.asdict(reviewer.outcome)
        result.trial_id = evaluation.trial_id
        result.route = str(evaluation.routing.route)
        result.reason = evaluation.routing.reason
        gave_up = False
        key = failure_key(spec.idea_id, config.venue)
        if result.route == "error":
            # The stand caught the strategy's exception and wrote an error trial: a crash all the
            # same, counted like one that left this function.
            failure = record_failure(key, result.reason or "error")
            result.failure = failure
            gave_up = failure["given_up"]
        else:
            clear_failure(key)  # it answered: earlier crashes no longer count
        result.status = "not-evaluable" if result.route == "not-evaluable" else "evaluated"
        metrics = evaluation.metrics or {}
        result.metrics = {k: metrics[k] for k in RESULT_METRICS if k in metrics}
        result.metrics.update({k: metrics[k] for k in year_keys(metrics)})
        if not measure:
            independent = ruleset.venue_split is not None and ruleset.venue_split.independent
            result.independent = independent
            given_up = bool(gave_up and independent)
            word = VENUE_TITLE.get(config.venue, config.venue)
            combined = combine_for(config, result.stage1_route or "",
                                   "none" if given_up else result.route,
                                   why=(f"{word} не проводился: сбой: {result.reason}"
                                        if given_up else result.reason or ""),
                                   best_of_venues=ruleset.venue_split is not None,
                                   independent=independent)
            if independent:
                _settle(session, result, combined, rules_version=ruleset.version,
                        end=child.data.end, parent_id=spec.idea_id, child_id=child.idea_id,
                        independent=True, note=settle_note)
                return
            if settle_note:
                combined = Combined(combined.route, f"{settle_note}; {combined.reason}")
            if combined.route not in PASSED and (only_a_pass_moves or stays_closed(
                    session, spec.idea_id, result.stage1_trial_id)):
                from qlab.registry.models import Idea

                result.combined_route = combined.route
                result.combined_reason = (f"{combined.reason}; the idea stays as it is: only a "
                                          "pass revives a closed idea")
                result.parent_status = session.get(Idea, spec.idea_id).status.value
                return
            _settle(session, result, combined, rules_version=ruleset.version,
                    end=child.data.end, parent_id=spec.idea_id, child_id=child.idea_id)


# --------------------------------------------------------------------------
# The morning report: both stages side by side, in plain Russian


def _pct(value: object) -> str:
    return f"{value:+.1%}" if isinstance(value, int | float) and value == value else "—"


def _route_ru(route: str | None) -> str:
    from qlab.funnel import ROUTE_TITLE

    return ROUTE_TITLE.get(route or "", route or "—")


def _numbers_ru(m: Mapping[str, object]) -> str:
    """The stage's numbers on the strategy's own money (docs/CAPITAL_BASE.md); the stand's, a
    share of the book's balance, small beside them where they differ."""
    return numbers_ru(m, noise="always")


def report_lines(stage1: Mapping[str, object], stage2: Mapping[str, object] | None) -> list[str]:
    """The two stages of one idea as report lines. `stage1` holds `route` and `metrics`
    (a `GeneratedResult` as a dict); `stage2` is a `Stage2Result` as a dict, or None for
    a run made before stage 2 existed (no lines)."""
    if stage2 is None:
        return []
    lines: list[str] = []
    stage1_route = stage2.get("stage1_route") or stage1.get("route")
    venue = str(stage2.get("venue") or HYPERLIQUID)
    word = VENUE_TITLE.get(venue, venue)
    suffix = "-bn" if venue == BINANCE else "-hl"
    config_file = "night/stage2_binance.yaml" if venue == BINANCE else "night/stage2.yaml"
    if stage2.get("status") == "skipped":
        if stage2.get("skipped", "").startswith("stage 2 is switched off"):
            return [f"второй этап (проверка на {word}) выключен в `{config_file}`"]
        if stage2.get("skipped", "").startswith("the second stage's idea id"):
            return [f"второй этап не проводился: имя для него (`<идея>{suffix}`) уже занято другой "
                    "идеей реестра, и писать в неё чужие испытания нельзя; идее нужно другое имя "
                    "второго этапа"]
        return [f"второй этап (тот же механизм на {word}) не проводился: на первом этапе "
                f"исход «{_route_ru(str(stage1_route))}», а на {word} проверяют только "
                "тех, кто прошёл первый"]
    head = (f"второй этап, {word} (тот же механизм и те же числа; комиссии {word}; "
            "код без агента, токены не тратятся)")
    if venue == BINANCE:
        head = ("второй этап, Binance (идея родилась на Hyperliquid; тот же механизм и те же "
                "числа на истории Binance; комиссии Binance; код без агента, токены не тратятся)")
    if stage2.get("measurement"):
        head = (f"ЗАМЕР на {word} (тот же механизм и те же числа; запущен вручную, статус "
                "идеи не меняет)")
    if stage2.get("status") == "error":
        lines.append(f"{head}: не завершился — {stage2.get('error')}; повторится при следующем "
                     "запуске, пока идея ждёт")
    elif stage2.get("status") == "not-evaluable":
        needed, most = stage2.get("needed_coins"), stage2.get("most_coins")
        why = (f"на {word} ни в один день не набиралось {needed} монет сразу, которые "
               f"нужны правилу портфеля (больше всего было {most})"
               if needed is not None and stage2.get("judged_from") is None
               and stage2.get("full_share") is None else stage2.get("reason"))
        lines.append(f"{head}: проверить нечем — {why}. Это не отказ: идея просто ждёт")
    else:
        from_day = stage2.get("judged_from")
        window = (f"считается с {from_day} — первого дня, когда на {word} хватает монет "
                  f"({stage2.get('needed_coins')} сразу), до {stage2.get('data_end')}"
                  if from_day else f"{stage2.get('data_start')} — {stage2.get('data_end')}")
        metrics = stage2.get("metrics") or {}
        days = metrics.get("forward_days")
        forward = (f"; проверка вперёд — {days:,.0f} дн. после даты фиксации параметров"
                   if isinstance(days, int | float) and days else "")
        numbers = f" — {_numbers_ru(metrics)}" if metrics else ""
        lines.append(f"{head}, прогон {stage2.get('trial_id')}: {_route_ru(stage2.get('route'))}"
                     f"{numbers}; {window}{forward}")
        if years := years_line(metrics, signs=True):
            lines.append(years)
        confirmation = stage2.get("confirmation")
        decided_by_history = bool(confirmation) and confirmation.get("status") != "not-applied"
        if stage2.get("route") not in ("paper", "shelf") and not decided_by_history:
            lines.append(f"причина: {stage2.get('reason')}")
        if confirmation:
            lines.extend(_confirmation_ru(confirmation))
    if stage2.get("dropped"):
        lines.append(f"монет первого этапа, которых нет на {word}, не торгуем: "
                     f"{len(stage2['dropped'])}")  # type: ignore[arg-type]
    if stage2.get("dropped_candidates"):
        lines.append("монет из списка, среди которых код выбирает, без контракта или без обычного "
                     f"рынка на {word}, в выбор не берём: "
                     f"{len(stage2['dropped_candidates'])}")  # type: ignore[arg-type]
    if stage2.get("no_data"):
        lines.append("из них в списке площадки, но без свечей в окне (сняты с торгов раньше), "
                     "поэтому считаются отсутствующими: "
                     + ", ".join(sorted(stage2["no_data"])))  # type: ignore[arg-type]
    if stage2.get("combined_route") is not None:
        lines.append("итог по двум этапам: " + _combined_ru(stage1_route, stage2))
    return lines


def _confirmation_ru(c: Mapping[str, object]) -> list[str]:
    """The consistency check with stage 1's long history (`qlab.pipeline.confirm`) in plain
    words: why the comparison with random books was replaced, what the check found."""
    power, need = c.get("noise_power"), c.get("noise_min_power")
    good = c.get("noise_good_return")
    why = ""
    if isinstance(power, int | float) and isinstance(need, int | float):
        yearly = f"{good:.0%} в год" if isinstance(good, int | float) else "хорошо"
        why = (f"сравнение со случайными книгами на таком коротком отрезке почти ничего не "
               f"различает: стратегия, которая на самом деле зарабатывает {yearly}, прошла бы "
               f"его лишь в {power:.0%} случаев (решает оно от {need:.0%}), поэтому его заменила "
               "сверка с длинной историей первого этапа (решение владельца 2026-10-27)")
    if c.get("status") == "not-applied":
        out = [f"сверка с длинной историей не проведена — {c.get('reason')}; остаётся прежний "
               "вывод: идея ждёт"]
        return [why, *out] if why else out
    percentile, alert = c.get("percentile"), c.get("alert_percentile")
    head = (f"сверка с длинной историей: хуже нынешнего было {percentile:.0f} из 100 таких же "
            f"отрезков на Binance (порог владельца — нижние {alert:g}); отрезков "
            f"{c.get('segments')}, из них не пересекающихся {c.get('independent_segments')}, "
            f"длина отрезка {c.get('window_days')} дн.; сравнивали с прогоном первого этапа "
            f"{c.get('stage1_trial_id')} ({c.get('stage1_from')} — {c.get('stage1_to')}); "
            f"правила {c.get('rules_version')}")
    shared, whole = c.get("stage1_coins_shared"), c.get("stage1_coins_total")
    if isinstance(shared, int | float) and isinstance(whole, int | float):
        head += (f"; история Binance взята на монетах, которые есть и на Hyperliquid: {shared:.0f} "
                 f"из {whole:.0f} колонок (docs/VENUE_SPLIT.md)")
    elif c.get("stage1_whole_history_reason"):
        head += ("; история Binance взята целиком, на монетах Hyperliquid её посчитать не "
                 f"удалось: {c['stage1_whole_history_reason']}")
    total = c.get("total_return")
    shown = f"{total:+.1%}" if isinstance(total, int | float) else "—"
    if c.get("status") == "confirmed":
        verdict = (f"итог сверки: за отрезок стратегия заработала {shown} после комиссий (больше "
                   "нуля) и не оказалась среди худших отрезков своей длинной истории — "
                   "подтверждено. Это проверка на согласие, а не второе доказательство: "
                   "доказывает первый этап; бумажная торговля 4–6 недель перед деньгами остаётся")
    elif c.get("status") == "not-positive":
        verdict = (f"итог сверки: за отрезок стратегия заработала {shown} после комиссий — не "
                   "больше нуля; отказ")
    else:
        verdict = (f"итог сверки: результат в нижних {alert:g} из 100 отрезков длинной истории — "
                   "на себя не похоже; отказ")
    return [line for line in (why, head, verdict) if line]


def _independent_ru(binance: object, stage2: Mapping[str, object]) -> str:
    """The total of two venues judged independently (rules 2026-10-30.2), in words. `binance` is
    stage 1's route; for an idea born on Hyperliquid (a Binance second stage) stage 1 is the
    Hyperliquid verdict and the second stage's route the Binance one."""
    status = stage2.get("parent_status")
    second = stage2.get("route")
    reverse = stage2.get("venue") == BINANCE
    first_state = venue_state(str(binance) if binance else None)
    second_state = venue_state(str(second) if second else None)
    if stage2.get("status") == "error":
        second_state = "pending"  # a crashed run has not spoken; it is tried again
    b, h = (second_state, first_state) if reverse else (first_state, second_state)
    routes = stage2.get("venue_routes")
    if isinstance(routes, Mapping):
        # what the shared status function read from the registry (both looks of a venue), where
        # it read anything
        if routes.get("binance") is not None:
            b = venue_state(routes["binance"])
        if routes.get("hyperliquid") is not None:
            h = venue_state(routes["hyperliquid"])
    words = {"pass": "прошла", "reject": "отвергнута", "pending": "ещё не решила",
             "none": "не проводилась"}
    confirmation = stage2.get("confirmation") or {}
    power = confirmation.get("noise_power")
    history = ""
    if confirmation.get("status") == "confirmed":
        # A pass that rests on the check against Binance's history alone must not read as a plain
        # one (owner, 2026-10-30): the noise bar had no power on Hyperliquid's short window.
        history = ("; на Hyperliquid — только сверкой с историей Binance, планка шума там не "
                   "решает (" + (f"сила {power:.0%}" if isinstance(power, int | float)
                                 else "сила не измерена") + ")")
    if h == "pass" and b == "pass":
        return f"прошла на обеих площадках, статус «{status}»{history}"
    if h == "pass":
        return (f"прошла на Hyperliquid, на Binance {words[b]}; статус «{status}». Площадки "
                "оцениваются независимо, итог — лучший из двух (docs/VENUE_SPLIT.md)")
    if b == "pass":
        tail = ("; отказ на Hyperliquid записан со своим диапазоном данных "
                f"(по {stage2.get('data_end')}), воскресить его можно только на данных, которых "
                "тогда не было" if h == "reject" and not reverse else "")
        return (f"прошла на Binance (нужна инфраструктура: адаптера Binance у нас нет), на "
                f"Hyperliquid {words[h]}; статус «{status}»{tail}")
    if b == "reject" and h == "reject":
        return ("отвергнута на обеих площадках: на Binance и на Hyperliquid (по "
                f"{stage2.get('data_end')}); воскресить можно только на данных, которых тогда не "
                "было")
    if "reject" in (b, h) and "pending" in (b, h):
        who = "Binance" if b == "reject" else "Hyperliquid"
        other = "Hyperliquid" if b == "reject" else "Binance"
        if status in ("rejected", "retired", "decayed"):
            # a closed idea is revived by a pass only: «рано судить» leaves it closed
            return (f"на {who} отвергнута, на {other} ещё не решила — идея остаётся закрытой, "
                    f"статус «{status}»: оживить её может только проход")
        return (f"на {who} отвергнута, на {other} ещё не решила — идея остаётся открытой, статус "
                f"«{status}»: вторая площадка не договорила")
    return (f"площадки вместе пока не решают: Binance {words[b]}, Hyperliquid {words[h]}; "
            f"статус «{status}»")


def _combined_ru(stage1_route: object, stage2: Mapping[str, object]) -> str:
    route = stage2.get("combined_route")
    status = stage2.get("parent_status")
    confirmation = stage2.get("confirmation") or {}
    if stage2.get("independent"):
        return _independent_ru(stage1_route, stage2)
    if (confirmation.get("status") == "confirmed"
            and route in ("paper", "shelf", "needs-infrastructure")):
        return (f"прошла оба этапа ({_route_ru(str(route))}), статус «{status}»: первый этап "
                "доказал, второй подтвердил согласие с ним")
    if (route == "needs-infrastructure" and str(stage1_route) in PASSED
            and stage2.get("route") not in ("paper", "shelf")):
        # Venue split (rules 2026-10-30.1, docs/VENUE_SPLIT.md): passed on Binance, not on
        # Hyperliquid; the idea stands on Binance alone, where the lab has no order adapter.
        on_hl = ("отвергнута" if stage2.get("route") == "reject"
                 else f"ещё не решила ({_route_ru(stage2.get('route'))})")
        tail = ("; отказ на Hyperliquid записан со своим диапазоном данных "
                f"(по {stage2.get('data_end')}), воскресить его можно только на данных, которых "
                "тогда не было" if stage2.get("route") == "reject" else "")
        return (f"прошла на Binance (нужна инфраструктура: адаптера Binance у нас нет), на "
                f"Hyperliquid {on_hl}; статус «{status}»{tail}. Площадки оцениваются раздельно, "
                "итог — лучший из двух (docs/VENUE_SPLIT.md)")
    if route == "reject":
        return (f"отвергнута на втором этапе — на данных Hyperliquid по {stage2.get('data_end')}; "
                "воскресить можно только на данных, которых тогда не было")
    if route in ("paper", "shelf", "needs-infrastructure") and stage2.get("route") == route:
        return (f"прошла оба этапа ({_route_ru(str(route))}), статус «{status}»")
    return (f"оба этапа вместе пока не решают: {_route_ru(str(route))}; статус «{status}». "
            "Пройти надо обоим")


__all__ = ["BINANCE_CONFIG", "BookWindow", "CONFIG", "Combined", "SPEC_DIR", "SpotLeg",
           "Stage2Config", "Stage2Result", "Translation", "absence_reason", "book_window",
           "canonical_name", "combine", "combine_for", "config_for", "failure_key",
           "hyperliquid_token_coins", "load_config", "map_instruments", "report_lines",
           "run_stage2", "second_venue_of", "spot_coin", "survives", "translate",
           "venue_listing", "venue_name", "write_spec"]
