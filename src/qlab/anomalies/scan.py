"""The detector's scan of one judged trial: re-run it, take it apart, ask each flag
(docs/ANOMALIES.md). Code only: no network, no agent, no tokens. Reads the registry and the
stored data snapshots; writes nothing (`qlab.anomalies.store` writes the result).

A scan has two kinds of flags:

* **from the re-run** -- `implausible_day`, `concentrated_instruments`, `concentrated_coin_days`
  and `untradeable_coin_days` need the book day by day and coin by coin, so the stored spec is
  run again on the stored snapshot (`qlab.anomalies.rerun`);
* **from the registry** -- `venue_divergence` (the idea's other stage), `stand_vs_paper` (the
  paper check) and `too_good_for_family` (the family's yearly returns) read stored numbers and,
  for the venues, the daily series of the two stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from qlab.anomalies import flags as fl
from qlab.anomalies.config import FLAGS, Config
from qlab.anomalies.rerun import Decomposition, StoredRun, daily, decompose, load_run
from qlab.capital_text import ann_of
from qlab.harness.strategy import ZERO_WEIGHT_TOL

BOOK_FLAGS = FLAGS[:7]
"""The flags that need the book re-run (the other three read the registry)."""


@dataclass
class ScanResult:
    trial_id: int
    idea_id: str
    route: str | None
    flags: list[fl.FlagResult]
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> list[str]:
        return [f.name for f in self.flags if f.fired]


# --------------------------------------------------------------------------
# Flags from the re-run


def coin_of(column: str) -> str:
    """The coin a column belongs to: `BTC` and `BTC-SPOT` are one coin, a hedged pair whose legs
    offset each other, so contributions are counted per coin."""
    return column[: -len("-SPOT")] if column.endswith("-SPOT") else column


def by_coin(frame: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
    """`frame` with the columns of one coin merged (summed, or `max`)."""
    grouped = frame.T.groupby([coin_of(c) for c in frame.columns])
    if how == "max_abs":  # the largest absolute value among a coin's legs, with its sign
        merged = frame.T.groupby([coin_of(c) for c in frame.columns]).agg(
            lambda col: col.iloc[int(col.abs().to_numpy().argmax())])
        return merged.T
    # A sum of legs that are all unknown stays unknown, never zero.
    return (grouped.sum(min_count=1) if how == "sum" else getattr(grouped, how)()).T


def coin_day_frames(dec: Decomposition) -> dict[str, pd.DataFrame]:
    """The decomposition by UTC day and coin: net contribution, whether the coin was held,
    traded dollar volume. A day sums its bars; a coin sums its perpetual and spot legs."""
    held_bar = (dec.weights.abs() > ZERO_WEIGHT_TOL).astype(float)
    volume = dec.volume_usd
    days = pd.DatetimeIndex(volume.index).tz_convert("UTC").normalize()
    return {
        "net": by_coin(daily(dec.net, "sum")),
        "held": by_coin(daily(held_bar, "max"), "max") > 0,
        "volume": by_coin(volume.groupby(days).sum(min_count=1)),
    }


def delist_mask(dec: Decomposition, days: pd.DatetimeIndex, window_days: float,
                coins: list[str]) -> pd.DataFrame:
    """Coin-days within `window_days` days up to a coin's last price, for coins that stopped
    trading before the panel's end (the earliest last price among a coin's legs)."""
    last = dec.last_priced.dropna()
    last = last.groupby([coin_of(c) for c in last.index]).min()
    stopped = last[last < dec.panel_end - pd.Timedelta(days=1)]
    mask = pd.DataFrame(False, index=days, columns=coins)
    window = pd.Timedelta(days=window_days)
    for name, when in stopped.items():
        if name in mask.columns:
            near = (days >= (when - window).normalize()) & (days <= when.normalize())
            mask.loc[near, name] = True
    return mask


def coin_day_details(dec: Decomposition, frames: dict[str, pd.DataFrame], moves: pd.DataFrame,
                     cells: list[tuple[pd.Timestamp, str]]) -> list[dict[str, Any]]:
    """For coin-days: the coin, the day (the DECISION day), what it earned, the coin's own move
    over the bar that earned it, that day's dollar volume and the coin's age in days."""
    first = dec.first_priced.groupby([coin_of(c) for c in dec.first_priced.index]).min()
    out = []
    for day, coin in cells:
        age = (day - first[coin]).days if coin in first.index and pd.notna(first[coin]) else None
        volume = frames["volume"].loc[day, coin]
        out.append({"day": str(day.date()), "instrument": str(coin),
                    "net": float(frames["net"].loc[day, coin]),
                    "move": float(moves.loc[day, coin]),
                    "volume_usd": None if pd.isna(volume) else float(volume), "age_days": age})
    return out


def frozen_stretches(zero: pd.DataFrame, min_days: int) -> tuple[pd.DataFrame, list[dict]]:
    """(the coin-days inside a stretch of `min_days` or more consecutive True days, the longest
    stretches as {instrument, days, from})."""
    inside = pd.DataFrame(False, index=zero.index, columns=zero.columns)
    longest: list[dict[str, Any]] = []
    for coin in zero.columns:
        col = zero[coin].to_numpy()
        if not col.any():
            continue
        start = None
        for i, flag in enumerate([*col, False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_days:
                    inside.iloc[start:i, inside.columns.get_loc(coin)] = True
                    longest.append({"instrument": str(coin), "days": i - start,
                                    "from": str(zero.index[start].date())})
                start = None
    longest.sort(key=lambda r: -r["days"])
    return inside, longest[:5]


def run_flags(dec: Decomposition, cfg: Config) -> tuple[list[fl.FlagResult], dict[str, Any]]:
    """The flags that read the book itself, and the facts measured on the way."""
    day_return = daily(dec.book_net, "compound")
    positions = daily(dec.positions.astype(float), "mean")
    gross = daily(dec.gross, "mean")
    net_exposure = daily(dec.weights.sum(axis=1), "mean")
    frames = coin_day_frames(dec)
    net_d, held_d, vol_d = frames["net"], frames["held"], frames["volume"]
    total = float(net_d.to_numpy().sum())
    held_count = int(held_d.to_numpy().sum())
    # A coin's own move over the bar that earns the weight set on the decision day; for a pair
    # of legs the larger of the two (a hedged pair's legs move together).
    moves = by_coin(daily(dec.move.fillna(0.0), "compound"), "max_abs")

    untradeable = cfg.flag("untradeable_coin_days")
    low = (vol_d < untradeable.need("min_volume_usd")) & held_d
    unknown = vol_d.isna() & held_d
    near = delist_mask(dec, net_d.index, untradeable.need("delist_days"),
                       list(net_d.columns)) & held_d
    union = low | near

    tail_line = cfg.flag("tail_move_coin_days")
    tail = (moves.abs() >= tail_line.need("min_move")) & held_d
    tail_cells = net_d.where(tail).stack().sort_values(ascending=False).head(10).index.tolist()

    frozen_line = cfg.flag("frozen_price")
    zero = (moves == 0.0) & held_d
    frozen, longest = frozen_stretches(zero, int(frozen_line.need("min_days")))

    coin_day = net_d.stack()
    coin_day = coin_day[coin_day != 0.0]
    day_flag = fl.implausible_day(day_return, positions, gross, net_exposure,
                                  cfg.flag("implausible_day"))
    for entry in day_flag.detail.get("top", []):  # which legs made the day
        day = pd.Timestamp(entry["day"], tz="UTC")
        legs = net_d.loc[day].sort_values(ascending=False).head(3).index
        entry["legs"] = coin_day_details(dec, frames, moves, [(day, c) for c in legs])
    results = [
        day_flag,
        fl.concentrated_instruments(net_d.sum(), cfg.flag("concentrated_instruments")),
        fl.concentrated_coin_days(coin_day, held_count, cfg.flag("concentrated_coin_days")),
        fl.tail_move_coin_days(
            total=total, tail_net=float(net_d.where(tail, 0.0).to_numpy().sum()),
            tail_days=int(tail.to_numpy().sum()), held_coin_days=held_count,
            top=coin_day_details(dec, frames, moves, tail_cells), line=tail_line),
        fl.untradeable_coin_days(
            total=total, low_volume=float(net_d.where(low, 0.0).to_numpy().sum()),
            near_delist=float(net_d.where(near, 0.0).to_numpy().sum()),
            union=float(net_d.where(union, 0.0).to_numpy().sum()),
            held_coin_days=held_count, low_volume_days=int(low.to_numpy().sum()),
            near_delist_days=int(near.to_numpy().sum()),
            unknown_volume_days=int(unknown.to_numpy().sum()), line=untradeable),
        fl.frozen_price(frozen_days=int(frozen.to_numpy().sum()), held_coin_days=held_count,
                        longest=longest, line=frozen_line),
    ]
    facts = {
        "days": int(len(day_return)), "instruments": int(net_d.shape[1]),
        "positions_median": float(positions.median()), "gross_median": float(gross.median()),
        "net_exposure_median": float(net_exposure.median()),
        "total_arithmetic": total, "reconciles": dec.reconciles, "reconcile_gap": dec.max_gap,
        "start": str(day_return.index.min().date()) if len(day_return) else None,
        "end": str(day_return.index.max().date()) if len(day_return) else None,
    }
    return results, facts


def listing_flag(session: Session, stored: StoredRun, dec: Decomposition, cfg: Config
                 ) -> fl.FlagResult:
    """The share of the profit earned on coins the venue the run is judged for does not list,
    from the first day its listing is known. A run is judged for its own venue (docs/VENUE_SPLIT.md,
    owner 2026-10-30): a Binance run is not flagged for coins Hyperliquid lacks, that is the other
    venue's verdict and `venue_divergence` reads the difference. The flag is for a run whose data
    is another venue's but which is judged for Hyperliquid (a run recorded before the split)."""
    from qlab.anomalies import toolkit

    line = cfg.flag("unlisted_on_execution_venue")
    if stored.judged_for == "hyperliquid" and stored.venue == "hyperliquid":
        return fl._undefined(line.name, "прогон идёт на данных самой биржи исполнения")
    if stored.judged_for != "hyperliquid":
        return fl._undefined(
            line.name, f"прогон судится для своей биржи ({stored.judged_for}): монеты, которых нет "
                       "на другой бирже, — не его дефект; разницу двух бирж показывает "
                       "признак «две биржи показывают разное»")
    mask = toolkit.listed_mask(session, stored)
    if mask is None:
        return fl._undefined(line.name, "в реестре нет снимка биржи исполнения со списком монет")
    frames = coin_day_frames(dec)
    net_d, held_d = frames["net"], frames["held"]
    listed = by_coin(daily(mask.astype(float), "max"), "max") > 0
    listed = listed.reindex(index=net_d.index, columns=net_d.columns).fillna(False)
    first = listed.any(axis=1)
    if not first.any():
        return fl._undefined(line.name, "список биржи исполнения не покрывает дни прогона")
    start = first.idxmax()
    window = net_d.index >= start
    net_w, held_w, listed_w = net_d[window], held_d[window], listed[window]
    unlisted = held_w & ~listed_w
    moves = by_coin(daily(dec.move.fillna(0.0), "compound"), "max_abs")
    cells = (net_w.where(unlisted).stack().sort_values(ascending=False).head(10).index.tolist())
    frames_w = {"net": net_d, "volume": frames["volume"]}
    return fl.unlisted_on_execution_venue(
        venue="hyperliquid", total=float(net_w.to_numpy().sum()),
        unlisted_net=float(net_w.where(unlisted, 0.0).to_numpy().sum()),
        unlisted_days=int(unlisted.to_numpy().sum()), held_coin_days=int(held_w.to_numpy().sum()),
        start=str(start.date()), top=coin_day_details(dec, frames_w, moves, cells), line=line)


# --------------------------------------------------------------------------
# Flags from the registry


def trial_daily_returns(session: Session, trial_id: int, *, run: StoredRun | None = None
                        ) -> pd.Series | None:
    """The judged part's net return by UTC day (compounded bars). Taken from the stored forward
    series when there is one -- exactly the run the verdict rests on -- else the stored spec is
    run again. None when neither is possible."""
    from qlab import forward_series

    payload = forward_series.load(trial_id)
    if payload is not None and payload.get("points"):
        frame = pd.DataFrame(payload["points"], columns=["day", "strategy", "hold"])
        growth = 1.0 + frame["strategy"].astype(float)
        growth.index = pd.DatetimeIndex(pd.to_datetime(frame["day"], utc=True))
        return (growth / growth.shift(1) - 1.0).iloc[1:]
    try:
        stored = run or load_run(session, trial_id)
    except Exception:  # noqa: BLE001 - a trial that cannot be re-run simply has no series
        return None
    return daily(stored.backtest().net_return, "compound")


def trial_daily_capital(session: Session, trial_id: int, *, run: StoredRun | None = None
                        ) -> tuple[pd.Series, bool] | None:
    """The judged part's return by UTC day on the strategy's OWN money (docs/CAPITAL_BASE.md) and
    whether its bars compound (True) or are summed (False: Bv2, FRAB, XSMOM, which do not put
    their profit back into positions). Two stages of one idea are set against each other on this
    one base. Taken from the stored forward series when it is already on the strategy's money,
    else the stored spec is run again. None when neither is possible."""
    from qlab import forward_series

    payload = forward_series.load(trial_id)
    older: tuple[pd.Series, bool] | None = None
    if payload is not None and payload.get("points"):
        frame = pd.DataFrame(payload["points"], columns=["day", "strategy", "hold"])
        curve = frame["strategy"].astype(float)
        curve.index = pd.DatetimeIndex(pd.to_datetime(frame["day"], utc=True))
        growth = 1.0 + curve
        ratio = (growth / growth.shift(1) - 1.0).iloc[1:]
        if payload.get("base") == "strategy_capital":
            if payload.get("reinvested", True) is False:
                return curve.diff().iloc[1:], False
            return ratio, True
        older = (ratio, True)  # a series from before the capital base: the stand's reading
    try:
        stored = run or load_run(session, trial_id)
    except Exception:  # noqa: BLE001 - a trial that cannot be re-run keeps what is stored, if any
        return older
    result = stored.backtest()
    refused = stored.capital is not None and stored.capital.refused is not None
    reinvested = True if refused else result.capital_reinvested
    series = result.net_return if refused else result.capital_return
    return daily(series, "compound" if reinvested else "sum"), reinvested


def counterpart_trial(session: Session, trial_id: int) -> tuple[int, int, str] | None:
    """(stage-1 trial, stage-2 trial, "this is stage 1 / 2") for a trial whose idea has both
    stages (docs/TWO_STAGE.md), else None. Each side is its idea's latest routed trial."""
    from qlab import forward_series
    from qlab.registry.models import Spec, Trial

    trial = session.get(Trial, trial_id)
    spec = None if trial is None else session.get(Spec, trial.spec_id)
    if spec is None:
        return None
    found = forward_series.stage_runs(session, spec.idea_id)
    if found is None or not found[0] or any(s.trial_id is None for s in found[1]):
        return None
    first, second = found[1][0], found[1][1]
    own = "1" if spec.idea_id == first.idea_id else "2"
    return first.trial_id, second.trial_id, own


def venue_flag(session: Session, trial_id: int, cfg: Config, *, run: StoredRun | None = None
               ) -> fl.FlagResult:
    line = cfg.flag("venue_divergence")
    pair = counterpart_trial(session, trial_id)
    if pair is None:
        return fl._undefined(line.name, "у идеи нет второй ступени на другой бирже")
    first, second, _ = pair
    # Both stages on the strategy's own money (docs/CAPITAL_BASE.md): a fresh second-stage book
    # against a first stage whose balance has grown with set-aside cash is not like with like.
    a = trial_daily_capital(session, first, run=run if first == trial_id else None)
    b = trial_daily_capital(session, second, run=run if second == trial_id else None)
    if a is None or b is None:
        return fl._undefined(line.name, "дневной ряд одной из ступеней построить не удалось")
    if a[1] != b[1]:
        return fl._undefined(line.name, "две ступени по-разному вкладывают прибыль: "
                                        "сравнивать их итоги нельзя")
    result = fl.venue_divergence(a[0], b[0], line, additive=not a[1])
    result.detail.update(stage1_trial=first, stage2_trial=second)
    return result


def paper_flag(session: Session, idea_id: str, trial_id: int, cfg: Config) -> fl.FlagResult:
    from qlab.paper_check import latest_check

    line = cfg.flag("stand_vs_paper")
    row = latest_check(session, idea_id)
    if row is None:
        return fl._undefined(line.name, "у идеи нет бумажной книги")
    if row.stand_trial_id != trial_id:
        return fl._undefined(line.name, "сверка с бумагой сделана по другому прогону стенда")
    return fl.stand_vs_paper(
        paper_return=row.paper_return, stand_return=row.stand_return,
        replay_return=row.replay_return,
        comparable=not (row.note or "").startswith("Отрезки разные"), line=line)


def family_returns(session: Session, idea_id: str) -> list[float]:
    """The yearly return of the latest judged run of each OTHER idea of `idea_id`'s family
    (`qlab.registry.families`): calibration noise, withdrawn ideas and runs with no number are
    left out."""
    from sqlalchemy import select

    from qlab.registry.families import family_keys
    from qlab.registry.models import Idea, IdeaStatus, Spec, Trial

    keys = family_keys(session)
    mine = keys.get(idea_id)
    if mine is None:
        return []
    out = []
    for other, key in keys.items():
        if key != mine or other == idea_id or other.startswith("noise-"):
            continue
        member = session.get(Idea, other)
        if member is None or member.status == IdeaStatus.RETIRED:
            continue
        last = session.execute(
            select(Trial).join(Spec, Trial.spec_id == Spec.id)
            .where(Spec.idea_id == other, Trial.metrics.is_not(None), Trial.route.is_not(None))
            .order_by(Trial.id.desc()).limit(1)).scalars().first()
        value = None if last is None else ann_of(last.metrics)
        if isinstance(value, int | float) and value == value:
            out.append(float(value))
    return out


def family_flag(session: Session, idea_id: str, metrics: dict, cfg: Config) -> fl.FlagResult:
    value = ann_of(metrics)
    return fl.too_good_for_family(
        float(value) if isinstance(value, int | float) else None,
        family_returns(session, idea_id), cfg.flag("too_good_for_family"))


# --------------------------------------------------------------------------
# The scan


def scan_trial(session: Session, trial_id: int, cfg: Config, *, run: StoredRun | None = None
               ) -> ScanResult:
    """Every flag for `trial_id`. A flag that cannot be computed is `not defined` with the
    reason; nothing raises over a missing snapshot (the result then has only the registry
    flags). Writes nothing."""
    from qlab.registry.models import Spec, Trial

    trial = session.get(Trial, trial_id)
    if trial is None:
        raise ValueError(f"no trial {trial_id}")
    idea_id = session.get(Spec, trial.spec_id).idea_id
    route = None if trial.route is None else str(trial.route.value)
    flags: list[fl.FlagResult] = []
    facts: dict[str, Any] = {}
    try:
        stored = run or load_run(session, trial_id)
        dec = decompose(stored)
        flags, facts = run_flags(dec, cfg)
        flags.append(listing_flag(session, stored, dec, cfg))
    except Exception as exc:  # noqa: BLE001 - the reason is the flags' not-defined text
        stored = None
        reason = f"прогон не удалось повторить: {exc}"
        flags = [fl._undefined(n, reason) for n in BOOK_FLAGS]
        facts = {"rerun_error": f"{type(exc).__name__}: {exc}"}
    flags += [venue_flag(session, trial_id, cfg, run=stored),
              paper_flag(session, idea_id, trial_id, cfg),
              family_flag(session, idea_id, trial.metrics or {}, cfg)]
    order = {name: i for i, name in enumerate(FLAGS)}
    flags.sort(key=lambda f: order[f.name])
    return ScanResult(trial_id=trial_id, idea_id=idea_id, route=route, flags=flags, facts=facts)


__all__ = ["ScanResult", "by_coin", "coin_day_details", "coin_day_frames", "coin_of",
           "counterpart_trial", "delist_mask", "family_flag", "frozen_stretches", "listing_flag",
           "family_returns", "paper_flag", "run_flags", "scan_trial", "trial_daily_capital",
           "trial_daily_returns",
           "venue_flag"]
