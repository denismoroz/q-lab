"""The anomaly detector's flags (docs/ANOMALIES.md): code only, no tokens.

Owner, 2026-10-28, after the XSMOM causal twin showed +313% in 2026 on Binance's history
(single days of +10 to +19% for a book of 20 positions) while the same mechanism on
Hyperliquid earned +3.7% where Binance showed +63% over the same three months: «давай введем
агента который будет искать такие аномалии и разбирать». The detector is the first half: for
every judged run of an idea that passed, it checks the result against a few declared lines
and writes flags.

**A flag decides nothing.** Like the year signs (`qlab.year_signs`), a flag is recorded and
counted; no verdict, status or route reads it. Which flag, if any, should one day block a
run is the owner's decision once the statistics exist. Each flag is one of three states --
`fires`, `quiet`, `not defined` (with the reason) -- and its line is a number in
`night/anomalies.yaml` with its rationale, not one in this file.

The flag functions are pure: they take series and a `FlagLine` and return a `FlagResult`.
`qlab.anomalies.scan` feeds them from a re-run of the trial and from the registry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qlab.anomalies.config import FlagLine

FIRES, QUIET, UNDEFINED = "fires", "quiet", "not defined"
FIRES_RU, QUIET_RU, UNDEFINED_RU = "сработал", "молчит", "не определено"

TITLES_RU = {
    "implausible_day": "день с невозможно большой прибылью для такого числа позиций",
    "concentrated_instruments": "прибыль держится на одной-трёх монетах",
    "concentrated_coin_days": "прибыль держится на нескольких «монето-днях»",
    "tail_move_coin_days": "прибыль держится на монето-днях с резким скачком цены монеты",
    "untradeable_coin_days": "прибыль с монето-дней, которые нельзя было бы торговать",
    "unlisted_on_execution_venue": "прибыль с монет, которых нет на бирже исполнения",
    "frozen_price": "позиции в монетах с замёрзшей ценой",
    "venue_divergence": "две биржи показывают разное на общем отрезке",
    "stand_vs_paper": "стенд и бумажная книга расходятся",
    "too_good_for_family": "слишком хорошо для своего семейства",
}
"""What each flag says, in plain Russian (a «монето-день» is one coin on one day)."""



@dataclass(frozen=True)
class FlagResult:
    name: str
    state: str
    value: float | None = None
    line: float | None = None
    reason: str | None = None
    """Why the flag is not defined, in plain Russian; None when it is defined."""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.state == FIRES

    @property
    def state_ru(self) -> str:
        return {FIRES: FIRES_RU, QUIET: QUIET_RU}.get(self.state, UNDEFINED_RU)

    @property
    def title_ru(self) -> str:
        return TITLES_RU[self.name]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "value": self.value, "line": self.line,
                "reason": self.reason, "detail": self.detail}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> FlagResult:
        return cls(name=raw["name"], state=raw["state"], value=raw.get("value"),
                   line=raw.get("line"), reason=raw.get("reason"),
                   detail=dict(raw.get("detail") or {}))


def _undefined(name: str, reason: str) -> FlagResult:
    return FlagResult(name=name, state=UNDEFINED, reason=reason)


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _state(fired: bool) -> str:
    return FIRES if fired else QUIET


# --------------------------------------------------------------------------
# 1. A day too good for the number of positions


def implausible_day(day_return: pd.Series, positions: pd.Series, gross: pd.Series,
                    net_exposure: pd.Series, line: FlagLine) -> FlagResult:
    """A day whose net return is larger than the book's positions could give.

    A book moves with the market by its NET exposure and by its coins' own moves, which
    average out over its N positions: one coin's own daily move is about `single_move` (a
    fraction), the market's about `market_move`, so a day's ordinary size is

        sd = sqrt( (|net| * market_move)^2 + (G * single_move)^2 / N )

    with G the gross exposure (the sum of absolute weights), `net` the signed sum, and N, G, net
    the day's means. A day beyond `sigmas` such sizes is not ordinary: the flag fires when the
    largest `|day return| / (sigmas * sd)` over the run passes 1. A dollar-neutral book of 20
    positions has almost no market part, so the breadth is what bounds it; a long-only book has
    the whole market part. Days with no position are left out."""
    n, g = positions.reindex(day_return.index), gross.reindex(day_return.index)
    net = net_exposure.reindex(day_return.index).abs()
    held = (n > 0) & (g > 0) & day_return.notna()
    if not held.any():
        return _undefined(line.name, "в прогоне нет дней с позициями")
    sd = np.sqrt((net[held] * line.need("market_move")) ** 2
                 + (g[held] * line.need("single_move")) ** 2 / n[held])
    day_line = line.need("sigmas") * sd
    ratio = (day_return[held].abs() / day_line).sort_values(ascending=False)
    top = ratio.head(5)
    return FlagResult(
        name=line.name, state=_state(float(ratio.iloc[0]) > 1.0), value=float(ratio.iloc[0]),
        line=1.0,
        detail={"days_over": int((ratio > 1.0).sum()), "days": int(held.sum()),
                "top": [{"day": str(pd.Timestamp(day).date()), "return": float(day_return[day]),
                         "day_line": float(day_line[day]), "positions": float(n[day]),
                         "gross": float(g[day]), "net": float(net_exposure.loc[day]),
                         "times_line": float(r)}
                        for day, r in top.items()]})


# --------------------------------------------------------------------------
# 2. Profit that rests on one to three coins


def concentrated_instruments(instrument_net: pd.Series, line: FlagLine) -> FlagResult:
    """The share of the run's total profit that its best one and best three instruments gave
    (net contributions summed over the run). Fires when either share reaches its line. Not
    defined when the run lost money, or traded fewer than `min_instruments` instruments (a
    three-coin book rests on three coins by design)."""
    total = float(instrument_net.sum())
    traded = int((instrument_net.abs() > 0).sum())
    if traded < line.need("min_instruments"):
        return _undefined(line.name, f"в книге меньше {line.need('min_instruments'):g} монет: "
                                     "опора на несколько монет у неё в устройстве")
    if total <= 0:
        return _undefined(line.name, "прибыль за весь период не положительна")
    best = instrument_net.sort_values(ascending=False)
    top1, top3 = float(best.iloc[0] / total), float(best.iloc[:3].sum() / total)
    fired = top1 >= line.need("top1_share") or top3 >= line.need("top3_share")
    return FlagResult(
        name=line.name, state=_state(fired), value=top1, line=line.need("top1_share"),
        detail={"top1_share": top1, "top3_share": top3, "instruments": traded,
                "top3_line": line.need("top3_share"),
                "top": [{"instrument": str(k), "net": float(v), "share": float(v / total)}
                        for k, v in best.head(5).items()]})


# --------------------------------------------------------------------------
# 3. Profit that rests on a few coin-days


def concentrated_coin_days(coin_day_net: pd.Series, held_coin_days: int, line: FlagLine
                           ) -> FlagResult:
    """The share of the total profit that the best coin-days (one coin on one day) gave: the
    best `top_fraction` of the held coin-days, at least `min_top` of them. XSMOM's causal twin
    earned 94% of its profit on its best 40 coin-days of about 45,000 held, each one a coin that
    rose 100 to 300% in a day. Fires at `share`. Not defined when the run lost money or earned
    less than `min_total` of the capital in all (shares of a tiny sum are noise)."""
    total = float(coin_day_net.sum())
    if total <= 0:
        return _undefined(line.name, "прибыль за весь период не положительна")
    if total < line.need("min_total"):
        return _undefined(line.name, f"вся прибыль за период меньше {line.need('min_total'):.0%} "
                                     "капитала: доли такой малой суммы определяются шумом")
    k = max(int(line.need("min_top")), int(np.ceil(line.need("top_fraction") * held_coin_days)))
    best = coin_day_net.sort_values(ascending=False).head(k)
    share = float(best.sum() / total)
    return FlagResult(
        name=line.name, state=_state(share >= line.need("share")), value=share,
        line=line.need("share"),
        detail={"top_k": k, "held_coin_days": held_coin_days,
                "coin_days": int((coin_day_net != 0).sum()),
                "top": [{"day": str(pd.Timestamp(d).date()), "instrument": str(c),
                         "net": float(v), "share": float(v / total)}
                        for (d, c), v in best.head(10).items()]})


# --------------------------------------------------------------------------
# 3b. Profit from a coin's sudden jump


def tail_move_coin_days(*, total: float, tail_net: float, tail_days: int, held_coin_days: int,
                        top: list[dict], line: FlagLine) -> FlagResult:
    """The share of the total profit earned on coin-days where the coin's own price moved by
    `min_move` or more in a day (up or down) while the book held it. XSMOM's causal twin
    (docs/research/xsmom-anomaly-2026-10-28.md): 187 such coin-days of 44,591 (0.4%) carry 59%
    of the profit, the Hyperliquid second stage 53%, the trend book 22%, Bv2 0%. Fires at
    `share`; not defined when the run lost money."""
    if total <= 0:
        return _undefined(line.name, "прибыль за весь период не положительна")
    share = tail_net / total
    return FlagResult(
        name=line.name, state=_state(share >= line.need("share")), value=float(share),
        line=line.need("share"),
        detail={"min_move": line.need("min_move"), "tail_coin_days": tail_days,
                "held_coin_days": held_coin_days,
                "tail_day_share_of_held": tail_days / held_coin_days if held_coin_days else None,
                "top": top})


# --------------------------------------------------------------------------
# 4. Profit from coin-days that could not have been traded


def untradeable_coin_days(*, total: float, low_volume: float, near_delist: float, union: float,
                          held_coin_days: int, low_volume_days: int, near_delist_days: int,
                          unknown_volume_days: int, line: FlagLine) -> FlagResult:
    """The share of the total profit earned on coin-days that are under the volume line
    (`min_volume_usd` traded that day, in dollars) or in the last `delist_days` days before the
    coin's last price (a coin that stopped trading). Such a day's price is thin, stale or the
    last one before an exit nobody could take. `union` is the profit of the coin-days that are
    either, so a day in both is counted once. Fires at `share`; not defined when the run lost
    money."""
    if total <= 0:
        return _undefined(line.name, "прибыль за весь период не положительна")
    share = union / total
    return FlagResult(
        name=line.name, state=_state(share >= line.need("share")), value=float(share),
        line=line.need("share"),
        detail={"low_volume_share": low_volume / total, "near_delist_share": near_delist / total,
                "held_coin_days": held_coin_days, "low_volume_days": low_volume_days,
                "near_delist_days": near_delist_days, "unknown_volume_days": unknown_volume_days,
                "min_volume_usd": line.need("min_volume_usd"),
                "delist_days": line.need("delist_days")})


# --------------------------------------------------------------------------
# 4b. Profit from coins the execution venue does not list


def unlisted_on_execution_venue(*, venue: str, total: float, unlisted_net: float,
                                unlisted_days: int, held_coin_days: int, start: str,
                                top: list[dict], line: FlagLine) -> FlagResult:
    """The share of the profit, over the days the execution venue's listing is known (from
    `start`), earned on coins the venue did not list that day. Stage 1 judges on Binance's long
    history but the lab trades on Hyperliquid: ten of the ten best days of 2026 of XSMOM's
    causal twin were coins Hyperliquid does not list. Fires at `share`; not defined when the
    window lost money."""
    if total <= 0:
        return _undefined(line.name, "прибыль за период с известным списком биржи исполнения "
                                     "не положительна")
    share = unlisted_net / total
    return FlagResult(
        name=line.name, state=_state(share >= line.need("share")), value=float(share),
        line=line.need("share"),
        detail={"venue": venue, "from": start, "unlisted_coin_days": unlisted_days,
                "held_coin_days": held_coin_days, "top": top})


# --------------------------------------------------------------------------
# 4c. Positions in coins whose price does not move


def frozen_price(*, frozen_days: int, held_coin_days: int, longest: list[dict],
                 line: FlagLine) -> FlagResult:
    """The share of held coin-days that sit in a stretch of `min_days` or more days in a row
    with exactly zero price change (a coin that is not trading, a stale price: BTCST held 243
    days with a frozen price). Fires at `share`; not defined when nothing was held."""
    if held_coin_days <= 0:
        return _undefined(line.name, "в прогоне нет позиций")
    share = frozen_days / held_coin_days
    return FlagResult(
        name=line.name, state=_state(share >= line.need("share")), value=float(share),
        line=line.need("share"),
        detail={"min_days": line.need("min_days"), "frozen_coin_days": frozen_days,
                "held_coin_days": held_coin_days, "longest": longest})


# --------------------------------------------------------------------------
# 5. Two venues that disagree


def _annual(total: float, days: int) -> float:
    return -1.0 if 1.0 + total <= 0 else (1.0 + total) ** (365.25 / days) - 1.0


def venue_divergence(stage1: pd.Series, stage2: pd.Series, line: FlagLine) -> FlagResult:
    """Stage 1 (Binance) and stage 2 (Hyperliquid) over the days both have (daily returns, UTC
    days): the same mechanism should earn about the same. Each side's total over the common
    window is turned into a yearly figure, and the flag fires when they differ by `gap_per_year`
    or more (the direction is in the detail). Not defined under `min_common_days`: a few weeks
    of two books differ by chance."""
    common = stage1.index.intersection(stage2.index)
    if len(common) < line.need("min_common_days"):
        return _undefined(line.name, f"общих дней меньше {line.need('min_common_days'):g}: "
                                     "на таком отрезке две книги расходятся и случайно")
    a, b = stage1.loc[common], stage2.loc[common]
    total_a, total_b = float((1 + a).prod() - 1), float((1 + b).prod() - 1)
    days = len(common)
    ann_a, ann_b = _annual(total_a, days), _annual(total_b, days)
    gap = ann_a - ann_b
    corr = _num(a.corr(b)) if a.std() > 0 and b.std() > 0 else None
    return FlagResult(
        name=line.name, state=_state(abs(gap) >= line.need("gap_per_year")), value=float(gap),
        line=line.need("gap_per_year"),
        detail={"days": days, "start": str(common.min().date()), "end": str(common.max().date()),
                "stage1_total": total_a, "stage2_total": total_b, "stage1_ann": ann_a,
                "stage2_ann": ann_b, "daily_correlation": corr})


# --------------------------------------------------------------------------
# 6. The stand against the paper book


def stand_vs_paper(*, paper_return: float | None, stand_return: float | None,
                   replay_return: float | None, comparable: bool, line: FlagLine) -> FlagResult:
    """The stand's forward total against the production paper book's over ONE window
    (`qlab.paper_check`), as a gap in total return. Fires at `gap` (absolute). Not defined when
    the idea has no paper book, the stand has no total, or the two windows differ."""
    if paper_return is None:
        return _undefined(line.name, "у идеи нет бумажной книги")
    if stand_return is None:
        return _undefined(line.name, "у стенда нет итога за отрезок бумаги")
    if not comparable:
        return _undefined(line.name, "отрезки стенда и бумаги разные: числа сравнивать нельзя")
    gap = stand_return - paper_return
    return FlagResult(
        name=line.name, state=_state(abs(gap) >= line.need("gap")), value=float(gap),
        line=line.need("gap"),
        detail={"paper_return": paper_return, "stand_return": stand_return,
                "replay_return": replay_return})


# --------------------------------------------------------------------------
# 7. Better than the strategy's own family


def too_good_for_family(value: float | None, others: Sequence[float], line: FlagLine
                        ) -> FlagResult:
    """The run's yearly return against the same figure of the other ideas of its family (the
    latest judged run of each, `qlab.registry.families`). Fires beyond the far-out fence of the
    family's spread: `Q3 + fence_iqr * (Q3 - Q1)` (Tukey's far-out fence at 3). Not defined with
    fewer than `min_members` other ideas: a spread of three numbers says nothing."""
    if value is None:
        return _undefined(line.name, "у прогона нет годовой доходности")
    members = [float(v) for v in others if _num(v) is not None]
    if len(members) < line.need("min_members"):
        return _undefined(line.name, f"в семействе меньше {line.need('min_members'):g} других "
                                     "идей с прогоном")
    q1, q3 = np.percentile(members, [25, 75])
    fence = float(q3 + line.need("fence_iqr") * (q3 - q1))
    return FlagResult(
        name=line.name, state=_state(value > fence), value=float(value), line=fence,
        detail={"members": len(members), "q1": float(q1), "median": float(np.median(members)),
                "q3": float(q3), "max_other": float(max(members))})


__all__ = [
    "FIRES", "FIRES_RU", "FlagResult", "QUIET", "QUIET_RU", "TITLES_RU",
    "UNDEFINED", "UNDEFINED_RU", "concentrated_coin_days", "concentrated_instruments",
    "frozen_price", "implausible_day", "stand_vs_paper", "tail_move_coin_days",
    "too_good_for_family", "unlisted_on_execution_venue", "untradeable_coin_days",
    "venue_divergence"]
