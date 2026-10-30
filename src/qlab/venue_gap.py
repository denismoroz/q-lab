"""Why one venue passed an idea and the other did not (docs/VENUE_SPLIT.md, "Почему площадки
расходятся").

Owner, 2026-10-30: «смотрим обе площадки независимо. проходить если где-то. если проходит одна но
не проходит другая нужно чтобы была причина почему не проходит». When one venue passed and the
other did not, the page and the morning report say why, and the words are written by CODE from
measurements, never by a model. A reading: nothing here decides a verdict or moves a status.

The reason has a fixed shape:

* **the rule that failed**, with its numbers (`net_edge_positive` -3.1%; the noise bar 87% against
  a 99% line, and the bar's power, which says whether that failure can mean anything);
* **the causes that apply**, from a CLOSED list (`CAUSES`), each with its numbers:
  `unlisted_coins` -- the share of the profit made on coins the other venue does not list, in both
  directions; `windows` -- the days each venue was judged on; `costs` -- the basis points each
  venue charges; `money` -- the capital each run needed against the strategy's own money;
  `no_run` -- the venue was not run, with the reason the ladder names; `history_only` -- a pass
  that rests only on the check against Binance's history, because the noise bar had no power;
  `recent_year` -- a venue whose latest year lost money;
* a cause that cannot be measured is still listed, as «причина не измерена» and why, never left out.

The profit share needs both runs taken apart coin by coin (`qlab.anomalies`), which re-runs a
strategy, so it is measured by `measure` (the night's stage, `qlab venue-gap run`), stored in
`data/venue_gap/<idea>.json` next to the two trials it was measured on, and only READ when a page is
built. A page built before the number exists says «причина не измерена: расчёт ещё не сделан».
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

STORE = Path("data/venue_gap")
CAUSES = ("unlisted_coins", "windows", "costs", "money", "no_run", "history_only", "recent_year")
"""The closed list of causes (a new one is a line here, in docs/VENUE_SPLIT.md and a test)."""
NOT_MEASURED = "причина не измерена"
SHARE_LINE = 0.2
"""From this share of the profit on coins the other venue does not list, the cause is named first
in the short line. Only the order of words in a sentence: no verdict reads it."""
MIN_POWER = 0.8
"""The noise bar's strength below which its failure means «рано судить» (rules 2026-10-29.2)."""
MINUS = "−"


# --------------------------------------------------------------------------
# What is known about one venue's run


@dataclass(frozen=True)
class FailedRule:
    """A rule of a run that did not pass: its id, the metric it reads, the value and the line."""

    rule_id: str
    metric: str
    value: float | None
    comparator: str | None
    threshold: float | None


@dataclass(frozen=True)
class VenueFacts:
    """What the ladder and the registry know of one venue of one idea, in plain data."""

    venue: str
    title: str
    outcome: str
    """`passed`, `failed`, `undecided` or `not_run` (`qlab.ladder.Outcome`)."""
    route: str | None = None
    run_idea: str | None = None
    trial_id: int | None = None
    data_start: str | None = None
    data_end: str | None = None
    forward_days: float | None = None
    forward_days_needed: float | None = None
    noise_percentile: float | None = None
    noise_power: float | None = None
    consistency_trial: int | None = None
    """The first-stage trial a pass rested on instead of the noise bar (`stage1_trial_id`)."""
    taker_fee_bps: float | None = None
    slippage_bps: float | None = None
    money_outcome: str | None = None
    money_detail: str | None = None
    failed_rules: tuple[FailedRule, ...] = ()
    not_run_reason: str | None = None
    not_run_detail: str | None = None
    years: tuple[tuple[int, float, int], ...] = ()
    """(year, return on the strategy's own money, days) of the judged part, oldest first."""


@dataclass(frozen=True)
class ProfitShare:
    """The share of one venue's profit that came from coins the other venue does not list."""

    venue: str
    """The venue whose profit it is."""
    other: str
    window_from: str | None
    window_to: str | None
    total: float | None
    """The profit over the window (share of the strategy's capital, arithmetic)."""
    unlisted: float | None
    share: float | None
    """`unlisted / total`; None when the total is not positive (then it means nothing)."""
    top: tuple[tuple[str, float], ...] = ()
    """The unlisted coins that earned most: (coin, profit)."""
    coins_unlisted: int = 0
    coins_held: int = 0
    why_not: str | None = None
    """Why it could not be measured (then every number above is None)."""

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "top": [list(t) for t in self.top]}


@dataclass(frozen=True)
class Cause:
    kind: str
    text_ru: str
    measured: bool
    numbers: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text_ru": self.text_ru, "measured": self.measured,
                "numbers": dict(self.numbers)}


@dataclass(frozen=True)
class VenueGap:
    passed: str
    """The venue that passed (`binance` / `hyperliquid`); for an uneven pair, the first of them."""
    other: str
    other_outcome: str
    rule_ru: str | None
    causes: tuple[Cause, ...]
    short_ru: str
    inputs: tuple = field(default=(), compare=False, repr=False)
    """The two `VenueFacts` the reason was made of (passed, other), for the stage that measures."""
    kind: str = "verdict"
    """`verdict`: one venue passed and the other did not. `uneven`: both passed the rules but not
    on equal terms (the money fits on one only, or a pass rests on the history check alone)."""

    @property
    def text_ru(self) -> str:
        if self.kind == "uneven":
            head = "обе площадки прошли правила, но не на равных: "
        else:
            head = (f"{_title(self.other)} не прошла, хотя {_title(self.passed)} прошла: "
                    if self.other_outcome == "failed" else
                    f"{_title(self.other)} не решила, хотя {_title(self.passed)} прошла: "
                    if self.other_outcome == "undecided" else
                    f"{_title(self.other)} не проводилась, хотя {_title(self.passed)} прошла: ")
        parts = [self.rule_ru] if self.rule_ru else []
        parts += [c.text_ru for c in self.causes]
        return head + " ".join(p if p.endswith(".") else p + "." for p in parts)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "passed": self.passed, "other": self.other,
                "other_outcome": self.other_outcome, "rule_ru": self.rule_ru,
                "causes": [c.to_dict() for c in self.causes],
                "text_ru": self.text_ru, "short_ru": self.short_ru}


TITLES = {"binance": "Binance", "hyperliquid": "Hyperliquid"}


def _title(venue: str) -> str:
    return TITLES.get(venue, venue)


# --------------------------------------------------------------------------
# Words and numbers


def pct(value: float | None, digits: int = 1) -> str:
    """«−3,1%»: a fraction as a percent with a decimal comma and a true minus."""
    if value is None or value != value:
        return "—"
    text = f"{value * 100:.{digits}f}".replace(".", ",")
    return text.replace("-", MINUS) + "%"


def _num(value: float | None) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.4g}".replace(".", ",").replace("-", MINUS)


PERCENT_METRICS = ("return", "percentile", "share", "drawdown", "dd")


def _is_percent(metric: str) -> bool:
    return any(word in metric for word in PERCENT_METRICS)


def _value(metric: str, value: float | None) -> str:
    return pct(value) if _is_percent(metric) else _num(value)


COMPARATOR_RU = {">": "больше", ">=": "не меньше", "<": "меньше", "<=": "не больше",
                 "==": "равно", "!=": "не равно"}


def rule_text(rule: FailedRule, facts: VenueFacts) -> str:
    """One failed rule with its numbers, in plain words."""
    want = ("" if rule.comparator is None or rule.threshold is None else
            f", нужно {COMPARATOR_RU.get(rule.comparator, rule.comparator)} "
            f"{_value(rule.metric, rule.threshold)}")
    text = f"правило `{rule.rule_id}`: {_value(rule.metric, rule.value)}{want}"
    if "noise" in rule.metric or rule.rule_id == "shape_aware_edge":
        power = facts.noise_power
        if power is None:
            text += "; сила планки шума не измерена"
        elif power < MIN_POWER:
            text += (f"; сила планки {pct(power, 0)} — меньше {pct(MIN_POWER, 0)}, на такой "
                     "длине она ничего не различает, и её провал значит «рано судить», "
                     "а не «не работает»")
        else:
            text += f"; сила планки {pct(power, 0)} — планка здесь различает"
    return text


def window_days(facts: VenueFacts) -> int | None:
    if not facts.data_start or not facts.data_end:
        return None
    try:
        return (date.fromisoformat(facts.data_end[:10])
                - date.fromisoformat(facts.data_start[:10])).days
    except ValueError:
        return None


def _days(n: int) -> str:
    return f"{n:,}".replace(",", " ") + " дн."


def window_text(facts: VenueFacts) -> str:
    days = window_days(facts)
    if days is None:
        return f"{facts.title}: окно не записано"
    forward = ("" if facts.forward_days is None or facts.forward_days >= days
               else f", из них проверка вперёд {_days(int(facts.forward_days))}")
    return f"{facts.title} {facts.data_start[:10]} — {facts.data_end[:10]} ({_days(days)}{forward})"


def costs_text(facts: VenueFacts) -> str:
    if facts.taker_fee_bps is None or facts.slippage_bps is None:
        return f"{facts.title}: издержки не записаны"
    return (f"{facts.title} {_num(facts.taker_fee_bps)} + {_num(facts.slippage_bps)} б.п. "
            "(комиссия + проскальзывание на ногу)")


# --------------------------------------------------------------------------
# The profit made on coins the other venue does not list


def profit_share(net_by_coin: Mapping[str, float], lists_coin: Callable[[str], bool], *,
                 venue: str, other: str, window_from: str | None = None,
                 window_to: str | None = None, held: int | None = None) -> ProfitShare:
    """The share of `venue`'s profit earned on coins `other` does not list (`lists_coin` says
    whether it does). `net_by_coin` is each coin's net profit over the window, as a share of the
    strategy's capital (a coin's perpetual and spot legs already summed). The share is the
    unlisted coins' profit over the whole profit; with a total that is not positive it means
    nothing and is None, with the reason."""
    total = float(sum(net_by_coin.values()))
    unlisted = {coin: float(v) for coin, v in net_by_coin.items() if not lists_coin(coin)}
    gone = float(sum(unlisted.values()))
    top = tuple(sorted(((c, v) for c, v in unlisted.items()), key=lambda cv: -cv[1])[:3])
    held_n = len(net_by_coin) if held is None else held
    if total <= 0:
        return ProfitShare(venue, other, window_from, window_to, total, gone, None, top,
                           len(unlisted), held_n,
                           why_not="за это окно прибыль не положительна, доля ничего не значит")
    return ProfitShare(venue, other, window_from, window_to, total, gone, gone / total, top,
                       len(unlisted), held_n)


def share_text(s: ProfitShare) -> str:
    """The profit share as one sentence about one direction."""
    own, other = _title(s.venue), _title(s.other)
    if s.why_not is not None and s.share is None and s.total is None:
        return f"прибыль {own} с монет, которых нет на {other}: {NOT_MEASURED}: {s.why_not}"
    window = "" if not s.window_from else f" за {s.window_from} — {s.window_to}"
    if s.share is None:
        return (f"прибыль {own}{window} не положительна ({pct(s.total, 2)} от денег стратегии): "
                f"доля монет, которых нет на {other}, ничего не значит")
    coins = ", ".join(f"{c} {pct(v, 2)}" for c, v in s.top if v > 0)
    best = f"; больше всего дали {coins}" if coins else ""
    return (f"{pct(s.share, 0)} прибыли {own}{window} — с монет, которых нет на {other} "
            f"({s.coins_unlisted} из {s.coins_held} монет книги){best}")


# --------------------------------------------------------------------------
# Putting the reason together


def _rule_ru(other: VenueFacts) -> str | None:
    if other.outcome == "not_run":
        return None
    if other.failed_rules:
        shown = [rule_text(r, other) for r in other.failed_rules[:3]]
        lead = ("не сошлось: " if other.outcome == "failed" else "не хватило: ")
        return lead + "; ".join(shown)
    if other.outcome == "undecided" and other.forward_days is not None \
            and other.forward_days_needed is not None:
        return (f"проверки вперёд {_days(int(other.forward_days))} из "
                f"{_days(int(other.forward_days_needed))} нужных: правила пока не решают")
    return NOT_MEASURED + ": прогон не записал ни одного непройденного правила"


def causes_of(passed: VenueFacts, other: VenueFacts, *, shares: Sequence[ProfitShare],
              shares_why: str | None) -> tuple[Cause, ...]:
    """The causes from the closed list that apply, in the order of the list."""
    out: list[Cause] = []
    # 1. the profit made on coins the other venue does not list, both directions
    if shares:
        for s in shares:
            out.append(Cause("unlisted_coins", share_text(s), s.share is not None or
                             s.total is not None, s.to_dict()))
    else:
        out.append(Cause(
            "unlisted_coins",
            f"прибыль с монет, которых нет на другой площадке: {NOT_MEASURED}: "
            f"{shares_why or 'расчёт ещё не сделан'}", False, {}))
    # 2. different windows
    if passed.data_start and other.data_start:
        out.append(Cause("windows", "окна разные: " + "; ".join(
            window_text(f) for f in (passed, other)), True,
            {f.venue: window_days(f) for f in (passed, other)}))
    elif other.outcome != "not_run":
        out.append(Cause("windows", f"окна разные: {NOT_MEASURED}: даты прогона не записаны",
                         False))
    # 3. different costs
    if passed.taker_fee_bps is not None and other.taker_fee_bps is not None:
        same = (passed.taker_fee_bps, passed.slippage_bps) == (other.taker_fee_bps,
                                                               other.slippage_bps)
        out.append(Cause("costs", ("издержки одинаковые: " if same else "издержки разные: ")
                         + "; ".join(costs_text(f) for f in (passed, other)), True,
                         {f.venue: [f.taker_fee_bps, f.slippage_bps] for f in (passed, other)}))
    elif other.outcome != "not_run":
        out.append(Cause("costs", f"издержки: {NOT_MEASURED}: в записи прогона их нет", False))
    # 4. the money check, per venue
    lines = [f"{f.title}: {f.money_detail}" for f in (passed, other)
             if f.money_detail and f.money_outcome not in (None, "not_run")]
    if lines:
        out.append(Cause("money", "деньги стратегии: " + "; ".join(lines), True,
                         {f.venue: f.money_outcome for f in (passed, other)}))
    # 5. a pass that rests on the history check alone
    for f in (passed, other):
        if f.outcome == "passed" and f.consistency_trial is not None:
            power = ("сила не измерена" if f.noise_power is None
                     else f"сила {pct(f.noise_power, 0)}")
            out.append(Cause(
                "history_only", f"на {f.title} прохождение держится только на сверке с историей "
                f"Binance (прогон {f.consistency_trial}), планка шума там не решает ({power})",
                True, {"venue": f.venue, "noise_power": f.noise_power,
                       "consistency_trial": f.consistency_trial}))
    # 6. a venue whose latest year lost money
    for f in (passed, other):
        if f.years and f.years[-1][1] < 0:
            year, ret, days = f.years[-1]
            out.append(Cause("recent_year", f"на {f.title} последний год {year} ({_days(days)}) "
                             f"{pct(ret)} от денег стратегии", True,
                             {"venue": f.venue, "year": year, "return": ret, "days": days}))
    # 7. no run on that venue, with its reason
    if other.outcome == "not_run":
        why = other.not_run_reason or NOT_MEASURED
        detail = f" ({other.not_run_detail})" if other.not_run_detail else ""
        out.append(Cause("no_run", f"на {other.title} прогона нет: {why}{detail}", True,
                         {"reason": other.not_run_reason}))
    return tuple(out)


def short_line(gap_rule: str | None, other: VenueFacts, causes: Sequence[Cause],
               passed: VenueFacts) -> str:
    """One short line for a table cell: the failed rule's numbers and the heaviest cause."""
    bits: list[str] = []
    if other.outcome == "not_run":
        bits.append(f"{other.title}: не проводилась ({other.not_run_reason or NOT_MEASURED})")
    elif other.failed_rules:
        r = other.failed_rules[0]
        bits.append(f"{other.title}: `{r.rule_id}` {_value(r.metric, r.value)}")
    else:
        bits.append(f"{other.title}: {NOT_MEASURED}")
    for c in causes:
        share = c.numbers.get("share") if c.kind == "unlisted_coins" else None
        if isinstance(share, int | float) and share >= SHARE_LINE:
            who = _title(str(c.numbers.get("venue")))
            bits.append(f"{pct(share, 0)} прибыли {who} — с монет без второй площадки")
            break
    return "; ".join(bits)


def compose(passed: VenueFacts, other: VenueFacts, *, shares: Sequence[ProfitShare] = (),
            shares_why: str | None = None, kind: str = "verdict") -> VenueGap:
    """The reason from plain facts (pure)."""
    causes = causes_of(passed, other, shares=shares, shares_why=shares_why)
    if kind == "uneven":
        # No failed rule to name: both passed. The unevenness itself is the first words.
        rule = None
        order = ("money", "history_only", "recent_year", "unlisted_coins", "windows", "costs")
        causes = tuple(sorted((c for c in causes if c.kind in order),
                              key=lambda c: order.index(c.kind)))
        return VenueGap(passed.venue, other.venue, other.outcome, rule, causes,
                        _uneven_short(passed, other), (passed, other), kind)
    rule = _rule_ru(other)
    return VenueGap(passed.venue, other.venue, other.outcome, rule, causes,
                    short_line(rule, other, causes, passed), (passed, other))


def _uneven_short(a: VenueFacts, b: VenueFacts) -> str:
    """The short line of an uneven pair: where the money does not fit, which pass rests on the
    history check alone, which venue's latest year lost money."""
    bits: list[str] = []
    fits = {f.money_outcome for f in (a, b) if f.money_outcome in ("passed", "failed")}
    if len(fits) > 1:
        bits += [f"деньги: {f.title} {f.money_detail}" for f in (a, b)
                 if f.money_outcome == "failed" and f.money_detail]
    for f in (a, b):
        if f.consistency_trial is not None:
            power = "" if f.noise_power is None else f" (сила планки {pct(f.noise_power, 0)})"
            bits.append(f"{f.title} — только сверкой с историей{power}")
    for f in (a, b):
        if f.years and f.years[-1][1] < 0:
            bits.append(f"{f.title} {f.years[-1][0]} {pct(f.years[-1][1])}")
    return "; ".join(bits) or "площадки не равны"


def applies(a: VenueFacts, b: VenueFacts) -> tuple[VenueFacts, VenueFacts] | None:
    """(the venue that passed, the other) when exactly one venue passed, else None."""
    if (a.outcome == "passed") == (b.outcome == "passed"):
        return None
    return (a, b) if a.outcome == "passed" else (b, a)


def uneven(a: VenueFacts, b: VenueFacts) -> bool:
    """Both venues passed the rules but not on equal terms: the strategy's money fits on one and
    not on the other, or a pass rests only on the history check (the noise bar had no power)."""
    if a.outcome != "passed" or b.outcome != "passed":
        return False
    fits = {f.money_outcome for f in (a, b) if f.money_outcome in ("passed", "failed")}
    return len(fits) > 1 or any(f.consistency_trial is not None for f in (a, b))


# --------------------------------------------------------------------------
# The measurement (re-runs the two trials; stored, read by the pages)


def store_path(idea_id: str, root: Path | None = None) -> Path:
    return (root or Path(".")) / STORE / f"{idea_id}.json"


def load_stored(idea_id: str, trials: tuple[int | None, int | None], root: Path | None = None
                ) -> tuple[list[ProfitShare], str | None]:
    """(the stored shares, why there are none): a file measured on exactly these two trials."""
    path = store_path(idea_id, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], "расчёт ещё не сделан, его делает ночь (`qlab venue-gap run`)"
    if data.get("trials") != list(trials):
        return [], "расчёт сделан на прежних прогонах, ночь его обновит"
    shares = [ProfitShare(**{**s, "top": tuple((c, v) for c, v in s.get("top", []))})
              for s in data.get("shares", [])]
    return shares, data.get("why")


def save_stored(idea_id: str, trials: tuple[int | None, int | None],
                shares: Sequence[ProfitShare], why: str | None, root: Path | None = None) -> Path:
    path = store_path(idea_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trials": list(trials), "why": why,
                                "shares": [s.to_dict() for s in shares]},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _net_by_coin(session, trial_id: int, since: str | None) -> tuple[dict[str, float], list[str],
                                                                      str, str]:
    """(each coin's net profit over the judged rows from `since`, the columns of the run's
    panel, first day, last day), the strategy's stored book run again and taken apart."""
    from qlab.anomalies.rerun import daily, decompose, load_run
    from qlab.anomalies.scan import by_coin

    run = load_run(session, trial_id)
    dec = decompose(run)
    net = by_coin(daily(dec.net, "sum"))
    if since:
        net = net[net.index >= _utc(since)]
    totals = net.sum()
    first = str(net.index.min().date()) if len(net) else ""
    last = str(net.index.max().date()) if len(net) else ""
    columns = [str(c) for c in run.panel.prices.columns]
    return {str(c): float(v) for c, v in totals.items()}, columns, first, last


def needs_measure(idea_id: str, trials: tuple[int | None, int | None],
                  root: Path | None = None) -> bool:
    """Whether the stored measurement is missing or was made on other trials."""
    try:
        data = json.loads(store_path(idea_id, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return data.get("trials") != list(trials)


def _utc(day: str):
    import pandas as pd

    return pd.Timestamp(day, tz="UTC")


def measure(session, idea_id: str, passed: VenueFacts, other: VenueFacts,
            root: Path | None = None) -> tuple[list[ProfitShare], str | None]:
    """The profit share in both directions over the days both venues were judged on, from the two
    trials; stored for the pages. Never raises: a trial that cannot be re-run is a stated reason."""
    trials = (passed.trial_id, other.trial_id)
    if None in trials:
        why = "одна из площадок не проводилась: сравнивать монеты не с чем"
        save_stored(idea_id, trials, [], why, root)
        return [], why
    try:
        shares, why = _measure(session, passed, other), None
    except Exception as exc:  # noqa: BLE001 - a stated reason, not a failed page
        shares, why = [], f"прогон не удалось повторить: {type(exc).__name__}: {exc}"
    save_stored(idea_id, trials, shares, why, root)
    return shares, why


def _measure(session, passed: VenueFacts, other: VenueFacts) -> list[ProfitShare]:
    from qlab.pipeline.stage2 import venue_name

    starts = [f.data_start[:10] for f in (passed, other) if f.data_start]
    since = max(starts) if starts else None
    sides = {}
    for facts in (passed, other):
        net, columns, first, last = _net_by_coin(session, int(facts.trial_id), since)
        sides[facts.venue] = (net, columns, first, last)
    out = []
    for facts, rival in ((passed, other), (other, passed)):
        net, _, first, last = sides[facts.venue]
        rival_columns = sides[rival.venue][1]
        out.append(profit_share(
            net, lambda coin, cols=rival_columns: venue_name(coin, cols) is not None,
            venue=facts.venue, other=rival.venue, window_from=first or None,
            window_to=last or None,
            held=len(net)))
    return out


def venue_gap_stage(day: date, end: date, state: dict, *, only: Sequence[str] | None = None,
                    max_per_night: int = 8, stop_at=None, root: Path | None = None) -> None:
    """The night's stage: measure, for every open idea whose two venues differ, the profit share
    on coins the other venue does not list, and store it for the pages (`measure`). Code only, no
    agent, no tokens. At most `max_per_night` ideas a night (the oldest measurements are the ones
    missing); the rest wait and the state says how many. `only` names ideas to measure now."""
    from qlab import ladder, night
    from qlab.registry.db import session_scope

    block = state.setdefault("venue_gap", {"entries": [], "pending": 0})
    block["entries"], block["pending"] = [], 0
    ledgers = ladder.load_ledgers(root or Path("."))
    with session_scope() as session:
        ids = list(only) if only else ladder.open_idea_ids(session)
        todo = []
        for idea_id in ids:
            found = ladder.build_ladder(session, idea_id, ledgers)
            gap = None if found is None else found.venue_gap
            if gap is None or len(gap.inputs) != 2:
                continue
            passed, other = gap.inputs
            trials = (passed.trial_id, other.trial_id)
            if only or needs_measure(idea_id, trials, root):
                todo.append((idea_id, passed, other))
        for idea_id, passed, other in todo:
            if len(block["entries"]) >= max_per_night:
                block["pending"] += 1
                continue
            if stop_at is not None and night.past_stop(stop_at, state, f"venue_gap:{idea_id}"):
                block["pending"] += 1
                continue
            shares, why = measure(session, idea_id, passed, other, root)
            block["entries"].append({
                "idea": idea_id, "measured": bool(shares), "why": why,
                "shares": {s.venue: s.share for s in shares}})
    night.save_state(day, state)


__all__ = ["venue_gap_stage", "needs_measure", "uneven", "CAUSES", "Cause", "FailedRule",
           "NOT_MEASURED", "ProfitShare", "VenueFacts",
           "VenueGap", "applies", "causes_of", "compose", "costs_text", "load_stored", "measure",
           "pct", "profit_share", "rule_text", "save_stored", "share_text", "short_line",
           "store_path", "window_days", "window_text"]
