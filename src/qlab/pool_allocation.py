"""How the owner's deployable capital could be split between the strategies of the pool.

T55, part 2 (docs/PLAN.md, M6; docs/TASKS.md). CODE PROPOSES, THE OWNER DECIDES: this module
computes a split and says why. It never moves money, never changes a status, never writes to
production and never calls an agent. The method, its sources and its parameters are declared in
`night/allocation.yaml` and were preregistered before any allocation was computed on real data
(`seed/preregistration/2026-10-25-capital-allocation.yaml`).

The method is inverse volatility ("simple risk parity", Asness-Frazzini-Pedersen 2012): a
strategy's weight is proportional to 1 / the standard deviation of its daily returns on ITS OWN
money. Nothing else is estimated: no expected return (weeks of returns say nothing about
profitability -- PREDICTION 0 of seed/preregistration/2026-09-21-paper-bv2-trend.yaml) and no
correlation matrix (DeMiguel-Garlappi-Uppal 2009: estimation error outweighs the gain on short
histories). Correlations are reported over the common window, not used.

Inputs, one strategy = one root of the registry's parent chain:

* `live` / `paper` -- a book's equity from the night's export of the production host, REALISED
  part only (from `realised_from`), on the book's `strategy_capital_usd`. Never an account's
  balance: the exchange account's free cash and transfers belong to no strategy (owner,
  2026-10-25; `qlab.strategy_pnl`).
* `stand_forward` -- the stand's run of a strategy that has no book yet, from the spec's
  `params_fixed_at` on. A strategy whose evidence is the fit period (`in_sample`) is listed and
  gets no weight, with that reason.

Constraints: the deployable capital, `max_active_strategies` of night/limits.yaml, and each
strategy's measured minimum capital (`min_capital_usd` of its latest trial): a strategy whose
share would be smaller than its minimum gets zero and the others are re-split.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qlab import decay, paper_rule
from qlab.paper_check import equity_curve

CONFIG = Path("night/allocation.yaml")
NIGHT_DIR = Path("data/night")

METHOD = "inverse_volatility"
EVIDENCE_KINDS = ("live", "paper", "stand_forward", "in_sample")
TERMINAL_STATUSES = ("rejected", "decayed", "retired")

SECTION_TITLE = "Распределение капитала (предложение кода; решает владелец)"


class MoneyError(ValueError):
    """A series that is not the strategy's own money (account money, or no stated capital)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> dict:
    """`night/allocation.yaml`, validated: the method must be the one this version knows
    (no silent fallback), the minimum days at least 2, and the capital positive."""
    config = yaml.safe_load((path or CONFIG).read_text(encoding="utf-8")) or {}
    if config.get("method") != METHOD:
        raise ValueError(f"allocation method must be {METHOD!r}, got {config.get('method')!r}")
    days = config.get("min_evidence_days")
    if not isinstance(days, int) or isinstance(days, bool) or days < 2:
        raise ValueError(f"min_evidence_days must be an integer of at least 2, got {days!r}")
    capital = config.get("deployable_capital_usd")
    if not isinstance(capital, int | float) or isinstance(capital, bool) or capital <= 0:
        raise ValueError(f"deployable_capital_usd must be a positive number, got {capital!r}")
    short = config.get("short_evidence") or {}
    if short.get("live") not in ("hold_current", "no_weight") or short.get("other") != "no_weight":
        raise ValueError("short_evidence needs live: hold_current|no_weight and other: no_weight")
    if config.get("on_decay_alert") not in ("report_only", "exclude"):
        raise ValueError("on_decay_alert must be report_only or exclude")
    config.setdefault("capital_ladder_usd", [])
    return config


def max_strategies(config: dict) -> int | None:
    """`max_active_strategies` of the limits file the config points to; None without one."""
    path = Path(config.get("limits_file") or "night/limits.yaml")
    if not path.is_file():
        return None
    value = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("max_active_strategies")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# Series: strategy money only
# ---------------------------------------------------------------------------


def assert_strategy_money(idea_id: str, cfg: dict) -> None:
    """Refuse a book whose series is not on the strategy's own money: it must state
    `strategy_capital_usd`, and a live book's equity must be built by `qlab.strategy_pnl`
    (the exchange account's total equity includes free cash and every transfer)."""
    capital = cfg.get("strategy_capital_usd")
    if not isinstance(capital, int | float) or isinstance(capital, bool) or capital <= 0:
        raise MoneyError(f"{idea_id}: no strategy_capital_usd; a return is only taken on the "
                         "strategy's own money")
    if cfg.get("live") and cfg.get("equity_from") != decay.STRATEGY_PNL:
        raise MoneyError(f"{idea_id}: a live book's series must be equity_from: "
                         f"{decay.STRATEGY_PNL}, never the account's own equity")


def daily_returns(equity: pd.Series) -> pd.Series:
    """Daily returns of an equity curve on its own base: the last value of each UTC day, COMPLETE
    days only (a day is complete once the curve reaches the next midnight), the return of a day
    on the previous day's close. Indexed by the day."""
    if equity.empty:
        return pd.Series(dtype=float)
    closes = decay.daily_closes(equity)
    complete = closes[closes.index + pd.Timedelta(days=1) <= equity.index[-1]]
    return complete.pct_change().dropna()


def daily_std(returns: pd.Series) -> float | None:
    """Sample standard deviation (ddof 1) of daily returns; None for fewer than two returns."""
    if len(returns) < 2:
        return None
    return float(np.std(returns.to_numpy(dtype=float), ddof=1))


@dataclass(frozen=True)
class Evidence:
    """What a strategy's weight rests on: daily returns on its own money, and what kind of
    evidence they are (`EVIDENCE_KINDS`)."""

    kind: str
    returns: pd.Series
    source: str

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.kind!r}")

    @property
    def days(self) -> int:
        return len(self.returns)

    @property
    def first(self) -> date | None:
        return self.returns.index[0].date() if len(self.returns) else None

    @property
    def last(self) -> date | None:
        return self.returns.index[-1].date() if len(self.returns) else None


@dataclass
class Strategy:
    """One strategy of the pool as the allocation sees it. `idea_id` is the variant that stands
    for the strategy `root`; `current_capital_usd` is a live strategy's money now."""

    root: str
    idea_id: str
    label: str
    role: str  # "live" | "paper" | "stand"
    evidence: Evidence | None
    min_capital_usd: float | None
    current_capital_usd: float | None = None
    other_books: list[str] = field(default_factory=list)
    decay_alert: str | None = None
    decay_percentile: float | None = None
    paper_ready: bool | None = None
    excluded: str | None = None  # a plain reason the strategy takes no part at all


# ---------------------------------------------------------------------------
# The allocation (pure)
# ---------------------------------------------------------------------------


@dataclass
class Line:
    """One strategy's row of the proposal."""

    root: str
    idea_id: str
    label: str
    role: str
    status: str  # "weighted" | "held" | "zero"
    evidence_kind: str | None
    days: int
    first: str | None
    last: str | None
    sigma: float | None
    dollars: float
    share: float
    equal_dollars: float | None
    risk_share: float | None
    equal_risk_share: float | None
    current_capital_usd: float | None
    min_capital_usd: float | None
    reason: str
    fits_from_usd: float | None = None
    ready_on: str | None = None
    decay_alert: str | None = None
    decay_percentile: float | None = None
    paper_ready: bool | None = None
    other_books: list[str] = field(default_factory=list)


@dataclass
class Proposal:
    version: str
    capital: float
    pool: float
    held: float
    min_days: int
    lines: list[Line]
    correlations: list[dict]
    correlation_note: str
    ladder: dict[str, dict[str, float]]
    needs_decision: list[str]
    start_capital: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def inverse_volatility(sigmas: dict[str, float]) -> dict[str, float]:
    """Weights proportional to 1/sigma, summing to 1. Every sigma must be positive."""
    if not sigmas:
        return {}
    if any(not s > 0 for s in sigmas.values()):
        raise ValueError("inverse volatility needs every volatility above zero")
    inverse = {k: 1.0 / s for k, s in sigmas.items()}
    total = sum(inverse.values())
    return {k: v / total for k, v in inverse.items()}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _usd(x: float) -> str:
    return f"${x:,.0f}".replace(",", " ")


def _split(pool: float, sigmas: dict[str, float], minimums: dict[str, float]
           ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Inverse-volatility dollars with the minimum-capital filter: while some strategy's dollars
    are below its minimum, drop the one with the lowest dollars-to-minimum ratio and re-split.
    Returns (dollars of the survivors, dollars each dropped strategy had when it was dropped,
    the total pool at which each dropped strategy would fit together with the survivors)."""
    alive = dict(sigmas)
    dropped_at: dict[str, float] = {}
    while alive:
        weights = inverse_volatility(alive)
        dollars = {k: pool * w for k, w in weights.items()}
        short = {k: dollars[k] / minimums[k] for k in alive if dollars[k] < minimums[k]}
        if not short:
            break
        worst = min(short, key=lambda k: (short[k], k))
        dropped_at[worst] = dollars[worst]
        del alive[worst]
    final = ({k: pool * w for k, w in inverse_volatility(alive).items()} if alive else {})
    fits: dict[str, float] = {}
    for k in dropped_at:
        group = {**alive, k: sigmas[k]}
        weights = inverse_volatility(group)
        fits[k] = max(minimums[j] / weights[j] for j in group)
    return final, dropped_at, fits


def allocate(strategies: list[Strategy], *, capital: float, max_count: int | None,
             config: dict, with_ladder: bool = True) -> Proposal:
    """The proposal: who gets how many dollars, and for strategies that get none, why."""
    min_days = int(config["min_evidence_days"])
    hold_live = config["short_evidence"]["live"] == "hold_current"
    exclude_on_alert = config["on_decay_alert"] == "exclude"
    lines: dict[str, Line] = {}
    needs: list[str] = []

    def base(s: Strategy, status: str, reason: str, **kw) -> Line:
        ev = s.evidence
        line = Line(
            root=s.root, idea_id=s.idea_id, label=s.label, role=s.role, status=status,
            evidence_kind=ev.kind if ev else None, days=ev.days if ev else 0,
            first=ev.first.isoformat() if ev and ev.first else None,
            last=ev.last.isoformat() if ev and ev.last else None,
            sigma=daily_std(ev.returns) if ev else None, dollars=0.0, share=0.0,
            equal_dollars=None, risk_share=None, equal_risk_share=None,
            current_capital_usd=s.current_capital_usd, min_capital_usd=s.min_capital_usd,
            reason=reason, decay_alert=s.decay_alert, decay_percentile=s.decay_percentile,
            paper_ready=s.paper_ready, other_books=list(s.other_books))
        for key, value in kw.items():
            setattr(line, key, value)
        lines[s.root] = line
        return line

    eligible: dict[str, Strategy] = {}
    held: dict[str, Strategy] = {}
    for s in strategies:
        ev = s.evidence
        if s.excluded:
            base(s, "zero", s.excluded)
        elif ev is None or ev.days == 0:
            base(s, "zero", "для неё нет ряда доходности на её собственных деньгах: считать нечего")
        elif ev.kind == "in_sample":
            base(s, "zero", "про неё известно только на периоде подбора параметров; на нём любая "
                 "стратегия выглядит хорошо, поэтому доли ей не даётся — нужны дни, которых при "
                 "подборе не было")
        elif exclude_on_alert and s.decay_alert == "below":
            base(s, "zero", "детектор затухания отметил её, а владелец велел в таком случае "
                 "исключать: доли нет, остальным делится заново")
        elif ev.days < min_days:
            ready = ev.last + timedelta(days=min_days - ev.days)
            if s.role == "live" and hold_live and s.current_capital_usd:
                held[s.root] = s
                base(s, "held", f"данных пока {ev.days} из {min_days} дней — на таких данных "
                     f"настоящие деньги не двигаем: остаётся как есть, "
                     f"{_usd(s.current_capital_usd)}", ready_on=ready.isoformat())
            else:
                base(s, "zero", f"данных пока {ev.days} из {min_days} дней — для расчёта доли "
                     "мало; хватит примерно к " + ready.strftime("%d.%m"),
                     ready_on=ready.isoformat())
        elif (sigma := daily_std(ev.returns)) is None or not sigma > 0:
            base(s, "zero", "дневной разброс равен нулю (ряд не меняется): доля по нему не "
                 "считается")
        elif s.min_capital_usd is None:
            base(s, "zero", "стенд не измерил, с какой суммы её можно запустить: без этого числа "
                 "доля не даётся")
        else:
            eligible[s.root] = s

    held_total = sum(float(s.current_capital_usd) for s in held.values())
    pool = capital - held_total
    if max_count is not None and len(held) + len(eligible) > max_count:
        needs.append(f"Кандидатов на долю больше, чем лимит стратегий ({max_count}): "
                     f"{len(held) + len(eligible)}. Варианты: поднять лимит в `night/limits.yaml` "
                     "или назвать, какие оставить. Код сам не выбирает.")
        for s in eligible.values():
            base(s, "zero", "лимит стратегий превышен: владелец называет, какие оставить")
        eligible = {}
    if eligible and pool <= 0:
        needs.append(f"Деньги, уже стоящие в живых стратегиях ({_usd(held_total)}), не меньше "
                     f"всей суммы ({_usd(capital)}): делить нечего. Нужно решение владельца.")
        for s in eligible.values():
            base(s, "zero", "делить нечего: вся сумма уже занята живыми стратегиями")
        eligible = {}

    if eligible:
        sigmas = {r: float(daily_std(s.evidence.returns)) for r, s in eligible.items()}
        minimums = {r: float(s.min_capital_usd) for r, s in eligible.items()}
        final, dropped_at, fits = _split(pool, sigmas, minimums)
        for root, s in eligible.items():
            if root in final:
                continue
            base(s, "zero",
                 f"при таком делении ей досталось бы {_usd(dropped_at[root])}, а меньше "
                 f"{_usd(minimums[root])} её не запустить (минимум, который измерил стенд). Она "
                 f"войдёт в расчёт, когда всех денег будет от {_usd(fits[root] + held_total)}",
                 fits_from_usd=fits[root] + held_total)
        n = len(final)
        risk_equal = {r: sigmas[r] for r in final}
        total_equal = sum(risk_equal.values())
        weights = inverse_volatility({r: sigmas[r] for r in final}) if final else {}
        for root, dollars in final.items():
            s = eligible[root]
            base(s, "weighted", "доля обратно пропорциональна дневному разбросу: чем спокойнее "
                 "стратегия, тем больше ей достаётся, и каждая вносит в общий разброс "
                 "примерно поровну",
                 dollars=dollars, share=dollars / capital, equal_dollars=pool / n,
                 risk_share=1.0 / n if n else None,
                 equal_risk_share=risk_equal[root] / total_equal)
            assert abs(weights[root] * pool - dollars) < 1e-6 * max(pool, 1.0)
    for root, s in held.items():
        lines[root].dollars = float(s.current_capital_usd)
        lines[root].share = lines[root].dollars / capital

    ordered = [lines[s.root] for s in strategies]
    correlations, corr_note = correlation_table(strategies, lines, min_days)
    ladder: dict[str, dict[str, float]] = {}
    if with_ladder:
        for step in config.get("capital_ladder_usd") or []:
            sub = allocate(strategies, capital=float(step), max_count=max_count, config=config,
                           with_ladder=False)
            ladder[str(int(step))] = {ln.root: ln.dollars for ln in sub.lines}
    return Proposal(
        version=str(config.get("version")), capital=float(capital), pool=float(pool),
        held=float(held_total), min_days=min_days, lines=ordered, correlations=correlations,
        correlation_note=corr_note, ladder=ladder, needs_decision=needs,
        start_capital=start_capital(config))


def start_capital(config: dict) -> float | None:
    """The money any strategy starts with (`start_capital_usd`), or None when unset."""
    value = config.get("start_capital_usd")
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"start_capital_usd must be a positive number, got {value!r}")
    return float(value)


def start_lines(p: Proposal) -> list[str]:
    """Who could start with the owner's starting sum: a strategy whose measured minimum fits in
    it, ready or not yet by the paper rule. A live strategy keeps what it has."""
    if p.start_capital is None:
        return []
    start = _usd(p.start_capital)
    out = [f"**Старт — {start} на любую стратегию** (решение владельца 26.10: «будет начинать "
           f"любая стратегия с 300»). Кто может стартовать:"]
    for ln in p.lines:
        if ln.role == "live":
            out.append(f"    {ln.label}: уже на настоящих деньгах, "
                       f"{_usd(ln.current_capital_usd or ln.dollars)} — не трогается.")
        elif ln.min_capital_usd is None:
            out.append(f"    {ln.label}: минимум, с которого её можно запустить, не измерен — "
                       "сказать нельзя.")
        elif ln.min_capital_usd > p.start_capital:
            out.append(f"    {ln.label}: в {start} не помещается — нужно от "
                       f"{_usd(ln.min_capital_usd)}.")
        elif ln.paper_ready:
            out.append(f"    {ln.label}: помещается (минимум {_usd(ln.min_capital_usd)}) и по "
                       "правилу бумаги готова — можно стартовать, решает владелец.")
        else:
            out.append(f"    {ln.label}: помещается (минимум {_usd(ln.min_capital_usd)}), но "
                       "по правилу бумаги ещё рано.")
    return [*out, "", "Ниже — как делились бы деньги, когда стратегий на деньгах станет "
            "несколько и суммы вырастут.", ""]


def correlation_table(strategies: list[Strategy], lines: dict[str, Line], min_days: int
                      ) -> tuple[list[dict], str]:
    """Pairwise correlations of daily returns over each PAIR's common window, for the strategies
    that have a real series, shown only where the common window has `min_days` days. Reported,
    never used in the weights."""
    usable = [s for s in strategies
              if s.evidence and s.evidence.kind != "in_sample" and s.evidence.days]
    out: list[dict] = []
    hidden = 0
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            joined = pd.concat([a.evidence.returns, b.evidence.returns], axis=1, join="inner")
            if len(joined) < min_days:
                hidden += 1
                continue
            out.append({"a": a.root, "b": b.root, "n": len(joined),
                        "first": joined.index[0].date().isoformat(),
                        "last": joined.index[-1].date().isoformat(),
                        "correlation": float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))})
    note = ""
    if hidden:
        note = (f"пар с общим окном короче {min_days} дней: {hidden} — связь на таком окне "
                "случайна, не показана")
    return out, note


# ---------------------------------------------------------------------------
# What changed since the previous proposal
# ---------------------------------------------------------------------------


def snapshot(proposal: Proposal) -> dict[str, dict]:
    """What the next night compares with: per strategy its status, whether it has enough days,
    the decay flag and the paper rule's `ready`."""
    return {ln.root: {"status": ln.status, "enough_days": ln.days >= proposal.min_days,
                      "decay_alert": ln.decay_alert, "paper_ready": ln.paper_ready}
            for ln in proposal.lines}


def changes_since(previous: dict[str, dict] | None, proposal: Proposal) -> list[str]:
    """The re-split triggers that fired since the previous proposal (preregistered list):
    a strategy enters or leaves the pool, its evidence reaches the minimum days, the paper rule
    passes it, the decay detector flags it. Weights drifting by themselves are not a trigger."""
    if previous is None:
        return ["Это первый расчёт: сравнивать с прошлой ночью не с чем."]
    out: list[str] = []
    now = snapshot(proposal)
    labels = {ln.root: ln.label for ln in proposal.lines}
    for root, cur in now.items():
        old = previous.get(root)
        label = labels[root]
        if old is None:
            out.append(f"{label}: появилась в пуле — доли пересчитаны.")
            continue
        if cur["enough_days"] and not old.get("enough_days"):
            out.append(f"{label}: набрала {proposal.min_days} дней данных — теперь ей "
                       "считается доля.")
        if cur["paper_ready"] and not old.get("paper_ready"):
            out.append(f"{label}: по правилу бумаги книга готова к решению владельца.")
        if cur["decay_alert"] and not old.get("decay_alert"):
            out.append(f"{label}: детектор затухания отметил стратегию.")
    for root in previous:
        if root not in now:
            out.append(f"{root}: ушла из пула (статус или ряд пропали) — доли пересчитаны.")
    return out or ["С прошлой ночи ничего из списка поводов не случилось."]


# ---------------------------------------------------------------------------
# Report text (Russian, plain words)
# ---------------------------------------------------------------------------

_ROLE = {"live": "настоящие деньги", "paper": "бумага", "stand": "только стенд"}


def _day(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%d.%m") if iso else "—"


def _days_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "день"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "дня"
    return "дней"


def _evidence_text(ln: Line) -> str:
    kind = {"live": "настоящие деньги", "paper": "бумага", "stand_forward": "проверка вперёд на "
            "стенде", "in_sample": "период подбора"}.get(ln.evidence_kind or "", "нет ряда")
    if not ln.days:
        return kind
    return f"{ln.days} {_days_word(ln.days)} ({_day(ln.first)}–{_day(ln.last)}; {kind})"


def _decay_text(ln: Line) -> str | None:
    if ln.decay_percentile is None:
        return None
    text = (f"Сверка с историей стенда: результат хуже, чем у {ln.decay_percentile:.0f} из 100 "
            "таких же отрезков")
    if ln.decay_alert:
        return text + "; детектор затухания отметил стратегию."
    return text + "; порог тревоги владелец не задал, на доли это не влияет."


def _held_warning(p: Proposal) -> list[str]:
    """If a strategy now held for lack of days would, by the same formula, take most of the pool,
    say so: that is where a ceiling decision matters."""
    sigmas = {ln.root: ln.sigma for ln in p.lines
              if ln.status in ("weighted", "held") and ln.sigma}
    if len(sigmas) < 2:
        return []
    weights = inverse_volatility(sigmas)
    out = []
    for ln in p.lines:
        if ln.status == "held" and weights.get(ln.root, 0) > 0.5:
            out.append(f"    Потолок на стратегию особенно важен для «{ln.label}»: по имеющимся "
                       f"{ln.days} дням её разброс так мал, что когда наберётся {p.min_days}, "
                       f"формула отдаст ей около {_pct(weights[ln.root])} общей суммы, если "
                       "потолка нет.")
    return out


def report_lines(proposal: Proposal, changes: list[str]) -> list[str]:
    """The proposal as the morning report's section."""
    p = proposal
    out = [
        "Код ничего не двигает и статусов не меняет: это расчёт «как можно поделить деньги», "
        "решение и перевод денег — за владельцем.",
        "",
        *start_lines(p),
        f"Всего денег для стратегий: {_usd(p.capital)} — рабочее допущение (так считает ночной "
        "прогон), владелец не называл эту сумму; кто из стратегий помещается при других суммах — "
        "в конце раздела.",
        "",
        "Как считается. Дневной разброс — насколько день в среднем отличается от обычного дня "
        "(стандартное отклонение дневной доходности на деньгах самой стратегии, а не на счёте "
        "биржи). Доля стратегии обратно пропорциональна её разбросу: спокойным достаётся больше, "
        "беспокойным меньше, и каждая вносит в общий разброс примерно поровну. Ожидаемая "
        "доходность в расчёте не участвует: за недели её не отличить от случайности (запись "
        "заранее от 21.09). Корреляции (насколько две стратегии ходят вместе) не оцениваются "
        "для долей: на коротком ряду они ошибаются сильнее, чем помогают "
        f"(DeMiguel, Garlappi, Uppal, 2009). Метод и источники — `night/allocation.yaml`, версия "
        f"{p.version}. Для сравнения показано деление поровну. Меньше {p.min_days} полных дней "
        "данных (минимум владельца — 4 недели) на расчёт доли не хватает.",
        "",
    ]
    for ln in p.lines:
        out.append(f"**{ln.label}** — {_ROLE[ln.role]}; данные: {_evidence_text(ln)}")
        if ln.status == "weighted":
            text = (f"    Предложено {_usd(ln.dollars)} ({_pct(ln.share)} от всей суммы). "
                    f"Дневной разброс {ln.sigma:.2%}. ")
            if sum(1 for x in p.lines if x.status == "weighted") > 1:
                text += (f"Поровну было бы {_usd(ln.equal_dollars)}, но тогда на неё пришлось бы "
                         f"{_pct(ln.equal_risk_share)} общего разброса вместо "
                         f"{_pct(ln.risk_share)}. ")
            out.append(text + f"Минимум, с которого её можно запустить, "
                       f"{_usd(ln.min_capital_usd)} — помещается.")
        elif ln.status == "held":
            seen = (f" Дневной разброс по этим дням {ln.sigma:.3%} — как справка."
                    if ln.sigma else "")
            out.append(f"    Остаётся {_usd(ln.dollars)} без изменений ({_pct(ln.share)}). "
                       f"{ln.reason[0].upper()}{ln.reason[1:]}.{seen}")
        else:
            out.append(f"    Доля: 0. {ln.reason[0].upper()}{ln.reason[1:]}.")
        if ln.status == "weighted" and ln.current_capital_usd:
            out.append(f"    Сейчас у стратегии {_usd(ln.current_capital_usd)}.")
        if text := _decay_text(ln):
            out.append(f"    {text}")
        if ln.paper_ready:
            out.append("    По правилу бумаги (4–6 недель, повтор на стенде сошёлся) книга готова "
                       "к решению владельца.")
        if ln.other_books:
            out.append("    Тот же замысел идёт ещё в книгах: " + ", ".join(ln.other_books)
                       + " — они не складываются, ряд берётся один.")
    out.append("")
    if p.held:
        out.append(f"Из общей суммы {_usd(p.held)} уже стоит в живых стратегиях и не трогается; "
                   f"делится остальное: {_usd(p.pool)}.")
    weighted = [ln for ln in p.lines if ln.status == "weighted"]
    if len(weighted) == 1:
        out.append("Делить удаётся между одной стратегией: это не набор из нескольких, "
                   "а вывод «остальные не помещаются или не набрали данных».")
    if p.correlations:
        out.append("Как стратегии ходят вместе (на общих днях пары):")
        names = {ln.root: ln.label for ln in p.lines}
        for c in p.correlations:
            out.append(f"    {names[c['a']]} и {names[c['b']]}: {c['correlation']:+.2f} за "
                       f"{c['n']} {_days_word(c['n'])} ({_day(c['first'])}–{_day(c['last'])}); "
                       "в доли не входит.")
    if p.correlation_note:
        out.append(f"    {p.correlation_note[0].upper()}{p.correlation_note[1:]}.")
    if p.ladder:
        out.append("")
        out.append("Что получилось бы при другой общей сумме:")
        names = {ln.root: ln.label for ln in p.lines}
        for step, row in p.ladder.items():
            got = [f"{names[r]} {_usd(d)}" for r, d in row.items() if d > 0]
            out.append(f"    {_usd(float(step))}: " + ("; ".join(got) if got else "никому"))
    out += ["", "Что изменилось с прошлой ночи (поводы пересчитать деление):"]
    out += [f"    {c}" for c in changes]
    soon = sorted((ln.ready_on, ln.label) for ln in p.lines if ln.ready_on)
    if soon:
        out.append("Когда у стратегий наберётся минимум данных: "
                   + "; ".join(f"{label} — {_day(day)}" for day, label in soon) + ".")
    out += ["", "Нужно решение владельца (кодом не выбирается):"]
    out += [f"    {n}" for n in p.needs_decision]
    out += _held_warning(p)
    out += [
        "    Общая сумма для стратегий: $1 000, $3 000, $5 000 или $15 000 (названные вами цифры "
        "разные; в расчёте $3 000).",
        "    Потолок на каждую стратегию: в расчёте его нет, и считается, что доходность на доллар "
        "не меняется с размером. Вариант: вы называете потолок для каждой; для тренда есть "
        "измерение ёмкости (`docs/T27_LIQUIDITY.md`), для остальных нет.",
        "    Что делать при тревоге детектора затухания: порог тревоги не задан "
        "(`night/decay.yaml`). Варианты: только показывать (так сейчас) или исключать стратегию "
        "и делить заново (`on_decay_alert` в `night/allocation.yaml`).",
        "    Как часто пересчитывать: сейчас только по поводам из списка выше. Варианты: по "
        "календарю (например, раз в месяц) или при сдвиге доли больше чем на заданное число "
        "пунктов — число задаёте вы.",
    ]
    return out


# ---------------------------------------------------------------------------
# Building the strategies from the night's data
# ---------------------------------------------------------------------------


def latest_min_capital(session, idea_id: str) -> float | None:
    """`min_capital_usd` of the idea's latest evaluated trial (the stand's measurement), or None."""
    from qlab.registry.models import Spec, Trial, TrialStatus

    rows = (session.query(Trial.metrics).join(Spec, Spec.id == Trial.spec_id)
            .filter(Spec.idea_id == idea_id, Trial.status == TrialStatus.OK,
                    Trial.metrics.isnot(None)).order_by(Trial.id.desc()).limit(50).all())
    for (metrics,) in rows:
        value = (metrics or {}).get("min_capital_usd")
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def book_evidence(idea_id: str, cfg: dict, *, root: Path | None = None) -> Evidence | None:
    """The realised daily returns of one book on its own money; None when the book has no export
    yet. Refuses (MoneyError) a book that is not on strategy money."""
    assert_strategy_money(idea_id, cfg)
    csv = paper_rule.latest_export(root if root is not None else paper_rule.PAPER_DIR,
                                   cfg["paper_file"])
    if csv is None:
        return None
    curve = equity_curve(csv, column=cfg.get("equity_column", "equity"),
                         strategy_id=cfg.get("strategy_id"), sum_coins=bool(cfg.get("sum_coins")))
    since = decay._as_datetime(cfg.get("realised_from"))
    if since is not None:
        curve = curve[curve.index >= pd.Timestamp(since)]
    if len(curve) < 2:
        return None
    if cfg.get("equity_from") == decay.STRATEGY_PNL:
        problem = decay.capital_mismatch(cfg, float(curve.iloc[0]), tolerance=1e-6)
        if problem:
            raise MoneyError(f"{idea_id}: {problem}")
    kind = "live" if cfg.get("live") else "paper"
    return Evidence(kind=kind, returns=daily_returns(curve), source=f"{cfg['paper_file']}")


def spec_path_for_idea(idea_id: str, specs: Path = Path("specs"),
                       watchlist: Path = Path("night/watchlist.yaml")) -> Path | None:
    """The spec file whose `idea_id` is the given one. Several specs can name one idea (the
    transcription and the live-code version): the one the night follows (night/watchlist.yaml)
    wins, then one that states `params_fixed_at`, then the last by name."""
    pattern = re.compile(rf"^idea_id:\s*{re.escape(idea_id)}\s*$", re.M)
    found = sorted(p for p in specs.rglob("*.yaml")
                   if pattern.search(p.read_text(encoding="utf-8")))
    if not found:
        return None
    followed = set()
    if watchlist.is_file():
        followed = {Path(x) for x in (yaml.safe_load(watchlist.read_text(encoding="utf-8")) or {})
                    .get("specs") or []}
    for tier in ([p for p in found if p in followed],
                 [p for p in found if re.search(r"^params_fixed_at:", p.read_text(encoding="utf-8"),
                                                re.M)],
                 found):
        if tier:
            return tier[-1]
    return None


def stand_forward_evidence(session, idea_id: str, *, end: date) -> Evidence | None:
    """The stand's run of an idea that has no book, from the spec's `params_fixed_at` on, folded
    to daily returns. A spec without `params_fixed_at` has only the fit period: its evidence is
    `in_sample`, which the allocation refuses to weigh. None when there is no spec or panel.
    Writes no trial."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.night import with_end
    from qlab.pipeline.evaluate import resolve_cached_panel, resolve_strategy
    from qlab.pipeline.spec import load_spec

    path = spec_path_for_idea(idea_id)
    if path is None:
        return None
    spec = with_end(load_spec(str(path)), end)
    if spec.params_fixed_at is None:
        return Evidence("in_sample", pd.Series(dtype=float), str(path))
    panel = resolve_cached_panel(session, spec)
    if panel is None:
        return None
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    net = run_backtest(panel, weights, costs, panel.funding).net_return
    forward = net[net.index >= pd.Timestamp(spec.params_fixed_at, tz="UTC")]
    if forward.empty:
        return Evidence("stand_forward", pd.Series(dtype=float), str(path))
    index = panel.prices.index
    bar = index[1] - index[0]
    day = (forward.index + bar - pd.Timedelta(microseconds=1)).normalize()
    daily = (1.0 + forward).groupby(day).prod() - 1.0
    daily = daily[daily.index + pd.Timedelta(days=1) <= index[-1] + bar]
    return Evidence("stand_forward", daily, str(path))


def build_strategies(session, *, books: dict[str, dict] | None = None, root: Path | None = None,
                     decay_results: list[dict] | None = None,
                     paper_rules: list | None = None, end: date | None = None,
                     stand_fn=None) -> tuple[list[Strategy], list[str]]:
    """The pool's strategies from the books, the registry and the stand, and report lines for
    anything that could not be read (one failing book does not stop the others)."""
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.registry.strategies import strategy_root

    books = books if books is not None else decay.load_all_books()
    alerts = {r["idea_id"]: r for r in decay_results or []}
    rules = {r.idea_id: r for r in paper_rules or []}
    problems: list[str] = []
    per_root: dict[str, list[tuple[str, dict, Evidence | None]]] = {}
    for idea_id, cfg in books.items():
        try:
            evidence = book_evidence(idea_id, cfg, root=root)
        except MoneyError as exc:
            problems.append(f"{cfg.get('label', idea_id)}: ряд не взят — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - a report line, the others go on
            problems.append(f"{cfg.get('label', idea_id)}: ряд не прочитан "
                            f"({type(exc).__name__}: {exc})")
            continue
        per_root.setdefault(strategy_root(session, idea_id), []).append((idea_id, cfg, evidence))

    strategies: list[Strategy] = []
    for root_id, group in per_root.items():
        def key(item, root_id=root_id):
            idea_id, cfg, ev = item
            return (not cfg.get("live"), -(ev.days if ev else 0), idea_id != root_id, idea_id)

        group.sort(key=key)
        idea_id, cfg, evidence = group[0]
        root_idea = session.get(Idea, root_id)
        excluded = None
        if root_idea is not None and str(root_idea.status) in TERMINAL_STATUSES:
            excluded = f"стратегия в статусе «{root_idea.status}»: в пул не входит"
        decayed = alerts.get(idea_id) or {}
        rule = rules.get(idea_id)
        strategies.append(Strategy(
            root=root_id, idea_id=idea_id, label=cfg.get("label", idea_id),
            role="live" if cfg.get("live") else "paper", evidence=evidence,
            min_capital_usd=latest_min_capital(session, idea_id),
            current_capital_usd=(float(cfg["strategy_capital_usd"]) if cfg.get("live") else None),
            other_books=[cfg2.get("label", i2) for i2, cfg2, _ in group[1:]],
            decay_alert=decayed.get("alert"),
            decay_percentile=((decayed.get("ret") or {}).get("percentile")),
            paper_ready=(rule.ready if rule is not None else None), excluded=excluded))

    # Strategies with no book that passed the stand and wait for the owner.
    from qlab.registry.models import Spec, Trial, TrialRoute, TrialStatus

    stand_fn = stand_fn or stand_forward_evidence
    for idea in (session.query(Idea).filter(Idea.status.in_([IdeaStatus.VALIDATED]))
                 .order_by(Idea.id).all()):
        root_id = strategy_root(session, idea.id)
        if root_id in per_root or any(s.root == root_id for s in strategies):
            continue
        passed = (session.query(Trial.id).join(Spec, Spec.id == Trial.spec_id)
                  .filter(Spec.idea_id == idea.id, Trial.status == TrialStatus.OK,
                          Trial.route == TrialRoute.PAPER).first())
        if passed is None:
            continue
        try:
            evidence = stand_fn(session, idea.id, end=end or date.today())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{idea.id}: ряд стенда не построен ({type(exc).__name__}: {exc})")
            continue
        strategies.append(Strategy(
            root=root_id, idea_id=idea.id, label=idea.id, role="stand", evidence=evidence,
            min_capital_usd=latest_min_capital(session, idea.id)))
    return strategies, problems


# ---------------------------------------------------------------------------
# The night step and the saved state
# ---------------------------------------------------------------------------


def latest_allocation(night_dir: Path | None = None, *, before: date | None = None
                      ) -> dict | None:
    """The newest saved night's allocation state (optionally strictly before `before`)."""
    folder = night_dir if night_dir is not None else NIGHT_DIR
    for path in sorted(folder.glob("*.json"), reverse=True):
        if before is not None and path.stem >= before.isoformat():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8")).get("allocation")
        except (OSError, ValueError):
            continue
        if state and state.get("proposal"):
            return dict(state, at=path.stem)
    return None


def run_allocation(session, day: date, end: date, *, books: dict[str, dict] | None = None,
                   config: dict | None = None, root: Path | None = None,
                   decay_results: list[dict] | None = None, paper_rules: list | None = None,
                   night_dir: Path | None = None, stand_fn=None) -> dict:
    """The night's allocation proposal: {"lines": report lines, "proposal": dict,
    "snapshot": dict, "changes": [...]}. Reads the night's exports, the registry and (for
    strategies without a book) the cached stand panels; writes nothing."""
    config = config if config is not None else load_config()
    strategies, problems = build_strategies(
        session, books=books, root=root, decay_results=decay_results, paper_rules=paper_rules,
        end=end, stand_fn=stand_fn)
    proposal = allocate(strategies, capital=float(config["deployable_capital_usd"]),
                        max_count=max_strategies(config), config=config)
    previous = latest_allocation(night_dir, before=day)
    changes = changes_since((previous or {}).get("snapshot") if previous else None, proposal)
    lines = report_lines(proposal, changes) + [f"    {p}" for p in problems]
    return {"lines": lines, "proposal": proposal.to_dict(), "snapshot": snapshot(proposal),
            "changes": changes}


def night_signals(session, state: dict | None, *, night_dir: Path | None = None,
                  root: Path | None = None) -> tuple[list[dict], list]:
    """The decay detector's results and the paper rule's states the proposal shows beside each
    strategy: from tonight's `state` when it holds them, else from the newest saved night."""
    state = state or {}
    results = (state.get("decay") or {}).get("results")
    if results is None:
        results = []
        for path in sorted((night_dir or NIGHT_DIR).glob("*.json"), reverse=True):
            try:
                found = (json.loads(path.read_text(encoding="utf-8")).get("decay") or {})
            except (OSError, ValueError):
                continue
            if found.get("results"):
                results = found["results"]
                break
    lines = state.get("paper") or paper_rule.latest_replay_lines(night_dir)
    rules, _ = paper_rule.paper_rules(session, root=root, replay_lines=lines)
    return results, rules


def night_stage(day: date, end: date, state: dict, *, root: Path | None = None) -> None:
    """The night's allocation proposal into `state["allocation"]` (report lines, the proposal,
    the snapshot the next night compares with). A failure is a report line, never a failed
    night. Reads exports, the registry and cached panels; writes nothing but the state."""
    from qlab.registry.db import session_scope

    try:
        with session_scope() as session:
            results, rules = night_signals(session, state, root=root)
            state["allocation"] = run_allocation(session, day, end, root=root,
                                                 decay_results=results, paper_rules=rules)
    except Exception as exc:  # noqa: BLE001
        state["allocation"] = {"lines": [f"не посчитано: {type(exc).__name__}: {exc}"],
                               "proposal": None, "snapshot": {}, "changes": []}


__all__ = [
    "CONFIG", "Evidence", "Line", "MoneyError", "Proposal", "SECTION_TITLE", "Strategy",
    "allocate", "assert_strategy_money", "book_evidence", "build_strategies", "changes_since",
    "correlation_table", "daily_returns", "daily_std", "inverse_volatility", "latest_allocation",
    "latest_min_capital", "load_config", "max_strategies", "night_signals", "night_stage",
    "report_lines", "run_allocation",
    "snapshot", "stand_forward_evidence",
]
