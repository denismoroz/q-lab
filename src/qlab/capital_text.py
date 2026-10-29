"""The strategy's own money in words, for the reports (docs/CAPITAL_BASE.md).

A run stores its summaries twice (`qlab.harness.metrics.capital_metrics`): the stand's, a share
of the book's own equity compounded bar to bar, and the `cap_` twins, a share of the capital the
strategy started with, summed when the strategy does not reinvest.

Since 2026-10-29 (owner: «мы вроде как решили отделять апельсины от яблок но тут все еще разная
доходность показана ... было бы здорово пофиксить это везде»; rules 2026-10-29.2 read the same
figures) the LEADING number everywhere the owner reads is the strategy's own money's:
`headline(m)` takes the `cap_` figure when the run has it and the stand's only for a run that
has none (an older run, or a base that is undefined), and says which. The stand's figure follows,
small, as «как считает стенд (от баланса книги)», only where it differs from the first
(`stand_suffix`): where they are the same (trend, most generated ideas: base 1, reinvesting)
nothing is added, so a line is not doubled for nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CAP_NUMBERS = ("cap_ann_return", "cap_total_return", "cap_sharpe", "cap_max_dd",
               "cap_ann_return_ex_best_1pct", "cap_reinvested", "cap_base_refused",
               "noise_cap_return_percentile", "noise_cap_power", "fit_cap_ann_return",
               "fit_cap_sharpe", "fit_cap_max_dd", "pre_cap_ann_return", "pre_cap_sharpe")
"""The capital summaries the reports carry along with the stand's (`RESULT_METRICS` and the
like)."""
TOLERANCE = 0.0005
"""Two figures closer than this (a twentieth of a percentage point) are the same figure."""


def _number(m: Mapping[str, object], key: str) -> float | None:
    value = m.get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


def _pct(value: float) -> str:
    return f"{value:+.1%}".replace("-", "−")


def differs(m: Mapping[str, object] | None) -> bool:
    """Whether the run's capital summaries say something the stand's do not: it has them, and
    the yearly figure or the worst fall is not the same figure."""
    m = m or {}
    for stand, cap in (("ann_return_net", "cap_ann_return"), ("max_dd", "cap_max_dd")):
        a, b = _number(m, stand), _number(m, cap)
        if a is not None and b is not None and abs(a - b) > TOLERANCE:
            return True
    return False


@dataclass(frozen=True)
class Headline:
    """The numbers a reader sees first, on the strategy's own money when the run has them."""

    ann: float | None
    sharpe: float | None
    max_dd: float | None
    total: float | None
    ex_best_1pct: float | None
    noise: float | None
    on_capital: bool
    """True: the figures are the `cap_` ones (a percent of the capital the strategy started
    with); False: the run has no such figures and these are the stand's."""
    reinvested: bool | None


def headline(m: Mapping[str, object] | None, prefix: str = "") -> Headline:
    """The run's leading numbers (`prefix`: "fit_" or "pre_" for a part's copy of the metrics).
    The capital set is used whole or not at all: a run with `cap_ann_return` shows every `cap_`
    figure it has and never fills a gap from the stand's, so one line never mixes two bases."""
    m = m or {}
    capital = _number(m, f"{prefix}cap_ann_return") is not None
    if capital:
        flag = _number(m, "cap_reinvested")
        return Headline(
            ann=_number(m, f"{prefix}cap_ann_return"), sharpe=_number(m, f"{prefix}cap_sharpe"),
            max_dd=_number(m, f"{prefix}cap_max_dd"),
            total=_number(m, f"{prefix}cap_total_return"),
            ex_best_1pct=_number(m, f"{prefix}cap_ann_return_ex_best_1pct"),
            noise=_number(m, "noise_cap_return_percentile") if not prefix else None,
            on_capital=True, reinvested=None if flag is None else bool(flag))
    return Headline(
        ann=_number(m, f"{prefix}ann_return_net"), sharpe=_number(m, f"{prefix}sharpe_net"),
        max_dd=_number(m, f"{prefix}max_dd"), total=_number(m, f"{prefix}total_return_net"),
        ex_best_1pct=_number(m, f"{prefix}ann_return_net_ex_best_1pct"),
        noise=_number(m, "noise_return_percentile") if not prefix else None,
        on_capital=False, reinvested=None)


def ann_of(m: Mapping[str, object] | None) -> float | None:
    """The yearly return a reader compares across runs: the strategy's own money's, else the
    stand's."""
    return headline(m).ann


def numbers_ru(m: Mapping[str, object] | None, *, noise: str = "always",
               ex_best: bool = False) -> str:
    """«в год +8.4%, Шарп 0.53, просадка −19.3%[, обгоняет шум 98%]» on the strategy's own money,
    followed by the stand's figures when they differ (`stand_suffix`). `noise`: "always" (a dash
    when there is no figure), "present" (only when there is one) or "never"."""
    h = headline(m)
    parts = [f"в год {_pct(h.ann) if h.ann is not None else '—'}",
             f"Шарп {h.sharpe:.2f}" if h.sharpe is not None else "Шарп —",
             f"просадка {_pct(h.max_dd) if h.max_dd is not None else '—'}"]
    if ex_best and h.ex_best_1pct is not None:
        parts.append(f"без лучшего 1% дней {_pct(h.ex_best_1pct)}")
    if noise == "always" or (noise == "present" and h.noise is not None):
        parts.append("обгоняет шум "
                     + (f"{h.noise:.0%}" if h.noise is not None else "—"))
    return ", ".join(parts) + stand_suffix(m)


def stand_suffix(m: Mapping[str, object] | None) -> str:
    """«; как считает стенд (от баланса книги): в год +2.0%, Шарп 0.55, просадка −4.5%» when the
    stand's figures say something the strategy's own money's do not, else an empty string. Small
    print: the leading figures are the strategy's own money's."""
    if not m or not differs(m):
        return ""
    parts = []
    ann, dd, sharpe = _number(m, "ann_return_net"), _number(m, "max_dd"), _number(m, "sharpe_net")
    if ann is not None:
        parts.append(f"в год {_pct(ann)}")
    if sharpe is not None:
        parts.append(f"Шарп {sharpe:.2f}")
    if dd is not None:
        parts.append(f"просадка {_pct(dd)}")
    return "; как считает стенд (от баланса книги): " + ", ".join(parts)


def basis_ru(m: Mapping[str, object] | None) -> str:
    """The base of the leading figures in words, for a line that wants to say it:
    «от денег стратегии, прибыль не вкладывается обратно (считается суммой)» or, for a run with no
    figures on the strategy's own money, «по счёту стенда (у прогона нет чисел на деньгах
    стратегии)»."""
    h = headline(m)
    if (m or {}).get("cap_base_refused"):
        return ("по счёту стенда: основа процентов у этой книги не определена (в ней пополнения), "
                "чисел на деньгах стратегии нет, и эти числа нельзя сравнивать с другими идеями")
    if not h.on_capital:
        return "по счёту стенда (у прогона нет чисел на деньгах стратегии)"
    how = ("прибыль не вкладывается обратно, считается суммой" if h.reinvested is False
           else "прибыль вкладывается, считается со сложным процентом")
    return f"от денег стратегии, {how}"


def capital_suffix(m: Mapping[str, object] | None) -> str:
    """«; на деньгах стратегии: в год +8.4%, просадка −19.3%» when the capital summaries differ
    from the stand's, else an empty string. Plain Russian; the stand's own words stay where they
    are. Kept for a line that leads with the stand's figures (an old run's); the reports now lead
    with `numbers_ru`."""
    if not m or not differs(m):
        return ""
    parts = []
    ann, dd = _number(m, "cap_ann_return"), _number(m, "cap_max_dd")
    sharpe = _number(m, "cap_sharpe")
    if ann is not None:
        parts.append(f"в год {_pct(ann)}")
    if sharpe is not None:
        parts.append(f"Шарп {sharpe:.2f}")
    if dd is not None:
        parts.append(f"просадка {_pct(dd)}")
    flag = _number(m, "cap_reinvested")
    how = ("прибыль не вкладывается обратно, считается суммой" if flag == 0.0
           else "прибыль вкладывается, считается со сложным процентом")
    return f"; на деньгах стратегии ({how}): " + ", ".join(parts)
