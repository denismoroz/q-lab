"""The strategy's own money in words, for the reports (docs/CAPITAL_BASE.md).

A run stores its summaries twice (`qlab.harness.metrics.capital_metrics`): the stand's, a share
of the book's own equity compounded bar to bar, and the `cap_` twins, a share of the capital the
strategy started with, summed when the strategy does not reinvest. Where the two differ the
reports show the second beside the first; where they are the same (trend, most generated
ideas: base 1, reinvesting) nothing is added, so a line is not doubled for nothing.
"""

from __future__ import annotations

from collections.abc import Mapping

CAP_NUMBERS = ("cap_ann_return", "cap_total_return", "cap_sharpe", "cap_max_dd",
               "cap_ann_return_ex_best_1pct", "cap_reinvested")
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


def capital_suffix(m: Mapping[str, object] | None) -> str:
    """«; на деньгах стратегии: в год +8.4%, просадка −19.3%» when the capital summaries differ
    from the stand's, else an empty string. Plain Russian; the stand's own words stay where they
    are."""
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
