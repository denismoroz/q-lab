"""What a strategy's forward series says about WHEN to be in it (docs/OVERLAYS.md).

Owner, 2026-10-28: «возможно имеет смысл подумать про точку входа по стратегии». Before any
overlay is built the series is looked at in three ways. **They describe; they decide nothing, and
no overlay is ever skipped because of them** (the stand's numbers decide, never one reading).

1. **Entry day.** What would an investor have had after holding 30, 60 or 90 days if they had
   started on each possible day of the judged part: the worst, the median and the best total,
   and the share of starts that ended in profit; for the strategy and for simply holding the
   coins it holds. A wide spread between worst and best says the day of entry matters a lot.
2. **Does a good period predict the next one?** The judged part is cut into whole,
   non-overlapping weeks, months (30 days) and quarters (91 days); the correlation between a
   period's return and the following period's return is the answer. A correlation inside the
   noise band (plus or minus two over the square root of the number of pairs) is
   indistinguishable from none; with fewer than eight pairs it is not defined.
3. **By market state known at the time.** The strategy's daily return split by BTC's 30-day
   direction (bull / flat / bear: the terciles of the days before, `qlab.strategies.detectors`)
   and by BTC's volatility (above or below the median of its own past), each state taken from
   what was known when the day began.

The series is the judged part's net daily return, from the same back-test the stand's comparison
uses (`qlab.overlay_compare`).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.overlay_compare import DAYS_PER_YEAR, store_dir

HOLD_DAYS = (30, 60, 90)
"""The holding periods of the entry-day reading: one, two and three months (the regime window
and its multiples, `qlab.regimes.WINDOW_DAYS`)."""
PERIOD_DAYS = {"week": 7, "month": 30, "quarter": 91}
MIN_PAIRS = 8
"""Fewer pairs of consecutive periods say nothing about a correlation."""
VOL_DAYS = 30


def _compound(daily: pd.Series) -> float:
    return float(np.prod(1.0 + daily.to_numpy(dtype=float)) - 1.0)


# --------------------------------------------------------------------------
# 1. From every possible entry day


def entry_day_spread(daily: pd.Series, hold_days: int) -> dict | None:
    """Worst / median / best total return after holding `hold_days` days from every start day,
    and the share of starts that ended above zero. None when the series is shorter."""
    values = np.log1p(daily.to_numpy(dtype=float))
    n = len(values)
    if n < hold_days:
        return None
    cum = np.r_[0.0, np.cumsum(values)]
    totals = np.expm1(cum[hold_days:] - cum[:-hold_days])
    return {"starts": int(len(totals)), "worst": float(totals.min()),
            "median": float(np.median(totals)), "best": float(totals.max()),
            "share_positive": float((totals > 0).mean())}


def entry_days(net: pd.Series, hold: pd.Series | None) -> dict[str, dict]:
    return {str(h): {"strategy": entry_day_spread(net, h),
                     "hold": None if hold is None else entry_day_spread(hold, h)}
            for h in HOLD_DAYS}


# --------------------------------------------------------------------------
# 2. Does a good period predict the next


def period_returns(daily: pd.Series, days: int) -> np.ndarray:
    """Compounded return of each whole, non-overlapping block of `days` days, oldest first
    (the incomplete first block is dropped, so every block ends on the last day's side)."""
    values = daily.to_numpy(dtype=float)
    blocks = len(values) // days
    if blocks == 0:
        return np.array([])
    tail = values[len(values) - blocks * days:].reshape(blocks, days)
    return np.prod(1.0 + tail, axis=1) - 1.0


def persistence(daily: pd.Series) -> dict[str, dict]:
    out = {}
    for name, days in PERIOD_DAYS.items():
        returns = period_returns(daily, days)
        pairs = len(returns) - 1
        entry: dict = {"period_days": days, "periods": int(len(returns)), "pairs": max(pairs, 0),
                       "correlation": None, "noise_band": None, "distinguishable": None}
        if pairs >= MIN_PAIRS and returns[:-1].std() > 0 and returns[1:].std() > 0:
            corr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
            band = 2.0 / math.sqrt(pairs)
            entry.update(correlation=corr, noise_band=band, distinguishable=abs(corr) > band)
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# 3. By market state known at the time


def _state_stats(daily: pd.Series) -> dict:
    n = len(daily)
    if n == 0:
        return {"days": 0, "total_return": None, "ann_return": None, "mean_daily": None,
                "share_positive": None}
    total = _compound(daily)
    return {"days": n, "total_return": total,
            "ann_return": -1.0 if total <= -1 else float((1 + total) ** (DAYS_PER_YEAR / n) - 1),
            "mean_daily": float(daily.mean()), "share_positive": float((daily > 0).mean())}


def direction_states(closes: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """BTC's 30-day direction known when each day began: terciles of the days before
    (`qlab.strategies.detectors.direction_terciles`), the last label stamped at or before
    the day."""
    from qlab.strategies.detectors import direction_terciles

    labels = direction_terciles(closes).dropna()
    return labels.reindex(index, method="ffill")


def volatility_states(closes: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """`high` when BTC's 30-day volatility is above the median of its own past values, `low`
    otherwise; unknown until a month of past values exists."""
    vol = np.log(closes).diff().rolling(VOL_DAYS, min_periods=VOL_DAYS).std()
    usual = vol.shift(1).expanding(min_periods=VOL_DAYS).median()
    state = pd.Series(np.where(vol > usual, "high", "low"), index=closes.index, dtype=object
                      ).where(vol.notna() & usual.notna())
    return state.dropna().reindex(index, method="ffill")


def by_state(daily: pd.Series, states: pd.Series, order: tuple[str, ...]) -> dict[str, dict]:
    known = states.reindex(daily.index)
    out = {name: _state_stats(daily[known == name]) for name in order}
    out["unknown"] = _state_stats(daily[known.isna()])
    return out


def market_states(net: pd.Series, closes: pd.Series | None) -> dict | None:
    if closes is None:
        return None
    return {"direction": by_state(net, direction_states(closes, net.index),
                                  ("bull", "flat", "bear")),
            "volatility": by_state(net, volatility_states(closes, net.index), ("high", "low"))}


# --------------------------------------------------------------------------
# Putting it together


def build(idea_id: str, net: pd.Series, hold: pd.Series | None, closes: pd.Series | None, *,
          trial_id: int | None = None) -> dict:
    """The three readings of one forward series (module docstring)."""
    if len(net) < 2:
        raise ValueError("no judged days to describe")
    return {
        "idea_id": idea_id, "trial_id": trial_id,
        "window": {"start": net.index[0].date().isoformat(),
                   "end": net.index[-1].date().isoformat(), "days": int(len(net))},
        "entry_days": entry_days(net, hold),
        "persistence": persistence(net),
        "market_state": market_states(net, closes),
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def compute_for_spec(spec, session, *, trial_id: int | None = None) -> dict:
    """Back-test `spec` on its snapshot and describe its judged part."""
    from qlab.overlay_compare import daily_from, prepare
    from qlab.regimes import market_closes

    run = prepare(spec, session)
    net, hold = daily_from(run, run.judged_from_row)
    return build(spec.idea_id, net, hold, market_closes(), trial_id=trial_id)


def path_for(idea_id: str, store: Path | None = None) -> Path:
    return (store or store_dir()) / "diagnostics" / f"{idea_id}.json"


def save(payload: dict, store: Path | None = None) -> Path:
    path = path_for(str(payload["idea_id"]), store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load(idea_id: str, store: Path | None = None) -> dict | None:
    try:
        return json.loads(path_for(idea_id, store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Words for the owner (console and report)


def pct(value: object, digits: int = 1) -> str:
    return f"{value:+.{digits}%}" if isinstance(value, int | float) and value == value else "—"


def lines_ru(payload: dict) -> list[str]:
    """The three readings in plain words, for the morning report and the CLI."""
    out = []
    window = payload["window"]
    out.append(f"Отрезок: {window['start']} — {window['end']}, {window['days']} дн. "
               "Это описание: оно ничего не решает, и ни одна надстройка из-за него не "
               "пропускается.")
    for h, row in payload["entry_days"].items():
        s, hold = row["strategy"], row["hold"]
        if s is None:
            out.append(f"- Войти в любой день и держать {h} дн.: отрезок короче.")
            continue
        line = (f"- Войти в любой день и держать {h} дн. ({s['starts']} возможных дней входа): "
                f"худший итог {pct(s['worst'])}, средний {pct(s['median'])}, лучший "
                f"{pct(s['best'])}, в плюсе {s['share_positive']:.0%} входов")
        if hold:
            line += (f"; просто держать те же монеты: худший {pct(hold['worst'])}, лучший "
                     f"{pct(hold['best'])}")
        out.append(line)
    names = {"week": "недели", "month": "месяцы", "quarter": "кварталы"}
    for key, row in payload["persistence"].items():
        if row["correlation"] is None:
            out.append(f"- Предсказывает ли хороший период следующий ({names[key]}): "
                       f"не определено, пар периодов {row['pairs']} (нужно хотя бы {MIN_PAIRS}).")
        else:
            verdict = ("связь заметна" if row["distinguishable"]
                       else "связи нет: в пределах случайного разброса")
            out.append(f"- Предсказывает ли хороший период следующий ({names[key]}): "
                       f"корреляция {row['correlation']:+.2f} при {row['pairs']} парах, "
                       f"разброс случайного ±{row['noise_band']:.2f}: {verdict}.")
    states = payload.get("market_state")
    if states:
        title = {"bull": "рост", "flat": "боковик", "bear": "падение",
                 "high": "высокая изменчивость", "low": "низкая изменчивость",
                 "unknown": "состояние ещё не известно"}
        for group, label in (("direction", "по направлению BTC за 30 дней"),
                             ("volatility", "по изменчивости BTC")):
            parts = [f"{title[k]} — {v['days']} дн., в год {pct(v['ann_return'])}"
                     for k, v in states[group].items() if v["days"]]
            out.append(f"- Результат {label} (состояние, известное на начало дня): "
                       + "; ".join(parts))
    return out


__all__ = ["HOLD_DAYS", "MIN_PAIRS", "PERIOD_DAYS", "build", "by_state", "compute_for_spec",
           "direction_states", "entry_day_spread", "entry_days", "lines_ru", "load",
           "market_states", "path_for", "period_returns", "persistence", "save",
           "volatility_states"]
